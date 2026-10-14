from pathlib import Path
import os,json,time,subprocess,signal,fcntl,hashlib,traceback
N=Path(__file__).resolve().parent;R=N.parents[1];M=R/'runs/pi05_debug_until_success_20261013'
plan=json.loads((N/'plan.json').read_text());T=Path(plan['training_run']);H=Path(plan['parent_handoff']);E=Path(plan['future_evaluation_run'])
lock=(N/'probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
policy=None;coordinator=None;paused=False
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def alive(pid,match=None):
 p=Path(f'/proc/{pid}/cmdline')
 if not p.exists():return False
 try:b=p.read_bytes()
 except FileNotFoundError:return False
 return bool(b) and (match is None or match.encode() in b)
def phase(state,**kw):
 d=dict(state=state,driver_pid=os.getpid(),probe_run=str(N),checkpoint_step=37000,training_run=str(T),future_evaluation_run=str(E),updated_at=time.time(),**kw)
 save(N/'driver_status.json',d);save(M/'status.json',d)
def stop(pid,match,group=False,interrupt=False):
 if not alive(pid,match):return
 sig=signal.SIGINT if interrupt else signal.SIGTERM
 (os.killpg if group else os.kill)(pid,sig)
 for _ in range(60):
  if not alive(pid,match):return
  time.sleep(1)
 raise RuntimeError(f'Process {pid} did not exit; no overlapping GPU job started')
def free_gpu():
 for _ in range(120):
  x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if all(int(v)<500 for v in x.splitlines()):return
  time.sleep(2)
 raise RuntimeError('GPU resources did not release')
def verify():
 m=read(N/'manifest.json')
 for path,d in m['shared_source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==d,path
 for name,d in m['code_sha256'].items():assert hashlib.sha256((N/'code'/name).read_bytes()).hexdigest()==d,name
 for name,d in m['input_sha256'].items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==d,name
 assert hashlib.sha256((N/'checkpoints/37000/_CHECKPOINT_METADATA').read_bytes()).hexdigest()==plan['checkpoint_metadata_sha256']
 return m
try:
 m=verify()
 assert alive(plan['training_pid'],str(T/'code/train_full_8gpu.py'))
 assert alive(plan['parent_driver_pid'],str(H/'continue_after_step37000_probe.py'))
 phase('pausing_training')
 stop(plan['parent_driver_pid'],str(H/'continue_after_step37000_probe.py'))
 phase('waiting_training_checkpoint_to_finish')
 for _ in range(600):
  current=read(T/'status.json')
  if current['state']=='failed':raise RuntimeError('Training failed before pause')
  if current['state']=='training':break
  time.sleep(1)
 else:raise RuntimeError('Training checkpoint or initialization has not finished; will not interrupt a save')
 last=read(T/'status.json')
 saved=read(T/'checkpoint_status.json')
 assert saved['saved']
 resume_step=int(saved['step'])
 assert (T/'checkpoints'/str(resume_step)/'_CHECKPOINT_METADATA').exists()
 save(N/'pause_evidence/last_training_status.json',last)
 stop(plan['training_pid'],str(T/'code/train_full_8gpu.py'),interrupt=True)
 paused=True
 receipt=dict(state='paused_for_five_task_probe',resume_step=resume_step,last_logged_step=last.get('step'),target_steps=46269,previous_training_pid=plan['training_pid'],probe_run=str(N),automatic_resume=True,updated_at=time.time())
 save(T/'status.json',receipt);save(N/'pause_receipt.json',receipt)
 save(M/'receipt.json',dict(driver_pid=os.getpid(),training_pid=None,training_run=str(T),evaluation_run=str(N),future_evaluation_run=str(E),training_mode='paused_for_50episode_probe',checkpoint=str(N/'checkpoints/37000'),resume_step=resume_step))
 phase('waiting_gpu_release');free_gpu()
 env=os.environ.copy()
 for key in ['LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','XLA_FLAGS','JAX_PLATFORMS','LD_PRELOAD','CUDA_VISIBLE_DEVICES']:env.pop(key,None)
 env.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME=str(R/'cache/openpi'),JAX_COMPILATION_CACHE_DIR=str(R/'cache/jax'),LD_PRELOAD=str(R/'runtime_overrides/nccl_2289/nvidia/nccl/lib/libnccl.so.2'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (N/'logs/policy.log').open('a') as f:policy=subprocess.Popen([str(R/'env/bin/python'),str(N/'code/policy_server.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 phase('inference_preflight',policy_pid=policy.pid)
 for _ in range(900):
  if (N/'policy_ready.json').exists() and read(N/'policy_ready.json')['pid']==policy.pid:break
  if policy.poll() is not None:raise RuntimeError('Probe policy failed; see policy log')
  time.sleep(1)
 else:raise RuntimeError('Policy initialization timeout')
 assert read(N/'preflight/inference.json')['passed']
 env2=os.environ.copy()
 for key in ['LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_PLATFORMS']:env2.pop(key,None)
 with (N/'logs/coordinator.log').open('a') as f:coordinator=subprocess.Popen(['python3',str(N/'code/coordinator.py')],env=env2,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 while coordinator.poll() is None:
  if policy.poll() is not None:raise RuntimeError('Policy exited while simulation active')
  st=read(N/'status.json') if (N/'status.json').exists() else {}
  phase('evaluating_five_tasks',policy_pid=policy.pid,coordinator_pid=coordinator.pid,evaluation=st)
  time.sleep(5)
 st=read(N/'status.json');rows=[read(p) for p in (N/'results').glob('*.json')]
 assert coordinator.returncode==0 and st['state']=='complete' and st['valid_completed']==m['total_episodes']
 assert {(x['task'],x['seed']) for x in rows}=={(t,seed) for t in m['evaluation_tasks'] for seed in m['seeds']}
 assert all(x['valid'] and x['video']['full_decode_passed'] and x['video']['decoded_frames']==x['frames'] for x in rows)
 assert all(not x['success'] or (x['geometry']['checked_frames']==x['frames'] and x['geometry']['frames_with_intersections']==0) for x in rows)
 res=dict(completed=True,checkpoint_step=37000,valid_episodes=m['total_episodes'],successes=sum(x['success'] for x in rows),rows=rows,scope='ten development resets per previously failing task, including all five previous first-seed results',finished_at=time.time())
 save(N/'COMPLETED.json',res);phase('probe_complete',successes=res['successes'],valid_episodes=m['total_episodes'])
except Exception:
 save(N/'ERROR.json',dict(error=traceback.format_exc(),time=time.time()))
 phase('probe_infrastructure_error',error=traceback.format_exc())
finally:
 if coordinator is not None and coordinator.poll() is None:stop(coordinator.pid,str(N/'code/coordinator.py'),group=True)
 if policy is not None and policy.poll() is None:stop(policy.pid,str(N/'code/policy_server.py'))
if paused:
 phase('resuming_training');free_gpu()
 out=subprocess.run(['python3',str(T/'launch_8gpu.py')],capture_output=True,text=True,check=True)
 launch=json.loads(out.stdout);pid=launch['pid']
 receipt=dict(training_pid=pid,driver_pid=os.getpid(),training_run=str(T),evaluation_run=str(E),resume_checkpoint=str(T/'checkpoints'/str(resume_step)),resume_step=resume_step,original_resume_step=17000,target_steps=46269,training_mode='seven_task_mixed_resume',global_batch=32,target_equivalent_frame_epochs=10,previous_probe=str(N),optimizer_restore_verification='pending training audit',resumed_at=time.time())
 save(M/'receipt.json',receipt);save(N/'training_resume_receipt.json',receipt)
 phase('training_resumed_after_probe',training_pid=pid)
 os.execv('/usr/bin/python3',['python3',str(H/'continue_after_five_tasks_x10.py')])
