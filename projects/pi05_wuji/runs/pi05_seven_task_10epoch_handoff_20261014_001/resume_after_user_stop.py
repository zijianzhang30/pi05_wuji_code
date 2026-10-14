from pathlib import Path
import os,json,time,subprocess,fcntl,hashlib,traceback,signal
H=Path(__file__).resolve().parent;R=H.parents[1];M=R/'runs/pi05_debug_until_success_20261013'
P=R.parent/'dexterity_living';plan=json.loads((H/'plan.json').read_text());T=Path(plan['training_run']);E=Path(plan['evaluation_run']);F=Path(plan['previous_formal_evaluation'])
guard=(H/'queue.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
policy=None;coordinator=None
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def alive(pid,expected=None):
 p=Path(f'/proc/{pid}/cmdline')
 if not p.exists():return False
 try:b=p.read_bytes()
 except FileNotFoundError:return False
 return bool(b) and (expected is None or expected.encode() in b)
def status(state,**kw):
 x=dict(state=state,driver_pid=os.getpid(),training_run=str(T),evaluation_run=str(E),resume_step=17000,target_steps=46269,global_batch=32,updated_at=time.time(),**kw)
 save(H/'status.json',x);save(M/'status.json',x)
def stop_owned(pid,expected,suspended=False):
 if not alive(pid,expected):return
 os.kill(pid,signal.SIGTERM)
 if suspended:os.kill(pid,signal.SIGCONT)
 for _ in range(30):
  if not alive(pid,expected):return
  time.sleep(1)
 raise RuntimeError(f'Owned process has not exited: {pid}; no new GPU workload started')
def free_gpus(timeout=600):
 until=time.monotonic()+timeout
 while time.monotonic()<until:
  memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if len(memory.splitlines())==8 and all(int(v)<500 for v in memory.splitlines()):return
  time.sleep(5)
 raise RuntimeError('GPU contexts still present; will not start a duplicate GPU workload')
def verify(manifest,folder):
 for path,digest in manifest.get('shared_source_sha256',{}).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 for name,digest in manifest.get('code_sha256',{}).items():assert hashlib.sha256((folder/'code'/name).read_bytes()).hexdigest()==digest,name
 for name,digest in manifest.get('input_sha256',{}).items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
def rows_complete(folder,manifest,st):
 expected={(task,seed) for task in manifest['tasks'] for seed in manifest['seeds']}
 rows=[read(p) for p in (folder/'results').glob('*.json')]
 assert len(rows)==len(expected) and {(x['task'],x['seed']) for x in rows}==expected
 assert st['state']=='complete' and st['valid_completed']==len(expected)
 assert all(x['valid'] and x['video']['full_decode_passed'] and x['video']['decoded_frames']==x['frames'] for x in rows)
 assert all(not x['success'] or (x['geometry']['checked_frames']==x['frames'] and x['geometry']['frames_with_intersections']==0) for x in rows)
 return rows
try:
 stopped=read(F/'STOPPED.json')
 assert stopped['state']=='stopped_by_user' and not stopped['completed']
 status('stopping_previous_policy',previous_valid_episodes=stopped['valid_episodes'],previous_successes=stopped['successes'])
 stop_owned(1076004,str(F/'code/pipeline.py'),suspended=True)
 stop_owned(1076107,str(F/'code/policy_server.py'))
 save(F/'pipeline_status.json',dict(state='stopped_by_user',valid_episodes=stopped['valid_episodes'],previous_pipeline_pid=1076004,automatic_piano_resume_superseded_by_user=True,next_training_run=str(T),stopped_at=time.time()))
 status('waiting_gpu_release');free_gpus()
 for name,digest in read(T/'source_manifest.json')['code_sha256'].items():assert hashlib.sha256((T/'code'/name).read_bytes()).hexdigest()==digest
 assert read(T/'preflight_static.json')['passed']
 out=subprocess.run(['python3',str(T/'launch_8gpu.py')],capture_output=True,text=True,check=True)
 receipt=json.loads(out.stdout);pid=receipt['pid']
 receipt.update(training_pid=pid,driver_pid=os.getpid(),training_run=str(T),evaluation_run=str(E),resume_checkpoint=plan['resume_checkpoint'],resume_step=17000,target_steps=46269,training_mode='seven_task_mixed_resume',global_batch=32,target_equivalent_frame_epochs=10,previous_evaluation_stop_receipt=str(F/'STOPPED.json'),optimizer_restore_verification='pending training audit')
 save(M/'receipt.json',receipt);save(H/'training_launch_receipt.json',receipt)
 mp=read(M/'plan.json');mp.update(status='training',active_iteration=T.name,active_training_run=str(T),active_evaluation_run=str(E),training_mode='seven_task_mixed_resume',active_task=None,global_batch=32,active_start_step=17000,active_target_step=46269,next_action='Train to10 total equivalent frame epochs, then all35 seven-task development evaluations',piano_single_task_automatic_resume=False);save(M/'plan.json',mp)
 while True:
  ts=read(T/'status.json')
  status(ts['state'],training_pid=pid,training_step=ts.get('step',17000),training_status=ts)
  if ts.get('state')=='failed':raise RuntimeError('Resume training failed: '+ts.get('error','see log'))
  if ts.get('state')=='complete' and not alive(pid,str(T/'code/train_full_8gpu.py')):break
  if not alive(pid,str(T/'code/train_full_8gpu.py')) and ts.get('state')!='complete':raise RuntimeError('Training exited before completion')
  if (T/'optimizer_continuation_audit.json').exists() and receipt['optimizer_restore_verification']=='pending training audit':
   assert read(T/'optimizer_continuation_audit.json')['passed'];receipt['optimizer_restore_verification']='passed';save(M/'receipt.json',receipt);save(H/'training_launch_receipt.json',receipt)
  time.sleep(10)
 assert ts['step']==46269 and (T/'checkpoints/46269/_CHECKPOINT_METADATA').exists()
 assert read(T/'checkpoint_roundtrip_8gpu.json')['passed'] and read(T/'full_gradient_check_8gpu.json')['passed']
 status('waiting_training_gpu_release');free_gpus()
 manifest=read(E/'manifest.json');verify(manifest,E);assert manifest['checkpoint_step']==46269
 save(E/'checkpoint_selection.json',dict(rule='User-requested final10 equivalent frame epochs; not selected from evaluation seeds',checkpoint=manifest['checkpoint'],step=46269))
 env=os.environ.copy()
 for k in ['LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','XLA_FLAGS','JAX_PLATFORMS','LD_PRELOAD','CUDA_VISIBLE_DEVICES']:env.pop(k,None)
 env.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME=str(R/'cache/openpi'),JAX_COMPILATION_CACHE_DIR=str(R/'cache/jax'),LD_PRELOAD=str(R/'runtime_overrides/nccl_2289/nvidia/nccl/lib/libnccl.so.2'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (E/'logs/policy.log').open('a') as log:policy=subprocess.Popen([str(R/'env/bin/python'),str(E/'code/policy_server.py')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 status('inference_preflight',policy_pid=policy.pid)
 for _ in range(1200):
  if (E/'policy_ready.json').exists() and read(E/'policy_ready.json')['pid']==policy.pid:break
  if policy.poll() is not None:raise RuntimeError('New policy preflight failed')
  time.sleep(1)
 else:raise RuntimeError('New policy startup timed out')
 assert read(E/'preflight/inference.json')['passed']
 env2=os.environ.copy()
 for k in ['CUDA_VISIBLE_DEVICES','LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','XLA_FLAGS','JAX_PLATFORMS']:env2.pop(k,None)
 with (E/'logs/coordinator.log').open('a') as log:coordinator=subprocess.Popen(['python3',str(E/'code/coordinator.py')],stdout=log,stderr=subprocess.STDOUT,env=env2,start_new_session=True)
 while coordinator.poll() is None:
  if policy.poll() is not None:raise RuntimeError('New policy exited during development evaluation')
  ds=read(E/'status.json') if (E/'status.json').exists() else {}
  status('evaluating_seven_tasks',policy_pid=policy.pid,coordinator_pid=coordinator.pid,development_status=ds)
  time.sleep(10)
 ds=read(E/'status.json');assert coordinator.returncode==0
 dev=rows_complete(E,manifest,ds);per_task=ds['per_task'];gate=all(x['valid_completed']==5 and x['successes']>=4 for x in per_task.values())
 result=dict(state='development_gate_passed' if gate else 'needs_further_debug',per_task=per_task,episodes=35,checkpoint=manifest['checkpoint'],development_only=True,training_mode='seven_task_mixed',total_equivalent_frame_epochs=plan['achieved_frame_epochs'],original_benchmark_unchanged=True)
 save(E/'iteration_result.json',result);save(E/'COMPLETED.json',dict(completed=True,development_only=True,valid_episodes=35,successes=sum(x['success'] for x in dev),checkpoint_step=46269,finished_at=time.time()))
 status(result['state'],per_task=per_task,episodes=35,result=str(E/'iteration_result.json'))
except Exception:
 status('infrastructure_error',error=traceback.format_exc());raise
finally:
 if policy is not None and policy.poll() is None and (coordinator is None or coordinator.poll() is not None):
  policy.terminate()
  try:policy.wait(timeout=15)
  except subprocess.TimeoutExpired:policy.kill();policy.wait()
