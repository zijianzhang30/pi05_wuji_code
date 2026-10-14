from pathlib import Path
import json,os,time,signal,subprocess,shlex,fcntl,traceback,hashlib,concurrent.futures
N=Path(__file__).resolve().parent;R=N.parents[1];P=R.parent/'dexterity_living';M=R/'runs/pi05_debug_until_success_20261013'
plan=json.loads((N/'plan.json').read_text());T=Path(plan['training_run']);E=Path(plan['future_evaluation_run']);H=Path(plan['parent_handoff'])
lock=(N/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
policy=None;paused=False;resume_step=None;workers=[]
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def alive(pid,match=None):
 try:b=Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:return False
 return bool(b) and (match is None or match.encode() in b)
def phase(state,**kw):
 d=dict(state=state,driver_pid=os.getpid(),diagnostic_run=str(N),training_run=str(T),checkpoint_step=37000,updated_at=time.time(),**kw)
 save(N/'driver_status.json',d)
 if paused:save(M/'status.json',d)
def stop(pid,match):
 if not alive(pid,match):return
 os.kill(pid,signal.SIGTERM)
 for _ in range(60):
  if not alive(pid,match):return
  time.sleep(1)
 raise RuntimeError(f'Owned process did not exit: {pid}')
def free():
 for _ in range(120):
  x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if all(int(v)<500 for v in x.splitlines()):return
  time.sleep(2)
 raise RuntimeError('GPU contexts not released')
env=os.environ.copy()
for k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_PLATFORMS']:env.pop(k,None)
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
FF=P/'runtime/python/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
def video(run,frames):
 x=subprocess.run([str(FF),'-v','error','-xerror','-nostats','-threads','2','-i',str(run/'rollout.mp4'),'-map','0:v:0','-f','null','-','-progress','pipe:1'],capture_output=True,text=True,timeout=180)
 counts=[int(l.split('=',1)[1]) for l in x.stdout.splitlines() if l.startswith('frame=')]
 assert x.returncode==0 and counts and counts[-1]==frames,(x.returncode,counts[-1:],frames,x.stderr[-300:])
 return dict(full_decode_passed=True,decoded_frames=counts[-1])
def simulate(mode,gpu):
 errors=[]
 for attempt in [1,2,3]:
  name=N.name+'_'+mode+'_a'+str(attempt);run=P/'runs/physical_v1'/name
  assert not run.exists()
  args=['/home/bowenjing/envs/isaaclab3/bin/python',str(N/'code/run_eval.py'),'--task','pick_place','--seed','12001','--gpu',str(gpu),'--name',name,'--socket',read(N/'policy_ready.json')['socket'],'--horizon','30','--mode',mode,'--reference',plan['reference']['source']]
  cmd='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(args)
  with (N/'logs'/f'{mode}_attempt{attempt}.log').open('w') as f:
   proc=subprocess.Popen(['/bin/bash','-c',cmd],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);workers.append(proc)
   try:proc.wait(timeout=900)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
  report=read(run/'report.json') if (run/'report.json').exists() else dict(status='infrastructure_error',success=None,error='no report')
  report.update(run=str(run),mode=mode,attempt=attempt,exit_code=proc.returncode,simulation_infrastructure_errors=list(errors))
  if (run/'reset_alignment.json').exists():report['reset_alignment']=read(run/'reset_alignment.json')
  if report.get('status')=='complete' and report.get('frames',0)>0:
   assert proc.returncode==0 and report['reset_alignment']['passed']
   report['video']=video(run,report['frames'])
   assert report['expert_actions_after_initialization']==(report['frames'] if mode=='expert' else 0)
   if report['success']:
    command='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(['/home/bowenjing/envs/isaaclab3/bin/python',str(N/'code/audit_geometry.py'),str(run),'--task-pairs','--output','diagnostic_geometry.json'])
    with (run/'geometry.log').open('w') as f:g=subprocess.run(['/bin/bash','-c',command],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
    assert g.returncode==0
    geometry=read(run/'diagnostic_geometry.json');assert geometry['checked_frames']==report['frames'];report['geometry']=geometry
    report['success']=geometry['frames_with_intersections']==0
   save(N/'results'/f'{mode}.json',report);return report
  errors.append(dict(attempt=attempt,report=report))
  save(N/'results'/f'{mode}_infrastructure_attempt{attempt}.json',report)
  if report.get('reset_alignment') and not report['reset_alignment']['passed']:break
 raise RuntimeError(f'{mode} diagnostic infrastructure failed; evidence retained')
try:
 m=read(N/'manifest.json')
 for name,h in m['code_sha256'].items():assert hashlib.sha256((N/'code'/name).read_bytes()).hexdigest()==h
 for name,h in m['input_sha256'].items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==h
 for path,h in m['shared_source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 ts=read(T/'status.json')
 assert ts['state']=='paused_for_training_scene_diagnostic'
 resume_step=int(ts['resume_step']);assert resume_step==38000
 assert not alive(read(M/'receipt.json')['training_pid'],str(T/'code/train_full_8gpu.py'))
 paused=True
 save(N/'training_before_pause.json',dict(previous_pause=ts,checkpoint_status=read(T/'checkpoint_status.json'),recovered_from_reference_schema_error=True))
 save(T/'status.json',dict(state='paused_for_training_scene_diagnostic',diagnostic_run=str(N),resume_step=resume_step,updated_at=time.time()))
 phase('waiting_gpu_release',resume_step=resume_step);free()
 pe=env.copy();pe.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME=str(R/'cache/openpi'),JAX_COMPILATION_CACHE_DIR=str(R/'cache/jax'),LD_PRELOAD=str(R/'runtime_overrides/nccl_2289/nvidia/nccl/lib/libnccl.so.2'))
 with (N/'logs/policy.log').open('w') as f:policy=subprocess.Popen([str(R/'env/bin/python'),str(N/'code/policy_server.py')],cwd=R/'openpi',env=pe,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 phase('starting_frozen_policy',policy_pid=policy.pid,resume_step=resume_step)
 for _ in range(900):
  if (N/'policy_ready.json').exists() and read(N/'policy_ready.json')['pid']==policy.pid:break
  if policy.poll() is not None:raise RuntimeError('Policy startup failed')
  time.sleep(1)
 else:raise RuntimeError('Policy startup timeout')
 assert read(N/'preflight/inference.json')['passed']
 phase('running_training_scene_pair',policy_pid=policy.pid,resume_step=resume_step)
 oe=env.copy();oe['JAX_PLATFORMS']='cpu'
 offline=None # Reuse completed identical teacher-forced data check.
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(simulate,mode,gpu) for mode,gpu in [('pi05',1),('expert',2)]]
  for fut in futures:fut.result()
 assert read(N/'offline/summary.json')['checkpoint_step']==37000
 res=dict(completed=True,diagnostic_only=True,checkpoint_step=37000,training_reference=plan['reference'],pi05=read(N/'results/pi05.json'),expert_comparator=read(N/'results/expert.json'),teacher_forced=read(N/'offline/summary.json'),finished_at=time.time())
 save(N/'COMPLETED.json',res);phase('diagnostic_complete',pi05_success=res['pi05']['success'],expert_success=res['expert_comparator']['success'],resume_step=resume_step)
except Exception:
 save(N/'ERROR.json',dict(error=traceback.format_exc(),time=time.time()));phase('diagnostic_infrastructure_error',error=traceback.format_exc())
finally:
 for proc in workers:
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
 if policy is not None and policy.poll() is None:stop(policy.pid,str(N/'code/policy_server.py'))
 if paused:
  phase('resuming_training',resume_step=resume_step);free()
  out=subprocess.run(['python3',str(T/'launch_8gpu.py')],capture_output=True,text=True,check=True);pid=json.loads(out.stdout)['pid']
  receipt=dict(training_pid=pid,driver_pid=os.getpid(),training_run=str(T),evaluation_run=str(E),resume_checkpoint=str(T/'checkpoints'/str(resume_step)),resume_step=resume_step,original_resume_step=17000,target_steps=46269,training_mode='seven_task_mixed_resume',global_batch=32,target_equivalent_frame_epochs=10,previous_probe=str(N),optimizer_restore_verification='pending training audit',resumed_at=time.time())
  save(M/'receipt.json',receipt);save(N/'training_resume_receipt.json',receipt);phase('training_resumed',training_pid=pid)
  os.execv('/usr/bin/python3',['python3',str(H/'continue_after_five_tasks_x10.py')])
