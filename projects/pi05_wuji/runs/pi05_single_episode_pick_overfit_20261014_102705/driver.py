from pathlib import Path
import os,json,time,subprocess,signal,fcntl,traceback,hashlib,shlex
N=Path(__file__).resolve().parent;R=N.parents[1];P=R.parent/'dexterity_living';M=R/'runs/pi05_debug_until_success_20261013'
plan=json.loads((N/'plan.json').read_text());E=Path(plan['evaluation_run']);T=Path(plan['parent_seven_training_run']);SE=Path(plan['parent_seven_evaluation_run']);H=Path(plan['parent_handoff']);B=Path(plan['baseline_run'])
guard=(N/'queue.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
paused=False;trainer=None;policy=None;sim=None;resume_step=None;original_plan=None
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def alive(pid,expected=None):
 try:b=Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:return False
 return bool(b) and (expected is None or expected.encode() in b)
def stop(pid,expected,group=False):
 if not alive(pid,expected):return
 (os.killpg if group else os.kill)(pid,signal.SIGTERM)
 for _ in range(60):
  if not alive(pid,expected):return
  time.sleep(1)
 raise RuntimeError(f'Owned process did not exit {pid}; no overlapping workload launched')
def phase(state,**kw):
 x=dict(state=state,driver_pid=os.getpid(),training_run=str(N),evaluation_run=str(E),diagnostic='single_training_episode_pick_overfit',updated_at=time.time(),**kw)
 save(N/'driver_status.json',x)
 if paused:save(M/'status.json',x)
def free():
 for _ in range(120):
  mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if all(int(v)<500 for v in mem.splitlines()):return
  time.sleep(2)
 raise RuntimeError('GPU memory not released')
env=os.environ.copy()
for k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_PLATFORMS']:env.pop(k,None)
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
def decode(run,frames):
 ff=P/'runtime/python/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
 p=subprocess.run([str(ff),'-v','error','-xerror','-nostats','-threads','2','-i',str(run/'rollout.mp4'),'-map','0:v:0','-f','null','-','-progress','pipe:1'],capture_output=True,text=True,timeout=180)
 count=[int(x.split('=',1)[1]) for x in p.stdout.splitlines() if x.startswith('frame=')]
 assert p.returncode==0 and count and count[-1]==frames,(p.returncode,count[-1:],frames)
 return dict(full_decode_passed=True,decoded_frames=count[-1])
try:
 assert read(N/'preflight/cpu_contract.json')['passed']
 manifest=read(N/'source_manifest.json')
 for name,digest in manifest['training_code_sha256'].items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==digest,name
 for path,digest in read(E/'manifest.json')['shared_source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 previous=read(M/'receipt.json');assert previous['training_run']==str(T)
 tp=previous['training_pid'];sp=previous['driver_pid']
 assert alive(tp,str(T/'code/train_full_8gpu.py')) and alive(sp,str(H/'continue_after_five_tasks_x10.py'))
 ck=read(T/'checkpoint_status.json');assert ck['saved'] and (Path(ck['path'])/'_CHECKPOINT_METADATA').exists()
 resume_step=ck['step'];original_plan=read(M/'plan.json')
 save(N/'seven_training_before_pause.json',dict(receipt=previous,status=read(T/'status.json'),checkpoint_status=ck,master_plan=original_plan))
 phase('pausing_seven_task_training',training_pid=tp,supervisor_pid=sp,resume_step=resume_step)
 stop(sp,str(H/'continue_after_five_tasks_x10.py'));stop(tp,str(T/'code/train_full_8gpu.py'));paused=True
 save(T/'status.json',dict(state='paused_for_single_episode_overfit',resume_step=resume_step,diagnostic_run=str(N),updated_at=time.time()))
 active=dict(original_plan);active.update(status='single_episode_overfit_diagnostic',scope='First isolate fit and closed-loop stability on one accepted training trajectory; seven-task model is preserved and resumes afterward',
 active_iteration=N.name,active_training_run=str(N),active_evaluation_run=str(E),next_action='1000 diagnostic updates on pick_place episode0, then exact training-scene evaluation, then resume parent seven-task model',changes=['One331-frame training episode; other training contracts unchanged from step37000 baseline'],checkpoint_warmstart_step=37000,first_budget_steps=1000)
 save(M/'plan.json',active);phase('waiting_gpu_release',parent_resume_step=resume_step);free()
 out=subprocess.run(['python3',str(N/'launch_8gpu.py')],capture_output=True,text=True,check=True);tid=json.loads(out.stdout)['pid']
 class Existing:
  pid=tid
  def poll(self):return None if alive(tid,str(N/'code/train_full_8gpu.py')) else 0
 trainer=Existing()
 save(M/'receipt.json',dict(training_pid=tid,driver_pid=os.getpid(),training_run=str(N),evaluation_run=str(E),resume_step=37000,target_steps=38000,new_updates=1000,training_mode='single_training_episode_overfit_diagnostic',global_batch=32,parent_seven_training_run=str(T),parent_seven_resume_step=resume_step,source_checkpoint=plan['source_checkpoint'],resumed_at=time.time()))
 while trainer.poll() is None:
  ts=read(N/'status.json');phase('training_single_episode',training_pid=tid,training_status=ts,new_updates=max(0,ts.get('step',37000)-37000),target_updates=1000)
  if ts['state']=='failed':raise RuntimeError('Single-episode training failed: '+ts.get('error','see log'))
  time.sleep(10)
 ts=read(N/'status.json');assert ts['state']=='complete' and ts['step']==38000
 assert (N/'checkpoints/38000/_CHECKPOINT_METADATA').exists() and read(N/'checkpoint_roundtrip_8gpu.json')['passed']
 assert read(N/'optimizer_continuation_audit.json')['passed'] and read(N/'optimizer_continuation_audit.json')['state_step']==37000
 phase('training_complete_waiting_gpu_release');free()
 em=read(E/'manifest.json')
 for name,digest in em['code_sha256'].items():assert hashlib.sha256((E/'code'/name).read_bytes()).hexdigest()==digest,name
 pe=env.copy();pe.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME=str(R/'cache/openpi'),JAX_COMPILATION_CACHE_DIR=str(R/'cache/jax'),LD_PRELOAD=str(R/'runtime_overrides/nccl_2289/nvidia/nccl/lib/libnccl.so.2'))
 with (E/'logs/policy.log').open('w') as f:policy=subprocess.Popen([str(R/'env/bin/python'),str(E/'code/policy_server.py')],cwd=R/'openpi',env=pe,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 phase('starting_overfit_policy',policy_pid=policy.pid)
 for _ in range(900):
  if (E/'policy_ready.json').exists() and read(E/'policy_ready.json')['pid']==policy.pid:break
  if policy.poll() is not None:raise RuntimeError('Overfit policy preflight failed')
  time.sleep(1)
 else:raise RuntimeError('Overfit policy startup timeout')
 assert read(E/'preflight/inference.json')['passed']
 oe=env.copy();oe['JAX_PLATFORMS']='cpu'
 with (E/'logs/offline.log').open('w') as f:offline=subprocess.Popen([str(R/'env/bin/python'),str(E/'offline_teacher.py')],env=oe,stdout=f,stderr=subprocess.STDOUT)
 phase('evaluating_same_training_scene',policy_pid=policy.pid)
 errors=[];row=None
 for attempt in [1,2,3]:
  name=N.name+'_pi05_a'+str(attempt);run=P/'runs/physical_v1'/name;assert not run.exists()
  args=['/home/bowenjing/envs/isaaclab3/bin/python',str(E/'code/run_eval.py'),'--task','pick_place','--seed','12001','--gpu','1','--name',name,'--socket',read(E/'policy_ready.json')['socket'],'--horizon','30','--mode','pi05','--reference',plan['training_reference']['source']]
  command='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(args)
  with (E/'logs'/f'simulation_attempt{attempt}.log').open('w') as f:
   sim=subprocess.Popen(['/bin/bash','-c',command],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:sim.wait(timeout=900)
   except subprocess.TimeoutExpired:os.killpg(sim.pid,signal.SIGTERM);sim.wait(timeout=30)
  report=read(run/'report.json') if (run/'report.json').exists() else dict(status='infrastructure_error',success=None,error='no report')
  if report.get('status')!='complete':
   errors.append(dict(attempt=attempt,report=report,run=str(run)));save(E/'results'/f'pi05_infrastructure_attempt{attempt}.json',errors[-1])
   if (run/'reset_alignment.json').exists() and not read(run/'reset_alignment.json')['passed']:break
   continue
  assert sim.returncode==0 and read(run/'reset_alignment.json')['passed'] and report['expert_actions_after_initialization']==0
  row=dict(report,run=str(run),mode='pi05',attempt=attempt,exit_code=sim.returncode,reset_alignment=read(run/'reset_alignment.json'),video=decode(run,report['frames']),simulation_infrastructure_errors=errors)
  if row['success']:
   command='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(['/home/bowenjing/envs/isaaclab3/bin/python',str(E/'code/audit_geometry.py'),str(run),'--task-pairs','--output','diagnostic_geometry.json'])
   with (run/'geometry.log').open('w') as f:g=subprocess.run(['/bin/bash','-c',command],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
   assert g.returncode==0;geo=read(run/'diagnostic_geometry.json');assert geo['checked_frames']==row['frames']
   row['geometry']=geo;row['success']=geo['frames_with_intersections']==0
  save(E/'results/pi05.json',row);break # Keep valid failure; never retry a policy failure.
 assert row is not None
 assert offline.wait(timeout=600)==0
 expert=read(B/'results/expert.json');expert['reuse_provenance']=dict(source=str(B/'results/expert.json'),reason='same accepted training initial scene and unchanged physical executor; expert actions do not depend on learned checkpoint',sha256=hashlib.sha256((B/'results/expert.json').read_bytes()).hexdigest())
 save(E/'results/expert.json',expert)
 analysis=subprocess.run([str(R/'env/bin/python'),str(E/'analyze.py')],env=oe,capture_output=True,text=True,check=True)
 baseline=read(B/'ANALYSIS.json');after=read(E/'ANALYSIS.json')
 comparison=dict(diagnostic_only=True,training_episode_index=0,training_reference=plan['training_reference'],source_checkpoint_step=37000,final_checkpoint_step=38000,new_updates=1000,
 baseline_success=baseline['pi05']['success'],overfit_success=row['success'],baseline_teacher_groups=baseline['teacher_forced']['groups'],overfit_teacher_groups=after['teacher_forced']['groups'],
 baseline_kinematic_error=baseline['teacher_forced']['kinematic_target_error'],overfit_kinematic_error=after['teacher_forced']['kinematic_target_error'],baseline_run=str(B),overfit_analysis=str(E/'ANALYSIS.json'),
 fit_loss_before=read(N/'fit_checks/step_037000.json'),fit_loss_after=read(N/'fit_checks/step_038000.json'),scope='one train-scene diagnostic, not held-out task success or a seven-task model release')
 save(N/'COMPARISON.json',comparison);save(E/'COMPLETED.json',dict(completed=True,diagnostic_only=True,checkpoint_step=38000,new_training_updates=1000,valid_model_episodes=1,model_successes=int(row['success']),expert_replay_reused=True,finished_at=time.time()))
 phase('single_episode_diagnostic_complete',success=row['success'],comparison=str(N/'COMPARISON.json'))
except Exception:
 save(N/'ERROR.json',dict(error=traceback.format_exc(),time=time.time()));phase('diagnostic_infrastructure_error',error=traceback.format_exc())
finally:
 if sim is not None and sim.poll() is None:stop(sim.pid,str(E/'code/run_eval.py'),group=True)
 if policy is not None and policy.poll() is None:stop(policy.pid,str(E/'code/policy_server.py'))
 if trainer is not None and trainer.poll() is None:stop(trainer.pid,str(N/'code/train_full_8gpu.py'))
 if paused:
  phase('resuming_original_seven_task_training',resume_step=resume_step);free()
  out=subprocess.run(['python3',str(T/'launch_8gpu.py')],capture_output=True,text=True,check=True);pid=json.loads(out.stdout)['pid']
  receipt=dict(training_pid=pid,driver_pid=os.getpid(),training_run=str(T),evaluation_run=str(SE),resume_checkpoint=str(T/'checkpoints'/str(resume_step)),resume_step=resume_step,original_resume_step=17000,target_steps=46269,training_mode='seven_task_mixed_resume',global_batch=32,target_equivalent_frame_epochs=10,previous_probe=str(N),optimizer_restore_verification='pending training audit',resumed_at=time.time())
  save(M/'receipt.json',receipt);save(N/'seven_training_resume_receipt.json',receipt)
  if original_plan is not None:save(M/'plan.json',dict(original_plan,last_single_episode_diagnostic=str(N)))
  phase('original_seven_task_training_resumed',training_pid=pid,resume_step=resume_step)
  os.execv('/usr/bin/python3',['python3',str(H/'continue_after_five_tasks_x10.py')])
