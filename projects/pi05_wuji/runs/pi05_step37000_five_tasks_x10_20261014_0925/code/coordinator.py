"""Fixed7 x5 development pi05 final-checkpoint evaluation. GPU rollouts and CPU geometry have separate queues."""
import os,json,time,subprocess,shlex,fcntl,hashlib,threading,math,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty

S=Path(__file__).resolve().parents[1];R=S.parents[1];P=R.parent/'dexterity_living'
lock=open(S/'coordinator.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
M=json.loads((S/'manifest.json').read_text());TASKS=M['evaluation_tasks'];jobs=Queue()
mutex=threading.RLock();active={};pending={};finished={};rollouts={};geometry_futures=[];started=time.time()
for t in TASKS:
 for seed in M['seeds']:
  p=S/'results'/f'{t}_{seed}.json'
  if p.exists():finished[f'{t}_{seed}']=json.loads(p.read_text())
for seed in M['seeds']:
 for task in TASKS:
  if f'{task}_{seed}' not in finished:jobs.put((task,seed))

def save(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(p)

def interval(k,n):
 if not n:return None
 z=1.959963984540054;den=1+z*z/n;center=(k/n+z*z/(2*n))/den;half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
 return [max(0,center-half),min(1,center+half)]

def status(state='running'):
 with mutex:
  rows={}
  for task in TASKS:
   rr=[x for x in finished.values() if x['task']==task];valid=[x for x in rr if x['valid']];k=sum(x['success'] for x in valid);n=len(valid)
   rows[task]={'target':len(M['seeds']),'valid_completed':n,'successes':k,'failures':n-k,'infrastructure_exhausted':sum(not x['valid'] for x in rr),'pending_geometry':sum(x['task']==task for x in pending.values()),'success_rate':k/n if n else None,'wilson_95_interval':interval(k,n)}
  save(S/'status.json',{'state':state,'pid':os.getpid(),'started_at':started,'updated_at':time.time(),'target_episodes':M['total_episodes'],'finished_records':len(finished),'valid_completed':sum(x['valid'] for x in finished.values()),'rollouts_returned_this_process':len(rollouts),'active_gpus':dict(active),'pending_geometry':dict(pending),'per_task':rows,'configuration':{'horizon':30,'chunk_size':30,'seeds':[M['seeds'][0],M['seeds'][-1]],'model':'intermediate seven-task pi05 step37000 checkpoint','environment':'v2 collection-aligned'},'results_dir':str(S/'results')})

def guard_shared_sources():
 for path,digest in {**M['shared_source_sha256'],**{str(S/'code'/k):v for k,v in M['code_sha256'].items()}}.items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen dependency changed: '+path)

def finish(key,row):
 with mutex:
  save(S/'results'/f'{key}.json',row);finished[key]=row;pending.pop(key,None);status()

def geometry(key,row):
 run=Path(row['run']);errors=[]
 for attempt in [1,2]:
  output=f'eval50_geometry_attempt{attempt}.json'
  cmd='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(['/home/bowenjing/envs/isaaclab3/bin/python',str(S/'code/audit_geometry.py'),str(run),'--task-pairs','--output',output])
  try:
   env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
   with (run/f'geometry_attempt{attempt}.log').open('w') as log:proc=subprocess.run(['/bin/bash','-c',cmd],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=1800)
   if proc.returncode:raise RuntimeError(f'geometry exit {proc.returncode}')
   d=json.loads((run/output).read_text());assert d['checked_frames']==row['frames']
   row.update(valid=True,success=d['frames_with_intersections']==0,reason='task_success' if d['frames_with_intersections']==0 else 'visible_mesh_intersection',geometry={'report':str(run/output),'checked_frames':d['checked_frames'],'frames_with_intersections':d['frames_with_intersections'],'intersection_counts':d['intersection_counts']},geometry_infrastructure_errors=errors,finished_at=time.time())
   finish(key,row);return
  except Exception as exc:errors.append(repr(exc))
 row.update(valid=False,success=None,reason='geometry_infrastructure_exhausted',geometry_infrastructure_errors=errors,finished_at=time.time());finish(key,row)

FFMPEG=P/'runtime/python/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
def check_video(run,frames):
 p=subprocess.run([str(FFMPEG),'-v','error','-xerror','-nostats','-threads','2','-i',str(run/'rollout.mp4'),'-map','0:v:0','-f','null','-','-progress','pipe:1'],capture_output=True,text=True,timeout=180)
 values=[int(x.split('=',1)[1]) for x in p.stdout.splitlines() if x.startswith('frame=')]
 if p.returncode or not values or values[-1]!=frames:raise RuntimeError(f'video validation failed: rc={p.returncode},frames={values[-1:]},expected={frames},error={p.stderr[-400:]}')
 return {'decoded_frames':values[-1],'full_decode_passed':True}

def gpu_worker(gpu,pool):
 while True:
  try:task,seed=jobs.get_nowait()
  except Empty:return
  key=f'{task}_{seed}';errors=[];row=None
  try:
   guard_shared_sources()
   for attempt in [1,2,3]:
    name=f'pi05_step37000_five_tasks_x10_20261014_0925_{task}_{seed}_h30_a{attempt}';episode=S/'episodes'/name
    with mutex:active[str(gpu)]={'task':task,'seed':seed,'attempt':attempt,'name':name,'started_at':time.time()};status()
    try:
     result=episode/'result.json'
     if not result.exists():
      # Recover only completed attempts; an interrupted attempt is never overwritten.
      if episode.exists():raise RuntimeError('incomplete existing attempt; retained for audit')
      with (S/'logs'/f'{name}.log').open('w') as log:proc=subprocess.run(['python3',str(S/'code/launch_eval.py'),'--task',task,'--seed',str(seed),'--gpu',str(gpu),'--horizon','30','--attempt',str(attempt)],stdout=log,stderr=subprocess.STDOUT)
     d=json.loads(result.read_text())
     if d.get('status')!='complete' or d.get('frames',0)<=0:raise RuntimeError('simulation infrastructure: '+str(d.get('reason',d.get('error','no valid report')))[:500])
     run=P/'runs/physical_v1'/name;video=check_video(run,d['frames'])
     assert d['expert_actions_after_initialization']==0 and d['object_pose_commands']==0 and d['physics_stepped'] and d['replan_steps']==30
     row={'task':task,'seed':seed,'gpu':gpu,'attempt':attempt,'run':str(run),'report':str(result),'frames':d['frames'],'task_success_before_geometry':bool(d['success']),'valid':True,'success':False,'reason':d['reason'],'video':video,'simulation_infrastructure_errors':errors,'checkpoint':d['checkpoint']}
     with mutex:rollouts[key]=row
     if d['success']:
      with mutex:pending[key]={'task':task,'seed':seed,'run':str(run)};geometry_futures.append(pool.submit(geometry,key,row))
     else:row['finished_at']=time.time();finish(key,row)
     break
    except Exception as exc:errors.append({'attempt':attempt,'error':repr(exc)})
   if row is None:finish(key,{'task':task,'seed':seed,'valid':False,'success':None,'reason':'simulation_infrastructure_exhausted','errors':errors,'finished_at':time.time()})
  except Exception as exc:finish(key,{'task':task,'seed':seed,'valid':False,'success':None,'reason':'dependency_or_scheduler_error','error':repr(exc),'finished_at':time.time()})
  finally:
   with mutex:active.pop(str(gpu),None);status()
   jobs.task_done()

stop=threading.Event()
def heartbeat():
 while not stop.wait(20):status()
guard_shared_sources();status('starting');threading.Thread(target=heartbeat,daemon=True).start()
try:
 with ThreadPoolExecutor(max_workers=8) as geometry_pool:
  with ThreadPoolExecutor(max_workers=7) as gpu_pool:list(gpu_pool.map(lambda gpu:gpu_worker(gpu,geometry_pool),range(1,8)))
  status('finishing_geometry')
  for future in geometry_futures:future.result()
 state='complete' if len(finished)==M['total_episodes'] and all(x['valid'] for x in finished.values()) else 'completed_with_infrastructure_errors'
 stop.set();status(state)
 save(S/'final_report.json',json.loads((S/'status.json').read_text()))
 print(state,flush=True)
except BaseException:
 stop.set();save(S/'coordinator_error.json',{'error':traceback.format_exc(),'time':time.time()});status('coordinator_error');raise
