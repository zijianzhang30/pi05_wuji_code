"""Expert endpoint replay through the unchanged v4 execution and task criteria.
This is an environment/data-action comparator, never a pi05 success result."""
from pathlib import Path
import os,sys,json,time,shutil,subprocess,threading,traceback,shlex,concurrent.futures
import numpy as np,h5py
from multiprocessing.connection import Listener
A=Path(__file__).resolve().parents[1]
R=Path('/media/external/bowenjing/projects/pi05_wuji')
P=R.parent/'dexterity_living'
E=R/'runs/pi05_repair_clean_image_v4_dev_20261013_001'
TASKS=json.loads((E/'manifest.json').read_text())['tasks']
jobs={}
for task in TASKS:
 sp=json.loads((R.parent/'act_wuji/datasets_v21'/task/'split_manifest.json').read_text())
 ep=next(x for x in sp['episodes'] if x['episode_index']==sp['validation'][0])
 with h5py.File(Path(ep['source'])/'episode.hdf5') as f:
  actions=f['action'][:]
  if task.startswith('gesture'):actions=actions[:99]
  jobs[task]=dict(actions=actions,names=f['metadata/joint_names'].asstr()[:].tolist(),seed=int(f.attrs['seed']),reference=ep['source'])
sock='/tmp/'+A.name+'_expert.sock'
listener=Listener(sock,family='AF_UNIX')
def client(conn):
 try:
  hello=conn.recv();job=jobs[hello['task']]
  conn.send(dict(joint_names=job['names'],policy='expert_endpoint_replay_diagnostic',source=job['reference'],seed=job['seed']))
  while True:
   request=conn.recv()
   if request is None:break
   frame=int(request['control_step']);ix=np.minimum(np.arange(frame,frame+30),len(job['actions'])-1)
   conn.send(dict(actions=job['actions'][ix].tolist(),inference_seconds=0.0))
 except (EOFError,BrokenPipeError):pass
 except Exception:print(traceback.format_exc(),flush=True)
 finally:conn.close()
def accept():
 while True:
  try:conn=listener.accept()
  except (OSError,EOFError):return
  threading.Thread(target=client,args=(conn,),daemon=True).start()
threading.Thread(target=accept,daemon=True).start()
env=os.environ.copy()
for k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_PLATFORMS']:
 env.pop(k,None)
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
def worker(task,gpu):
 job=jobs[task];name=A.name+'_expert_'+task;run=P/'runs/physical_v1'/name
 if run.exists():raise FileExistsError(run)
 args=['/home/bowenjing/envs/isaaclab3/bin/python',str(A/'code/run_eval.py'),'--task',task,'--gpu',str(gpu),'--seed',str(job['seed']),'--name',name,'--socket',sock,'--horizon','30']
 cmd='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(args)
 with (A/'logs'/('expert_'+task+'.log')).open('w') as log:
  proc=subprocess.run(['/bin/bash','-c',cmd],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 f=run/'report.json'
 report=json.loads(f.read_text()) if f.exists() else dict(status='infrastructure_error',success=None)
 report.update(diagnostic='expert_endpoint_replay',source=job['reference'],run=str(run),exit_code=proc.returncode)
 (A/'evidence'/('expert_'+task+'.json')).write_text(json.dumps(report,indent=2))
 print('EXPERT_RESULT',json.dumps({k:report.get(k) for k in ['task','status','success','reason','frames','exit_code','run']}),flush=True)
 return report
results=[]
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
  futures=[]
  for gpu,task in enumerate(TASKS,1):
   futures.append(pool.submit(worker,task,gpu));time.sleep(2)
  for f in concurrent.futures.as_completed(futures):
   results.append(f.result())
   (A/'evidence/expert_endpoint_results.json').write_text(json.dumps(results,indent=2))
finally:
 listener.close()
print('EXPERT_DONE',sum(x.get('success') is True for x in results),'/',len(results),flush=True)
