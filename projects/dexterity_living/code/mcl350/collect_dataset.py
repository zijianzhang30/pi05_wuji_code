"""Bounded, resumable eight-GPU collection; only gate-approved episodes meet quota."""
import fcntl,json,os,subprocess,time,sys,collections
from pathlib import Path
P=Path(__file__).resolve().parents[2];D=P/'datasets/wuji_mcl350_v1';R=P/'reports/mcl350';R.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
lock=open(R/'coordinator.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec=json.loads((P/'configs/mcl350_collection.json').read_text());tasks=list(spec['controllers']);active={};counter=collections.Counter();consecutive=collections.Counter();history=[]
for f in R.glob('mcl350_v1_*_013*.json'):
 d=json.loads(f.read_text());counter[d['task']]+=1
 started=time.time()
def accepted():
 out={t:[] for t in tasks}
 for t in tasks:
  for f in sorted((D/t).glob('episode_*/acceptance.json')):
   a=json.loads(f.read_text())
   if a.get('accepted'):out[t].append(a)
 return out
def write_status(data):
 f=R/'status.tmp';f.write_text(json.dumps(data,indent=2));f.replace(R/'status.json')
start=time.time()
while True:
 for gpu,item in list(active.items()):
  code=item['worker'].poll()
  if code is None:continue
  f=R/(item['name']+'.json');d=json.loads(f.read_text()) if f.exists() else {'stage':'rejected','error':'worker exited without status'}
  consecutive[item['task']]=0 if d['stage']=='accepted' else consecutive[item['task']]+1
  history.append(dict(task=item['task'],name=item['name'],gpu=gpu,exit_code=code,stage=d['stage'],error=d.get('error')))
  item['lock'].close();item['log'].close();del active[gpu]
 got=accepted();counts={t:len(v) for t,v in got.items()};inflight=collections.Counter(x['task'] for x in active.values());blocked={t:'5 consecutive failures require review' if consecutive[t]>=5 else '100 attempt budget reached' for t in tasks if consecutive[t]>=5 or counter[t]>=100}
 eligible=[t for t in tasks if counts[t]+inflight[t]<50 and t not in blocked]
 for gpu in range(8):
  if gpu in active or not eligible:continue
  g=open(P/f'reports/process_manager/gpu{gpu}.lock','a')
  try:fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:g.close();continue
  task=min(eligible,key=lambda t:(counts[t]+inflight[t],tasks.index(t)))
  counter[task]+=1;seed=13000+counter[task];name=f'mcl350_v1_{task}_{seed:06d}'
  # Never overwrite an existing raw candidate after coordinator restart.
  while (P/'runs/physical_v1'/name).exists():
   counter[task]+=1;seed=13000+counter[task];name=f'mcl350_v1_{task}_{seed:06d}'
  log=open(R/(name+'_worker.log'),'w');args=[sys.executable,str(P/'code/mcl350/collect_one.py'),'--task',task,'--seed',str(seed),'--name',name,'--gpu',str(gpu)]
  worker=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  active[gpu]=dict(worker=worker,lock=g,log=log,name=name,task=task,seed=seed,started=time.time());inflight[task]+=1;eligible=[t for t in eligible if counts[t]+inflight[t]<50]
 finished=all(x==50 for x in counts.values());stopped=not eligible and not active and not finished
 state=dict(state='complete' if finished else 'needs_review' if stopped else 'collecting',target_per_task=50,target_total=350,accepted=counts,accepted_total=sum(counts.values()),attempts=dict(counter),blocked=blocked,running=[dict(gpu=g,task=x['task'],name=x['name'],pid=x['worker'].pid,seconds=time.time()-x['started']) for g,x in active.items()],recent_results=history[-20:],seconds=time.time()-start,updated_at=time.time(),dataset=str(D),qualification='physics + recorded-frame mesh + RGB/state/action alignment + no duplicate trajectory',generation='scripted feedback control; narrow initial-condition variation; not human teleoperation')
 write_status(state)
 if finished or stopped:
  manifest=dict(specification=spec,status=state,episodes=got);temp=D/'manifest.tmp';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temp.replace(D/'manifest.json');break
 # Periodic discovery manifest allows consumers to use already accepted episodes safely.
 temp=D/'manifest.tmp';temp.write_text(json.dumps(dict(specification=spec,status=state,episodes=got),ensure_ascii=False,indent=2));temp.replace(D/'manifest.json')
 time.sleep(3)
print(json.dumps(state,indent=2))
