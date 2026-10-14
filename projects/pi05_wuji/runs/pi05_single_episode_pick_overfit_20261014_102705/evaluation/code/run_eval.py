"""Reuse source task initialization, replacing all demonstration motion with pi05."""
import argparse,sys,json,os,fcntl,hashlib,traceback
from pathlib import Path
import numpy as np
P=Path('/media/external/bowenjing/projects/dexterity_living')
ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--name',required=True);ap.add_argument('--socket',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--mode',choices=['pi05','expert'],required=True);ap.add_argument('--horizon',type=int,default=5);args=ap.parse_args();assert 1<=args.horizon<=30
lock=open(P/f'reports/process_manager/gpu{args.gpu}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
OUT=P/'runs/physical_v1'/args.name
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
spec=json.loads((P/'configs/mcl350_collection.json').read_text());cfg=spec['controllers'][args.task]
source=P/cfg['script'];text=source.read_text()
sys.path[:0]=[str(Path(__file__).parent),str(P/'code/physical_v11'),str(P/'code/physical_v10')]
from model import joints,arm_names,hand_names
rng=np.random.default_rng(args.seed);initial_deltas={}
micro_approach_seconds=1.8+float(rng.uniform(-.04,.04)) if args.task=='microwave_open' else None
def jitter(q,side):
 if side!=('left' if args.task=='microwave_open' else 'right'):return
 d={}
 for n in arm_names(side)+hand_names(side):
  old=q[n];amount=.012 if n in arm_names(side) else .018
  q[n]=float(np.clip(old+rng.uniform(-amount,amount),joints[n]['lo']+1e-4,joints[n]['hi']-1e-4));d[n]=q[n]-old
 initial_deltas[side]=d
if args.task in ['pick_place','stack']:
 text=text.replace('qinit[side].update(open_q(side))','qinit[side].update(open_q(side)); jitter(qinit[side],side)')
elif args.task=='microwave_open':
 text=text.replace("initial['left']=approach[0]","initial['left']=approach[0].copy(); jitter(initial['left'],'left')")
elif args.task=='piano':
 text=text.replace("initial['right']=above.copy()","initial['right']=above.copy(); jitter(initial['right'],'right')")
else:text=text.replace('initial[side].update(open_q(side))','initial[side].update(open_q(side)); jitter(initial[side],side)')
if args.task.startswith('gesture'):
 text=text.replace("raised.update(zip(arm_names('right'),qa))","raised.update(zip(arm_names('right'),qa)); T=fk('right_palm_link','right',raised); T[:3,3]+=EVAL_RNG.uniform(-.008,.008,3); qa,pe,re=solve('right',T,qa); assert pe<.001; raised.update(zip(arm_names('right'),qa))")
markers={'pick_place':' for phase_index,','stack':' for phase_index,','gesture_one':' for frame in range(174):','gesture_two':' for frame in range(174):','piano':' for duration,end,phase,key in sequence:','produce_sort':' for _ in range(360*a.substeps):','microwave_open':' for frame in range(30*32):'}
pos=text.index(markers[args.task]);text=text[:pos]
lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('app=SimulationApp('))
lines[i+1:i+1]=['from eval_environment import install_reset_hook, run_policy', 'install_reset_hook(EVAL_ARGS)']
text='\n'.join(lines)+'\n report=run_policy(globals(),EVAL_ARGS)\nexcept Exception:\n report={"status":"infrastructure_error","success":None,"error":traceback.format_exc()}\n print(report["error"],flush=True)\nfinally:\n (out/"report.json").write_text(json.dumps(report,indent=2))\n print("EVAL_REPORT",json.dumps(report),flush=True)\n app.close()\n'
(OUT/'instrumented_source.py').write_text(text)
(OUT/'provenance.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'instrumented_sha256':hashlib.sha256(text.encode()).hexdigest(),'task':args.task,'seed':args.seed,'protocol':f'pi05 final checkpoint; 30 Hz absolute 52D actions; replan every {args.horizon} steps; no expert actions after reset'},indent=2))
sys.argv=[str(source),'--name',args.name,'--gpu',str(args.gpu),*cfg['args']]
if micro_approach_seconds is not None:sys.argv[sys.argv.index('--approach-seconds')+1]=str(micro_approach_seconds)
ns={'__name__':'__main__','__file__':str(source),'EVAL_ARGS':args,'EVAL_RNG':rng,'jitter':jitter}
exec(compile(text,str(source),'exec'),ns)
