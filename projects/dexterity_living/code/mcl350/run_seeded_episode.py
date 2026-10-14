"""Run existing contact controllers with seeded initialization and complete logging."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--name',required=True)
a=ap.parse_args();spec=json.loads((P/'configs/mcl350_collection.json').read_text());cfg=spec['controllers'][a.task]
source=P/cfg['script'];text=source.read_text();out=P/'runs/physical_v1'/a.name
if out.exists(): raise FileExistsError(out)
sys.path.insert(0,str(P/'code/physical_v10'));sys.path.insert(0,str(P/'code/physical_v11'));sys.path.insert(0,str(Path(__file__).parent))
from model import joints,arm_names,hand_names,fk,solve
rng=np.random.default_rng(a.seed)
active='left' if a.task=='microwave_open' else 'right'
initial_log=[]
def jitter_side(q,side):
    if side!=active:return
    noise={}
    for n in arm_names(side)+hand_names(side):
        amount=.012 if n in arm_names(side) else .018
        delta=float(rng.uniform(-amount,amount));old=q[n]
        q[n]=float(np.clip(old+delta,joints[n]['lo']+1e-4,joints[n]['hi']-1e-4));noise[n]=q[n]-old
    initial_log.append(noise)
if a.task in ['pick_place','stack']:
    text=text.replace("qinit[side].update(open_q(side))","qinit[side].update(open_q(side)); jitter_side(qinit[side],side)")
elif a.task=='microwave_open':
    text=text.replace("initial['left']=approach[0]","initial['left']=approach[0].copy(); jitter_side(initial['left'],'left'); approach[0]=initial['left'].copy()")
elif a.task=='piano':
    text=text.replace("initial['right']=above.copy()","initial['right']=above.copy(); jitter_side(initial['right'],'right')")
else:
    text=text.replace("initial[side].update(open_q(side))","initial[side].update(open_q(side)); jitter_side(initial[side],side)")
if a.task.startswith('gesture'):
    text=text.replace("raised.update(zip(arm_names('right'),qa))","raised.update(zip(arm_names('right'),qa)); T=fk('right_palm_link','right',raised); T[:3,3]+=rng.uniform(-.008,.008,3); qa,pe,re=solve('right',T,qa); assert pe<.001; raised.update(zip(arm_names('right'),qa))")
if a.task=='piano':
    text=text.replace('from isaacsim import SimulationApp',"sequence=[(max(.04,d+float(rng.uniform(-.025,.025))) if ph in ['move','press','release'] else d,q,ph,key) for d,q,ph,key in sequence]\nfrom isaacsim import SimulationApp")
# Install after SimulationApp initialization, before importing core simulation classes.
lines=text.splitlines();idx=next(i for i,l in enumerate(lines) if l.startswith('app=SimulationApp('))
lines[idx+1:idx+1]=["from dataset_recorder import Recorder", "collector=Recorder(out,COLLECT_TASK,COLLECT_SEED,COLLECT_SPEC); collector.install()"]
text='\n'.join(lines).replace('app.close()',"collector.finish(); save_provenance(); app.close()")
arguments=['--name',a.name,'--gpu',str(a.gpu),*cfg['args']]
if a.task=='microwave_open':
    i=arguments.index('--approach-seconds')+1;arguments[i]=str(1.8+float(rng.uniform(-.04,.04)))
sys.argv=[str(source),*arguments]
spec['initial_robot_variation']={'arm_rad':.012,'finger_rad':.018,'active_side':active,'seed':a.seed}
def save_provenance():
    out.mkdir(parents=True,exist_ok=True)
    (out/'collection_provenance.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'instrumented_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':a.seed,'initial_joint_deltas':initial_log,'args':arguments},indent=2))
namespace={'save_provenance':save_provenance,'__name__':'__main__','__file__':str(source),'jitter_side':jitter_side,'rng':rng,'COLLECT_TASK':a.task,'COLLECT_SEED':a.seed,'COLLECT_SPEC':spec}
try:exec(compile(text,str(source),'exec'),namespace)
finally:
    out.mkdir(parents=True,exist_ok=True)
    (out/'collection_provenance.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'instrumented_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':a.seed,'initial_joint_deltas':initial_log,'args':arguments},indent=2))
