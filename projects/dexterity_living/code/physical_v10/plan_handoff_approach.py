from model import *
from dual_collision import DualCollision
import json,time
cand=json.loads((P/'runs/physical_v1/left_folded.json').read_text())[0];q=cand['q'];axis=np.array([-1.,0,0]);wide=q.copy()
wide=json.loads((P/'runs/physical_v1/left_folded_wide.json').read_text())[0]['q']
audit=DualCollision(P/'runs/physical_v1/pick_tip_v18_gpu1/initial_scene.usda',cand['right_q']);from cube_obstacle import attach_cube
attach_cube(audit,cand)
ns=arm_names('left');start=np.array([.65,2.12,-.95,.17,-1.18,.08]);lo=np.array([joints[n]['lo'] for n in ns]);hi=np.array([joints[n]['hi'] for n in ns]);rng=np.random.default_rng(42)
def valid(x):
 d=wide.copy();d.update(zip(ns,x));return not audit.check(d)
def edge(a,b):return all(valid(a+(b-a)*u) for u in np.linspace(0,1,max(2,int(np.max(np.abs(a-b))/.035)+1)))
assert valid(start),'wide preshape invalid'
target=fk('left_palm_link','left',wide)
end=np.array([wide[n] for n in ns]);assert valid(end),'wide endpoint collides with right hand'
# Find a Cartesian escape from the narrow final grasp corridor, then RRT-connect to it.
final_end=end.copy();tail=None
for offset in [np.array([0,-.12,.02]),np.array([-.1,0,0]),np.array([0,0,.10]),np.array([-.08,-.06,.04]),np.array([0,-.05,0]),np.array([.06,-.06,.05])]:
 backwards=[final_end];seed=final_end.copy();T=fk('left_palm_link','left',wide);ok=True
 for u in np.linspace(.05,1,20):
  X=T.copy();X[:3,3]+=offset*u;qa,pe,re=solve('left',X,seed)
  if pe>.003 or re>.03 or not edge(seed,qa):
   print('ESCAPE_FAIL',offset,u,pe,re,audit.check({**wide,**dict(zip(ns,qa))}),flush=True);ok=False;break
  backwards.append(qa);seed=qa
 if ok:tail=list(reversed(backwards));end=tail[0];print('ESCAPE',offset,flush=True);break
assert tail is not None,'no Cartesian escape from final grasp'
path=None
if edge(start,end):path=[start,end]
if path is None:
 trees=[dict(nodes=[start],parents=[-1],label=0),dict(nodes=[end],parents=[-1],label=1)];beg=time.time()
 def extend(tree,target):
  nearest=int(np.argmin([np.linalg.norm(x-target) for x in tree['nodes']]));diff=target-tree['nodes'][nearest];x=tree['nodes'][nearest]+diff*min(1,.20/max(np.linalg.norm(diff),1e-9))
  if not edge(tree['nodes'][nearest],x):return None
  tree['nodes'].append(x);tree['parents'].append(nearest);return len(tree['nodes'])-1
 def trace(tree,idx):
  result=[]
  while idx>=0:result.append(tree['nodes'][idx]);idx=tree['parents'][idx]
  return list(reversed(result))
 for k in range(3000):
  A,B=trees;sample=np.clip(start+(end-start)*rng.random()+rng.normal(0,.65,6),lo,hi);i=extend(A,sample)
  if i is not None:
   for _ in range(50):
    j=extend(B,A['nodes'][i])
    if j is None:break
    if np.linalg.norm(B['nodes'][j]-A['nodes'][i])<1e-5:
     pa,pb=trace(A,i),trace(B,j);path=pa+list(reversed(pb))[1:] if A['label']==0 else pb+list(reversed(pa))[1:];break
  if path is not None:break
  trees.reverse()
  if time.time()-beg>60:break
 assert path is not None,'RRT-connect did not find a safe approach'
path+=tail[1:]
for _ in range(40):
 if len(path)<3:break
 i,j=sorted(rng.choice(len(path),2,replace=False))
 if j>i+1 and edge(path[i],path[j]):path=path[:i+1]+path[j:]
dur=[max(1.0,float(np.max(np.abs(b-a))/.4)) for a,b in zip(path[:-1],path[1:])]
out=dict(candidate=cand,wide=wide,arm_waypoints=[x.tolist() for x in path],segment_seconds=dur,scope="Static inter-arm and conservative table clearance; cube contact must be audited dynamically")
(P/'runs/physical_v1/handoff_approach.json').write_text(json.dumps(out,indent=2));print("DONE",sum(dur),flush=True)
