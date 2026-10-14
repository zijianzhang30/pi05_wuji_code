from model import *
from table_collision import TableCollision
import json,time
best=json.loads((P/'runs/physical_v1/pick_pad_candidate_3.json').read_text())[0];goal=best['q'];ns=arm_names('right');T=fk('right_palm_link','right',goal);T[:3,3]+=np.array([0,-.015,.10]);qa,pe,re=solve('right',T,[goal[n] for n in ns]);goal=goal.copy();goal.update(zip(ns,qa));end=np.array([goal[n] for n in ns]);start=np.array([-.65,2.12,-.95,-.17,-1.18,-.08]);audit=TableCollision(P/'runs/physical_v1/pick_enclosure_v11_gpu0/initial_scene.usda');rng=np.random.default_rng(421);checks=0
lo=np.array([joints[n]['lo'] for n in ns]);hi=np.array([joints[n]['hi'] for n in ns])
def valid(x):
 global checks
 checks+=1;q=goal.copy();q.update(zip(ns,x));return not audit.check(q)
def edge(a,b):return all(valid(a+(b-a)*u) for u in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.02)+1)))
# Try a wrist-first path before sampling.
via=end.copy();via[0]=start[0];path=None
if edge(start,via) and edge(via,end):path=[start,via,end]
if path is None:
 nodes=[start];parents=[-1];beg=time.time()
 for iteration in range(1400):
  target=end if rng.random()<.3 else np.clip(start+(end-start)*rng.random()+rng.normal(0,.6,6),lo,hi)
  near=int(np.argmin([np.linalg.norm(x-target) for x in nodes]));d=target-nodes[near];x=nodes[near]+d*min(1,.3/max(np.linalg.norm(d),1e-10))
  if not edge(nodes[near],x):continue
  nodes.append(x);parents.append(near)
  if np.linalg.norm(x-end)<.75 and edge(x,end):
   path=[end];i=len(nodes)-1
   while i>=0:path.append(nodes[i]);i=parents[i]
   path.reverse();break
  if iteration%100==0:print('RRT',iteration,'nodes',len(nodes),'seconds',time.time()-beg,flush=True)
assert path is not None,('no path',checks)
for _ in range(70):
 if len(path)<3:break
 i,j=sorted(rng.choice(len(path),2,replace=False))
 if j>i+1 and edge(path[i],path[j]):path=path[:i+1]+path[j:]
assert all(edge(a,b) for a,b in zip(path[:-1],path[1:]));durations=[max(1.,float(np.max(np.abs(b-a))/.35)) for a,b in zip(path[:-1],path[1:])]
report={'pregrasp_q':goal,'arm_waypoints':[q.tolist() for q in path],'segment_seconds':durations,'duration':sum(durations),'geometry_checks':checks,'nonhandle_intersections':0,'discrete_joint_step_rad':.045,'scope':'microwave mesh clearance only; dynamics and table/self clearance require runtime validation'}
(P/'runs/physical_v1/approach_pad_3.json').write_text(json.dumps(report,indent=2));print('PATH',json.dumps(report),flush=True)
