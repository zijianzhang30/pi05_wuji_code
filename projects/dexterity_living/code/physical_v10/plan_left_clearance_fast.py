from model import *
from pxr import Usd,UsdGeom
from precise_table_collision import TableCollision
from dual_collision import DualCollision
import itertools,json,argparse
p=argparse.ArgumentParser();p.add_argument('--direction',type=float,default=0);p.add_argument('--orientation-weight',type=float,default=.03);p.add_argument('--height',type=float,default=.705);p.add_argument('--tag',default='tip_full');a=p.parse_args();source=P/'runs/physical_v1/pick_tip_v18_gpu1';rows=json.loads((source/'contacts.json').read_text());report=json.loads((source/'report.json').read_text());handoff_frame=max(i for i,r in enumerate(rows) if r['phase']=='transfer');handoff_row=rows[handoff_frame];object_center=np.array(handoff_row['object_pose'][:3]);right_q=dict(zip(report['dof_names'],handoff_row['q']));old={}
for n in hand_names('right'):
 if any(f'finger{fi}_' in n for fi in [3,4,5]):
  ji=int(n[-1]);right_q[n]=float(np.clip([0.,0.,.9,1.,.7][ji],joints[n]['lo']+1e-5,joints[n]['hi']-1e-5))

for n in arm_names('left')+hand_names('left'):
 rn=n.replace('fl_joint','fr_joint').replace('left_','right_');v=right_q[rn]
 if n in ['fl_joint1','fl_joint4','fl_joint6'] or (joints[n]['lo']>0 and joints[rn]['hi']<0) or (joints[n]['hi']<0 and joints[rn]['lo']>0):v=-v
 old[n]=float(np.clip(v,joints[n]['lo']+1e-5,joints[n]['hi']-1e-5))
scene=P/'runs/physical_v1/pick_tip_v18_gpu1/initial_scene.usda';s=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache(0);centers={};probes=[];surfaces=[]
for link in [n for n in links('left') if 'finger' in n or 'palm' in n or n in ['fl_link5','fl_link6']]:
 body=s.GetPrimAtPath('/World/Robot_left/'+link);meshes=[p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath()) and UsdGeom.Imageable(p).ComputeVisibility(0)!='invisible']
 if not meshes:continue
 vertices=[]
 for pr in meshes:
  v=np.array(UsdGeom.Mesh(pr).GetPointsAttr().Get());T=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)@np.array(cache.GetLocalToWorldTransform(pr)).T;vertices.append(v@T[:3,:3].T+T[:3,3])
 v=np.concatenate(vertices);center=(v.min(0)+v.max(0))/2
 if link in ['left_finger1_tip_link','left_finger2_tip_link']:centers[link]=center
 surfaces.append((link,v[::max(1,len(v)//16)]));probes.append((link,np.array(list(itertools.product(*zip(v.min(0),v.max(0)))))))
old=json.loads((P/'runs/physical_v1/handoff_ranked.json').read_text())[0]['q']
ordered=sorted([j for j in joints.values() if j['parent'] in set(links('left')) and j['child'] in set(links('left'))],key=lambda j:len(chain(j['child'],'left')))
def all_fk(q):
 d={'fl_base_link':base['left']}
 for j in ordered:
  T=d[j['parent']]@j['T']
  if j['kind']=='revolute':T=T.copy();T[:3,:3]=T[:3,:3]@Rot.from_rotvec(j['axis']*q.get(j['name'],0)).as_matrix()
  d[j['child']]=T
 return d
names=arm_names('left')+[n for n in hand_names('left') if 'finger1_' in n or 'finger2_' in n];lo=np.array([joints[n]['lo']+1e-6 for n in names]);hi=np.array([joints[n]['hi']-1e-6 for n in names]);rng=np.random.default_rng(210+int(a.direction));direction=np.array([np.cos(np.deg2rad(a.direction)),np.sin(np.deg2rad(a.direction)),0]);targets=object_center+np.array([-1,1])[:,None]*direction*.030
obstacles=[]
for pr in s.Traverse():
 path=str(pr.GetPath())
 if not(path.startswith('/World/Robot_right/') and '/visuals/' in path and pr.IsA(UsdGeom.Mesh)) or UsdGeom.Imageable(pr).ComputeVisibility(0)=='invisible':continue
 link=path.split('/')[3];body=s.GetPrimAtPath('/World/Robot_right/'+link);v=np.asarray(UsdGeom.Mesh(pr).GetPointsAttr().Get());T=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)@np.array(cache.GetLocalToWorldTransform(pr)).T;v=v@T[:3,:3].T+T[:3,3];X=fk(link,'right',right_q);center=(v.min(0)+v.max(0))/2;extent=(v.max(0)-v.min(0))/2
 if np.linalg.norm(X[:3,:3]@center+X[:3,3]-object_center)<.22:obstacles.append((X,center,extent))
check=DualCollision(scene,right_q);results=[]
for trial in range(4):
 seed=np.clip(np.array([old[n] for n in names])+rng.normal(0,.25 if trial else 0,len(names)),lo,hi)
 def fun(x):
  q=old.copy();q.update(zip(names,x));d=all_fk(q);got=np.array([d[link][:3,3]+d[link][:3,:3]@center for link,center in centers.items()]);pen=[]
  for link,corners in probes:
   T=d[link];v=corners@T[:3,:3].T+T[:3,3];pen.extend(np.maximum(.686-v[:,2],0))
  for link,v in surfaces:
   T=d[link];world=v@T[:3,:3].T+T[:3,3];outside=np.abs(world-object_center)-np.array([.0238,.0238,.0238]);dist=np.linalg.norm(np.maximum(outside,0),axis=1)+np.minimum(outside.max(1),0);pen.extend(np.maximum(.0005-dist,0))
   for X,cc,ee in obstacles:
    local=(world-X[:3,3])@X[:3,:3];outside=np.abs(local-cc)-ee;dist=np.linalg.norm(np.maximum(outside,0),axis=1)+np.minimum(outside.max(1),0);pen.extend(np.maximum(.002-dist,0))
  directions=np.array([d[l][:3,2] for l in centers]);desired=np.array([direction,-direction]);return np.r_[100*(got-targets).ravel(),100*a.orientation_weight*(directions-desired).ravel(),200*np.array(pen),.003*(x-seed)]
 opt=least_squares(fun,seed,bounds=(lo,hi),max_nfev=150);q=old.copy();q.update(zip(names,opt.x));d=all_fk(q);err=np.array([d[l][:3,3]+d[l][:3,:3]@c for l,c in centers.items()])-targets;hits=check.check(q);res=dict(handoff_frame=handoff_frame,object_center=object_center.tolist(),right_q=right_q,q=q,errors_m=np.linalg.norm(err,axis=1).tolist(),table_hits=hits,direction_deg=a.direction,score=float(np.linalg.norm(fun(opt.x)))+len(hits),pad_centers={k:v.tolist() for k,v in centers.items()});results.append(res);print('RESULT',trial,res['errors_m'],len(hits),'palm',d['left_palm_link'][:3,3],flush=True)
results.sort(key=lambda x:x['score']);(P/'runs/physical_v1'/f'{a.tag}.json').write_text(json.dumps(results,indent=2))
