from model import *
from pxr import Usd,UsdGeom
from precise_table_collision import TableCollision
import itertools,json,argparse
p=argparse.ArgumentParser();p.add_argument('--direction',type=float,default=0);p.add_argument('--orientation-weight',type=float,default=.03);p.add_argument('--tag',default='tip_full');a=p.parse_args();old=json.loads((P/'runs/physical_v1/pick_grip.json').read_text())[0]['q'];scene=P/'runs/physical_v1/pick_enclosure_v11_gpu0/initial_scene.usda';s=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache(0);centers={};probes=[]
for link in [n for n in links('right') if 'finger' in n or 'palm' in n]:
 body=s.GetPrimAtPath('/World/Robot_right/'+link);meshes=[p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath()) and UsdGeom.Imageable(p).ComputeVisibility(0)!='invisible']
 if not meshes:continue
 vertices=[]
 for pr in meshes:
  v=np.array(UsdGeom.Mesh(pr).GetPointsAttr().Get());T=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)@np.array(cache.GetLocalToWorldTransform(pr)).T;vertices.append(v@T[:3,:3].T+T[:3,3])
 v=np.concatenate(vertices);center=(v.min(0)+v.max(0))/2
 if link in ['right_finger1_tip_link','right_finger2_tip_link']:centers[link]=center
 probes.append((link,np.array(list(itertools.product(*zip(v.min(0),v.max(0)))))))
ordered=sorted([j for j in joints.values() if j['parent'] in set(links('right')) and j['child'] in set(links('right'))],key=lambda j:len(chain(j['child'],'right')))
def all_fk(q):
 d={'fr_base_link':base['right']}
 for j in ordered:
  T=d[j['parent']]@j['T']
  if j['kind']=='revolute':T=T.copy();T[:3,:3]=T[:3,:3]@Rot.from_rotvec(j['axis']*q.get(j['name'],0)).as_matrix()
  d[j['child']]=T
 return d
names=arm_names('right')+[n for n in hand_names('right') if 'finger1_' in n or 'finger2_' in n];lo=np.array([joints[n]['lo']+1e-6 for n in names]);hi=np.array([joints[n]['hi']-1e-6 for n in names]);rng=np.random.default_rng(210+int(a.direction));direction=np.array([np.cos(np.deg2rad(a.direction)),np.sin(np.deg2rad(a.direction)),0]);targets=np.array([-1.87,-3.495,.710])+np.array([-1,1])[:,None]*direction*.034
check=TableCollision(scene);results=[]
for trial in range(8):
 seed=np.clip(np.array([old[n] for n in names])+rng.normal(0,.25 if trial else 0,len(names)),lo,hi)
 def fun(x):
  q=old.copy();q.update(zip(names,x));d=all_fk(q);got=np.array([d[link][:3,3]+d[link][:3,:3]@center for link,center in centers.items()]);pen=[]
  for link,corners in probes:
   T=d[link];v=corners@T[:3,:3].T+T[:3,3];pen.extend(np.maximum(.686-v[:,2],0))
  directions=np.array([d[l][:3,2] for l in centers]);desired=np.array([direction,-direction]);return np.r_[100*(got-targets).ravel(),100*a.orientation_weight*(directions-desired).ravel(),200*np.array(pen),.003*(x-seed)]
 opt=least_squares(fun,seed,bounds=(lo,hi),max_nfev=350);q=old.copy();q.update(zip(names,opt.x));d=all_fk(q);err=np.array([d[l][:3,3]+d[l][:3,:3]@c for l,c in centers.items()])-targets;hits=check.check(q);res=dict(q=q,errors_m=np.linalg.norm(err,axis=1).tolist(),table_hits=hits,direction_deg=a.direction,score=float(np.linalg.norm(err))+len(hits),pad_centers={k:v.tolist() for k,v in centers.items()});results.append(res);print('RESULT',trial,res['errors_m'],len(hits),'palm',d['right_palm_link'][:3,3],flush=True)
results.sort(key=lambda x:x['score']);(P/'runs/physical_v1'/f'{a.tag}.json').write_text(json.dumps(results,indent=2))
