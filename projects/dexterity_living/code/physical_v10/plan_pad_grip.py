from model import *
from pxr import Usd,UsdGeom
from table_collision import TableCollision
import itertools,json
old=json.loads((P/'runs/physical_v1/pick_grip.json').read_text())[0]['q'];s=Usd.Stage.Open(str(P/'runs/physical_v1/pick_enclosure_v11_gpu0/initial_scene.usda'));cache=UsdGeom.XformCache(0);centers={}
for fi in [1,2]:
 link=f'right_finger{fi}_link4';body=s.GetPrimAtPath('/World/Robot_right/'+link);pr=next(p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh) and '/visuals/' in str(p.GetPath()) and not str(p.GetPath()).endswith('_enclosing_collision'));v=np.array(UsdGeom.Mesh(pr).GetPointsAttr().Get());T=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)@np.array(cache.GetLocalToWorldTransform(pr)).T;v=v@T[:3,:3].T+T[:3,3];centers[fi]=(v.min(0)+v.max(0))/2
check=TableCollision(P/'runs/physical_v1/pick_enclosure_v11_gpu0/initial_scene.usda');baseT=fk('right_palm_link','right',old);results=[]
for x,y,z in itertools.product([-1.82,-1.84],[-3.56,-3.58,-3.60],[.755,.765,.775,.785,.795]):
 T=baseT.copy();T[:3,3]=[x,y,z];qa,pe,re=solve('right',T,[old[n] for n in arm_names('right')]);q=old.copy();q.update(zip(arm_names('right'),qa));errors=[]
 for fi,sgn in [(1,-1),(2,1)]:
  ns=[n for n in hand_names('right') if f'finger{fi}_' in n];target=np.array([-1.87+sgn*.032,-3.495,.7053]);center=centers[fi]
  def residual(xx):
   d=q.copy();d.update(zip(ns,xx));F=fk(f'right_finger{fi}_link4','right',d);return F[:3,3]+F[:3,:3]@center-target
  opt=least_squares(residual,[q[n] for n in ns],bounds=([joints[n]['lo'] for n in ns],[joints[n]['hi'] for n in ns]),max_nfev=180);q.update(zip(ns,opt.x));errors.append(float(np.linalg.norm(residual(opt.x))))
 hits=check.check(q);score=sum(errors)+pe+.1*re+len(hits);results.append(dict(q=q,pad_errors_m=errors,arm_errors=[pe,re],table_hits=hits,score=score,pad_centers_local={str(k):v.tolist() for k,v in centers.items()}));print(x,y,z,'error',errors,'table',len(hits),flush=True)
results.sort(key=lambda x:x['score']);(P/'runs/physical_v1/pick_pad_grips.json').write_text(json.dumps(results,indent=2));print('BEST',results[0],flush=True)
