"""Kinematic clearance plan for an extended index to push the inside upper door frame."""
import sys,json,itertools,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
from pxr import Usd,UsdGeom
ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--x',type=float,default=.41);ap.add_argument('--z',type=float,default=.025);a=ap.parse_args()
g=P/'runs/physical_v1/gesture_one_fluid_v12';r=json.loads((g/'report.json').read_text());cs=json.loads((g/'contacts.json').read_text());print('RECORD',cs[-1].keys(),flush=True);fixed=dict(zip(r['dof_names'],cs[120]['q']))
s=Usd.Stage.Open(str(g/'initial_scene.usda'));cache=UsdGeom.XformCache();probes={}
for l in links('right'):
 if not ('finger' in l or 'palm' in l or l in ['fr_link4','fr_link5','fr_link6']):continue
 body=s.GetPrimAtPath('/World/Robot_right/'+l);vs=[]
 for pr in Usd.PrimRange(body):
  if pr.IsA(UsdGeom.Mesh) and '/visuals/' in str(pr.GetPath()):
   T=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)@np.array(cache.GetLocalToWorldTransform(pr)).T;v=np.array(UsdGeom.Mesh(pr).GetPointsAttr().Get());vs.append(v@T[:3,:3].T+T[:3,3])
 if vs:
  v=np.concatenate(vs);probes[l]=np.array(list(itertools.product(*zip(v.min(0),v.max(0)))))
ns=arm_names('right');lo=np.array([joints[n]['lo']+1e-5 for n in ns]);hi=np.array([joints[n]['hi']-1e-5 for n in ns]);h=np.array([-2.2972766,-3.4886971,.9657585]);rng=np.random.default_rng(5);out=P/'runs/physical_v1'/a.tag;out.mkdir(parents=True,exist_ok=True);results=[]
for trial in range(12):
 seed=np.array([fixed[n] for n in ns])
 if trial:seed=rng.uniform(lo,hi)
 path=[]
 for angle in range(20,77,2):
  R=Rot.from_euler('z',-angle,degrees=True).as_matrix();target=h+R@np.array([a.x,.012,a.z]);normal=.6*np.array([0,0,1])+.8*R[:,1]
  def fun(x):
   q=fixed.copy();q.update(zip(ns,x));Ts=fk_many(list(probes),'right',q);T=Ts['right_finger2_tip_link'];got=T[:3,3]+T[:3,:3]@np.array([0,0,-.00755]);pen=[]
   for l,v in probes.items():
    T=Ts[l];v=v@T[:3,:3].T+T[:3,3];local=(v-h)@R;pen.extend(np.maximum(np.minimum.reduce([1.027-v[:,2],.004-local[:,1],local[:,0]+.015,.48-local[:,0]]),0));pen.extend(np.maximum(np.minimum(1.04-v[:,2],v[:,1]-(h[1]-.06)),0));pen.extend(np.maximum(.704-v[:,2],0))
   return np.r_[5*(got-target),.003*(Ts['right_finger2_tip_link'][:3,2]-normal),3*np.array(pen),.0005*(x-seed)]
  opt=least_squares(fun,np.clip(seed,lo,hi),bounds=(lo if angle==20 else np.maximum(lo,seed-.3),hi if angle==20 else np.minimum(hi,seed+.3)),max_nfev=300 if angle==20 else 100)
  e=fun(opt.x);pe=float(np.max(np.abs(e[:3]))/5);ge=float(np.max(e[6:-6]))
  if pe>.002 or ge>.0015:break
  seed=opt.x;q=fixed.copy();q.update(zip(ns,map(float,seed)));path.append(dict(angle_deg=angle,q=q,position_error_m=pe))
 print('PLAN',trial,'last',path[-1]['angle_deg'] if path else -1,'pos',pe,'geo',ge,flush=True);results.append(dict(path=path,last_angle=path[-1]['angle_deg'] if path else -1))
 if len(path)==29:break
results.sort(key=lambda x:-x['last_angle']);(out/'plan.json').write_text(json.dumps(dict(candidates=results,parameters=vars(a)),indent=2))
