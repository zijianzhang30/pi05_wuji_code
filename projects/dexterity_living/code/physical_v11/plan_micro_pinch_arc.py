"""Plan an opposed thumb-index grip along the complete passive door arc."""
import sys,json,argparse,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
from pxr import Usd,UsdGeom
ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--height',type=float,default=-.12);ap.add_argument('--sign',type=int,default=1);a=ap.parse_args()
out=P/'runs/physical_v1'/a.tag;out.mkdir(parents=True,exist_ok=True)
stage=Usd.Stage.Open(str(P/'runs/physical_v1/micro_self_v22_gpu1/initial_scene.usda'));cache=UsdGeom.XformCache(0);probes={}
for l in links('right'):
 if 'finger' not in l and 'palm' not in l:continue
 body=stage.GetPrimAtPath('/World/Robot_right/'+l);vs=[]
 for pr in Usd.PrimRange(body):
  if not pr.IsA(UsdGeom.Mesh) or '/visuals/' not in str(pr.GetPath()):continue
  T=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(pr)).T;v=np.asarray(UsdGeom.Mesh(pr).GetPointsAttr().Get());vs.append(v@T[:3,:3].T+T[:3,3])
 if vs:
  v=np.concatenate(vs);probes[l]=np.array(list(itertools.product(*zip(v.min(0),v.max(0)))))
seedq=json.loads((P/'runs/physical_v1/contact_tip_1.json').read_text())[0]['q']
for fi in [3,4,5]:
 for ji,v in enumerate([0,.1,.8,.5],1):
  n=f'right_finger{fi}_joint{ji}';seedq[n]=float(np.clip(v,joints[n]['lo']+.001,joints[n]['hi']-.001))
ns=arm_names('right')+hand_names('right');lo=np.array([joints[n]['lo']+1e-5 for n in ns]);hi=np.array([joints[n]['hi']-1e-5 for n in ns]);rng=np.random.default_rng(31+a.sign)
h=np.array([-2.2972766,-3.4886971,.9657585]);padlinks=['right_finger1_tip_link','right_finger2_tip_link'];offsets=[np.array([0,0,-.003083]),np.array([0,0,-.00755])]
targets0=np.array([[.4391-a.sign*.024,-.063,a.height],[.4391+a.sign*.024,-.063,a.height]])
results=[]
for trial in range(8):
 seed=np.array([seedq[n] for n in ns])
 if trial:seed=np.clip(seed+rng.normal(0,.5,len(seed)),lo+.001,hi-.001)
 path=[];angle_done=-1
 for angle in range(0,71,2):
  R=Rot.from_euler('z',-angle,degrees=True).as_matrix();targets=targets0@R.T+h;want=np.array([[a.sign,0,0],[-a.sign,0,0]])@R.T
  def fun(x):
   qq=dict(zip(ns,x));Ts=fk_many(list(probes), 'right',qq);got=np.array([Ts[l][:3,3]+Ts[l][:3,:3]@o for l,o in zip(padlinks,offsets)]);normals=np.array([Ts[l][:3,2] for l in padlinks]);pen=[]
   for l,verts in probes.items():
    T=Ts[l];v=verts@T[:3,:3].T+T[:3,3];local=(v-h)@R
    pen.extend(np.maximum(local[:,1]+.043,0));pen.extend(np.maximum(.704-v[:,2],0))
   # Keep index and unused fingers separated, without changing collision filters.
   tips=np.array([Ts[f'right_finger{i}_tip_link'][:3,3] for i in [2,3,4,5]])
   separate=np.maximum(0,.025-np.linalg.norm(tips[:,None]-tips[None,:],axis=-1))[np.triu_indices(4,1)]
   return np.r_[5*(got-targets).ravel(),.003*(normals-want).ravel(),3*np.asarray(pen),2*separate,.0005*(x-seed)]
  opt=least_squares(fun,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=250 if angle==0 else 100,ftol=1e-7)
  residual=fun(opt.x);error=float(np.max(np.abs(residual[:6]))/5);geom=float(np.max(residual[12:-len(ns)])) if len(residual)>12+len(ns) else 0.
  if error>.002 or geom>.0005:break
  seed=opt.x;path.append(dict(angle_deg=angle,q=dict(zip(ns,map(float,seed))),position_error_m=error));angle_done=angle
 print('PLAN',trial,'reached',angle_done,'error',error,'geom',geom,flush=True)
 results.append(dict(trial=trial,last_angle=angle_done,path=path,error_m=error,geometry_residual=geom))
 if angle_done>=70:break
results.sort(key=lambda x:-x['last_angle']);(out/'plan.json').write_text(json.dumps(dict(parameters=vars(a),targets_local=targets0.tolist(),candidates=results),indent=2))
