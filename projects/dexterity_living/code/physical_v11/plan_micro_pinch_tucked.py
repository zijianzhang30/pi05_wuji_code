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
caps={}
for l,v in probes.items():
 if 'finger' not in l or not any(x in l for x in ['link3','link4','tip_link']):continue
 mn,mx=v.min(0),v.max(0);span=mx-mn;axis=int(np.argmax(span));radius=float(np.linalg.norm(np.delete(span/2,axis)))+.0005
 samples=np.tile((mx+mn)/2,(5,1));samples[:,axis]=np.linspace(mn[axis],mx[axis],5);caps[l]=(samples,radius,int(l.split('finger')[1][0]))
capnames=list(caps);cappairs=[(x,y) for k,x in enumerate(capnames) for y in capnames[k+1:] if caps[x][2]!=caps[y][2] and min(caps[x][2],caps[y][2])<=2]
seedq=json.loads((P/'runs/physical_v1/micro_pinch_plan_b0/plan.json').read_text())['candidates'][0]['path'][0]['q']
g=P/'runs/physical_v1/gesture_one_fluid_v12';rr=json.loads((g/'report.json').read_text());cc=json.loads((g/'contacts.json').read_text());gq=dict(zip(rr['dof_names'],cc[120]['q']))
for n in hand_names('right'):
 if any(f'finger{fi}_' in n for fi in [3,4,5]):seedq[n]=gq[n]
ns=arm_names('right')+[n for n in hand_names('right') if 'finger1_' in n or 'finger2_' in n];lo=np.array([joints[n]['lo']+1e-5 for n in ns]);hi=np.array([joints[n]['hi']-1e-5 for n in ns]);rng=np.random.default_rng(31+a.sign)
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
   qq=seedq.copy();qq.update(zip(ns,x));Ts=fk_many(list(probes), 'right',qq);got=np.array([Ts[l][:3,3]+Ts[l][:3,:3]@o for l,o in zip(padlinks,offsets)]);normals=np.array([Ts[l][:3,2] for l in padlinks]);pen=[]
   for l,verts in probes.items():
    T=Ts[l];v=verts@T[:3,:3].T+T[:3,3];local=(v-h)@R
    pen.extend(np.maximum(local[:,1]+.043,0));pen.extend(np.maximum(.704-v[:,2],0))
   # Keep index and unused fingers separated, without changing collision filters.
   tips=np.array([Ts[f'right_finger{i}_tip_link'][:3,3] for i in [2,3,4,5]])
   separate=np.maximum(0,.025-np.linalg.norm(tips[:,None]-tips[None,:],axis=-1))[np.triu_indices(4,1)]
   cps={l:caps[l][0]@Ts[l][:3,:3].T+Ts[l][:3,3] for l in capnames};handlepen=[]
   for l,points in cps.items():
    local=(points-h)@R;d=np.abs(local-np.array([.4391,-.05775,-.11155]))-np.array([.0125,.01795,.12445]);sdf=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0);handlepen.extend(np.maximum(caps[l][1]+.001-sdf,0))
   selfpen=[max(0,caps[x][1]+caps[y][1]+.0005-float(np.min(np.linalg.norm(cps[x][:,None,:]-cps[y][None,:,:],axis=-1)))) for x,y in cappairs]
   return np.r_[5*(got-targets).ravel(),.005*(normals-want).ravel(),3*np.asarray(pen),3*np.asarray(handlepen),3*np.asarray(selfpen),2*separate,.001*(x-seed)]
  opt=least_squares(fun,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo if angle==0 else np.maximum(lo,seed-.20),hi if angle==0 else np.minimum(hi,seed+.20)),max_nfev=300 if angle==0 else 140,ftol=1e-7)
  residual=fun(opt.x);error=float(np.max(np.abs(residual[:6]))/5);geom=float(np.max(residual[12:-len(ns)])) if len(residual)>12+len(ns) else 0.
  if error>.001 or geom>.003:break
  seed=opt.x;path.append(dict(angle_deg=angle,q={**seedq,**dict(zip(ns,map(float,seed)))},position_error_m=error));angle_done=angle
 print('PLAN',trial,'reached',angle_done,'error',error,'geom',geom,flush=True)
 results.append(dict(last_q=dict(zip(ns,map(float,opt.x))),trial=trial,last_angle=angle_done,path=path,error_m=error,geometry_residual=geom))
 if angle_done>=70:break
results.sort(key=lambda x:-x['last_angle']);(out/'plan.json').write_text(json.dumps(dict(parameters=vars(a),targets_local=targets0.tolist(),candidates=results),indent=2))
