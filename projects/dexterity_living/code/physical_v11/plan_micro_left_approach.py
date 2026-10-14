
import sys,json,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
from pxr import Usd,UsdGeom,UsdPhysics
scene=P/'runs/physical_v1/micro_self_v22_gpu1/initial_scene.usda';stage=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache();bodies=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)];Ts=[np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T for p in bodies]
for idx in [0,1]:
 plan=json.loads((P/f'runs/physical_v1/micro_left_hook_plan_{idx}/plan.json').read_text());goal=plan['candidates'][0]['path'][0]['q'];palm=fk('left_palm_link','left',goal);previous=[goal[n] for n in arm_names('left')];path=[];errors=[]
 for distance in np.linspace(0,-.075,26):
  T=palm.copy();T[0,3]+=distance;x,pe,re=solve('left',T,previous);previous=x;path.append({**goal,**dict(zip(arm_names('left'),map(float,x)))});errors.append([pe,re])
 path=path[::-1];out=P/f'runs/physical_v1/micro_left_hook_approach_{idx}';out.mkdir(exist_ok=True);shutil.copy2(scene,out/'initial_scene.usda');poses=[]
 for q in path:
  frame=[]
  for name,T in zip(bodies,Ts):
   if name.startswith('/World/Robot_left/'):T=fk(name.split('/')[-1],'left',q)
   elif name.startswith('/World/Robot_right/'):
    qq=dict(zip(arm_names('right'),[-.65,2.12,-.95,-.17,-1.18,-.08]));qq.update(open_q('right'));T=fk(name.split('/')[-1],'right',qq)
   frame.append(np.r_[T[:3,3],Rot.from_matrix(T[:3,:3]).as_quat()])
  poses.append(frame)
 np.savez_compressed(out/'trajectory.npz',body_paths=bodies,poses=np.array(poses),time=np.arange(len(poses))/30)
 (out/'plan.json').write_text(json.dumps(dict(path=path,errors=errors,duration=1.2),indent=2));(out/'report.json').write_text(json.dumps(dict(status='planning_only',physics_stepped=False)))
 print('APPROACH',idx,'max pe/re',np.max(errors,axis=0),flush=True)
 subprocess.run([sys.executable,str(P/'code/physical_v11/audit_full_geometry.py'),str(out)],check=True)
