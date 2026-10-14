"""Geometry-only candidate audit; synthetic poses are NOT a physics demo."""
import sys,json,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
from pxr import Usd,UsdGeom,UsdPhysics
scene=P/'runs/physical_v1/micro_self_v22_gpu1/initial_scene.usda';stage=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache();bodies=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)];Ts=[np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))).T for p in bodies];h=np.array([-2.2972766,-3.4886971,.9657585])
for i in range(2):
 p=P/f'runs/physical_v1/micro_left_tucked_v2_{i}';out=p/'geometry_only';out.mkdir(exist_ok=True);shutil.copy2(scene,out/'initial_scene.usda');plan=json.loads((p/'plan.json').read_text());poses=[]
 for row in plan['candidates'][0]['path']:
  R=Rot.from_euler('z',-row['angle_deg'],degrees=True).as_matrix();Q=np.eye(4);Q[:3,:3]=R;Q[:3,3]=h-R@h;frame=[]
  for name,T in zip(bodies,Ts):
   if name.startswith('/World/Robot_left/'):T=fk(name.split('/')[-1],'left',row['q'])
   elif name.startswith('/World/Robot_right/'):
    qq=dict(zip(arm_names('right'),[-.65,2.12,-.95,-.17,-1.18,-.08]));qq.update(open_q('right'));T=fk(name.split('/')[-1],'right',qq)
   elif name in ['/World/TaskMicrowave/E_door_2','/World/TaskMicrowave/SpringLatchSlider']:T=Q@T
   frame.append(np.r_[T[:3,3],Rot.from_matrix(T[:3,:3]).as_quat()])
  poses.append(frame)
 np.savez_compressed(out/'trajectory.npz',body_paths=bodies,poses=np.array(poses),time=np.arange(len(poses))/30)
 (out/'report.json').write_text(json.dumps(dict(status='planning_only',physics_stepped=False,source='synthetic joint plan for collision rejection, not execution evidence')))
 log=(p/'geometry_audit.log').open('w');proc=subprocess.Popen([sys.executable,str(P/'code/physical_v11/audit_full_geometry.py'),str(out)],stdout=log,stderr=subprocess.STDOUT);print(i,proc.pid,flush=True)
