from pathlib import Path
import json,numpy as np
from isaacsim import SimulationApp
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=7);ap.add_argument('--torque',type=float,default=5);a=ap.parse_args()
app=SimulationApp({'headless':True,'active_gpu':a.gpu,'multi_gpu':False,'limit_cpu_threads':8})
import omni.usd
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
p=Path('/media/external/bowenjing/projects/dexterity_living');omni.usd.get_context().open_stage(str(p/'runs/physical_v1/micro_adaptive_v11_gpu3/initial_scene.usda'));stage=omni.usd.get_context().get_stage()
for pr in stage.Traverse():
 if '/Robot_' in str(pr.GetPath()) and pr.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(pr).GetCollisionEnabledAttr().Set(False)
world=World(physics_dt=1/480,rendering_dt=1/30,physics_prim_path='/World/PhysicsScene');micro=world.scene.add(SingleArticulation('/World/TaskMicrowave/E_body_9',name='microwave'));world.reset();names=list(micro.dof_names);di=next(i for i,n in enumerate(names) if 'left' in n);si=names.index('SpringLatchJoint');records=[]
for k in range(480*8):
 efforts=np.zeros(len(names),np.float32)
 if k>480:efforts[di]=-a.torque
 micro.apply_action(ArticulationAction(joint_efforts=efforts));world.step(render=False)
 if k%16==0:
  q=micro.get_joint_positions();records.append(dict(t=k/480,angle=float(-np.rad2deg(q[di])),latch=float(q[si])))
(p/f'reports/v12_micro_native_torque_{a.torque}_diagnostic.json').write_text(json.dumps({'diagnostic_only':True,'external_door_torque_Nm':a.torque,'no_robot_collision':True,'records':records},indent=2));print('ANGLE_RANGE',min(r['angle'] for r in records),max(r['angle'] for r in records),flush=True);app.close()
