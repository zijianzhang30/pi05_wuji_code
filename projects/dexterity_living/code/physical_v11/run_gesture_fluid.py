"""Finite-actuator single-hand gesture demo. No scene-object or robot body pose commands after initialization."""
import sys,argparse,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['gesture_one','gesture_two'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
out=P/'runs/physical_v1'/a.name;old=P/'runs/wuji_blackwhite_smooth_pt4k_20261011/living_mcl_wuji_v1_20261011'/a.task
data=json.loads((old/'refined_joints.json').read_text())
initial={}
for side,sgn in [('left',1),('right',-1)]:
 initial[side]=dict(zip(arm_names(side),[sgn*.65,2.12,-.95,sgn*.17,-1.18,sgn*.08]));initial[side].update(open_q(side))
selected=json.loads((P/'runs/physical_v1/gesture_pose_centered_v11.json').read_text());qa=np.array(selected['arm_q'])
raised=initial['right'].copy();raised.update(zip(arm_names('right'),qa))
def blend(x,y,u):
 u=np.clip(u,0,1);u=10*u**3-15*u**4+6*u**5
 return {n:float(x[n]*(1-u)+y[n]*u) for n in x}
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,'limit_cpu_threads':8})
from pxr import Usd,UsdGeom,UsdPhysics
records=[];poses=[];report=dict(status='failed',physics_stepped=True,object_pose_commands=0,object_attachments=0,task=a.task)
try:
 from build import build
 from conservative_collision import validate_cooked
 import omni.usd
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 build(out,initial,task=a.task);omni.usd.get_context().open_stage(str(out/'initial_scene.usda'));stage=omni.usd.get_context().get_stage();validate_cooked(stage,out)
 world=World(physics_dt=1/480,rendering_dt=1/30,physics_prim_path='/World/PhysicsScene')
 robots={s:world.scene.add(SingleArticulation('/World/Robot_'+s,name=s)) for s in initial};world.reset()
 names={s:list(r.dof_names) for s,r in robots.items()};last={s:np.array([initial[s][n] for n in names[s]],np.float32) for s in initial}
 for s,r in robots.items():r.set_joint_positions(last[s]);r.set_joint_velocities(np.zeros_like(last[s]))
 paths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bv=world.physics_sim_view.create_rigid_body_view(paths);assert list(bv.prim_paths)==paths
 sensors=['/World/Robot_right/'+n for n in links('right')];contact=world.physics_sim_view.create_rigid_contact_view(sensors,filter_patterns=[['/World/V2Mat','/World/V2Tabletop']]*len(sensors),max_contact_data_count=8192)
 qarr=np.array(data['q']['right']);hn=data['joint_names']['right'];maxforce=0;maxdepth=0;maxtrack=0;reason='completed'
 sign=raised.copy();sign.update(zip(hn,qarr[len(qarr)//2]))
 for frame in range(174):
  t=frame/30
  if t<1:q=initial['right'];phase='settle'
  elif t<2.6:
   q=blend(initial['right'],raised,(t-1)/1.6);shape=blend(raised,sign,(t-1.6)/1.);q.update({n:shape[n] for n in hand_names('right')});phase='raise_and_form'
  elif t<4.3:q=sign.copy();phase='gesture'
  else:
   q=blend(raised,initial['right'],(t-4.3)/1.5);shape=blend(sign,initial['right'],(t-4.3)/1.2);q.update({n:shape[n] for n in hand_names('right')});phase='open_and_return'
  target=np.array([q[n] for n in names['right']],np.float32);depth=0;force=0
  for sub in range(16):
   for side in robots:
    value=last[side] if side=='left' else last[side]+(target-last[side])*(sub+1)/16
    robots[side].apply_action(ArticulationAction(joint_positions=value,joint_efforts=robots[side]._articulation_view.get_generalized_gravity_forces()[0]))
   world.step(render=False);force=max(force,float(np.linalg.norm(contact.get_contact_force_matrix(1/480),axis=-1).sum()))
   cf,cp,cn,cd,counts,starts=contact.get_contact_data(1/480)
   for ri,ci in np.argwhere(np.asarray(counts)>0):
    i,n=int(starts[ri,ci]),int(counts[ri,ci]);depth=max(depth,-float(np.min(cd[i:i+n])))
  last['right']=target.copy();actual=robots['right'].get_joint_positions();maxtrack=max(maxtrack,float(np.max(np.abs(actual-target))));maxforce=max(force,maxforce);maxdepth=max(depth,maxdepth)
  records.append(dict(t=(frame+1)/30,phase=phase,q=actual.tolist(),target=target.tolist(),q_left=robots['left'].get_joint_positions().tolist(),unintended_contact_N=force,contact_penetration_m=depth))
  poses.append(np.asarray(bv.get_transforms()).copy())
  if frame%30==0:print('FRAME',frame,phase,force,depth,flush=True)
  if force>5 or depth>.0001:reason='unintended_contact';break
 report.update(status='passed_gesture_dynamics' if reason=='completed' and maxtrack<.20 else 'failed',reason=reason,frames=len(records),max_unintended_contact_N=maxforce,max_contact_penetration_m=maxdepth,max_joint_tracking_error_rad=maxtrack,dof_names=names['right'],dof_names_left=names['left'],body_paths=paths,sim_hz=480,scope='Finite-actuator gesture; no object manipulation or standardized sign-language certification')
except Exception:report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 out.mkdir(exist_ok=True,parents=True)
 if poses:np.savez_compressed(out/'trajectory.npz',poses=np.array(poses),body_paths=paths,time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records))
 (out/'report.json').write_text(json.dumps(report,indent=2));print('REPORT',json.dumps(report),flush=True);app.close()