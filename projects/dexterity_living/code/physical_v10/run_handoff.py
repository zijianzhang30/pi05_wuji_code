"""Physical right-to-left handoff. Only finite robot joint motor commands are applied."""
import argparse,json,traceback
from model import *
from scipy.spatial.transform import Slerp
ap=argparse.ArgumentParser();ap.add_argument("--name",default="handoff01");ap.add_argument("--gpu",type=int,default=5);ap.add_argument("--contact-stop",type=float,default=.3);ap.add_argument("--hold-force",type=float,default=3);a=ap.parse_args()
out=P/"runs/physical_v1"/a.name;out.mkdir(parents=True,exist_ok=True)
source=P/"runs/physical_v1/pick_tip_v18_gpu1";original=json.loads((source/"contacts.json").read_text());old_report=json.loads((source/"report.json").read_text())
plan=json.loads((P/"runs/physical_v1/handoff_approach.json").read_text());cand=plan["candidate"];wide=plan["wide"];closed=cand["q"];right_fold=cand["right_q"];axis=np.array([np.cos(np.deg2rad(cand['direction_deg'])),np.sin(np.deg2rad(cand['direction_deg'])),0.])
initial={}
for side,sign in [("left",1),("right",-1)]:
 initial[side]=dict(zip(arm_names(side),[sign*.65,2.12,-.95,sign*.17,-1.18,sign*.08]));initial[side].update(open_q(side))
def shift(q,side,delta):
 T=fk(side+"_palm_link",side,q);T[:3,3]+=delta;qa,pe,re=solve(side,T,[q[n] for n in arm_names(side)]);assert pe<.012,(side,delta,pe);result=q.copy();result.update(zip(arm_names(side),qa));return result
def finger_offset(q,side,fi,delta,center):
 names=[n for n in hand_names(side) if f"finger{fi}_" in n];T=fk(f"{side}_finger{fi}_tip_link",side,q);goal=T[:3,3]+T[:3,:3]@center+delta
 def res(x):
  d=q.copy();d.update(zip(names,x));X=fk(f"{side}_finger{fi}_tip_link",side,d);return X[:3,3]+X[:3,:3]@center-goal
 opt=least_squares(res,[q[n] for n in names],bounds=([joints[n]["lo"] for n in names],[joints[n]["hi"] for n in names]),max_nfev=80);d=q.copy();d.update(zip(names,opt.x));return d
for fi,sign in [(1,1),(2,-1)]:
 closed=finger_offset(closed,"left",fi,sign*axis*.003,np.array(cand["pad_centers"][f"left_finger{fi}_tip_link"]))
right_end=dict(zip(old_report["dof_names"],original[cand["handoff_frame"]]["target"]))
right_fold={**right_end,**{n:v for n,v in right_fold.items() if any(f"finger{fi}_" in n for fi in [3,4,5])}}
left_pre=initial["left"].copy();left_pre.update({n:v for n,v in wide.items() if "finger" in n})
sequence=[(2.,initial["left"],right_fold,"fold_unused"),(1.5,left_pre,right_fold,"left_preshape")]
for duration,qarm in zip(plan["segment_seconds"],plan["arm_waypoints"][1:]):
 q=wide.copy();q.update(zip(arm_names("left"),qarm));sequence.append((duration,q,right_fold,"left_approach"))
sequence.append((3.,closed,right_fold,"left_close"))
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"multi_gpu":False,"active_gpu":a.gpu,"physics_gpu":a.gpu,"limit_cpu_threads":8})
records=[];poses=[];report=dict(status="failed",object_pose_commands=0,object_attachments=0,physics_backend="PhysX CPU");abort=False
try:
 import omni.usd
 from pxr import UsdPhysics
 from build import build
 from conservative_collision import validate_cooked
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 build(out,initial,task="pick_place");omni.usd.get_context().open_stage(str(out/"initial_scene.usda"));stage=omni.usd.get_context().get_stage();validate_cooked(stage,out)
 world=World(physics_dt=1/480,rendering_dt=1/30,stage_units_in_meters=1,physics_prim_path="/World/PhysicsScene")
 robots={s:world.scene.add(SingleArticulation("/World/Robot_"+s,name=s)) for s in ["left","right"]};world.reset()
 names={s:list(r.dof_names) for s,r in robots.items()};last={s:np.array([initial[s][n] for n in names[s]],np.float32) for s in robots}
 for side,r in robots.items():r.set_joint_positions(last[side]);r.set_joint_velocities(np.zeros(len(last[side]),np.float32))
 sensor_paths=["/World/Robot_"+s+"/"+n for s in ["left","right"] for n in links(s)]
 contact=world.physics_sim_view.create_rigid_contact_view(sensor_paths,filter_patterns=[["/World/TaskObjects/object","/World/V2Mat","/World/V2Tabletop"]]*len(sensor_paths),max_contact_data_count=32768)
 left_paths=[x for x in sensor_paths if "Robot_left" in x];cross=world.physics_sim_view.create_rigid_contact_view(left_paths,filter_patterns=[["/World/Robot_right/*"]]*len(left_paths),max_contact_data_count=8192)
 obj=world.physics_sim_view.create_rigid_body_view("/World/TaskObjects/object");body_paths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bodies=world.physics_sim_view.create_rigid_body_view(body_paths);assert list(bodies.prim_paths)==body_paths
 frame=0;max_pen=0.;first_contact=None;reference=None;drift=0.;reason="completed";left_locked=None;force_axis=np.zeros(2);left_servo=None;left_only_frames=0
 def tick(qs,phase):
  global frame,max_pen,first_contact,reference,drift,abort,reason,force_axis,left_only_frames
  targets={s:np.array([qs[s][n] for n in names[s]],np.float32) for s in robots};samples=[];cross_force=0.;depth=0.
  for sub in range(16):
   for s,r in robots.items():r.apply_action(ArticulationAction(joint_positions=last[s]+(targets[s]-last[s])*(sub+1)/16,joint_efforts=r._articulation_view.get_generalized_gravity_forces()[0]))
   world.step(render=False);samples.append(np.asarray(contact.get_contact_force_matrix(1/480)).copy());cross_force=max(cross_force,float(np.linalg.norm(cross.get_contact_force_matrix(1/480),axis=-1).sum()))
   cf,cp,cn,cd,counts,starts=contact.get_contact_data(1/480)
   for ri,ci in np.argwhere(np.asarray(counts)>0):
    first=int(starts[ri,ci]);count=int(counts[ri,ci]);depth=max(depth,-float(np.min(cd[first:first+count])))
  for s in last:last[s]=targets[s].copy()
  avg=np.mean(samples,axis=0);mag=np.max(np.linalg.norm(samples,axis=-1),axis=0);state=np.asarray(obj.get_transforms())[0].copy();vel=np.asarray(obj.get_velocities())[0].copy()
  leftforce=sum(float(mag[i,0]) for i,x in enumerate(sensor_paths) if "Robot_left" in x);rightforce=sum(float(mag[i,0]) for i,x in enumerate(sensor_paths) if "Robot_right" in x);table=float(mag[:,1:].sum())
  force_axis=np.array([sum(np.dot(avg[i,0],axis) for i,x in enumerate(sensor_paths) if f"left_finger{fi}_" in x) for fi in [1,2]])
  max_pen=max(max_pen,depth)
  if frame>=30:
   if reference is None:reference=state[:3].copy()
   if first_contact is None and leftforce+rightforce>.01:first_contact=frame
   if first_contact is None:drift=max(drift,float(np.linalg.norm(state[:3]-reference)))
  if phase in ["verify_left","left_transfer"] and leftforce>.2 and rightforce<.05 and state[2]>.75:left_only_frames+=1
  records.append(dict(t=(frame+1)/30,phase=phase,q_left=robots["left"].get_joint_positions().tolist(),q_right=robots["right"].get_joint_positions().tolist(),target_left=targets["left"].tolist(),target_right=targets["right"].tolist(),object_pose=state.tolist(),object_velocity=vel.tolist(),left_object_force_N=leftforce,right_object_force_N=rightforce,left_opposed_forces_N=force_axis.tolist(),cross_hand_force_N=cross_force,table_contact_N=table,contact_penetration_m=depth))
  poses.append(np.asarray(bodies.get_transforms()).copy())
  if frame%30==0:print("HANDOFF",frame/30,phase,state[:3],"L/R",leftforce,rightforce,"cross",cross_force,flush=True)
  frame+=1
  if depth>.0001 or table>12 or cross_force>.5 or leftforce+rightforce>50 or drift>.0001:
   abort=True;reason="abort penetration, unintended contact, or precontact motion"
  if phase not in ["right_pick","fold_unused","left_preshape","left_approach","left_close","left_lower","left_release","left_retreat","settle_end"] and state[2]<.745:
   abort=True;reason="abort lost object"
 for row in original[:cand["handoff_frame"]+1]:
  tick({"left":initial["left"],"right":dict(zip(old_report["dof_names"],row["target"]))},"right_pick")
  if abort:break
 start={"left":initial["left"].copy(),"right":right_end.copy()}
 if not abort:
  for phase_index,(duration,end_left,end_right,phase) in enumerate(sequence):
   end={"left":end_left,"right":end_right}
   for k in range(round(duration*30)):
    u=(k+1)/(duration*30);u=u*u*(3-2*u);q={s:{n:start[s][n]*(1-u)+end[s][n]*u for n in start[s]} for s in robots}
    if phase=="left_close" and left_locked is not None:q["left"]=left_locked.copy()
    if phase in ["left_transfer","left_lower","left_retreat","right_retreat"]:
     side="right" if phase=="right_retreat" else "left";A=fk(side+"_palm_link",side,start[side]);B=fk(side+"_palm_link",side,end[side]);A[:3,3]=(1-u)*A[:3,3]+u*B[:3,3];A[:3,:3]=Slerp([0,1],Rot.from_matrix(np.stack([A[:3,:3],B[:3,:3]])))([u]).as_matrix()[0];qa,pe,re=solve(side,A,[q[side][n] for n in arm_names(side)]);assert pe<.003 and re<.04,(phase,pe,re);q[side].update(zip(arm_names(side),qa))
    if left_locked is not None and phase in ["double_hold","right_release","verify_left","right_retreat","left_transfer","left_lower"]:
     if left_servo is None:left_servo={n:left_locked[n] for n in hand_names("left")}
     q["left"].update(left_servo)
     for fi,sign in [(1,1),(2,-1)]:
      delta=float(np.clip((a.hold_force-abs(force_axis[fi-1]))*.00001,-.00004,.00004));q["left"]=finger_offset(q["left"],"left",fi,sign*axis*delta,np.array(cand["pad_centers"][f"left_finger{fi}_tip_link"]))
     left_servo={n:q["left"][n] for n in hand_names("left")}
    tick(q,phase)
    if phase=="left_close" and left_locked is None and force_axis.prod()<-(a.contact_stop**2):
     left_locked=q["left"].copy();measured=dict(zip(names["left"],robots["left"].get_joint_positions().tolist()));left_locked.update({n:measured[n] for n in arm_names("left")});end["left"]=left_locked.copy()
     released=right_fold.copy()
     for fi,sign in [(1,-1),(2,1)]:
      offset=np.array([0,0,-.00308 if fi==1 else -.00755]);released=finger_offset(released,"right",fi,np.array([0,sign*.020,0]),offset)
     right_retreat=shift(released,"right",np.array([.08,-.04,.07]));left_transfer=shift(left_locked,"left",np.array([-2.17,-3.505,0])-np.r_[cand["object_center"][:2],0]);left_lower=shift(left_transfer,"left",np.array([0,0,-.095]));left_release=left_lower.copy()
     for fi,sign in [(1,-1),(2,1)]:left_release=finger_offset(left_release,"left",fi,sign*axis*.016,np.array(cand["pad_centers"][f"left_finger{fi}_tip_link"]))
     left_retreat=shift(left_release,"left",np.array([0,-.04,.10]))
     sequence[phase_index+1:]=[(1.,left_locked,right_fold,"double_hold"),(2.,left_locked,released,"right_release"),(1.,left_locked,released,"verify_left"),(2.,left_locked,right_retreat,"right_retreat"),(3.,left_transfer,right_retreat,"left_transfer"),(3.,left_lower,right_retreat,"left_lower"),(2.,left_release,right_retreat,"left_release"),(2.,left_retreat,right_retreat,"left_retreat"),(2.,left_retreat,right_retreat,"settle_end")]
     print("LEFT_CONTACT_CAPTURED",frame,force_axis,flush=True)
    if abort:break
   if abort:break
   if phase=="left_close" and left_locked is None:abort=True;reason="no opposed left grasp";break
   if left_servo is not None and phase in ["double_hold","right_release","verify_left","right_retreat","left_transfer","left_lower"]:end["left"].update(left_servo)
   start={s:end[s].copy() for s in robots}
 final=np.array(records[-1]["object_pose"][:3]);success=not abort and left_only_frames>=30 and np.linalg.norm(final[:2]-[-2.17,-3.505])<.035 and np.linalg.norm(records[-1]["object_velocity"][:3])<.03
 report.update(status="passed_handoff_trial" if success else "failed",reason=reason,physics_stepped=True,left_only_holding_frames=left_only_frames,max_contact_penetration_m=max_pen,precontact_max_drift_m=drift,first_contact_frame=first_contact,final_object_position=final.tolist(),body_paths=body_paths,joint_names=names,sim_hz=480,frames=len(records),goal_center_m=[-2.17,-3.505,.6823])
except Exception:
 report["error"]=traceback.format_exc();print(report["error"],flush=True)
finally:
 if poses:np.savez_compressed(out/"trajectory.npz",poses=np.asarray(poses),body_paths=body_paths,time=np.array([r["t"] for r in records]))
 (out/"contacts.json").write_text(json.dumps(records));(out/"report.json").write_text(json.dumps(report,indent=2));print("REPORT",json.dumps({k:v for k,v in report.items() if k not in ["body_paths","joint_names"]}),flush=True);app.close()
