import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
import argparse,json,traceback,time
import numpy as np
from model import *
p=argparse.ArgumentParser();p.add_argument('--name',default='micro02');p.add_argument('--gpu',type=int,default=0);p.add_argument('--lock-force',type=float,default=2);p.add_argument('--hand-kp',type=float,default=3);p.add_argument('--arm-kp',type=float,default=200);p.add_argument('--lead-deg',type=float,default=1.5);p.add_argument('--pressure-m',type=float,default=0);p.add_argument('--clearance-mm',type=int,default=4);p.add_argument('--grip-id',type=int,default=0);p.add_argument('--total-contact-cap',type=float,default=45);p.add_argument('--measured-lock',action='store_true');a=p.parse_args();out=P/'runs/physical_v1'/a.name
candidate=json.loads((P/'runs/physical_v1'/f'micro_tip_{a.grip_id}.json').read_text())[0];goal=candidate['q'];qinit={}
for side,sgn in [('left',1),('right',-1)]:qinit[side]=dict(zip(arm_names(side),[sgn*.65,2.12,-.95,sgn*.17,-1.18,sgn*.08]));qinit[side].update(open_q(side))
pre=qinit['right'].copy();pre.update({n:v for n,v in goal.items() if n in hand_names('right')})
approach=json.loads((P/'runs/physical_v1'/f'micro_approach_v11_{a.grip_id}.json').read_text());close_path=json.loads((P/'runs/physical_v1'/f'micro_close_v11_{a.grip_id}.json').read_text());approach_end=2.5+approach['duration'];close_move_end=approach_end+close_path['duration'];close_end=close_move_end+1.5;pull_end=close_end+35;release_end=pull_end
qinit['right'].update(zip(arm_names('right'),approach['arm_waypoints'][0]));qinit['right'].update({n:v for n,v in goal.items() if n in hand_names('right')});pre=qinit['right'].copy()
closed=goal.copy()
for fi in [2,3,4,5]:
 ns=[n for n in hand_names('right') if f'finger{fi}_' in n];lo=[joints[n]['lo'] for n in ns];hi=[joints[n]['hi'] for n in ns];T=fk(f'right_finger{fi}_tip_link','right',closed);offset=np.array([0,0,-.00755]);target=T[:3,3]+T[:3,:3]@offset;target[1]-=.0035+a.pressure_m
 def fun(v):
  q=closed.copy();q.update(zip(ns,v));T=fk(f'right_finger{fi}_tip_link','right',q);return 100*(T[:3,3]+T[:3,:3]@offset-target)
 opt=least_squares(fun,[closed[n] for n in ns],bounds=(lo,hi),max_nfev=150);closed.update(zip(ns,opt.x))
# Only robot motor targets are planned. Door/latch transforms are never commanded.
pull=[];hinge=np.array([-2.2972766,-3.4886971,.9657585]);ns=arm_names('right');seed=np.array([closed[n] for n in ns]);lo=np.array([joints[n]['lo'] for n in ns]);hi=np.array([joints[n]['hi'] for n in ns]);tips=np.array([fk(f'right_finger{fi}_tip_link','right',closed)[:3,3] for fi in [2,3]])
for angle in np.linspace(0,-np.deg2rad(45),151):
 R=Rot.from_euler('z',angle).as_matrix();targets=(tips-hinge)@R.T+hinge
 def fun(x):
  q=closed.copy();q.update(zip(ns,x));got=np.array([fk(f'right_finger{fi}_tip_link','right',q)[:3,3] for fi in [2,3]]);return np.r_[(got-targets).ravel(),.0001*(x-seed)]
 opt=least_squares(fun,seed,bounds=(lo,hi),max_nfev=150);seed=opt.x;pull.append({'q':seed.tolist(),'pe':float(np.linalg.norm(fun(seed)[:6])),'angle':angle})
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','multi_gpu':False,'active_gpu':a.gpu,'physics_gpu':a.gpu,'limit_cpu_threads':8})
report={'status':'failed','physics_stepped':False,'object_pose_commands':0};records=[];poses=[]
try:
 from build import build
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 import omni.usd
 from pxr import UsdPhysics
 build(out,qinit);(out/'plan.json').write_text(json.dumps({'preshape':pre,'approach':goal,'closed':closed,'pull':pull},indent=2));omni.usd.get_context().open_stage(str(out/'initial_scene.usda'));stage=omni.usd.get_context().get_stage()
 for pr in stage.Traverse():
  if '/Robot_' in str(pr.GetPath()) and pr.IsA(UsdPhysics.RevoluteJoint):
   drive=UsdPhysics.DriveAPI(pr,'angular');is_arm='finger' not in pr.GetName();drive.GetMaxForceAttr().Set(min(joints[pr.GetName()]['effort'],60.) if is_arm else joints[pr.GetName()]['effort']);drive.GetStiffnessAttr().Set(800. if is_arm else 8.);drive.GetDampingAttr().Set(60. if is_arm else .3)
 from conservative_collision import validate_cooked
 validate_cooked(stage,out)
 world=World(physics_dt=1/480,rendering_dt=1/30,stage_units_in_meters=1,physics_prim_path='/World/PhysicsScene');robots={side:world.scene.add(SingleArticulation(prim_path='/World/Robot_'+side,name=side)) for side in ['left','right']};micro=world.scene.add(SingleArticulation(prim_path='/World/TaskMicrowave/E_body_9',name='microwave'));world.reset()
 for side,r in robots.items():r.set_joint_positions(np.array([qinit[side][n] for n in r.dof_names],np.float32));r.set_joint_velocities(np.zeros(len(r.dof_names),np.float32))
 names=robots['right'].dof_names;mnames=micro.dof_names;di=next(i for i,n in enumerate(mnames) if 'left' in n);si=mnames.index('SpringLatchJoint')
 sensor_paths=['/World/Robot_right/'+n for n in links('right')];filters=['/World/TaskMicrowave/E_door_2','/World/TaskMicrowave/E_body_9','/World/TaskMicrowave/SpringLatchSlider','/World/V2Mat','/World/V2Tabletop'];contact=world.physics_sim_view.create_rigid_contact_view(sensor_paths,filter_patterns=[filters]*len(sensor_paths),max_contact_data_count=16384)
 body_paths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bodyview=world.physics_sim_view.create_rigid_body_view(body_paths);assert list(bodyview.prim_paths)==body_paths;print('BODY_PATHS',len(body_paths),'contact_sensors',contact.sensor_count,flush=True)
 def mix(x,y,u):u=float(np.clip(u,0,1));u=u*u*(3-2*u);return {n:x[n]*(1-u)+y[n]*u for n in x}
 qleft=np.array([qinit['left'][n] for n in robots['left'].dof_names],np.float32);reason='maximum_duration';maxbody=0;maxgrip=0;pull_index=0;maxpenetration=0;last_phase=None;lock_q=None;lock_t=None;feedback_pull=None
 for frame in range(int((release_end+2)*30)):
  t=frame/30
  if t<1:q=qinit['right'];phase='settle'
  elif t<2.5:q=mix(qinit['right'],pre,(t-1)/1.5);phase='preshape'
  elif t<approach_end:
   elapsed=t-2.5;seg=0
   while seg<len(approach['segment_seconds'])-1 and elapsed>approach['segment_seconds'][seg]:elapsed-=approach['segment_seconds'][seg];seg+=1
   w=float(np.clip(elapsed/approach['segment_seconds'][seg],0,1));w=w*w*(3-2*w);qa=np.array(approach['arm_waypoints'][seg])*(1-w)+np.array(approach['arm_waypoints'][seg+1])*w;q=goal.copy();q.update(zip(arm_names('right'),qa));phase='approach'
  elif t<close_move_end:
   elapsed=t-approach_end;seg=0
   while seg<len(close_path['segment_seconds'])-1 and elapsed>close_path['segment_seconds'][seg]:elapsed-=close_path['segment_seconds'][seg];seg+=1
   w=float(np.clip(elapsed/close_path['segment_seconds'][seg],0,1));w=w*w*(3-2*w);qa=np.array(close_path['arm_waypoints'][seg])*(1-w)+np.array(close_path['arm_waypoints'][seg+1])*w;q=goal.copy();q.update(zip(arm_names('right'),qa));phase='close_approach'
  elif t<close_end:q=mix(goal,closed,(t-close_move_end)/1.5);phase='close'
  elif t<pull_end:
   pull_index=min(150,pull_index+1,int(max(0,angle+3)/.3));q=closed.copy();q.update(zip(arm_names('right'),pull[int(pull_index)]['q']));phase='pull'
  elif t<release_end:
   qend=closed.copy();qend.update(zip(arm_names('right'),pull[-1]['q']));q=mix(qend,{**qend,**open_q('right')},(t-pull_end)/2);phase='release'
  else:phase='hold'
  if lock_q is not None:
   q=lock_q.copy();phase='grip_hold'
   if t>lock_t+.7:
    if feedback_pull is None:
     feedback_names=arm_names('right')+[n for n in hand_names('right') if any(f'finger{i}_' in n for i in [1,2,3,4,5])];feedback_pull=[];previous=np.array([lock_q[n] for n in feedback_names]);tip_links=[f'right_finger{i}_tip_link' for i in [1,2,3,4,5]];offsets={l:np.array([0,0,-.0030825 if 'finger1_' in l else -.00755]) for l in tip_links};tips0=np.array([fk(l,'right',lock_q)[:3,3]+fk(l,'right',lock_q)[:3,:3]@offsets[l] for l in tip_links]);palmR0=fk('right_palm_link','right',lock_q)[:3,:3]
     for theta in np.linspace(0,-np.deg2rad(45),91):
      R=Rot.from_euler('z',theta).as_matrix();targets=(tips0-hinge)@R.T+hinge
      def residual(x):
       qq=lock_q.copy();qq.update(zip(feedback_names,x));got=np.array([fk(l,'right',qq)[:3,3]+fk(l,'right',qq)[:3,:3]@offsets[l] for l in tip_links]);rotation=fk('right_palm_link','right',qq)[:3,:3];orientation=Rot.from_matrix((R@palmR0)@rotation.T).as_rotvec();return np.r_[(got-targets).ravel(),.04*orientation,.001*(x-previous)]
      opt=least_squares(residual,np.clip(previous,[joints[n]['lo']+1e-6 for n in feedback_names],[joints[n]['hi']-1e-6 for n in feedback_names]),bounds=([joints[n]['lo']+1e-6 for n in feedback_names],[joints[n]['hi']-1e-6 for n in feedback_names]),max_nfev=250);assert np.linalg.norm(residual(opt.x)[:15])<.004;feedback_pull.append(opt.x.tolist());previous=opt.x
     pull_index=0
    pull_index=min(90.,pull_index+.3,max(0,(angle+a.lead_deg)/.5));ii=int(pull_index);jj=min(90,ii+1);u=pull_index-ii;q.update(zip(feedback_names,(1-u)*np.array(feedback_pull[ii])+u*np.array(feedback_pull[jj])));phase='feedback_pull'
  if phase!=last_phase:
   arm_kp=a.arm_kp if phase in ['close_approach','close','pull','hold','feedback_pull','grip_hold'] else 800.;hand_kp=.7 if phase=='close_approach' else a.hand_kp if phase in ['feedback_pull','grip_hold'] else 8.;robots['right'].get_articulation_controller().set_gains(kps=np.array([arm_kp if n in arm_names('right') else hand_kp for n in names]),kds=np.array([30. if n in arm_names('right') else .2 for n in names]));last_phase=phase
  qr=np.array([q[n] for n in names],np.float32);forces=[];depths=[]
  for _ in range(16):
   robots['left'].apply_action(ArticulationAction(joint_positions=qleft,joint_efforts=robots['left']._articulation_view.get_generalized_gravity_forces()[0]));robots['right'].apply_action(ArticulationAction(joint_positions=qr,joint_efforts=robots['right']._articulation_view.get_generalized_gravity_forces()[0]));world.step(render=False);forces.append(np.asarray(contact.get_contact_force_matrix(1/480)).copy())
   cf,cp,cn,cd,counts,starts=contact.get_contact_data(1/480);minsep=0.
   for ri,ci in np.argwhere(np.asarray(counts)>0):
    begin=int(starts[ri,ci]);count=int(counts[ri,ci]);minsep=min(minsep,float(np.min(cd[begin:begin+count])))
   depths.append(-minsep)
  force=np.mean(forces,axis=0);magnitudes=np.linalg.norm(force,axis=-1);doorforce=float(magnitudes[:,0].sum());bodyforce=float(magnitudes[:,1].sum());tableforce=float(magnitudes[:,3:].sum());mq=micro.get_joint_positions();mvel=micro.get_joint_velocities();angle=float(-np.rad2deg(mq[di]));slider=float(mq[si]);actual=robots['right'].get_joint_positions();thumb=float(sum(magnitudes[i,0] for i,path in enumerate(sensor_paths) if 'finger1_' in path));index=float(sum(magnitudes[i,0] for i,path in enumerate(sensor_paths) if 'finger2_' in path));middle=float(sum(magnitudes[i,0] for i,path in enumerate(sensor_paths) if 'finger3_' in path));maxbody=max(maxbody,bodyforce);maxgrip=max(maxgrip,min(middle,index))
  if phase in ['close_approach','close','pull'] and min(middle,index)>a.lock_force and lock_q is None:
   lock_q=dict(zip(names,actual.tolist())) if a.measured_lock else q.copy();lock_t=t;print('LOCK_MULTI_FINGER_GRIP',t,doorforce,flush=True)
  maxpenetration=max(maxpenetration,max(depths))
  records.append(dict(contact_penetration_m=max(depths),t=t+1/30,phase=phase,q=actual.tolist(),target=qr.tolist(),door_angle_deg=angle,latch_m=slider,door_force_N=doorforce,body_force_N=bodyforce,table_force_N=tableforce,thumb_force_N=thumb,index_force_N=index,middle_force_N=middle,door_velocity=float(mvel[di]),spring_force_N=float(-1200*(slider+.0025)-mvel[si])))
  poses.append(np.asarray(bodyview.get_transforms()).copy())
  if frame%30==0:print('STATE',t,phase,'door',round(angle,3),'latch',round(slider,6),'thumb/index/middle',round(thumb,3),round(index,3),round(middle,3),'body',round(bodyforce,3),flush=True)
  if angle>40:reason='opened above 40 degrees';break
  if tableforce>12:reason='abort unintended table contact';break
  if max(depths)>.0001:reason='abort contact penetration above 0.1 mm';break
  if doorforce>a.total_contact_cap:reason='abort excessive handle contact force';break
  if bodyforce>12:reason='abort unintended body contact';break
  if not np.isfinite(actual).all() or abs(slider)>.013:reason='abort invalid state';break
  if lock_q is None and close_end-1/30<=t<close_end and (middle<.1 or index<.1):reason='abort no opposed finger contact';break
 report={'status':'passed_open_contact_trial' if reason=='opened above 40 degrees' and records[-1]['door_angle_deg']>40 and maxgrip>.2 and maxbody<2 and maxpenetration<=.0001 else 'failed','physics_stepped':True,'object_pose_commands':0,'door_drive_max_force':0,'reason':reason,'frames':len(records),'max_contact_penetration_m':maxpenetration,'max_nonhandle_body_force_N':maxbody,'max_multi_finger_force_N':maxgrip,'last':records[-1],'body_paths':body_paths,'dof_names':names,'microwave_dof_names':mnames,'sim_hz':480,'sweep_parameters':vars(a),'physics_backend':'PhysX CPU'}
 np.savez_compressed(out/'trajectory.npz',body_paths=body_paths,poses=np.asarray(poses),time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records));print('REPORT',json.dumps({k:v for k,v in report.items() if k not in ['body_paths','last','dof_names']}),flush=True)
except Exception:
 report['error']=traceback.format_exc();print(report['error'],flush=True)
 if poses:
  np.savez_compressed(out/'trajectory.npz',body_paths=body_paths,poses=np.asarray(poses),time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records));report.update(physics_stepped=True,frames=len(records),last=records[-1])

finally:
 out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2));app.close()
