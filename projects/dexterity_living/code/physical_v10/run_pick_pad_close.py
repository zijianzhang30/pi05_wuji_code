"""Finite joint drives only. Cube motion is exclusively solver-generated."""
import argparse,json,traceback
from model import *
p=argparse.ArgumentParser();p.add_argument('--name',default='pick01');p.add_argument('--gpu',type=int,default=1);p.add_argument('--contact-stop',type=float,default=1);p.add_argument('--pinch',type=float,default=.001);p.add_argument('--move-seconds',type=float,default=5);p.add_argument('--candidate',type=int,default=0);p.add_argument('--open-gap',type=float,default=.008);a=p.parse_args();out=P/'runs/physical_v1'/a.name
candidate=json.loads((P/'runs/physical_v1'/f'pick_pad_candidate_{a.candidate}.json').read_text())[0];grasp=candidate['q'];axis=np.array([np.cos(np.deg2rad(candidate['direction_deg'])),np.sin(np.deg2rad(candidate['direction_deg'])),0.]);qinit={}
for side,sgn in [('left',1),('right',-1)]:qinit[side]=dict(zip(arm_names(side),[sgn*.65,2.12,-.95,sgn*.17,-1.18,sgn*.08]));qinit[side].update(open_q(side))
wide=grasp.copy()
for fi,sign in [(1,-1),(2,1)]:
 link=f'right_finger{fi}_link4';off=np.array(candidate['pad_centers'][link]);nsf=[n for n in hand_names('right') if f'finger{fi}_' in n];T=fk(link,'right',grasp);target=T[:3,3]+T[:3,:3]@off+sign*axis*a.open_gap
 def rr(x):
  d=wide.copy();d.update(zip(nsf,x));X=fk(link,'right',d);return X[:3,3]+X[:3,:3]@off-target
 opt=least_squares(rr,[wide[n] for n in nsf],bounds=([joints[n]['lo'] for n in nsf],[joints[n]['hi'] for n in nsf]),max_nfev=200);wide.update(zip(nsf,opt.x));print('OPEN_PAD_ERROR',fi,np.linalg.norm(rr(opt.x)),flush=True)
preshape=qinit['right'].copy();preshape.update({n:v for n,v in wide.items() if 'finger' in n});palm=fk('right_palm_link','right',grasp)
def offset_pose(q,offset):
 T=fk('right_palm_link','right',q);T[:3,3]+=offset;qa,pe,re=solve('right',T,[q[n] for n in arm_names('right')]);assert pe<.012,(offset,pe);d=q.copy();d.update(zip(arm_names('right'),qa));return d
above=offset_pose(grasp,np.array([0,-.015,.10]));lift=offset_pose(grasp,np.array([0,0,.10]));transfer=offset_pose(lift,np.array([-.14,-.01,0]));lower=offset_pose(transfer,np.array([0,0,-.09]));opened=lower.copy();opened.update(open_q('right'));retreat=offset_pose(opened,np.array([0,-.04,.10]));sequence=[(1.,qinit['right'],'settle'),(1.5,preshape,'preshape'),(4.,above,'approach'),(3.,grasp,'grasp'),(1.,grasp,'grip_hold'),(a.move_seconds,lift,'lift'),(a.move_seconds,transfer,'transfer'),(a.move_seconds,lower,'lower'),(2.,opened,'release'),(2.,retreat,'retreat'),(2.,retreat,'settle_end')]
ap=json.loads((P/'runs/physical_v1'/f'approach_pad_{a.candidate}.json').read_text());approach_steps=[]
for duration,qa in zip(ap['segment_seconds'],ap['arm_waypoints'][1:]):
 d=wide.copy();d.update(zip(arm_names('right'),qa));approach_steps.append((duration,d,'approach'))
sequence=sequence[:2]+approach_steps+[(3.,wide,'descend_open'),(1.,wide,'settle_grasp'),(3.,grasp,'grasp')]+sequence[4:]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','multi_gpu':False,'active_gpu':a.gpu,'physics_gpu':a.gpu,'limit_cpu_threads':8});report={'status':'failed','object_pose_commands':0};records=[];poses=[]
try:
 from build import build
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 from pxr import UsdPhysics
 import omni.usd
 build(out,qinit,task='pick_place');omni.usd.get_context().open_stage(str(out/'initial_scene.usda'));stage=omni.usd.get_context().get_stage()
 world=World(physics_dt=1/480,rendering_dt=1/30,stage_units_in_meters=1,physics_prim_path='/World/PhysicsScene');robots={s:world.scene.add(SingleArticulation('/World/Robot_'+s,name=s)) for s in ['left','right']};world.reset();from conservative_collision import validate_cooked;validate_cooked(stage,out);world.reset()
 for s,r in robots.items():r.set_joint_positions(np.array([qinit[s][n] for n in r.dof_names],np.float32));r.set_joint_velocities(np.zeros(len(r.dof_names),np.float32))
 sensor_paths=['/World/Robot_right/'+n for n in links('right')];obj_path='/World/TaskObjects/object';contact=world.physics_sim_view.create_rigid_contact_view(sensor_paths,filter_patterns=[[obj_path,'/World/V2Mat','/World/V2Tabletop']]*len(sensor_paths),max_contact_data_count=16384);obj=world.physics_sim_view.create_rigid_body_view(obj_path);body_paths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bv=world.physics_sim_view.create_rigid_body_view(body_paths);assert list(bv.prim_paths)==body_paths
 qleft=np.array([qinit['left'][n] for n in robots['left'].dof_names],np.float32);names=robots['right'].dof_names;start=qinit['right'];frame=0;maxz=-1;reason='completed';abort=False;grasp_lock=None;maxpenetration=0.;pending_lock=None;last_cmd=np.array([qinit['right'][n] for n in names],np.float32);cart_seed=[qinit['right'][n] for n in arm_names('right')]
 for phase_index,(duration,end,phase) in enumerate(sequence):
  for k in range(round(duration*30)):
   u=(k+1)/(duration*30);u=u*u*(3-2*u);q={n:start[n]*(1-u)+end[n]*u for n in start}
   if phase=='grasp' and grasp_lock is not None:q=grasp_lock.copy()
   if phase in ['lift','transfer','lower','retreat']:
    A=fk('right_palm_link','right',start);B=fk('right_palm_link','right',end);A[:3,3]=(1-u)*A[:3,3]+u*B[:3,3];qa,pe,re=solve('right',A,[q[n] for n in arm_names('right')]);assert pe<.002 and re<.02,(phase,pe,re);q.update(zip(arm_names('right'),qa))
   qr=np.array([q[n] for n in names],np.float32);ff=[];depths=[]
   for sub in range(16):
    for side,target in [('left',qleft),('right',last_cmd+(qr-last_cmd)*(sub+1)/16)]:
     r=robots[side];r.apply_action(ArticulationAction(joint_positions=target,joint_efforts=r._articulation_view.get_generalized_gravity_forces()[0]))
    world.step(render=False);ff.append(np.asarray(contact.get_contact_force_matrix(1/480)).copy())
    cf,cp,cn,cd,counts,starts=contact.get_contact_data(1/480);minsep=0.
    for ri,ci in np.argwhere(np.asarray(counts)>0):
     begin=int(starts[ri,ci]);count=int(counts[ri,ci]);minsep=min(minsep,float(np.min(cd[begin:begin+count])))
    depths.append(-minsep)
    if phase=='grasp' and grasp_lock is None and pending_lock is None:
     instant=np.asarray(ff[-1])[:,0];ta=sum(np.dot(instant[i],axis) for i,path in enumerate(sensor_paths) if 'finger1_' in path);ia=sum(np.dot(instant[i],axis) for i,path in enumerate(sensor_paths) if 'finger2_' in path)
     if ta*ia<-(a.contact_stop**2):
      pending_lock=dict(zip(names,qr.tolist()));measured=dict(zip(names,robots['right'].get_joint_positions().tolist()));pending_lock.update({n:measured[n] for n in arm_names('right')});q=pending_lock.copy();qr=np.array([q[n] for n in names],np.float32);last_cmd=qr.copy();print('SUBSTEP_OPPOSED_CONTACT',frame,sub,ta,ia,flush=True)

   last_cmd=qr.copy()
   vec=np.mean(ff,axis=0)[:,0];thumb_axis=float(sum(np.dot(vec[i],axis) for i,path in enumerate(sensor_paths) if 'finger1_' in path));index_axis=float(sum(np.dot(vec[i],axis) for i,path in enumerate(sensor_paths) if 'finger2_' in path))
   forces=np.max(np.linalg.norm(ff,axis=-1),axis=0);normal=float(forces[:,0].sum());table=float(forces[:,1:].sum());thumb=float(sum(forces[i,0] for i,p in enumerate(sensor_paths) if 'finger1_' in p));index=float(sum(forces[i,0] for i,p in enumerate(sensor_paths) if 'finger2_' in p));state=np.asarray(obj.get_transforms())[0].copy();vel=np.asarray(obj.get_velocities())[0].copy();actual=robots['right'].get_joint_positions();maxz=max(maxz,float(state[2]));maxpenetration=max(maxpenetration,max(depths));rec=dict(thumb_axis_N=thumb_axis,index_axis_N=index_axis,contact_penetration_m=max(depths),t=(frame+1)/30,phase=phase,q=actual.tolist(),target=qr.tolist(),object_pose=state.tolist(),object_velocity=vel.tolist(),object_contact_N=normal,table_contact_N=table,table_contact_pairs=[(sensor_paths[i],float(forces[i,1:].sum())) for i in range(len(sensor_paths)) if forces[i,1:].sum()>.1],thumb_N=thumb,index_N=index);records.append(rec);poses.append(np.asarray(bv.get_transforms()).copy())
   if phase=='grasp' and grasp_lock is None and pending_lock is not None:
    grasp_lock=q.copy()
    for fi,off,sign in [(1,np.array([0,0,-.00308]),1),(2,np.array([0,0,-.00755]),-1)]:
     nn=[n for n in hand_names('right') if f'finger{fi}_' in n];T=fk(f'right_finger{fi}_tip_link','right',grasp_lock);target=T[:3,3]+T[:3,:3]@off;target[0]+=sign*a.pinch
     def rr(x):
      dd=grasp_lock.copy();dd.update(zip(nn,x));X=fk(f'right_finger{fi}_tip_link','right',dd);return X[:3,3]+X[:3,:3]@off-target
     opt=least_squares(rr,[grasp_lock[n] for n in nn],bounds=([joints[n]['lo'] for n in nn],[joints[n]['hi'] for n in nn]),max_nfev=150);grasp_lock.update(zip(nn,opt.x))
    end=grasp_lock.copy();lift=offset_pose(grasp_lock,np.array([0,0,.10]));transfer=offset_pose(lift,np.array([-.14,-.01,0]));lower=offset_pose(transfer,np.array([0,0,-.095]));opened=lower.copy()
    for fi,sign in [(1,-1),(2,1)]:
     link=f'right_finger{fi}_link4';offset=np.array(candidate['pad_centers'][link]);nsf=[n for n in hand_names('right') if f'finger{fi}_' in n];T=fk(link,'right',lower);target=T[:3,3]+T[:3,:3]@offset+sign*axis*.020
     def residual(x):
      d=opened.copy();d.update(zip(nsf,x));X=fk(link,'right',d);return X[:3,3]+X[:3,:3]@offset-target
     opt=least_squares(residual,[opened[n] for n in nsf],bounds=([joints[n]['lo'] for n in nsf],[joints[n]['hi'] for n in nsf]),max_nfev=150);opened.update(zip(nsf,opt.x))
    retreat=offset_pose(opened,np.array([0,-.04,.10]));sequence[phase_index+1:]=[(1.,grasp_lock,'grip_hold'),(a.move_seconds,lift,'lift'),(a.move_seconds,transfer,'transfer'),(a.move_seconds,lower,'lower'),(2.,opened,'release'),(2.,retreat,'retreat'),(2.,retreat,'settle_end')];print('CONTACT_STOP_DESCENT',rec['t'],thumb,index,flush=True)
   if frame%30==0:print('STATE',frame/30,phase,'obj',state[:3],'force',normal,'table',table,flush=True)
   frame+=1
   if maxpenetration>.0001 or table>12 or normal>35 or not np.isfinite(actual).all():reason='abort excessive contact or invalid state';abort=True;break
   if phase=='grip_hold' and k==round(duration*30)-1 and (min(thumb,index)<.1 or thumb_axis*index_axis>=-.01):reason='abort no opposed grasp';abort=True;break
  if abort:break
  start=end.copy()
 target=np.array([-2.01,-3.505]);final=np.array(records[-1]['object_pose'][:3]);success=(not abort and maxz>.763 and np.linalg.norm(final[:2]-target)<.04 and np.linalg.norm(records[-1]['object_velocity'][:3])<.03 and all(r['object_pose'][2]>.76 for r in records if r['phase']=='transfer'))
 report={'status':'passed_pick_place_trial' if success else 'failed','reason':reason,'physics_stepped':True,'object_pose_commands':0,'object_attachments':0,'max_contact_penetration_m':maxpenetration,'max_object_z_m':maxz,'final_object_position':final.tolist(),'frames':len(records),'dof_names':names,'body_paths':body_paths,'sim_hz':480}
 np.savez_compressed(out/'trajectory.npz',poses=np.asarray(poses),body_paths=body_paths,time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records));print('REPORT',json.dumps(report),flush=True)
except Exception:report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2));app.close()
