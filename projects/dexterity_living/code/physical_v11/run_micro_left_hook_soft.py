"""Two-finger hook microwave pull using finite actuators and native passive latch."""
import sys,json,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--force',type=float,default=12);ap.add_argument('--lead',type=float,default=1.5);ap.add_argument('--grip-shift',type=float,default=.010);ap.add_argument('--soft-kp',type=float,default=100);ap.add_argument('--follow-speed',type=float,default=.7);a=ap.parse_args()
out=P/'runs/physical_v1'/a.name;plan=json.loads(Path(a.plan).read_text());path=plan['candidates'][0]['path'];assert path[-1]['angle_deg']>=60
sgn=plan['parameters']['sign'];h=np.array([-2.2972766,-3.4886971,.9657585]);offset={2:np.array([0,0,-.00755]),3:np.array([0,0,-.00755])}
def interp(angle):
 angle=np.clip(angle,0,path[-1]['angle_deg']);i=min(int(angle/2),len(path)-2);u=(angle-path[i]['angle_deg'])/2
 return {n:(1-u)*path[i]['q'][n]+u*path[i+1]['q'][n] for n in path[i]['q']}
def finger_delta(q,fi,delta):
 ns=[n for n in hand_names('left') if f'finger{fi}_' in n];l=f'left_finger{fi}_tip_link';T=fk(l,'left',q);goal=T[:3,3]+T[:3,:3]@offset[fi]+delta;seed=np.array([q[n] for n in ns]);lo=np.array([joints[n]['lo']+1e-6 for n in ns]);hi=np.array([joints[n]['hi']-1e-6 for n in ns])
 def residual(v):
  d=q.copy();d.update(zip(ns,v));X=fk(l,'left',d);return np.r_[X[:3,3]+X[:3,:3]@offset[fi]-goal,.00001*(v-seed)]
 opt=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=70);d=q.copy();d.update(zip(ns,opt.x));return d
initial={}
for side,sign in [('left',1),('right',-1)]:
 initial[side]=dict(zip(arm_names(side),[sign*.65,2.12,-.95,sign*.17,-1.18,sign*.08]));initial[side].update(open_q(side))
initial['left']=interp(0)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':a.gpu,'physics_gpu':a.gpu,'limit_cpu_threads':8})
records=[];poses=[];report=dict(status='failed',object_pose_commands=0,door_drive_max_force=0,object_attachments=0,physics_stepped=False)
try:
 from build import build
 from conservative_collision import validate_cooked
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 import omni.usd
 from pxr import UsdPhysics
 build(out,initial);omni.usd.get_context().open_stage(str(out/'initial_scene.usda'));stage=omni.usd.get_context().get_stage();validate_cooked(stage,out)
 w=World(physics_dt=1/960,rendering_dt=1/30,physics_prim_path='/World/PhysicsScene');robots={s:w.scene.add(SingleArticulation('/World/Robot_'+s,name=s)) for s in initial};m=w.scene.add(SingleArticulation('/World/TaskMicrowave/E_body_9',name='micro'));w.reset()
 names=robots['left'].dof_names;lnames=robots['right'].dof_names;last=np.array([initial['left'][n] for n in names],np.float32);ql=np.array([initial['right'][n] for n in lnames],np.float32)
 for side,r in robots.items():r.set_joint_positions(last if side=='left' else ql);r.set_joint_velocities(np.zeros(len(r.dof_names),np.float32))
 robots['left'].get_articulation_controller().set_gains(kps=np.array([600 if n in arm_names('left') else 10 for n in names]),kds=np.array([35 if n in arm_names('left') else .3 for n in names]))
 di=next(i for i,n in enumerate(m.dof_names) if 'left' in n);si=m.dof_names.index('SpringLatchJoint')
 assert m._articulation_view.get_max_efforts()[0,di]==0
 bodypaths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bv=w.physics_sim_view.create_rigid_body_view(bodypaths);assert list(bv.prim_paths)==bodypaths
 sensors=['/World/Robot_left/'+l for l in links('left')];filters=['/World/TaskMicrowave/E_door_2','/World/TaskMicrowave/E_body_9','/World/TaskMicrowave/SpringLatchSlider','/World/V2Mat','/World/V2Tabletop']
 ct=w.physics_sim_view.create_rigid_contact_view(sensors,filter_patterns=[filters]*len(sensors),max_contact_data_count=32768)
 angle=0.;depthmax=0.;close_amount=np.zeros(2);normal=np.zeros(2);locktime=None;holdtime=None;command=0.;reason='duration';maxbody=0.
 for frame in range(30*32):
  t=frame/30
  if t<.7:phase='settle';q=interp(0)
  else:
   if locktime is None:
    phase='pinch';close_amount[:]=min(.004,max(0,t-.7)*.002)
   elif t<locktime+.7:phase='grip_hold'
   else:
    phase='pull'
    if holdtime is None:
     command=min(command+(.12 if angle<4 else a.follow_speed),max(0,angle)+a.lead,path[-1]['angle_deg'])
     if angle>61:holdtime=t
    else:phase='hold_open'
   q=interp(command);axis=Rot.from_euler('z',-command,degrees=True).apply([0,-1,0])
   if locktime is not None:close_amount+=np.clip((a.force-np.abs(normal))*.00002,-.00006,.00006);close_amount=np.clip(close_amount,.001,.008)
   for k,(fi,sign) in enumerate([(2,1),(3,1)]):q=finger_delta(q,fi,sign*axis*close_amount[k])
   palm=fk('left_palm_link','left',q);palm[:3,3]+=axis*min(a.grip_shift,max(0,t-.7)*.003);armq,pe,re=solve('left',palm,[q[n] for n in arm_names('left')]);q.update(zip(arm_names('left'),armq))
  blend=float(np.clip((angle-42)/13,0,1));akp=600*(1-blend)+a.soft_kp*blend;hkp=10*(1-blend)+2*blend
  robots['left'].get_articulation_controller().set_gains(kps=np.array([akp if n in arm_names('left') else hkp for n in names]),kds=np.array([35*(1-blend)+12*blend if n in arm_names('left') else .3 for n in names]))
  target=np.array([q[n] for n in names],np.float32);target=np.clip(target,last-.055,last+.055);samples=[];depth=0;deepest=None
  for sub in range(32):
   robots['right'].apply_action(ArticulationAction(joint_positions=ql,joint_efforts=robots['right']._articulation_view.get_generalized_gravity_forces()[0]))
   robots['left'].apply_action(ArticulationAction(joint_positions=last+(target-last)*(sub+1)/32,joint_efforts=robots['left']._articulation_view.get_generalized_gravity_forces()[0]));w.step(render=False);samples.append(np.asarray(ct.get_contact_force_matrix(1/960)).copy())
   cf,cp,cn,cd,counts,starts=ct.get_contact_data(1/960)
   for ri,ci in np.argwhere(np.asarray(counts)>0):
    b,n=int(starts[ri,ci]),int(counts[ri,ci]);d=-float(np.min(cd[b:b+n]))
    if d>depth:depth=d;deepest=[sensors[ri],filters[ci],d]
  last=target;force=np.mean(samples,axis=0);mag=np.max(np.linalg.norm(samples,axis=-1),axis=0);axis=Rot.from_euler('z',-command,degrees=True).apply([0,-1,0]);normal=np.array([sum(np.dot(force[i,0]+force[i,2],axis) for i,p in enumerate(sensors) if f'finger{fi}_' in p) for fi in [2,3]])
  mq=m.get_joint_positions();angle=-float(np.rad2deg(mq[di]));vel=float(m.get_joint_velocities()[di]);bodyforce=float(mag[:,1].sum());tableforce=float(mag[:,3:].sum());handleforce=float(mag[:,[0,2]].sum());depthmax=max(depthmax,depth);maxbody=max(maxbody,bodyforce)
  if locktime is None and phase=='pinch' and np.min(np.abs(normal))>1.0:locktime=t;print('HOOK_LOCK',t,normal.tolist(),flush=True)
  actual=robots['left'].get_joint_positions();records.append(dict(t=t+1/30,phase=phase,q=actual.tolist(),target=target.tolist(),door_angle_deg=angle,door_velocity=vel,latch_m=float(mq[si]),door_force_N=float(mag[:,0].sum()),latch_contact_N=float(mag[:,2].sum()),body_force_N=bodyforce,table_force_N=tableforce,hook_pull_force_N=normal.tolist(),close_amount_m=close_amount.tolist(),contact_penetration_m=depth,deepest_contact=deepest,all_contact_links={sensors[i]:mag[i].tolist() for i in range(len(sensors)) if mag[i].max()>.02}));poses.append(np.asarray(bv.get_transforms()).copy())
  if frame%30==0:print('STATE',t,phase,angle,normal.tolist(),bodyforce,depth,flush=True)
  if t<.6 and handleforce>.3:reason='initial hand contact';break
  if depth>.0001:reason='penetration gate';break
  if bodyforce>5 or tableforce>5:reason='unintended contact';break
  if handleforce>60:reason='excessive contact force';break
  if locktime is None and t>7:reason='no two-finger hook contact';break
  if holdtime is not None and t>holdtime+1.5 and abs(vel)<.08 and angle>60:reason='opened and held';break
  if holdtime is not None and t>holdtime+8:reason='open settling timeout';break
 report.update(status='passed_open_contact_trial' if reason=='opened and held' and angle>60 and abs(vel)<.1 else 'failed',reason=reason,physics_stepped=True,frames=len(records),sim_hz=960,body_paths=bodypaths,dof_names=names,last=records[-1],max_contact_penetration_m=depthmax,max_nonhandle_body_force_N=maxbody,sweep_parameters=vars(a))
except Exception:report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 if poses:
  np.savez_compressed(out/'trajectory.npz',body_paths=bodypaths,poses=np.asarray(poses),time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records))
 out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2));print('REPORT',json.dumps({k:v for k,v in report.items() if k not in ['last','body_paths','dof_names']}),flush=True);app.close()
