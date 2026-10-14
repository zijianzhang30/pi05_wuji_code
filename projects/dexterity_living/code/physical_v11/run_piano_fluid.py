import sys,argparse,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--name',default='piano_v11');a=ap.parse_args();out=P/'runs/physical_v1'/a.name
initial={}
for side,sgn in [('left',1),('right',-1)]:initial[side]=dict(zip(arm_names(side),[sgn*.65,2.12,-.95,sgn*.17,-1.18,sgn*.08]));initial[side].update(open_q(side))
hand=initial['right'].copy()
for n in hand_names('right'):
 fi=int(n.split('finger')[1][0]);ji=int(n[-1]);v=([0,.12,.35,.45,.2] if fi==2 else [0,.15,.5,.8,.45] if fi>2 else [0,.15,.45,.12,.1])[ji];hand[n]=float(np.clip(v,joints[n]['lo']+1e-5,joints[n]['hi']-1e-5))
R=np.array([[0,-1,0],[-.64278761,0,.76604444],[-.76604444,0,-.64278761]])
center=np.array([0,0,-.00755]);palm=fk('right_palm_link','right',hand);tip=fk('right_finger2_tip_link','right',hand);rel=np.linalg.inv(palm)@tip;off=rel[:3,3]+rel[:3,:3]@center
ns=arm_names('right');seed=np.array([-.16687,1.446,-1.048,.322,.414,-1.975])
def pose_at(target):
 X=np.eye(4);X[:3,:3]=R;X[:3,3]=target-R@off;qa,pe,re=solve('right',X,seed);assert pe<.003 and re<.02,(target,pe,re);d=hand.copy();d.update(zip(ns,qa));return d
xs=[-2.1323,-2.109,-2.08555,-2.0621,-2.0386,-2.0149,-1.9913,-1.9677]
hovers={i:pose_at(np.array([xs[i],-3.515,.7545])) for i in range(4,8)}
press={}
for i,h in hovers.items():
 names=[n for n in hand_names('right') if 'finger2_' in n];X=fk('right_finger2_tip_link','right',h);goal=X[:3,3]+X[:3,:3]@center-np.array([0,0,.010])
 def fun(x):
  d=h.copy();d.update(zip(names,x));X=fk('right_finger2_tip_link','right',d);return np.r_[100*(X[:3,3]+X[:3,:3]@center-goal),.01*(x-np.array([h[n] for n in names]))]
 opt=least_squares(fun,[h[n] for n in names],bounds=([joints[n]['lo'] for n in names],[joints[n]['hi'] for n in names]),max_nfev=200);d=h.copy();d.update(zip(names,opt.x));press[i]=d
above=pose_at(np.array([xs[5],-3.515,.90]));initial['right']=above.copy()
sequence=[(1.,above,'settle',-1),(.8,hovers[5],'approach',-1)]
for key in [5,5,6,6,7,7,6,4]:
 sequence.extend([(.20,hovers[key],'move',key),(.22,press[key],'press',key),(.08,press[key],'hold',key),(.20,hovers[key],'release',key)])
sequence.extend([(.7,above,'retreat',-1),(.5,above,'settle_end',-1)])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':a.gpu,'physics_gpu':a.gpu,'multi_gpu':False,'limit_cpu_threads':8})
records=[];poses=[];report=dict(status='failed',physics_stepped=True,object_pose_commands=0,object_attachments=0,task='piano');events=[]
try:
 from pxr import Usd,UsdPhysics
 from build import build,CURRENT
 from conservative_collision import validate_cooked
 from keyboard import add_keyboard
 import omni.usd
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 build(out,initial,task='piano');omni.usd.get_context().open_stage(str(out/'initial_scene.usda'));stage=omni.usd.get_context().get_stage();validate_cooked(stage,out);info=add_keyboard(stage,Usd.Stage.Open(str(CURRENT/'piano/animation.usdc')));(out/'keyboard_physics.json').write_text(json.dumps(info,indent=2))
 world=World(physics_dt=1/480,rendering_dt=1/30,physics_prim_path='/World/PhysicsScene');robots={s:world.scene.add(SingleArticulation('/World/Robot_'+s,name=s)) for s in initial};world.reset();names={s:list(r.dof_names) for s,r in robots.items()};last={s:np.array([initial[s][n] for n in names[s]],np.float32) for s in initial}
 for s,r in robots.items():r.set_joint_positions(last[s]);r.set_joint_velocities(np.zeros_like(last[s]))
 paths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bv=world.physics_sim_view.create_rigid_body_view(paths);assert list(bv.prim_paths)==paths;keys=world.physics_sim_view.create_rigid_body_view(info['paths']);assert list(keys.prim_paths)==info['paths']
 sensors=['/World/Robot_right/'+n for n in links('right')];filters=info['paths']+['/World/V2Mat','/World/V2Tabletop','/World/MusicRig/RealKeyboard/Model'];ct=world.physics_sim_view.create_rigid_contact_view(sensors,filter_patterns=[filters]*len(sensors),max_contact_data_count=16384)
 start=initial['right'];frame=0;abort=False;reason='completed';maxdepth=0;key0=np.array(info['centers'])[:,2];lock=None
 for duration,end,phase,key in sequence:
  if phase=='press':lock=None
  fired=False
  for k in range(round(duration*30)):
   u=(k+1)/round(duration*30);u=10*u**3-15*u**4+6*u**5;q={n:start[n]*(1-u)+end[n]*u for n in start}
   if phase in ['press','hold'] and lock is not None:q=lock.copy()
   target=np.array([q[n] for n in names['right']],np.float32);force=[];depth=0
   for sub in range(16):
    for side,r in robots.items():
     value=last[side] if side=='left' else last[side]+(target-last[side])*(sub+1)/16;r.apply_action(ArticulationAction(joint_positions=value,joint_efforts=r._articulation_view.get_generalized_gravity_forces()[0]))
    world.step(render=False);force.append(np.asarray(ct.get_contact_force_matrix(1/480)).copy());cf,cp,cn,cd,counts,starts=ct.get_contact_data(1/480)
    for ri,ci in np.argwhere(np.asarray(counts)>0):
     i,n=int(starts[ri,ci]),int(counts[ri,ci]);depth=max(depth,-float(np.min(cd[i:i+n])))
   last['right']=target.copy();mag=np.max(np.linalg.norm(force,axis=-1),axis=0);kf=mag[:,:8].sum(0);unintended=float(mag[:,8:].sum());state=np.asarray(keys.get_transforms());travel=key0-state[:,2];actual=robots['right'].get_joint_positions()
   if phase=='press' and lock is None and (kf[key]>.8 or travel[key]>.0022):lock=dict(zip(names['right'],actual.tolist()))
   if phase=='press' and not fired and kf[key]>.05 and travel[key]>.0008:events.append(dict(frame=frame,key=key,force_N=float(kf[key]),travel_m=float(travel[key])));fired=True
   records.append(dict(t=(frame+1)/30,phase=phase,q=actual.tolist(),target=target.tolist(),key_contact_N=kf.tolist(),key_travel_m=travel.tolist(),unintended_contact_N=unintended,contact_penetration_m=depth));poses.append(np.asarray(bv.get_transforms()).copy());maxdepth=max(depth,maxdepth)
   if frame%30==0:print('FRAME',frame,phase,'key',key,'force',kf,'travel',travel,flush=True)
   frame+=1
   if depth>.0001 or unintended>2 or float(kf.max())>5:abort=True;reason='contact_gate';break
  if abort:break
  start=q.copy()
 report.update(status='passed_piano_contact_trial' if not abort and len(events)>=8 else 'failed',reason=reason,events=events,max_contact_penetration_m=maxdepth,frames=len(records),dof_names=names['right'],body_paths=paths,sim_hz=480,physics_backend='PhysX CPU',key_drive='passive 220 N/m spring, 0.4 Ns/m damping, 3 mm travel; keys never commanded')
except Exception:report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 out.mkdir(parents=True,exist_ok=True)
 if poses:np.savez_compressed(out/'trajectory.npz',poses=np.asarray(poses),body_paths=paths,time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records))
 (out/'report.json').write_text(json.dumps(report,indent=2));print('REPORT',json.dumps(report),flush=True);app.close()
