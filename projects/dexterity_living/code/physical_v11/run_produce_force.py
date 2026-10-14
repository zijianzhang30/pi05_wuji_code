"""Single-hand sequential orange/pepper sorting. Robot motors only; food remains free rigid bodies."""
import sys,argparse,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--name',default='produce_v11');ap.add_argument('--force',type=float,default=3);ap.add_argument('--gap',type=float,default=.02);ap.add_argument('--close-distance',type=float,default=.035);ap.add_argument('--substeps',type=int,default=16);a=ap.parse_args();out=P/'runs/physical_v1'/a.name
initial={}
for side,sgn in [('left',1),('right',-1)]:initial[side]=dict(zip(arm_names(side),[sgn*.65,2.12,-.95,sgn*.17,-1.18,sgn*.08]));initial[side].update(open_q(side))
candidate=json.loads((P/'runs/physical_v1/contact_tip_1.json').read_text())[0];centers={l:np.array(v) for l,v in candidate['pad_centers'].items()};axis=np.array([0,1.,0])
def finger_delta(q,fi,delta):
 ns=[n for n in hand_names('right') if f'finger{fi}_' in n];l=f'right_finger{fi}_tip_link';X=fk(l,'right',q);target=X[:3,3]+X[:3,:3]@centers[l]+delta
 def fun(x):
  d=q.copy();d.update(zip(ns,x));T=fk(l,'right',d);return T[:3,3]+T[:3,:3]@centers[l]-target
 opt=least_squares(fun,[float(np.clip(q[n],joints[n]['lo']+1e-6,joints[n]['hi']-1e-6)) for n in ns],bounds=([joints[n]['lo']+1e-6 for n in ns],[joints[n]['hi']-1e-6 for n in ns]),max_nfev=80);d=q.copy();d.update(zip(ns,opt.x));return d
def shift(q,delta):
 T=fk('right_palm_link','right',q);T[:3,3]+=delta;ns=arm_names('right');seed=np.array([q[n] for n in ns]);lo=np.array([joints[n]['lo']+1e-6 for n in ns]);hi=np.array([joints[n]['hi']-1e-6 for n in ns])
 def residual(x):
  d=q.copy();d.update(zip(ns,x));X=fk('right_palm_link','right',d);return np.r_[X[:3,3]-T[:3,3],.025*Rot.from_matrix(T[:3,:3]@X[:3,:3].T).as_rotvec(),.0001*(x-seed)]
 opt=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=250);assert np.linalg.norm(residual(opt.x)[:3])<.003,(delta,residual(opt.x));d=q.copy();d.update(zip(ns,opt.x));return d
def grasp_plan(center,half):
 q=candidate['q'].copy()
 for n in hand_names('right'):
  if any(f'finger{fi}_' in n for fi in [3,4,5]):
   ji=int(n[-1]);q[n]=float(np.clip([0,0,.35,.45,.25][ji],joints[n]['lo']+1e-5,joints[n]['hi']-1e-5))
 ns=arm_names('right')+[n for n in hand_names('right') if 'finger1_' in n or 'finger2_' in n];targets=np.array([center-axis*(half[1]+a.gap),center+axis*(half[1]+a.gap)]);seed=np.array([q[n] for n in ns])
 def fun(x):
  d=q.copy();d.update(zip(ns,x));got=[]
  for fi in [1,2]:
   l=f'right_finger{fi}_tip_link';T=fk(l,'right',d);got.append(T[:3,3]+T[:3,:3]@centers[l])
  return np.r_[100*(np.array(got)-targets).ravel(),.02*(x-seed)]
 opt=least_squares(fun,seed,bounds=([joints[n]['lo']+1e-6 for n in ns],[joints[n]['hi']-1e-6 for n in ns]),max_nfev=250);assert np.max(np.abs(fun(opt.x)[:6]))<.1; q.update(zip(ns,opt.x));return q
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':a.gpu,'physics_gpu':a.gpu,'limit_cpu_threads':8})
records=[];poses=[];report=dict(status='failed',physics_stepped=True,object_pose_commands=0,object_attachments=0,task='produce_sort');out.mkdir(parents=True,exist_ok=True)
try:
 from build import build
 from conservative_collision import validate_cooked
 from produce_enclosure import enclose
 from pxr import Usd,UsdGeom,UsdPhysics,Gf
 import omni.usd
 from isaacsim.core.api import World
 from isaacsim.core.prims import SingleArticulation
 from isaacsim.core.utils.types import ArticulationAction
 build(out,initial,task='produce_sort');omni.usd.get_context().open_stage(str(out/'initial_scene.usda'));stage=omni.usd.get_context().get_stage();validate_cooked(stage,out)
 foodpaths=['/World/TaskFood_orange','/World/TaskFood_pepper'];binpaths=['/World/TaskBin_fruit','/World/TaskBin_vegetable'];recipes=[enclose(stage,x) for x in foodpaths];(out/'food_enclosures.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='body_vertices'} for r in recipes],indent=2))
 # Initial placement correction only, before simulation and recording.
 bbox=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
 for path in foodpaths+binpaths:
  prim=stage.GetPrimAtPath(path);low=float(bbox.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2]);xf=UsdGeom.Xformable(prim);T=np.asarray(xf.GetLocalTransformation()).T;T[2,3]+=.6816-low;xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix='settleInit').Set(Gf.Matrix4d(T.T.tolist()))
 stage.GetRootLayer().Save()
 world=World(physics_dt=1/(30*a.substeps),rendering_dt=1/30,physics_prim_path='/World/PhysicsScene');robots={s:world.scene.add(SingleArticulation('/World/Robot_'+s,name=s)) for s in initial};world.reset();names={s:list(r.dof_names) for s,r in robots.items()};last={s:np.array([initial[s][n] for n in names[s]],np.float32) for s in initial}
 for s,r in robots.items():r.set_joint_positions(last[s]);r.set_joint_velocities(np.zeros_like(last[s]))
 paths=[str(pr.GetPath()) for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)];bv=world.physics_sim_view.create_rigid_body_view(paths);assert list(bv.prim_paths)==paths;fv=world.physics_sim_view.create_rigid_body_view(foodpaths);assert list(fv.prim_paths)==foodpaths;bins=world.physics_sim_view.create_rigid_body_view(binpaths)
 sensors=['/World/Robot_right/'+n for n in links('right')];filters=foodpaths+binpaths+['/World/V2Mat','/World/V2Tabletop'];ct=world.physics_sim_view.create_rigid_contact_view(sensors,filter_patterns=[filters]*len(sensors),max_contact_data_count=16384)
 for _ in range(360*a.substeps):
  for s,r in robots.items():r.apply_action(ArticulationAction(joint_positions=last[s],joint_efforts=r._articulation_view.get_generalized_gravity_forces()[0]))
  world.step(render=False)
 current=initial['right'].copy();reference=np.asarray(fv.get_transforms())[:,:3].copy();touched=[False,False];drift=[0.,0.];maxdepth=0;frame=0;active=0;force_axis=np.zeros(2);gripped=False;servo=None;descend_stop=False;peaks=np.asarray(fv.get_transforms())[:,2].copy();plans=[];abort=False
 def tick(q,phase):
  global current,last,frame,force_axis,gripped,servo,maxdepth,abort,descend_stop
  target=np.array([q[n] for n in names['right']],np.float32);samples=[];depth=0;deepest=None
  for sub in range(a.substeps):
   for s,r in robots.items():
    value=last[s] if s=='left' else last[s]+(target-last[s])*(sub+1)/a.substeps;r.apply_action(ArticulationAction(joint_positions=value,joint_efforts=r._articulation_view.get_generalized_gravity_forces()[0]))
   world.step(render=False);instant=np.asarray(ct.get_contact_force_matrix(1/(30*a.substeps))).copy();samples.append(instant)
   if phase=='descend' and not descend_stop and np.linalg.norm(instant[:,active],axis=-1).sum()>.5:
    descend_stop=True;target=robots['right'].get_joint_positions().copy();last['right']=target.copy();q=dict(zip(names['right'],target.tolist()))
   cf,cp,cn,cd,counts,starts=ct.get_contact_data(1/(30*a.substeps))
   for ri,ci in np.argwhere(np.asarray(counts)>0):
    i,n=int(starts[ri,ci]),int(counts[ri,ci]);dd=-float(np.min(cd[i:i+n]))
    if dd>depth:depth=dd;deepest=dict(link=sensors[ri],object=filters[ci],depth=dd)
  last['right']=target.copy();vec=np.mean(samples,axis=0)[:,active];mag=np.max(np.linalg.norm(samples,axis=-1),axis=0);force_axis=np.array([sum(np.dot(vec[i],axis) for i,p in enumerate(sensors) if f'finger{fi}_' in p) for fi in [1,2]]);foodforce=mag[:,:2].sum(0);unwanted=float(mag[:,2:].sum());state=np.asarray(fv.get_transforms()).copy();maxdepth=max(maxdepth,depth)
  for j in [0,1]:
   peaks[j]=max(peaks[j],state[j,2])
   if foodforce[j]>.01:touched[j]=True
   if not touched[j]:drift[j]=max(drift[j],float(np.linalg.norm(state[j,:3]-reference[j])))
  actual=robots['right'].get_joint_positions();records.append(dict(t=(frame+1)/30,phase=phase,active_object=active,q=actual.tolist(),target=target.tolist(),q_left=robots['left'].get_joint_positions().tolist(),food_poses=state.tolist(),food_contact_N=foodforce.tolist(),contact_links_N={sensors[i]:float(mag[i,active]) for i in range(len(sensors)) if mag[i,active]>.01},thumb_axis_N=float(force_axis[0]),index_axis_N=float(force_axis[1]),unintended_contact_N=unwanted,contact_penetration_m=depth,deepest_contact=deepest));poses.append(np.asarray(bv.get_transforms()).copy());frame+=1
  if phase=='close' and not gripped and force_axis[0]*force_axis[1]<-.04:gripped=True;servo={n:q[n] for n in hand_names('right')}
  if depth>.0001 or unwanted>5 or max(drift)>.0001 or foodforce.max()>40:abort=True;raise RuntimeError(('contact_gate',phase,depth,unwanted,drift,foodforce.tolist()))
  current=q.copy()
  if frame%30==0:print('FRAME',frame,phase,active,state[:,:3].tolist(),foodforce.tolist(),flush=True)
 def move(end,duration,phase,cart=False,hold=False):
  global current,servo,descend_stop
  start=current.copy();A=fk('right_palm_link','right',start);B=fk('right_palm_link','right',end)
  for k in range(round(duration*30)):
   u=(k+1)/round(duration*30);u=10*u**3-15*u**4+6*u**5;q={n:start[n]*(1-u)+end[n]*u for n in start}
   if cart:
    X=A.copy();X[:3,3]=(1-u)*A[:3,3]+u*B[:3,3];X[:3,:3]=Rot.from_rotvec(u*Rot.from_matrix(B[:3,:3]@A[:3,:3].T).as_rotvec()).as_matrix()@A[:3,:3];qa,pe,re=solve('right',X,[q[n] for n in arm_names('right')]);assert pe<.003 and re<.03;q.update(zip(arm_names('right'),qa))
   if phase=='close' and gripped:q.update(servo)
   if hold:
    q.update(servo)
    for fi,sign,normal in [(1,1,abs(force_axis[0])),(2,-1,abs(force_axis[1]))]:q=finger_delta(q,fi,sign*axis*float(np.clip((a.force-normal)*.00005,-.0001,.0001)))
    servo={n:q[n] for n in hand_names('right')}
   tick(q,phase)
   if phase=='descend' and descend_stop:break
 move(current,1.,'settle')
 for active in [0,1]:
  pos=np.asarray(fv.get_transforms())[active];v=np.array(recipes[active]['body_vertices']);R=Rot.from_quat(pos[3:]).as_matrix();wv=v@R.T+pos[:3];center=(wv.min(0)+wv.max(0))/2;half=(wv.max(0)-wv.min(0))/2
  openhand=grasp_plan(center,half);above=shift(openhand,np.array([0,0,.08]));plans.append(dict(active=active,center=center.tolist(),half=half.tolist(),q=openhand));(out/'grasp_plans.json').write_text(json.dumps(plans,indent=2))
  # Preshape while raised, then approach from above.
  descend_stop=False;pre=current.copy();pre.update({n:openhand[n] for n in hand_names('right')});move(pre,.8,'preshape');move(above,2.,'approach');move(openhand,1.5,'descend',cart=True)
  closed=current.copy()
  for fi,sign in [(1,1),(2,-1)]:closed=finger_delta(closed,fi,sign*axis*a.close_distance)
  gripped=False;servo=None;move(closed,2.,'close');assert gripped,('no_opposed_grasp',active);move(current,.7,'grip_hold',hold=True)
  lifted=shift(current,np.array([0,0,.10]));move(lifted,1.5,'lift',cart=True,hold=True);assert np.asarray(fv.get_transforms())[active,2]>reference[active,2]+.06,('object_not_lifted',active)
  goal=np.asarray(bins.get_transforms())[active,:3];obj=np.asarray(fv.get_transforms())[active,:3];delta=np.r_[goal[:2]-obj[:2],0.];transfer=shift(current,delta);move(transfer,2.,'transfer',cart=True,hold=True)
  obj=np.asarray(fv.get_transforms())[active,:3];lower=shift(current,np.array([0,0,max(-.13,.735+half[2]-obj[2]) ]));move(lower,1.5,'lower',cart=True,hold=True)
  release=current.copy()
  for fi,sign in [(1,-1),(2,1)]:release=finger_delta(release,fi,sign*axis*.020)
  move(release,.8,'release');move(shift(current,np.array([0,0,.08])),1.,'retreat',cart=True);move(current,1.,'settle_item')
 move(current,1.5,'settle_end');final=np.asarray(fv.get_transforms())[:,:3];goals=np.asarray(bins.get_transforms())[:,:3];ok=all(np.linalg.norm(final[i,:2]-goals[i,:2])<.065 and .69<final[i,2]<.80 and peaks[i]>.79 for i in [0,1]);report.update(status='passed_produce_sort_trial' if ok else 'failed',reason='completed',final_food_positions=final.tolist(),bin_positions=goals.tolist(),max_food_z=peaks.tolist(),precontact_drift_m=drift,max_contact_penetration_m=maxdepth)
except Exception:report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 if poses:
  np.savez_compressed(out/'trajectory.npz',poses=np.asarray(poses),body_paths=paths,time=np.array([r['t'] for r in records]));(out/'contacts.json').write_text(json.dumps(records));report.update(frames=len(records),body_paths=paths,dof_names=names['right'],dof_names_left=names['left'],sim_hz=30*a.substeps)
 (out/'report.json').write_text(json.dumps(report,indent=2));print('REPORT',json.dumps(report),flush=True);app.close()
