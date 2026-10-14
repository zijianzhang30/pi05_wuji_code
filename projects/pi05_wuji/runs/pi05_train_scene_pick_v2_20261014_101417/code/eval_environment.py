"""Shared online Wuji environment and task-state evaluation, independent of policy."""
import json,time,os
from pathlib import Path
import numpy as np
from task_criteria import produce_collection_criteria
P=Path('/media/external/bowenjing/projects/dexterity_living')
CURRENT=P/'runs/wuji_blackwhite_smooth_pt4k_20261011/living_mcl_wuji_v1_20261011'

def save(path,value):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def install_reset_hook(args):
 from isaacsim.core.api import World
 original=World.reset
 def reset(world,*a,**kw):
  import omni.usd
  from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
  stage=omni.usd.get_context().get_stage();src=Usd.Stage.Open(str(CURRENT/args.task/'animation.usdc'));flat=src.Flatten()
  excluded={'DeploymentRobot','BasicRig','TaskObjects','TaskMicrowave','MusicRig','TaskFood_orange','TaskFood_pepper','TaskBin_fruit','TaskBin_vegetable','PhysicsScene'}
  copied=[]
  for pr in src.GetPrimAtPath('/World').GetChildren():
   path=str(pr.GetPath())
   if pr.GetName() in excluded or stage.GetPrimAtPath(path):continue
   Sdf.CopySpec(flat,path,stage.GetRootLayer(),path);copied.append(path)
   # Background is visual only, frozen at time zero. Task physics stays untouched.
   for leaf in list(Usd.PrimRange(stage.GetPrimAtPath(path))):
    if leaf.IsA(UsdPhysics.Joint) or leaf.IsA(UsdPhysics.Scene):stage.RemovePrim(leaf.GetPath());continue
    for schema in list(leaf.GetAppliedSchemas()):
     if 'Physics' in schema or 'Physx' in schema:leaf.RemoveAPI(schema)
    for attr in leaf.GetAttributes():
     ts=attr.GetTimeSamples()
     if ts:
      value=attr.Get(0)
      for t in ts:attr.ClearAtTime(t)
      if value is not None:attr.Set(value)
  material=UsdShade.Material.Define(stage,'/World/ReplayBlack');shader=UsdShade.Shader.Define(stage,'/World/ReplayBlack/Shader');shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.018,.023,.028));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.48);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
  for side in ['left','right']:UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/Robot_'+side)).Bind(material,UsdShade.Tokens.strongerThanDescendants)
  for path,color,rough in [('/World/WujiMaterials/Hands/Shader',(.018,.023,.028),.48),('/World/V2Materials/MatGray/Surface',(.8,.81,.82),.7),('/World/V2Materials/TableEdge/Surface',(.8,.81,.82),.55)]:
   sh=UsdShade.Shader.Get(stage,path)
   if sh:
    sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));sh.GetInput('diffuseColor').DisconnectSource();sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough)
  if stage.GetPrimAtPath('/World/Floor'):UsdGeom.Imageable(stage.GetPrimAtPath('/World/Floor')).MakeInvisible()
  cam=UsdGeom.Camera.Define(stage,'/World/TableEgo');xf=UsdGeom.Xformable(cam);xf.ClearXformOpOrder()
  for attr in list(cam.GetPrim().GetAttributes()):
   if attr.GetName().startswith('xformOp:'):cam.GetPrim().RemoveProperty(attr.GetName())
  xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(-2.05,-4.15,1.72),Gf.Vec3d(-2.05,-3.51,.72),Gf.Vec3d(0,0,1)).GetInverse())
  cam.CreateFocalLengthAttr(16.5);cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(15.2908);cam.CreateClippingRangeAttr(Gf.Vec2f(.025,100))
  roots=['/World/TaskObjects/object'] if args.task in ['pick_place','stack'] else ['/World/TaskFood_orange','/World/TaskFood_pepper'] if args.task=='produce_sort' else []
  rng=np.random.default_rng(args.seed+70000);changes=[]
  for path in roots:
   prim=stage.GetPrimAtPath(path);xf=UsdGeom.Xformable(prim);T=np.array(xf.GetLocalTransformation()).T;delta=np.r_[rng.uniform(-.003,.003,2),0.];T[:3,3]+=delta;xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix='evaluation_init').Set(Gf.Matrix4d(T.T.tolist()));changes.append({'path':path,'delta':delta.tolist()})
  stage.GetRootLayer().Save()
  save(P/'runs/physical_v1'/args.name/'environment.json',{'camera':'/World/TableEgo','resolution':[640,480],'seed':args.seed,'object_initialization':changes,'copied_background':copied,'physics': 'unchanged source task physics; background copies visual only'})
  result=original(world,*a,**kw)
  # Explicit render-purpose bindings avoid the runtime's fallback material on
  # collision geometry. The physics-purpose binding and friction stay intact.
  white=UsdShade.Material.Define(stage,'/World/EvalWhiteDesk');sh=UsdShade.Shader.Define(stage,'/World/EvalWhiteDesk/Surface');sh.CreateIdAttr('UsdPreviewSurface')
  sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8,.81,.82));sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.7);sh.CreateOutput('surface',Sdf.ValueTypeNames.Token);white.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
  for purpose in ['', 'preview','full']:
   UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/V2Mat')).Bind(white,UsdShade.Tokens.strongerThanDescendants,purpose)
  return result
 World.reset=reset

def run_policy(ns,args):
 from multiprocessing.connection import Client
 from isaacsim.core.utils.types import ArticulationAction
 from model import joints,arm_names
 from PIL import Image
 import omni.replicator.core as rep
 import carb,imageio.v2 as imageio
 task=args.task;out=ns['out'];world=ns.get('world',ns.get('w'));robots=ns['robots'];bv=ns['bv'];ct=ns.get('ct',ns.get('contact'))
 names={s:list(robots[s].dof_names) for s in ['left','right']};allnames=names['left']+names['right'];dt=world.get_physics_dt();substeps=round(1/(30*dt));assert substeps in [16,32]
 initial=ns.get('initial',ns.get('qinit'));last=np.concatenate([np.array([initial[s][n] for n in names[s]],np.float32) for s in ['left','right']])
 lower=np.array([joints[n]['lo'] for n in allnames]);upper=np.array([joints[n]['hi'] for n in allnames])
 def qpos():return np.concatenate([robots[s].get_joint_positions() for s in ['left','right']])
 def command(target,previous,record_contact=True):
  samples=[];depth=0.;surface_failure=False
  for sub in range(substeps):
   value=previous+(target-previous)*(sub+1)/substeps
   for i,s in enumerate(['left','right']):
    r=robots[s];r.apply_action(ArticulationAction(joint_positions=value[i*26:(i+1)*26].astype(np.float32),joint_efforts=r._articulation_view.get_generalized_gravity_forces()[0]))
   world.step(render=False)
   if record_contact:
    samples.append(np.asarray(ct.get_contact_force_matrix(dt)).copy())
    cf,cp,cn,cd,counts,starts=ct.get_contact_data(dt)
    for ri,ci in np.argwhere(np.asarray(counts)>0):
     start,count=int(starts[ri,ci]),int(counts[ri,ci]);d=max(0.,-float(np.min(cd[start:start+count])));depth=max(depth,d)
     if task=='produce_sort' and d>.0001:
      if ci>=2 or d>=.003:surface_failure=True
      else:
       check=ns['watch'].check(ns['sensors'][ri],ns['filters'][ci],np.asarray(bv.get_transforms()))
       if check['intersections']:surface_failure=True
  return (np.asarray(samples),depth,surface_failure)
 save(out/'eval_status.json',{'state':'settling','task':task,'seed':args.seed,'updated_at':time.time()})
 start_target=last.copy()
 for frame in range(390 if task=='produce_sort' else 30):
  target=start_target.copy()
  if task=='piano':
   u=(frame+1)/30;u=10*u**3-15*u**4+6*u**5
   above=np.array([ns['above'][n] for n in names['right']]);target[26:]=start_target[26:]*(1-u)+above*u
  command(target,last,False);last=target
 # Inference process uses its own verified Python environment.
 conn=Client(args.socket,family='AF_UNIX');conn.send({'task':task,'seed':args.seed,'mode':args.mode,'denoising_steps':int(os.environ.get('DIAG_DENOISING_STEPS','10'))});meta=conn.recv();assert meta['joint_names']==allnames
 settings=carb.settings.get_settings();settings.set('/rtx/rendermode','RayTracedLighting');settings.set('/rtx/post/motionblur/enabled',False)
 rp=rep.create.render_product('/World/TableEgo',(640,480));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([rp])
 def capture(warm=False):
  before=qpos().copy();poses=np.asarray(bv.get_transforms()).copy();t=world.current_time
  for _ in range(20 if warm else 3):world.render()
  image=np.asarray(rgb.get_data())
  assert image.shape[:2]==(480,640),('invalid render',image.shape)
  assert world.current_time==t and np.max(abs(qpos()-before))<1e-7 and np.max(abs(np.asarray(bv.get_transforms())-poses))<1e-7,'render advanced physics'
  return image[:,:,:3].copy()
 frame0=capture(True);assert frame0.std()>5
 Image.fromarray(frame0).save(out/'initial_observation.png')
 if args.reference:
  import h5py,cv2
  from pxr import UsdGeom
  with h5py.File(Path(args.reference)/'episode.hdf5') as f:
   refq=f['observations/qpos'][0];refv=f['observations/qvel'][0];refg=f['controller/gains_kp_kd'][0];refp=f['observations/palm_pose_world_xyzw'][0]
   refimage=cv2.cvtColor(cv2.imdecode(np.asarray(f['observations/images/ego'][0],np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
   refcam=f['camera/camera_to_world'][:];refintr=f['camera/intrinsics'][:]
  actualv=np.concatenate([robots[s].get_joint_velocities() for s in ['left','right']]);gains=np.concatenate([np.stack(robots[s].get_articulation_controller().get_gains(),axis=0) for s in ['left','right']],axis=-1)
  paths=list(bv.prim_paths);palms=np.asarray(bv.get_transforms())[[paths.index('/World/Robot_'+s+'/'+s+'_palm_link') for s in ['left','right']]]
  cam=UsdGeom.Camera(ns['stage'].GetPrimAtPath('/World/TableEgo'));matrix=np.array(cam.GetLocalTransformation()).T
  intr=np.array([[640*16.5/20.955,0,320],[0,480*16.5/15.2908,240],[0,0,1]])
  checks={'qpos_max_abs_rad':float(abs(qpos()-refq).max()),'qvel_max_abs_rad_s':float(abs(actualv-refv).max()),'gains_max_abs':float(abs(gains-refg).max()),'palm_pose_max_abs':float(abs(palms-refp).max()),'camera_matrix_max_abs':float(abs(matrix-refcam).max()),'intrinsics_max_abs':float(abs(intr-refintr).max()),'image_mean_abs_8bit':float(abs(frame0.astype(float)-refimage).mean())}
  passed=checks['qpos_max_abs_rad']<1e-4 and checks['qvel_max_abs_rad_s']<.001 and checks['gains_max_abs']<1e-5 and checks['palm_pose_max_abs']<1e-4 and checks['camera_matrix_max_abs']<1e-6 and checks['intrinsics_max_abs']<1e-3
  with h5py.File(Path(args.reference)/'episode.hdf5') as f:
   body_names=f['metadata/body_paths'].asstr()[:].tolist()
  acceptance=json.loads((Path(args.reference)/'acceptance.json').read_text())
  with np.load(Path(acceptance['source_run'])/'observation_trajectory.npz') as recorded:refposes=recorded['poses'][int(acceptance['start_frame'])]
  actualposes=np.asarray(bv.get_transforms());dynamic_checks={}
  for path in ['/World/TaskObjects/object']:
   ai=paths.index(path);ri=body_names.index(path)
   dynamic_checks[path]=float(abs(actualposes[ai]-refposes[ri]).max())
  checks['task_body_pose_max_abs']=dynamic_checks
  passed=passed and all(v<1e-4 for v in dynamic_checks.values())
  Image.fromarray(refimage).save(out/'reference_initial_observation.png')
  report={'status':'reset_check','passed':bool(passed),'success':None,'task':task,'seed':args.seed,'reference':args.reference,'checks':checks}
  save(out/'reset_alignment.json',report)
  assert passed,('training scene reset alignment failed',checks)

 writer=imageio.get_writer(str(out/'rollout.mp4'),fps=30,codec='libx264',macro_block_size=16,ffmpeg_params=['-crf','18','-preset','fast','-threads','2'])
 maxsteps={'gesture_one':180,'gesture_two':180,'pick_place':450,'stack':540,'piano':330,'produce_sort':1200,'microwave_open':450}[task]
 records=[];poses=[];observations=[];commands=[];reason='time_limit';success=False;streak=0;maxstreak=0;eventkeys=[];pressed=np.zeros(8,bool);maxdepth=0.;inference=[];clip_count=0
 object_start=None;maxz=-np.inf;food_peaks=None;first_contact=False;drift=0.;food_touched=np.zeros(2,bool);fourstreak=0;maxfour=0
 if task in ['pick_place','stack']:object_start=np.asarray(ns['obj'].get_transforms())[0,:3].copy()
 if task=='produce_sort':object_start=np.asarray(ns['fv'].get_transforms())[:,:3].copy();food_peaks=object_start[:,2].copy()
 maxsteps*=int(os.environ.get('DIAG_MAX_STEPS_MULT','1'))
 chunks=[];query_frames=[];raw_commands=[]
 began=time.monotonic();actions=None
 try:
  for frame in range(maxsteps):
   image=frame0 if frame==0 else capture();q=qpos();assert np.isfinite(q).all();writer.append_data(image)
   if frame%args.horizon==0:
    conn.send({'q':q.tolist(),'rgb_bytes':image.tobytes(),'control_step':frame});reply=conn.recv();actions=np.asarray(reply['actions'],np.float32)[:args.horizon];inference.append(reply['inference_seconds']);chunks.append(np.asarray(reply['actions'],np.float32));query_frames.append(frame)
   raw=actions[frame%args.horizon];raw_commands.append(raw.copy());target=np.clip(raw,lower,upper);clip_count+=int(np.any(target!=raw))
   if task=='microwave_open':
    target[:26]=np.clip(target[:26],last[:26]-.055,last[:26]+.055)
    angle=-float(np.rad2deg(ns['m'].get_joint_positions()[ns['di']]));blend=float(np.clip((angle-42)/13,0,1))
    robots['left'].get_articulation_controller().set_gains(kps=np.array([(600*(1-blend)+80*blend) if n in arm_names('left') else (10*(1-blend)+2*blend) for n in names['left']]),kds=np.array([35*(1-blend)+12*blend if n in arm_names('left') else .3 for n in names['left']]))
   samples,depth,surface_failure=command(target,last);last=target.copy();mag=np.max(np.linalg.norm(samples,axis=-1),axis=0);maxdepth=max(maxdepth,depth)
   observations.append(q);commands.append(target);poses.append(np.asarray(bv.get_transforms()).copy())
   rec={'frame':frame,'t':(frame+1)/30,'q':qpos().tolist(),'target':target.tolist(),'contact_penetration_m':depth};failure=None;candidate=False
   if task.startswith('gesture'):
    goal=np.array([ns['sign'][n] for n in names['right']]);error=float(np.max(abs(qpos()[26:]-goal)));unwanted=float(mag.sum());candidate=error<.2;rec.update(goal_max_joint_error_rad=error,unintended_contact_N=unwanted)
    if unwanted>5 or depth>.0001:failure='unintended_contact'
   elif task in ['pick_place','stack']:
    pos=np.asarray(ns['obj'].get_transforms())[0,:3];vel=np.asarray(ns['obj'].get_velocities())[0,:3];normal=float(mag[:,0].sum());table=float(mag[:,1:].sum());maxz=max(maxz,float(pos[2]));first_contact=first_contact or normal>.01
    if not first_contact:drift=max(drift,float(np.linalg.norm(pos-object_start)))
    candidate=(maxz>(.79 if task=='stack' else .763) and np.linalg.norm(pos[:2]-[-2.17,-3.505])<.04 and np.linalg.norm(vel)<.03 and normal<.05 and (abs(pos[2]-.752)<.007 if task=='stack' else abs(pos[2]-object_start[2])<.012))
    rec.update(object_pose=pos.tolist(),object_velocity=vel.tolist(),object_contact_N=normal,table_contact_N=table,max_object_z_m=maxz)
    if depth>.0001 or table>12 or normal>35:failure='contact_gate'
    if drift>.0001:failure='precontact_object_drift'
   elif task=='piano':
    kf=mag[:,:8].sum(0);travel=ns['key0']-np.asarray(ns['keys'].get_transforms())[:,2];unwanted=float(mag[:,8:].sum());down=(kf>.05)&(travel>.0008)
    for key in np.where(down&~pressed)[0]:eventkeys.append(int(key))
    pressed=np.where((travel<.0004)|(kf<.01),False,pressed|down)
    rec.update(key_contact_N=kf.tolist(),key_travel_m=travel.tolist(),events=list(eventkeys))
    if eventkeys!=[5,5,6,6,7,7,6,4][:len(eventkeys)]:failure='wrong_note_sequence'
    candidate=eventkeys==[5,5,6,6,7,7,6,4] and not down.any()
    if depth>.0001 or unwanted>2 or kf.max()>5:failure='contact_gate'
   elif task=='produce_sort':
    pos=np.asarray(ns['fv'].get_transforms())[:,:3];goals=np.asarray(ns['bins'].get_transforms())[:,:3];food=mag[:,:2].sum(0);unwanted=float(mag[:,2:].sum());food_peaks=np.maximum(food_peaks,pos[:,2]);food_touched|=food>.01
    for j in range(2):
     if not food_touched[j]:drift=max(drift,float(np.linalg.norm(pos[j]-object_start[j])))
    vel=np.asarray(ns['fv'].get_velocities())[:,:3]
    history=records[-14:];criteria=produce_collection_criteria([x['food_poses'] for x in history]+[pos.tolist()],goals,food_peaks,[x['food_contact_N'] for x in history]+[food.tolist()]);candidate=criteria['success'];rec.update(collection_criteria=criteria,food_velocity_m_s=vel.tolist())
    rec.update(food_poses=pos.tolist(),food_contact_N=food.tolist(),bin_positions=goals.tolist(),unintended_contact_N=unwanted)
    if surface_failure or unwanted>5 or food.max()>40:failure='contact_gate'
    if drift>.0001:failure='precontact_object_drift'
   else:
    m=ns['m'];angle=-float(np.rad2deg(m.get_joint_positions()[ns['di']]));vel=float(m.get_joint_velocities()[ns['di']]);axis=ns['Rot'].from_euler('z',-angle,degrees=True).apply([0,-1,0]);force=np.mean(samples,axis=0)
    normal=np.array([sum(np.dot(force[i,0]+force[i,2],axis) for i,p in enumerate(ns['sensors']) if f'finger{fi}_' in p) for fi in [2,3,4,5]])
    fourstreak=fourstreak+1 if np.min(abs(normal))>.4 else 0;maxfour=max(maxfour,fourstreak);body=float(mag[:,1].sum());table=float(mag[:,3:].sum());handle=float(mag[:,[0,2]].sum())
    candidate=angle>60 and abs(vel)<.08 and maxfour>=5
    rec.update(door_angle_deg=angle,door_velocity=vel,four_finger_forces_N=normal.tolist(),four_finger_max_streak=maxfour,body_force_N=body,table_force_N=table)
    if depth>.0001 or body>5 or table>5 or handle>60:failure='contact_gate'
   if not np.isfinite(qpos()).all():failure='nonfinite_robot_state'
   streak=streak+1 if candidate else 0;maxstreak=max(maxstreak,streak);records.append(rec)
   if frame%5==0:save(out/'eval_status.json',{'state':'evaluating','task':task,'seed':args.seed,'frame':frame+1,'max_steps':maxsteps,'policy_queries':len(inference),'updated_at':time.time(),'elapsed_seconds':time.monotonic()-began})
   if frame%30==0:print('POLICY_FRAME',task,frame,'candidate',bool(candidate),'depth',depth,flush=True)
   if failure:reason=failure;break
   if streak>=(30 if task.startswith('gesture') else 1 if task=='produce_sort' else 15):success=True;reason='task_success';break
  Image.fromarray(capture()).save(out/'final_observation.png')
 finally:
  writer.close()
  try:conn.send(None)
  except (BrokenPipeError,EOFError):pass
  conn.close();rgb.detach([rp])
  save(out/'contacts.json',records)
  paths=list(bv.prim_paths)
  np.savez_compressed(out/'trajectory.npz',poses=np.asarray(poses),body_paths=paths,time=np.arange(1,len(poses)+1)/30)
  np.savez_compressed(out/'policy_rollout.npz',observations=np.asarray(observations),actions=np.asarray(commands),raw_commands=np.asarray(raw_commands),chunks=np.asarray(chunks),query_frames=np.asarray(query_frames),joint_names=allnames)
 report={'status':'complete','success':bool(success),'reason':reason,'task':task,'seed':args.seed,'frames':len(records),'max_steps':maxsteps,'diagnostic_denoising_steps':int(os.environ.get('DIAG_DENOISING_STEPS','10')),'policy_queries':len(inference),'checkpoint':meta,'physics_stepped':True,'object_pose_commands':0,'object_attachments':0,'expert_actions_after_initialization':len(records) if args.mode=='expert' else 0,'online_elapsed_control_step_input':True,'control_hz':30,'physics_hz':round(1/dt),'replan_steps':args.horizon,'predicted_chunk_steps':30,'criteria_version':'v2_produce_collection_position_stability','camera':'ego','max_contact_penetration_m':maxdepth,'max_success_streak':maxstreak,'action_limit_clipped_frames':clip_count,'elapsed_seconds':time.monotonic()-began,'mean_inference_seconds':float(np.mean(inference)),'precontact_drift_m':drift,'geometry_audit':'pending; contact gate only','evaluation_scope':'accepted training episode initial scene; diagnostic only','diagnostic_mode':args.mode,'training_reference':args.reference}
 save(out/'eval_status.json',dict(report,updated_at=time.time()));return report
