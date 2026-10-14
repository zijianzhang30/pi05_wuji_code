"""PRO5000 offline USD playback: 4K 4:3 PathTracing. No physics stepping."""
import argparse,json,os,time,traceback,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--root',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--pilot',action='store_true');p.add_argument('--preview-video',action='store_true');p.add_argument('--width',type=int,default=3840);p.add_argument('--height',type=int,default=2880);p.add_argument('--spp',type=int,default=1024);p.add_argument('--solver-run',required=True);p.add_argument('--realtime',action='store_true');p.add_argument('--asset-detail',action='store_true');a=p.parse_args()
ROOT=Path(a.root);out=ROOT/a.task;cfg=json.loads((out/'task.json').read_text());W,H=a.width,a.height
scene_out=out
if a.output_dir:out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
if a.preview_video:out=out/'quick_preview';out.mkdir(exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':W,'height':H,'renderer':'RayTracedLighting' if a.realtime else 'PathTracing','samples_per_pixel_per_frame':32,'max_bounces':12,'max_specular_transmission_bounces':16,'multi_gpu':False,'active_gpu':a.gpu,'physics_gpu':a.gpu,'limit_cpu_threads':8})
exitcode=0
try:
 import carb,omni.usd,omni.timeline,omni.replicator.core as rep
 from pxr import UsdGeom,UsdShade,Gf,Sdf,UsdPhysics
 import numpy as np
 from scipy.spatial.transform import Rotation
 solver_data=np.load(Path(a.solver_run)/'observation_trajectory.npz')
 from PIL import Image
 import imageio.v2 as imageio
 import h5py,io
 st=carb.settings.get_settings()
 settings={'/omni/replicator/pathTracedMotionBlurSubSamples':a.spp//32,'/rtx/rendermode':'PathTracing','/rtx/pathtracing/spp':32,'/rtx/pathtracing/totalSpp':a.spp,'/rtx/pathtracing/maxBounces':12,'/rtx/pathtracing/maxSpecularAndTransmissionBounces':16,'/rtx/pathtracing/optixDenoiser/enabled':True,'/rtx/pathtracing/optixDenoiser/blendFactor':0.0,'/rtx/pathtracing/optixDenoiser/temporalMode/enabled':False,'/rtx/pathtracing/adaptiveSampling/enabled':False,'/rtx/pathtracing/cached/enabled':False,'/rtx/resetPtAccumOnAnimTimeChange':True,'/rtx/post/motionblur/enabled':False}
 if a.realtime:settings['/rtx/rendermode']='RayTracedLighting'
 for k,v in settings.items():st.set(k,v)
 assert omni.usd.get_context().open_stage(str(scene_out/'animation.usdc'))
 s=omni.usd.get_context().get_stage();s.SetEditTarget(s.GetSessionLayer())
 for pr in list(s.Traverse()):
  if pr.IsA(UsdPhysics.Joint) or pr.IsA(UsdPhysics.Scene):s.RemovePrim(pr.GetPath());continue
  for schema in list(pr.GetAppliedSchemas()):
   if 'Physics' in schema or 'Physx' in schema:pr.RemoveAPI(schema)
 def set_solver_frame(frame):
  for path,pose in zip(solver_data['body_paths'],solver_data['poses'][frame]):
   matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();matrix[:3,3]=pose[:3];s.GetPrimAtPath(str(path)).GetAttribute('xformOp:transform').Set(Gf.Matrix4d(matrix.T.tolist()))
  for _ in range(5):app.update()
  print('POSE_FRAME',frame,s.GetPrimAtPath('/World/Robot_right/right_palm_link').GetAttribute('xformOp:transform').Get().ExtractTranslation(),flush=True)
 # Visual-only overrides. Preserve transforms, collision geometry and friction.
 for path,color,roughness in [('/World/WujiMaterials/Hands/Shader',(.018,.023,.028),.48),('/World/V2Materials/MatGray/Surface',(.8,.81,.82),.7),('/World/V2Materials/TableEdge/Surface',(.8,.81,.82),.55)]:
  sh=UsdShade.Shader.Get(s,path);assert sh,path
  sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));sh.GetInput('diffuseColor').DisconnectSource();sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
 camera='/World/TableEgo';cam=UsdGeom.Camera.Define(s,camera)
 cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(-2.0,-3.87,1.02) if a.asset_detail else Gf.Vec3d(-2.05,-4.15,1.72),Gf.Vec3d(-2.0,-3.58,.72) if a.asset_detail else Gf.Vec3d(-2.05,-3.51,.72),Gf.Vec3d(0,0,1)).GetInverse())
 cam.CreateFocalLengthAttr(34. if a.asset_detail else 16.5);cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(15.2908);cam.CreateClippingRangeAttr(Gf.Vec2f(.025,100))
 rp=rep.create.render_product(camera,(W,H));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([rp]);timeline=omni.timeline.get_timeline_interface();timeline.stop()
 first=int(cfg.get('presentation_start_frame',0)) if a.start is None else a.start
 end=int(s.GetEndTimeCode()) if a.end is None else a.end
 end=min(end,first+179) if a.preview_video else end
 frames=[first, min(first+360,end)] if a.pilot else list(range(first,end+1))
 chunk=out/'chunks';chunk.mkdir(exist_ok=True);tag=f'{first:05d}_{end:05d}'
 status=out/('pilot_status.json' if a.pilot else f'chunk_{tag}.json');video=(out/'ego.mp4') if a.preview_video else chunk/f'{tag}.mp4';tmp=chunk/f'{tag}.partial.mp4'
 writer=None; image_file=None
 def save_status(state,**extra):
  d={'state':state,'task':a.task,'gpu':a.gpu,'first_frame':first,'last_frame':end,'frame_count':len(frames),'resolution':[W,H],'fps':30,'renderer':'RayTracedLighting' if a.realtime else 'PathTracing','spp':a.spp,'max_bounces':12,'physics_stepped':False,'source_motion':'recorded solver poses','source_run':a.solver_run,'settings':{k:st.get(k) for k in settings},**extra}
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(status)
 save_status('initializing')
 timeline.set_current_time(first/30);set_solver_frame(first)
 for _ in range(20):app.update()
 if a.realtime:settings['/rtx/rendermode']='RayTracedLighting'
 for k,v in settings.items():st.set(k,v)
 print("CAPTURE_SETTINGS",json.dumps({k:st.get(k) for k in settings}),flush=True)
 # Replicator handles path-traced accumulation from totalSpp/spp.
 for attempt in range(5):
  rep.orchestrator.step(delta_time=0.,pause_timeline=True)
  pixels=np.asarray(rgb.get_data())
  if pixels.shape[:2]==(H,W):break
 assert pixels.shape[:2]==(H,W),pixels.shape
 s.GetSessionLayer().Export(str(out/f'render_session_gpu{a.gpu}.usda'))
 if not a.pilot:
  writer=imageio.get_writer(str(tmp),fps=30,codec='libx264',macro_block_size=16,ffmpeg_params=['-crf','12','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2'])
 image_file=h5py.File(out/'rgb.partial.hdf5','w')
 images=image_file.create_dataset('ego',shape=(len(frames),),dtype=h5py.vlen_dtype(np.dtype('uint8')))
 image_file.create_dataset('source_frame',data=np.array(frames))
 image_file.attrs.update(encoding='JPEG',quality=95,color_order='RGB',width=W,height=H,fps=30,camera_path=camera)
 image_file.create_dataset('camera_to_world',data=np.array(cam.GetLocalTransformation()).T)
 image_file.create_dataset('intrinsics',data=np.array([[W*float(cam.GetFocalLengthAttr().Get())/float(cam.GetHorizontalApertureAttr().Get()),0,W/2],[0,H*float(cam.GetFocalLengthAttr().Get())/float(cam.GetVerticalApertureAttr().Get()),H/2],[0,0,1]]))
 started=time.monotonic()
 for n,k in enumerate(frames):
  ft=time.monotonic();timeline.set_current_time(k/30);set_solver_frame(k)
  for _ in range(3):rep.orchestrator.step(delta_time=0.,pause_timeline=True)
  img=np.asarray(rgb.get_data())[:,:,:3].copy();assert img.shape==(H,W,3)
  buf=io.BytesIO();Image.fromarray(img).save(buf,format='JPEG',quality=95);images[n]=np.frombuffer(buf.getvalue(),dtype=np.uint8)
  if writer:writer.append_data(img)
  if a.pilot or k==first or k==end or k==int(s.GetEndTimeCode())//2:
   Image.fromarray(img).save(out/f'pt_{k:05d}.png')
  if n==0 and first==int(cfg.get('presentation_start_frame',0)):
   Image.fromarray(img).resize((1280,960),Image.Resampling.LANCZOS).save(out/'poster.jpg',quality=95)
  if True:
   save_status('rendering',completed_frames=n+1,current_frame=k,seconds=time.monotonic()-started,seconds_last_frame=time.monotonic()-ft)
   print('FRAME',a.task,k,end,'seconds',round(time.monotonic()-started,2),flush=True)
 if writer:writer.close();writer=None;tmp.replace(video)
 image_file.close();image_file=None;(out/'rgb.partial.hdf5').replace(out/'rgb.hdf5')
 save_status('complete',completed_frames=len(frames),seconds=time.monotonic()-started,video=str(video) if not a.pilot else None)
 print('DONE',a.task,tag,flush=True)
except Exception:
 exitcode=1;err=traceback.format_exc();print(err,flush=True)
 if 'save_status' in locals():save_status('failed',error=err)
finally:
 if 'writer' in locals() and writer:writer.close()
 if 'image_file' in locals() and image_file:image_file.close()
 app.close()
sys.exit(exitcode)
