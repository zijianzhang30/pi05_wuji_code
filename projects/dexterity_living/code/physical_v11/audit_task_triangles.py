"""Audit solver-recorded rigid poses; never reconstruct object motion from authored tracks."""
import argparse,json,itertools
from pathlib import Path
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--stride',type=int,default=1);a=p.parse_args();out=Path(a.run);data=np.load(out/'trajectory.npz');paths=list(map(str,data['body_paths']));indices={v:i for i,v in enumerate(paths)};s=Usd.Stage.Open(str(out/'initial_scene.usda'));cache=UsdGeom.XformCache(0);parts=[]
for pr in s.Traverse():
 path=str(pr.GetPath());robot=path.startswith('/World/Robot_') and '/visuals/' in path;micro=path.startswith('/World/TaskObjects/') and path.endswith('/visual')
 if not (robot or micro) or not pr.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(pr).ComputeVisibility(0)=='invisible':continue
 parent=pr
 while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
 if not parent:continue
 body=str(parent.GetPath());mesh=UsdGeom.Mesh(pr);v=np.asarray(mesh.GetPointsAttr().Get(),dtype=float);T=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(parent)).T)@np.array(cache.GetLocalToWorldTransform(pr)).T;v=v@T[:3,:3].T+T[:3,3]
 ids=list(mesh.GetFaceVertexIndicesAttr().Get());tri=[];off=0
 for n in mesh.GetFaceVertexCountsAttr().Get():
  for k in range(1,n-1):tri.append([ids[off],ids[off+k],ids[off+k+1]])
  off+=n
 model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,np.asarray(tri,dtype=np.int32));model.endModel();parts.append((path,indices[body],model,np.array(list(itertools.product(*zip(v.min(0),v.max(0))))),robot))
assert sum(x[4] for x in parts)>=60,'robot geometry missing'
max_initial_mapping_error=0.
for path,i in indices.items():
 T=np.asarray(cache.GetLocalToWorldTransform(s.GetPrimAtPath(path))).T
 max_initial_mapping_error=max(max_initial_mapping_error,float(np.linalg.norm(T[:3,3]-data['poses'][0,i,:3])))
assert max_initial_mapping_error<.03,('body view path order may be wrong',max_initial_mapping_error)
compiled={path:fcl.CollisionObject(model) for path,i,model,v,robot in parts}
frames=[]
for frame in range(0,len(data['poses']),a.stride):
 objs=[[],[]]
 for path,i,model,v,robot in parts:
  pos=data['poses'][frame,i];R=Rotation.from_quat(pos[3:]).as_matrix();vv=v@R.T+pos[:3];ob=compiled[path];objs[int(robot)].append((path,ob,vv.min(0),vv.max(0),R.copy(),pos[:3].astype(float)))
 hits=[];transformed=set()
 for x,y in itertools.product(objs[1],objs[0]):
  if np.any(x[2]>y[3]) or np.any(y[2]>x[3]):continue
  for item in [x,y]:
   if item[0] not in transformed:item[1].setTransform(fcl.Transform(item[4],item[5]));transformed.add(item[0])
  r=fcl.CollisionResult();fcl.collide(x[1],y[1],fcl.CollisionRequest(),r)
  if r.is_collision:hits.append([x[0],y[0]])
 bad=hits
 frames.append({'frame':frame,'t':float(data['time'][frame]),'nonhandle_pairs':bad,'handle_intersection_pairs':len(hits)-len(bad)})
 if frame%100==0:print('AUDIT',frame,len(data['poses']),'nonhandle',len(bad),flush=True)
report={'source':'solver-recorded rigid poses','sample_hz':30/a.stride,'checked_frames':len(frames),'frames_with_nonhandle_intersections':sum(bool(f['nonhandle_pairs']) for f in frames),'max_initial_mapping_error_m':max_initial_mapping_error,'scope':'triangle surface intersection; handle contact depth and subframe penetration require contact separation checks','frames':frames}
(out/'triangle_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames'}))
