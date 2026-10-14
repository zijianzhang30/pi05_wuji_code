"""Recorded-frame geometry audit: robot/task, hand self-intersection and inter-arm intersections."""
import argparse,json,itertools,re
from pathlib import Path
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--stride',type=int,default=1);ap.add_argument('--task-pairs',action='store_true');ap.add_argument('--start-frame',type=int,default=0);ap.add_argument('--end-frame',type=int);ap.add_argument('--output');a=ap.parse_args();out=Path(a.run);data=np.load(out/'observation_trajectory.npz');paths=list(map(str,data['body_paths']));index={p:i for i,p in enumerate(paths)}
s=Usd.Stage.Open(str(out/'initial_scene.usda'));cache=UsdGeom.XformCache(0);parts=[]
cube_faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)
for pr in s.Traverse():
 path=str(pr.GetPath());robot=path.startswith('/World/Robot_') and '/visuals/' in path
 task=path.startswith(('/World/TaskObjects/','/World/TaskFood_','/World/TaskBin_','/World/TaskMicrowave/','/World/MusicRig/'))
 table=path in ['/World/V2Mat','/World/V2Tabletop']
 if not(robot or task or table) or not(pr.IsA(UsdGeom.Mesh) or pr.IsA(UsdGeom.Cube)) or UsdGeom.Imageable(pr).ComputeVisibility(0)=='invisible':continue
 if not robot and any(x in path.lower() for x in ['collision','collider','/material','/looks']):continue
 body=pr
 while body and str(body.GetPath()) not in index:body=body.GetParent()
 bp=str(body.GetPath()) if body else None;bi=index.get(bp,-1)
 W=np.asarray(cache.GetLocalToWorldTransform(pr)).T;B=np.asarray(cache.GetLocalToWorldTransform(body)).T if body else np.eye(4);T=np.linalg.inv(B)@W
 if pr.IsA(UsdGeom.Mesh):
  mesh=UsdGeom.Mesh(pr);v=np.asarray(mesh.GetPointsAttr().Get(),float);ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());tri=[];off=0
  for n in counts:tri.extend([[ids[off],ids[off+k],ids[off+k+1]] for k in range(1,n-1)]);off+=n
  tri=np.asarray(tri,np.int32)
 else:
  sz=float(UsdGeom.Cube(pr).GetSizeAttr().Get())/2;v=np.array(list(itertools.product([-sz,sz],repeat=3)));tri=cube_faces
 v=v@T[:3,:3].T+T[:3,3];bounds=np.array(list(itertools.product(*zip(v.min(0),v.max(0)))));model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,tri);model.endModel()
 link=path.split('/')[3] if robot else '';match=re.search(r'finger([1-5])_(link([1-4])|tip_link)',link);finger=(int(match[1]),int(match[3]) if match[3] else 5) if match else None
 parts.append(dict(path=path,body=bp,index=bi,bounds=bounds,ob=fcl.CollisionObject(model),robot=robot,side='left' if '/Robot_left/' in path else 'right',finger=finger,link=link))
pairs=[]
for i,x in enumerate(parts):
 for j in range(i+1,len(parts)):
  y=parts[j]
  if not x['robot'] and not y['robot']:
   xp,yp=x['path'],y['path'];moving=lambda z:z.startswith(('/World/TaskFood_','/World/TaskBin_','/World/TaskObjects/'))
   table=lambda z:z in ['/World/V2Mat','/World/V2Tabletop']
   if a.task_pairs and x['body']!=y['body'] and ((moving(xp) and moving(yp)) or (moving(xp) and table(yp)) or (moving(yp) and table(xp))):pairs.append((i,j,'object_object_or_support'))
   continue
  if x['robot']!=y['robot']:pairs.append((i,j,'task'));continue
  if x['side']!=y['side']:pairs.append((i,j,'inter_arm'));continue
  if x['body']==y['body']:continue
  f,g=x['finger'],y['finger']
  if f and g and (f[0]!=g[0] or abs(f[1]-g[1])>1):pairs.append((i,j,'finger_self'))
  elif f and 'palm_link' in y['link'] and f[1]>=3:pairs.append((i,j,'finger_palm'))
  elif g and 'palm_link' in x['link'] and g[1]>=3:pairs.append((i,j,'finger_palm'))
pair_i=np.array([x[0] for x in pairs],dtype=int);pair_j=np.array([x[1] for x in pairs],dtype=int)
results=[];counts={};exact_pair_cache={}
frames=range(a.start_frame,min(len(data['poses']),a.end_frame or len(data['poses'])),a.stride)
for frame in frames:
 boxes=[];trans=[];updated=set()
 body_stamps=[x.tobytes() for x in data['poses'][frame]]
 for x in parts:
  if x['index']>=0:
   pos=data['poses'][frame,x['index']];R=Rotation.from_quat(pos[3:]).as_matrix();t=pos[:3].astype(float)
  else:R=np.eye(3);t=np.zeros(3)
  v=x['bounds']@R.T+t;boxes.append((v.min(0),v.max(0)));trans.append((R,t))
 hits=[]
 boxes=np.array(boxes)
 overlap=np.all(boxes[pair_i,0]<=boxes[pair_j,1],axis=1)&np.all(boxes[pair_j,0]<=boxes[pair_i,1],axis=1)
 for pair_id in np.flatnonzero(overlap):
  i,j,kind=pairs[pair_id]
  stamp=tuple(body_stamps[parts[k]['index']] if parts[k]['index']>=0 else b'static' for k in [i,j])
  previous=exact_pair_cache.get(pair_id)
  if previous is not None and previous[0]==stamp:collision=previous[1]
  else:
   for k in [i,j]:
    if k not in updated:parts[k]['ob'].setTransform(fcl.Transform(*trans[k]));updated.add(k)
   r=fcl.CollisionResult();fcl.collide(parts[i]['ob'],parts[j]['ob'],fcl.CollisionRequest(),r);collision=r.is_collision
   exact_pair_cache[pair_id]=(stamp,collision)
  if collision:hits.append([kind,parts[i]['path'],parts[j]['path']]);counts[kind]=counts.get(kind,0)+1
 if hits:results.append(dict(frame=frame,hits=hits))
 if frame%100==0:print('GEOMETRY',frame,len(data['poses']),len(hits),flush=True)
report=dict(checked_frames=len(frames),frames_with_intersections=len(results),intersection_counts=counts,failed_frames=results,scope='Recorded frames only; checks task objects, table, inter-arm, nonadjacent finger segments, cross-finger and distal finger/palm. Adjacent mechanical joint interfaces excluded.')
(out/(a.output or ('task_geometry_audit.json' if a.task_pairs else 'full_geometry_audit.json'))).write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='failed_frames'})
