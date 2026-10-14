
"""Check actual visible surfaces at the solver substep when proxy contact reports a large separation."""
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from pxr import UsdGeom
class SurfaceWatch:
 def __init__(self,stage,body_paths):
  self.index={str(p):i for i,p in enumerate(body_paths)};self.parts=[];cache=UsdGeom.XformCache(0)
  for pr in stage.Traverse():
   path=str(pr.GetPath());robot=path.startswith('/World/Robot_') and '/visuals/' in path;food=path.startswith('/World/TaskFood_')
   if not(robot or food) or not pr.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(pr).ComputeVisibility(0)=='invisible':continue
   if any(x in path.lower() for x in ['collision','collider']):continue
   body=pr
   while body and str(body.GetPath()) not in self.index:body=body.GetParent()
   if not body:continue
   m=UsdGeom.Mesh(pr);v=np.asarray(m.GetPointsAttr().Get(),float);ids=np.asarray(m.GetFaceVertexIndicesAttr().Get());counts=m.GetFaceVertexCountsAttr().Get();tri=[];off=0
   for n in counts:tri.extend([[ids[off],ids[off+k],ids[off+k+1]] for k in range(1,n-1)]);off+=n
   T=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(pr)).T;v=v@T[:3,:3].T+T[:3,3]
   model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,np.asarray(tri,np.int32));model.endModel()
   self.parts.append(dict(path=path,body=str(body.GetPath()),index=self.index[str(body.GetPath())],ob=fcl.CollisionObject(model)))
 def check(self,link,food,poses):
  # The invisible distal sensor is fixed to link4: check every visible segment of that finger.
  finger=link.split('_finger')[-1].split('_')[0] if '_finger' in link else None
  prefix=link.split('_finger')[0]+'_finger'+finger+'_' if finger else link
  left=[x for x in self.parts if x['body'].startswith(prefix)]
  right=[x for x in self.parts if x['body']==food]
  assert left and right,('missing_visible_mesh_for_contact',link,food)
  for x in left+right:
   pos=poses[x['index']];x['ob'].setTransform(fcl.Transform(Rotation.from_quat(pos[3:]).as_matrix(),pos[:3].astype(float)))
  distance=float('inf');hits=[]
  for x in left:
   for y in right:
    result=fcl.CollisionResult();fcl.collide(x['ob'],y['ob'],fcl.CollisionRequest(),result)
    if result.is_collision:hits.append([x['path'],y['path']])
    else:distance=min(distance,float(fcl.distance(x['ob'],y['ob'])))
  return dict(intersections=hits,min_surface_distance_m=distance if np.isfinite(distance) else None)
