import numpy as np,fcl
from pxr import Usd,UsdGeom
from scipy.linalg import polar
from model import fk
class CollisionAudit:
 def __init__(self,scene):
  s=Usd.Stage.Open(str(scene));self.parts=[]
  for p in s.Traverse():
   path=str(p.GetPath());robot=path.startswith('/World/Robot_right/') and '/visuals/' in path;micro=path.startswith('/World/TaskMicrowave/')
   if not p.IsA(UsdGeom.Mesh) or not (robot or micro) or UsdGeom.Imageable(p).ComputeVisibility(0)=='invisible':continue
   mesh=UsdGeom.Mesh(p);v=np.array(mesh.GetPointsAttr().Get(),dtype=float);inds=list(mesh.GetFaceVertexIndicesAttr().Get());counts=list(mesh.GetFaceVertexCountsAttr().Get());tri=[];off=0
   for n in counts:
    for k in range(1,n-1):tri.append([inds[off],inds[off+k],inds[off+k+1]])
    off+=n
   world=np.array(UsdGeom.XformCache(0).GetLocalToWorldTransform(p)).T
   if robot:
    link=path.split('/')[3];body=s.GetPrimAtPath('/World/Robot_right/'+link);T=np.linalg.inv(np.array(UsdGeom.XformCache(0).GetLocalToWorldTransform(body)).T)@world
   else:link=None;T=world
   rot,stretch=polar(T[:3,:3]);v=v@stretch.T;T[:3,:3]=rot
   model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,np.array(tri,dtype=np.int32));model.endModel();self.parts.append((path,link,model,v,T))
  assert sum(bool(p[1]) for p in self.parts)>=30,'Missing robot geometry'
 def check(self,q):
  rob=[];env=[]
  for path,link,model,v,local in self.parts:
   T=fk(link,'right',q)@local if link else local;verts=v@T[:3,:3].T+T[:3,3];o=fcl.CollisionObject(model,fcl.Transform(T[:3,:3].copy(),T[:3,3].copy()));(rob if link else env).append((path,o,verts.min(0),verts.max(0)))
  hits=[]
  for a in rob:
   for b in env:
    if np.any(a[2]>b[3]) or np.any(b[2]>a[3]):continue
    result=fcl.CollisionResult();fcl.collide(a[1],b[1],fcl.CollisionRequest(),result)
    if result.is_collision:hits.append((a[0],b[0]))
  return hits
