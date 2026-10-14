import itertools
import numpy as np,fcl
from pxr import Usd,UsdGeom
from model import fk
class DualCollision:
 def __init__(self,scene,right_q):
  s=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache(0);self.left=[];self.right=[]
  for p in s.Traverse():
   path=str(p.GetPath())
   if not p.IsA(UsdGeom.Mesh) or not path.startswith('/World/Robot_') or '/visuals/' not in path or UsdGeom.Imageable(p).ComputeVisibility(0)=='invisible':continue
   side='left' if '/Robot_left/' in path else 'right';link=path.split('/')[3]
   if link.endswith('base_link') or link in ['fl_link1','fr_link1']:continue
   body=s.GetPrimAtPath('/World/Robot_'+side+'/'+link);mesh=UsdGeom.Mesh(p);v=np.asarray(mesh.GetPointsAttr().Get(),float)
   T=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(p)).T;v=v@T[:3,:3].T+T[:3,3]
   ids=list(mesh.GetFaceVertexIndicesAttr().Get());off=0;tri=[]
   for n in mesh.GetFaceVertexCountsAttr().Get():
    tri.extend([[ids[off],ids[off+k],ids[off+k+1]] for k in range(1,n-1)]);off+=n
   model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,np.asarray(tri,np.int32));model.endModel();obj=fcl.CollisionObject(model)
   bounds=np.array(list(itertools.product(*zip(v.min(0),v.max(0)))))
   if side=='left':self.left.append((path,link,obj,bounds))
   else:
    X=fk(link,'right',right_q);vv=bounds@X[:3,:3].T+X[:3,3];obj.setTransform(fcl.Transform(X[:3,:3].copy(),X[:3,3].copy()));self.right.append((path,obj,vv.min(0),vv.max(0)))
 def check(self,q):
  hits=[]
  for path,link,obj,bounds in self.left:
   X=fk(link,'left',q);v=bounds@X[:3,:3].T+X[:3,3];lo,hi=v.min(0),v.max(0)
   if lo[2]<.682:hits.append((path,'table conservative bounds'));continue
   near=[r for r in self.right if not(np.any(lo>r[3]) or np.any(r[2]>hi))]
   if not near:continue
   obj.setTransform(fcl.Transform(X[:3,:3].copy(),X[:3,3].copy()))
   for path2,other,_,_ in near:
    result=fcl.CollisionResult();fcl.collide(obj,other,fcl.CollisionRequest(),result)
    if result.is_collision:hits.append((path,path2))
  return hits
