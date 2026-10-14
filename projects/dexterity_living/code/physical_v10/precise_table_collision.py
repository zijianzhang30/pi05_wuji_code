from collision import *
from pxr import UsdPhysics
class TableCollision(CollisionAudit):
 def __init__(self,scene):
  super().__init__(scene);s=Usd.Stage.Open(str(scene));cache=UsdGeom.BBoxCache(0,['default','render','proxy']);self.boxes=[]
  for path in ['/World/V2Tabletop','/World/V2Mat','/World/V2Mount_right','/World/TaskObjects/object/RealAsset/Model/visual']:
   pr=s.GetPrimAtPath(path)
   if not pr:continue
   b=cache.ComputeWorldBound(pr).ComputeAlignedBox();lo=np.array(b.GetMin());hi=np.array(b.GetMax());lo-=.002;hi+=.002
   if path in ['/World/V2Tabletop','/World/V2Mat']:lo[2]=-.5;self.boxes.append((path,fcl.CollisionObject(fcl.Box(*(hi-lo)),fcl.Transform((hi+lo)/2)),lo,hi))
 def check(self,q):
  hits=[]
  for path,link,model,v,local in self.parts:
   if not link or link in ['fr_base_link','fr_link1']:continue
   T=fk(link,'right',q)@local;vv=v@T[:3,:3].T+T[:3,3];lo=vv.min(0);hi=vv.max(0)
   for b in self.boxes:
    if np.any(lo>b[3]) or np.any(b[2]>hi):continue
    ob=fcl.CollisionObject(model,fcl.Transform(T[:3,:3].copy(),T[:3,3].copy()));r=fcl.CollisionResult();fcl.collide(ob,b[1],fcl.CollisionRequest(),r)
    if r.is_collision:hits.append((path,b[0]))
  return hits
