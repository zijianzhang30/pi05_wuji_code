import numpy as np,json
from scipy.spatial import ConvexHull,HalfspaceIntersection
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf,PhysicsSchemaTools
import omni.usd,omni.physx,omni.kit.app
def enclose(stage,root):
 body=stage.GetPrimAtPath(root);cache=UsdGeom.XformCache(0);B=np.asarray(cache.GetLocalToWorldTransform(body)).T;vertices=[]
 for pr in Usd.PrimRange(body):
  if pr.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr(False)
  if pr.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(pr).ComputeVisibility(0)!='invisible':
   T=np.linalg.inv(B)@np.asarray(cache.GetLocalToWorldTransform(pr)).T;v=np.asarray(UsdGeom.Mesh(pr).GetPointsAttr().Get(),float);vertices.append(v@T[:3,:3].T+T[:3,3])
 original=np.concatenate(vertices);verts=original.copy();path=root+'/PhysicsHull';m=UsdGeom.Mesh.Define(stage,path);m.CreateSubdivisionSchemeAttr('none');UsdGeom.Imageable(m).MakeInvisible();UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull');PhysxSchema.PhysxConvexHullCollisionAPI.Apply(m.GetPrim()).CreateHullVertexLimitAttr(255);pc=PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim());pc.CreateContactOffsetAttr(.0005);pc.CreateRestOffsetAttr(0)
 margin=.0001
 for attempt in range(10):
  hull=ConvexHull(verts);faces=hull.simplices.copy()
  for i,f in enumerate(faces):
   tri=verts[f]
   if np.dot(np.cross(tri[1]-tri[0],tri[2]-tri[0]),hull.equations[i,:3])<0:faces[i]=f[::-1]
  m.CreatePointsAttr([Gf.Vec3f(*v) for v in verts]);m.CreateFaceVertexCountsAttr([3]*len(faces));m.CreateFaceVertexIndicesAttr(faces.ravel().tolist());m.CreateExtentAttr([Gf.Vec3f(*verts.min(0)),Gf.Vec3f(*verts.max(0))]);result=[]
  omni.kit.app.get_app().update()
  def cb(status,convexes):
   assert len(convexes)==1
   result.append(np.array([[v.x,v.y,v.z] for v in convexes[0].vertices]))
  omni.physx.get_physx_cooking_interface().request_convex_collision_representation(omni.usd.get_context().get_stage_id(),PhysicsSchemaTools.sdfPathToInt(path),False,cb)
  assert result
  cooked=ConvexHull(result[0]);error=float((original@cooked.equations[:,:3].T+cooked.equations[:,3]).max())
  print('FOOD_ENCLOSURE',root,attempt,error,'extent',original.max(0)-original.min(0),flush=True)
  if error<=1e-5:return dict(root=root,attempts=attempt+1,max_graphics_outside_collision_m=error,body_vertices=original.tolist())
  margin+=0 if attempt==0 else max(error,0)+.0001;assert margin<.003,('excessive food enclosure margin',root,margin);normals=np.unique(cooked.equations[:,:3].round(9),axis=0);dist=(original@normals.T).max(0)+margin;verts=HalfspaceIntersection(np.column_stack([normals,-dist]),original.mean(0)).intersections
 raise RuntimeError('Food collision enclosure did not converge')
