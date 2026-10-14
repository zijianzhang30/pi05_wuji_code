"""Watertight convex strips matching the native handle profile; native latch untouched."""
import numpy as np,trimesh
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Vt

def repair_handle(stage):
 reports=[]
 for pr in list(stage.Traverse()):
  if '/E_handle_8/' not in str(pr.GetPath()) or not pr.IsA(UsdGeom.Mesh):continue
  mesh=UsdGeom.Mesh(pr);tm=trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=True)
  lo,hi=tm.bounds;zs=np.linspace(lo[2]+1e-6,hi[2]-1e-6,65);sections=[]
  for z in zs:
   cut=tm.section(plane_origin=[0,0,z],plane_normal=[0,0,1]);assert cut is not None
   a,b=cut.vertices.min(0),cut.vertices.max(0);sections.append(np.array([[a[0],a[1],z],[b[0],a[1],z],[b[0],b[1],z],[a[0],b[1],z]]))
  # Preserve the visible mesh; replace only its unreliable non-watertight SDF collider.
  UsdPhysics.CollisionAPI.Apply(pr).CreateCollisionEnabledAttr(False)
  triangles=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
  for i in range(64):
   v=np.vstack([sections[i],sections[i+1]]).astype(np.float32);m=UsdGeom.Mesh.Define(stage,str(pr.GetPath())+f'/ContactSegment_{i:02d}');m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v));m.CreateFaceVertexCountsAttr([3]*12);m.CreateFaceVertexIndicesAttr(triangles.ravel().tolist());m.CreateSubdivisionSchemeAttr('none');m.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull');c=PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim());c.CreateContactOffsetAttr(.0002);c.CreateRestOffsetAttr(0)
  reports.append({'source_mesh':str(pr.GetPath()),'source_watertight':bool(tm.is_watertight),'source_signed_volume':float(tm.volume),'segments':64,'profile_step_m':float(zs[1]-zs[0]),'method':'convex strips from original mesh cross sections; close missing back skin in collision only','visual_mesh_changed':False,'latch_geometry_or_spring_changed':False})
 assert len(reports)==1,reports
 return reports
