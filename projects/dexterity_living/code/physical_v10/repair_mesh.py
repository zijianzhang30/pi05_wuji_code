"""Remove confirmed detached wrist triangle artifacts from run copies, not source assets."""
import numpy as np
from pxr import UsdGeom,Usd,Vt

def repair_wrist(stage,body):
 cache=UsdGeom.XformCache();inv=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T);logs=[]
 for pr in Usd.PrimRange(body):
  if not pr.IsA(UsdGeom.Mesh):continue
  m=UsdGeom.Mesh(pr);points=np.asarray(m.GetPointsAttr().Get());counts=np.asarray(m.GetFaceVertexCountsAttr().Get());indices=np.asarray(m.GetFaceVertexIndicesAttr().Get());assert np.all(counts==3)
  ids=indices.reshape(-1,3);T=inv@np.asarray(cache.GetLocalToWorldTransform(pr)).T;world=points@T[:3,:3].T+T[:3,3];tri=world[ids];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;remote=np.any(tri[:,:,0]>.12,axis=1)
  if not remote.any():continue
  assert np.all(area[remote]<1e-6),'Unexpected large wrist geometry; manual inspection required'
  keep=~remote;kept_face_vertices=np.flatnonzero(np.repeat(keep,3));used,newids=np.unique(ids[keep].ravel(),return_inverse=True)
  normals=m.GetNormalsAttr().Get()
  if normals is not None:
   n=np.asarray(normals);interp=m.GetNormalsInterpolation()
   if interp=='faceVarying':m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n[kept_face_vertices].astype(np.float32)))
   elif interp in ['vertex','varying']:m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n[used].astype(np.float32)))
  for pv in UsdGeom.PrimvarsAPI(pr).GetPrimvars():
   interp=pv.GetInterpolation()
   if interp not in ['faceVarying','vertex','varying','uniform']:continue
   value=pv.ComputeFlattened()
   if value is None:continue
   sel=kept_face_vertices if interp=='faceVarying' else used if interp in ['vertex','varying'] else np.flatnonzero(keep)
   if len(value)!= (len(indices) if interp=='faceVarying' else len(points) if interp in ['vertex','varying'] else len(counts)):continue
   pv.Set(type(value)([value[int(i)] for i in sel]));pv.GetIndicesAttr().Clear()
  for sub in UsdGeom.Subset.GetAllGeomSubsets(m):
   remap=np.full(len(counts),-1);remap[np.flatnonzero(keep)]=np.arange(keep.sum());old=np.asarray(sub.GetIndicesAttr().Get());sub.GetIndicesAttr().Set(remap[old][remap[old]>=0].tolist())
  m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points[used].astype(np.float32)));m.GetFaceVertexCountsAttr().Set([3]*int(keep.sum()));m.GetFaceVertexIndicesAttr().Set(newids.tolist());m.GetExtentAttr().Set([points[used].min(0).tolist(),points[used].max(0).tolist()])
  logs.append({'mesh':str(pr.GetPath()),'removed_faces':int(remote.sum()),'removed_area_m2':float(area[remote].sum()),'old_body_bounds':[world.min(0).tolist(),world.max(0).tolist()],'new_body_bounds':[world[used].min(0).tolist(),world[used].max(0).tolist()]})
 return logs
