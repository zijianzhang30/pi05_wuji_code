"""Convert eight existing white-key meshes into passive spring keys. Original shell and texture remain."""
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Sdf,Gf,Vt
def add_keyboard(stage,source):
 flat=source.Flatten()
 for name in ['MusicRig','MusicMaterials']:
  Sdf.CopySpec(flat,'/World/'+name,stage.GetRootLayer(),'/World/'+name)
 cache=UsdGeom.XformCache(0);root=stage.GetPrimAtPath('/World/MusicRig')
 for pr in list(Usd.PrimRange(root)):
  if pr.HasAPI(UsdPhysics.RigidBodyAPI):pr.RemoveAPI(UsdPhysics.RigidBodyAPI)
  if pr.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr(False)
  for at in pr.GetAttributes():
   if at.GetNumTimeSamples():
    value=at.Get(0)
    for t in at.GetTimeSamples():at.ClearAtTime(t)
    at.Set(value)
 for i in range(8):stage.RemovePrim('/World/MusicRig/Number_'+str(i))
 bounds=[];key_data=[]
 for i in range(8):
  path=f'/World/MusicRig/RealKeyboard/Keys/Key_{i}';pr=stage.GetPrimAtPath(path+'/Mesh');m=UsdGeom.Mesh(pr);v=np.asarray(m.GetPointsAttr().Get(),float);T=np.asarray(cache.GetLocalToWorldTransform(pr)).T;v=v@T[:3,:3].T+T[:3,3];lo,hi=v.min(0),v.max(0);center=(lo+hi)/2;bounds.append((lo,hi));key_data.append((path,v,center,lo,hi))
 # Remove original duplicate white-key faces where the separately movable key meshes sit.
 visual=stage.GetPrimAtPath('/World/MusicRig/RealKeyboard/Model/visual');m=UsdGeom.Mesh(visual);v=np.asarray(m.GetPointsAttr().Get(),float);T=np.asarray(cache.GetLocalToWorldTransform(visual)).T;world=v@T[:3,:3].T+T[:3,3];counts=np.asarray(m.GetFaceVertexCountsAttr().Get());ids=np.asarray(m.GetFaceVertexIndicesAttr().Get());assert np.all(counts==3)
 cent=world[ids.reshape(-1,3)].mean(1);drop=np.zeros(len(counts),bool)
 for lo,hi in bounds:drop|=np.all(cent[:,:2]>=lo[:2]-.0003,axis=1)&np.all(cent[:,:2]<=hi[:2]+.0003,axis=1)&(cent[:,2]>.726)&(cent[:,2]<.746)
 keep=~drop;newids=ids.reshape(-1,3)[keep].ravel();remap=np.full(len(counts),-1);remap[keep]=np.arange(keep.sum())
 for pv in UsdGeom.PrimvarsAPI(visual).GetPrimvars():
  if pv.GetInterpolation()==UsdGeom.Tokens.faceVarying:
   if pv.IsIndexed():pv.SetIndices(np.asarray(pv.GetIndices())[np.repeat(keep,3)].tolist())
   else:
    values=pv.Get()
    if values is not None and len(values)==len(ids):pv.Set(type(values)([x for x,k in zip(values,np.repeat(keep,3)) if k]))
 for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(visual)):
  old=np.asarray(subset.GetIndicesAttr().Get());subset.GetIndicesAttr().Set(remap[old][remap[old]>=0].tolist())
 m.GetFaceVertexIndicesAttr().Set(newids.tolist());m.GetFaceVertexCountsAttr().Set([3]*int(keep.sum()));m.GetNormalsAttr().Clear()
 collision='/World/MusicRig/RealKeyboard/Model/collision';stage.RemovePrim(collision);Sdf.CopySpec(stage.GetRootLayer(),str(visual.GetPath()),stage.GetRootLayer(),collision);pr=stage.GetPrimAtPath(collision);UsdGeom.Imageable(pr).MakeInvisible();UsdPhysics.CollisionAPI.Apply(pr).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(pr).CreateApproximationAttr('none')
 key_paths=[]
 for path,v,center,lo,hi in key_data:
  body=stage.GetPrimAtPath(path);xf=UsdGeom.Xformable(body);xf.ClearXformOpOrder();xf.SetResetXformStack(True);xf.AddTranslateOp(opSuffix='physics').Set(Gf.Vec3d(*center))
  m=UsdGeom.Mesh(stage.GetPrimAtPath(path+'/Mesh'));xx=UsdGeom.Xformable(m);xx.ClearXformOpOrder();m.GetPointsAttr().Set([Gf.Vec3f(*x) for x in v-center]);m.GetNormalsAttr().Clear();m.CreateExtentAttr([Gf.Vec3f(*(lo-center)),Gf.Vec3f(*(hi-center))])
  UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(False);UsdPhysics.MassAPI.Apply(body).CreateMassAttr(.02);PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateEnableCCDAttr(True);PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0)
  col=UsdGeom.Cube.Define(stage,path+'/Collider');col.CreateSizeAttr(1);col.AddScaleOp().Set(Gf.Vec3f(*(hi-lo+np.array([.0001,.0001,.0001]))));UsdGeom.Imageable(col).MakeInvisible();UsdPhysics.CollisionAPI.Apply(col.GetPrim());pc=PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim());pc.CreateContactOffsetAttr(.0002);pc.CreateRestOffsetAttr(0)
  j=UsdPhysics.PrismaticJoint.Define(stage,path+'/Spring');j.CreateBody1Rel().SetTargets([path]);j.CreateAxisAttr('Z');j.CreateLocalPos0Attr(Gf.Vec3f(*center));j.CreateLowerLimitAttr(-.003);j.CreateUpperLimitAttr(0);d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');d.CreateTypeAttr('force');d.CreateStiffnessAttr(220);d.CreateDampingAttr(.4);d.CreateMaxForceAttr(2);d.CreateTargetPositionAttr(.001)
  key_paths.append(path)
 stage.GetRootLayer().Save()
 return dict(paths=key_paths,centers=[x[2].tolist() for x in key_data],removed_duplicate_faces=int(drop.sum()),spring_N_per_m=220,damping_Ns_per_m=.4,travel_m=.003)
