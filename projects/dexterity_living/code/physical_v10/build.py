from model import *
from repair_mesh import repair_wrist
from handle_collider import repair_handle
import json,os
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Gf,Sdf
ROBOT_SOURCE=P/'sources/20261011/robots/wuji/assembled_standalone.usdc'
CURRENT=P/'runs/wuji_blackwhite_smooth_pt4k_20261011/living_mcl_wuji_v1_20261011'
def set_pose(pr,T):
 x=UsdGeom.Xformable(pr);x.ClearXformOpOrder()
 for a in list(pr.GetAttributes()):
  if a.GetName().startswith('xformOp:'):pr.RemoveProperty(a.GetName())
 x.AddTranslateOp().Set(Gf.Vec3d(*T[:3,3]));r=Rot.from_matrix(T[:3,:3]).as_quat();x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(r[3]),Gf.Vec3f(*r[:3])))
def physics_material(s,path,mu_s,mu_d):
 m=UsdShade.Material.Define(s,path);a=UsdPhysics.MaterialAPI.Apply(m.GetPrim());a.CreateStaticFrictionAttr(mu_s);a.CreateDynamicFrictionAttr(mu_d);a.CreateRestitutionAttr(0)
 PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim()).CreateFrictionCombineModeAttr('average');return m

def build(out,qinit,task='microwave_open'):
 out.mkdir(parents=True,exist_ok=True);s=Usd.Stage.CreateNew(str(out/'initial_scene.usda'));UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.SetStageMetersPerUnit(s,1);s.SetTimeCodesPerSecond(30);s.SetStartTimeCode(0);s.SetEndTimeCode(30000)
 UsdGeom.Xform.Define(s,'/World');s.SetDefaultPrim(s.GetPrimAtPath('/World'))
 handmat=physics_material(s,'/World/PhysicsMaterials/Hand',.8,.6);tablemat=physics_material(s,'/World/PhysicsMaterials/Table',.7,.55)
 current=Usd.Stage.Open(str(CURRENT/task/'animation.usdc'))
 for name in ['V2Tabletop','V2Mat','V2Mount_left','V2Mount_right']:
  if current.GetPrimAtPath('/World/'+name):Sdf.CopySpec(current.Flatten(),'/World/'+name,s.GetRootLayer(),'/World/'+name)
 for pr in list(s.Traverse()):
  if pr.HasAPI(UsdPhysics.RigidBodyAPI):pr.RemoveAPI(UsdPhysics.RigidBodyAPI)
  if pr.HasAPI(UsdPhysics.CollisionAPI):UsdShade.MaterialBindingAPI.Apply(pr).Bind(tablemat,UsdShade.Tokens.strongerThanDescendants,'physics')
 # Small collision-only floor. All approved desk/mount coordinates are copied above.
 floor=UsdGeom.Cube.Define(s,'/World/Floor');floor.CreateSizeAttr(1);floor.AddTranslateOp().Set(Gf.Vec3d(-2,-3.5,-.06));floor.AddScaleOp().Set(Gf.Vec3f(4,4,.1));UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
 src=Usd.Stage.Open(str(ROBOT_SOURCE));src.SetEditTarget(src.GetSessionLayer())
 while any(pr.IsInstance() for pr in src.Traverse()):
  for pr in list(src.Traverse()):
   if pr.IsInstance():pr.SetInstanceable(False)
 flat=src.Flatten();arts={};mesh_repairs=[]
 Sdf.CopySpec(flat,'/World/Materials',s.GetRootLayer(),'/World/Materials')
 for side in ['left','right']:
  root='/World/Robot_'+side;UsdGeom.Xform.Define(s,root);UsdGeom.Scope.Define(s,root+'/joints');ns=links(side);nset=set(ns)
  for n in ns:
   Sdf.CopySpec(flat,'/World/Robot/'+n,s.GetRootLayer(),root+'/'+n);pr=s.GetPrimAtPath(root+'/'+n);set_pose(pr,fk(n,side,qinit[side]));a=UsdPhysics.RigidBodyAPI.Apply(pr);a.CreateRigidBodyEnabledAttr(True);a.CreateKinematicEnabledAttr(False)
   pa=PhysxSchema.PhysxRigidBodyAPI.Apply(pr);pa.CreateDisableGravityAttr(False);pa.CreateEnableCCDAttr(True);pa.CreateSolverPositionIterationCountAttr(64);pa.CreateSolverVelocityIterationCountAttr(16);pa.CreateMaxDepenetrationVelocityAttr(1)
   PhysxSchema.PhysxContactReportAPI.Apply(pr).CreateThresholdAttr(0)
   for c in list(Usd.PrimRange(pr)):
    if c.HasAPI(UsdPhysics.CollisionAPI):
     leaves=[c] if c.IsA(UsdGeom.Gprim) else [p for p in Usd.PrimRange(c) if p.IsA(UsdGeom.Mesh)]
     if not c.IsA(UsdGeom.Gprim):c.RemoveAPI(UsdPhysics.CollisionAPI)
     for leaf in leaves:
      UsdPhysics.CollisionAPI.Apply(leaf).CreateCollisionEnabledAttr(True)
      if leaf.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(leaf).CreateApproximationAttr('convexDecomposition' if n.endswith('palm_link') else 'convexHull')
      pc=PhysxSchema.PhysxCollisionAPI.Apply(leaf);pc.CreateContactOffsetAttr(.0005);pc.CreateRestOffsetAttr(0);UsdShade.MaterialBindingAPI.Apply(leaf).Bind(handmat,UsdShade.Tokens.strongerThanDescendants,'physics')
  # Cover the actual visible finger geometry, rather than undersized simplified pads.
  for n in ns:
   if 'finger' not in n or not n.endswith('_link4'):continue
   body=s.GetPrimAtPath(root+'/'+n)
   for leaf in Usd.PrimRange(body):
    if leaf.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(leaf).CreateCollisionEnabledAttr(False)
   for leaf in Usd.PrimRange(body):
    if not leaf.IsA(UsdGeom.Mesh) or '/visuals/' not in str(leaf.GetPath()):continue
    UsdPhysics.CollisionAPI.Apply(leaf).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(leaf).CreateApproximationAttr('convexHull');pc=PhysxSchema.PhysxCollisionAPI.Apply(leaf);pc.CreateContactOffsetAttr(.0005);pc.CreateRestOffsetAttr(.0001);UsdShade.MaterialBindingAPI.Apply(leaf).Bind(handmat,UsdShade.Tokens.strongerThanDescendants,'physics')
  mesh_repairs.extend(repair_wrist(s,s.GetPrimAtPath(root+'/'+('fl_link6' if side=='left' else 'fr_link6'))))
  for j in joints.values():
   if j['parent'] not in nset or j['child'] not in nset:continue
   path=root+'/joints/'+j['name'];srcpath='/World/Robot/joints/'+j['name'];assert src.GetPrimAtPath(srcpath),srcpath
   Sdf.CopySpec(flat,srcpath,s.GetRootLayer(),path);pr=s.GetPrimAtPath(path);pr.SetActive(True);jj=UsdPhysics.Joint(pr);jj.CreateBody0Rel().SetTargets([root+'/'+j['parent']]);jj.CreateBody1Rel().SetTargets([root+'/'+j['child']]);jj.CreateJointEnabledAttr(True)
   if j['kind']=='revolute':
    arm=j['name'] in arm_names(side);d=UsdPhysics.DriveAPI.Apply(pr,'angular');d.CreateTypeAttr('force');d.CreateStiffnessAttr(800 if arm else 12);d.CreateDampingAttr(60 if arm else .5);d.CreateMaxForceAttr(min(j['effort'],60) if arm else j['effort']);d.CreateTargetPositionAttr(float(np.rad2deg(qinit[side].get(j['name'],0))));d.CreateTargetVelocityAttr(0)
    state=PhysxSchema.JointStateAPI.Apply(pr,'angular');state.CreatePositionAttr(float(np.rad2deg(qinit[side].get(j['name'],0))));state.CreateVelocityAttr(0)
  # Socket knuckles are mechanically adjacent across the 4.6 mm two-axis joint.
  for fi in [2,3,4,5]:UsdPhysics.FilteredPairsAPI.Apply(s.GetPrimAtPath(root+f'/{side}_finger{fi}_link2')).CreateFilteredPairsRel().AddTarget(root+'/'+side+'_palm_link')
  anchor=UsdPhysics.FixedJoint.Define(s,root+'/root_joint');anchor.CreateBody1Rel().SetTargets([root+'/'+ns[0]]);anchor.CreateLocalPos0Attr(Gf.Vec3f(*base[side][:3,3]));r=Rot.from_matrix(base[side][:3,:3]).as_quat();anchor.CreateLocalRot0Attr(Gf.Quatf(float(r[3]),Gf.Vec3f(*r[:3])));UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim());a=PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim());a.CreateEnabledSelfCollisionsAttr(True);a.CreateSolverPositionIterationCountAttr(64);a.CreateSolverVelocityIterationCountAttr(16);arts[side]=root
 if task=='microwave_open':
  flat=current.Flatten();Sdf.CopySpec(flat,'/World/TaskMicrowave',s.GetRootLayer(),'/World/TaskMicrowave')
  for pr in Usd.PrimRange(s.GetPrimAtPath('/World/TaskMicrowave')):
   if pr.IsA(UsdGeom.Xformable):
    x=UsdGeom.Xformable(pr)
    if any(a.GetNumTimeSamples() for a in pr.GetAttributes() if a.GetName().startswith('xformOp:')):
     T=x.GetLocalTransformation(0);x.ClearXformOpOrder();x.AddTransformOp(opSuffix='dynamicInit').Set(T)
     for a in pr.GetAttributes():
      if a.GetName().startswith('xformOp:'):
       for t in a.GetTimeSamples():a.ClearAtTime(t)
   if pr.HasAPI(UsdPhysics.RigidBodyAPI):
    set_pose(pr,np.asarray(UsdGeom.Xformable(pr).GetLocalTransformation(0)).T)
    UsdPhysics.RigidBodyAPI(pr).CreateRigidBodyEnabledAttr(True);UsdPhysics.RigidBodyAPI(pr).CreateKinematicEnabledAttr(False);PhysxSchema.PhysxRigidBodyAPI.Apply(pr).CreateDisableGravityAttr(False);PhysxSchema.PhysxContactReportAPI.Apply(pr).CreateThresholdAttr(0)
  body=s.GetPrimAtPath('/World/TaskMicrowave/E_body_9');T=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(body)).T
  anchor=UsdPhysics.FixedJoint.Get(s,'/World/TaskMicrowave/E_body_9/FixedJoint');anchor.CreateLocalPos0Attr(Gf.Vec3f(*T[:3,3]));r=Rot.from_matrix(T[:3,:3]).as_quat();anchor.CreateLocalRot0Attr(Gf.Quatf(float(r[3]),Gf.Vec3f(*r[:3])))
  arts['microwave']='/World/TaskMicrowave/E_body_9'
  (out/'handle_collision_repair.json').write_text(json.dumps(repair_handle(s),indent=2))
 if task in ['pick_place','stack','produce_sort']:
  task_roots=['TaskObjects'] if task in ['pick_place','stack'] else ['TaskBin_vegetable','TaskBin_fruit','TaskFood_orange','TaskFood_pepper']
  current.SetEditTarget(current.GetSessionLayer())
  while any(pr.IsInstance() for pr in current.Traverse()):
   for pr in list(current.Traverse()):
    if pr.IsInstance():pr.SetInstanceable(False)
  source=current.Flatten()
  for name in task_roots:Sdf.CopySpec(source,'/World/'+name,s.GetRootLayer(),'/World/'+name)
  for name in ['MusicMaterials','TaskMaterials','TaskFoodLooks','RefinedMaterials','V2Materials','WujiMaterials']:
   if current.GetPrimAtPath('/World/'+name):Sdf.CopySpec(source,'/World/'+name,s.GetRootLayer(),'/World/'+name)
  bodies=[]
  for name in task_roots:
   for pr in Usd.PrimRange(s.GetPrimAtPath('/World/'+name)):
    if pr.IsA(UsdGeom.Xformable) and any(a.GetNumTimeSamples() for a in pr.GetAttributes() if a.GetName().startswith('xformOp:')):set_pose(pr,np.array(UsdGeom.Xformable(pr).GetLocalTransformation(0)).T)
    if pr.HasAPI(UsdPhysics.RigidBodyAPI):
     ancestor=pr.GetParent();nested=False
     while ancestor and ancestor.GetPath()!=Sdf.Path.absoluteRootPath:
      if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):nested=True;break
      ancestor=ancestor.GetParent()
     if nested:pr.RemoveAPI(UsdPhysics.RigidBodyAPI);pr.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
     else:
      bodies.append(str(pr.GetPath()));UsdPhysics.RigidBodyAPI(pr).CreateRigidBodyEnabledAttr(True);UsdPhysics.RigidBodyAPI(pr).CreateKinematicEnabledAttr(False);rb=PhysxSchema.PhysxRigidBodyAPI.Apply(pr);rb.CreateDisableGravityAttr(False);rb.CreateEnableCCDAttr(True);rb.CreateMaxDepenetrationVelocityAttr(.5);PhysxSchema.PhysxContactReportAPI.Apply(pr).CreateThresholdAttr(0)
    if pr.HasAPI(UsdPhysics.CollisionAPI):
     if pr.GetName()=='Shape' and s.GetPrimAtPath(str(pr.GetParent().GetPath())+'/RealAsset'):pr.RemoveAPI(UsdPhysics.CollisionAPI);continue
     UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr(True)
     if pr.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(pr).CreateApproximationAttr('convexDecomposition' if 'TaskBin_' in str(pr.GetPath()) else 'convexHull')
     pc=PhysxSchema.PhysxCollisionAPI.Apply(pr);pc.CreateContactOffsetAttr(.0005);pc.CreateRestOffsetAttr(0);UsdShade.MaterialBindingAPI.Apply(pr).Bind(tablemat,UsdShade.Tokens.strongerThanDescendants,'physics')
  arts['task_bodies']=bodies
 scene=UsdPhysics.Scene.Define(s,'/World/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81);p=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());p.CreateEnableCCDAttr(True);p.CreateSolverTypeAttr('TGS');p.CreateEnableGPUDynamicsAttr(False);p.CreateBroadphaseTypeAttr('MBP')
 
 for side in ['left','right']:
  root=s.GetPrimAtPath('/World/Robot_'+side);mesh_count=sum(pr.IsA(UsdGeom.Mesh) for pr in Usd.PrimRange(root));collision_meshes=sum(pr.IsA(UsdGeom.Mesh) and pr.HasAPI(UsdPhysics.CollisionAPI) for pr in Usd.PrimRange(root));assert mesh_count>=60 and collision_meshes>=30,(side,mesh_count,collision_meshes)
 from conservative_collision import install
 install(s,out)
 s.GetRootLayer().Save();(out/'mesh_repairs.json').write_text(json.dumps(mesh_repairs,indent=2));return s,arts
