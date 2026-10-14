"""Restore source shape proportions; nominal scale and mass are not measured calibration."""
def apply(stage,out):
 from pxr import UsdGeom,UsdPhysics,Gf
 import numpy as np,json
 changes=[]
 for name,longest,mass in [('orange',.060,.080),('pepper',.080,.080)]:
  root=stage.GetPrimAtPath('/World/TaskFood_'+name)
  vis=stage.GetPrimAtPath(str(root.GetPath())+'/ReferenceVisuals')
  asset=stage.GetPrimAtPath(str(vis.GetPath())+'/Asset')
  ext=np.array(asset.GetCustomData()['metadata']['link_bounding_boxes']['base_link']['visual']['axis_aligned']['extent'])
  old=np.diag(np.asarray(UsdGeom.Xformable(vis).GetLocalTransformation()))[:3].copy()
  scale=float(longest/ext.max())
  xf=UsdGeom.Xformable(vis);xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix='uniformReferenceScale').Set(Gf.Matrix4d().SetScale(Gf.Vec3d(scale)))
  UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
  changes.append(dict(asset=name,old_scale=old.tolist(),uniform_scale=scale,source_bounds_m=ext.tolist(),new_bounds_m=(ext*scale).tolist(),mass_kg=mass,mass_status='engineering nominal; not measured',friction_status='engineering nominal; not measured',original_textures_preserved=True))
 (out/'asset_proportions.json').write_text(json.dumps(changes,indent=2))
 return changes
