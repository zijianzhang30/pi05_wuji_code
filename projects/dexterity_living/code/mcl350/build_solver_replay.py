"""Render-only replay of recorded solver body poses, with explicit provenance."""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--task", default="pick_place")
p.add_argument("--output", required=True)
a = p.parse_args()
run, out = Path(a.run), Path(a.output)
project = run.parents[2]
old = project / "runs/wuji_blackwhite_smooth_pt4k_20261011/living_mcl_wuji_v1_20261011" / a.task
report = json.loads((run / "report.json").read_text())
assert report["physics_stepped"] and report["object_pose_commands"] == 0
data = np.load(run / "observation_trajectory.npz")
paths = list(map(str, data["body_paths"]))
out.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.Open(str(old / "animation.usdc"))
stage.Flatten().Export(str(out / "animation.usdc"))
stage = Usd.Stage.Open(str(out / "animation.usdc"))
for name in ["DeploymentRobot", "BasicRig", "TaskObjects", "TaskMicrowave", "MusicRig", "TaskFood_orange", "TaskFood_pepper", "TaskBin_fruit", "TaskBin_vegetable"]:
    stage.RemovePrim("/World/" + name)
source = Usd.Stage.Open(str(run / "initial_scene.usda"))
flat = source.Flatten()
for name in ["Robot_left", "Robot_right", "Materials", "TaskObjects", "TaskMicrowave", "MusicRig", "MusicMaterials", "TaskFood_orange", "TaskFood_pepper", "TaskFoodLooks", "TaskBin_fruit", "TaskBin_vegetable", "TaskMaterials", "RefinedMaterials", "PhysicsMaterials"]:
    path = "/World/" + name
    if source.GetPrimAtPath(path):
        Sdf.CopySpec(flat, path, stage.GetRootLayer(), path)

# A black material applies only to robot render geometry. Physical run is unchanged.
material = UsdShade.Material.Define(stage, "/World/ReplayBlack")
shader = UsdShade.Shader.Define(stage, "/World/ReplayBlack/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.018, .023, .028))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.48)
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
for side in ["left", "right"]:
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Robot_" + side)).Bind(material, UsdShade.Tokens.strongerThanDescendants)
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

cache = UsdGeom.XformCache(0)
ops = []
for path in paths:
    prim = stage.GetPrimAtPath(path)
    assert prim, path
    # Reset stack allows exact WORLD poses even for nested microwave rigid bodies.
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            prim.RemoveProperty(attr.GetName())
    op = xf.AddTransformOp()
    xf.SetResetXformStack(True)
    ops.append(op)
for frame, poses in enumerate(data["poses"]):
    for op, pose in zip(ops, poses):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        matrix[:3, 3] = pose[:3]
        op.Set(Gf.Matrix4d(matrix.T.tolist()), frame)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(len(data["poses"]) - 1)
stage.SetTimeCodesPerSecond(30)
stage.SetFramesPerSecond(30)
stage.GetRootLayer().Save()
cfg = json.loads((old / "task.json").read_text())
cfg.update(source="recorded_physics", scope=f"Replay of actual {report.get('sim_hz', 480)} Hz PhysX CPU simulation; 30 Hz body pose recording", new_physics_validation=False,
           presentation_start_frame=0, duration_s=(len(data["poses"])-1)/30, source_run=str(run), physics_report=report,
           motion_revision="solver body transforms only; no scripted object trajectory", style="black Wuji and Piper, white tabletop")
surface_path=run / "surface_audit.json"
triangle_path=run / "triangle_audit.json"
if surface_path.exists() and triangle_path.exists():
    surf=json.loads(surface_path.read_text()); tri=json.loads(triangle_path.read_text())
    cfg["new_physics_validation"]=(report.get("status")=="passed_pick_place_trial" and surf.get("maximum_sampled_penetration_m")==0 and tri.get("frames_with_nonhandle_intersections")==0)
    cfg["validation_scope"]="Dynamics report plus recorded-frame visual surface and triangle checks; no continuous-time guarantee"
(out / "task.json").write_text(json.dumps(cfg, indent=2))
print(json.dumps({"output": str(out), "frames": len(data["poses"]), "body_count": len(paths)}))
