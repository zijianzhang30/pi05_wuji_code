"""Circumscribe graphics using support planes retained by PhysX cooking."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

P = Path(__file__).resolve().parents[2]

def install(stage, out, margins=None):
    profiles = json.loads((P / "reports/cooked_collision_all_profiles.json").read_text())
    margin = float(os.environ.get("SUPPORT_MARGIN", ".0001"))
    rows = []
    for profile in profiles:
        path = profile["path"]
        local_margin = (margins or {}).get(path, margin)
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        mesh = UsdGeom.Mesh(prim)
        original = np.asarray(mesh.GetPointsAttr().Get(), float)
        assert len(profile["convexes"]) == 1, path
        old = np.asarray(profile["convexes"][0]["vertices"])
        normals = np.unique(ConvexHull(old).equations[:, :3].round(9), axis=0)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        distances = (original @ normals.T).max(0) + local_margin
        halfspaces = np.column_stack([normals, -distances])
        vertices = HalfspaceIntersection(halfspaces, original.mean(0)).intersections
        vertices = np.unique(vertices.round(10), axis=0)
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for index, face in enumerate(faces):
            tri = vertices[face]
            if np.dot(np.cross(tri[1]-tri[0], tri[2]-tri[0]), hull.equations[index, :3]) < 0:
                faces[index] = face[::-1]
        proxy_path = path + "_enclosing_collision"
        Sdf.CopySpec(stage.GetRootLayer(), path, stage.GetRootLayer(), proxy_path)
        proxy = stage.GetPrimAtPath(proxy_path)
        for attr in list(proxy.GetAttributes()):
            if attr.GetName().startswith("primvars:") or attr.GetName() == "normals":
                proxy.RemoveProperty(attr.GetName())
        pm = UsdGeom.Mesh(proxy)
        pm.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        pm.CreateFaceVertexCountsAttr([3] * len(faces))
        pm.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        pm.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
        pm.CreateSubdivisionSchemeAttr("none")
        UsdGeom.Imageable(proxy).CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
        UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(proxy).CreateApproximationAttr("convexHull")
        ca = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(proxy)
        ca.CreateHullVertexLimitAttr(255)
        ca.CreateMinThicknessAttr(.0001)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(proxy)
        pc.CreateRestOffsetAttr(0.)
        pc.CreateContactOffsetAttr(.0005)
        rows.append(dict(source_mesh=path, collision_mesh=proxy_path, vertex_count=len(vertices),
                         support_margin_local_m=local_margin, original_cooked_max_outside_local_m=profile["convexes"][0]["max_input_outside_m"]))
    (out / "collision_enclosure_recipe.json").write_text(json.dumps(rows, indent=2))
    return rows

def check_once(stage, out):
    import omni.usd
    import omni.physx
    from pxr import PhysicsSchemaTools
    rows = json.loads((out / "collision_enclosure_recipe.json").read_text())
    cache = UsdGeom.XformCache(0)
    results = []
    for row in rows:
        prim = stage.GetPrimAtPath(row["source_mesh"])
        original = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        scale = max(np.linalg.norm(np.asarray(cache.GetLocalToWorldTransform(prim))[:3, :3], axis=1))
        def callback(result, convexes):
            assert len(convexes) == 1, (row, result, len(convexes))
            vertices = np.asarray([[p.x, p.y, p.z] for p in convexes[0].vertices])
            planes = ConvexHull(vertices).equations
            outside = float((original @ planes[:, :3].T + planes[:, 3]).max()) * scale
            results.append(dict(mesh=row["source_mesh"], actual_cooked_vertices=len(vertices),
                                max_graphics_outside_collision_m=outside, world_scale=scale))
        omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
            omni.usd.get_context().get_stage_id(), PhysicsSchemaTools.sdfPathToInt(row["collision_mesh"]), False, callback)
    assert len(results) == len(rows), "Missing synchronous cooking callbacks"
    report = dict(status="passed" if all(r["max_graphics_outside_collision_m"] <= .00001 for r in results) else "failed", meshes=results)
    (out / "cooked_enclosure_check.json").write_text(json.dumps(report, indent=2))
    return report


def validate_cooked(stage, out):
    import omni.kit.app
    attempts=[]
    for attempt in range(8):
        report=check_once(stage,out);attempts.append(report)
        if report['status']=='passed':
            (out/'cooked_enclosure_iterations.json').write_text(json.dumps(attempts,indent=2))
            stage.GetRootLayer().Save()
            return report
        rows=json.loads((out/'collision_enclosure_recipe.json').read_text())
        margins={r['source_mesh']:r['support_margin_local_m'] for r in rows}
        for r in report['meshes']:
            if r['max_graphics_outside_collision_m']>0.00001:
                margins[r['mesh']]+=r['max_graphics_outside_collision_m']/r['world_scale']+.0001
        assert max(margins.values())<.002, ('enclosure would exceed 2 mm local support margin',margins)
        install(stage,out,margins)
        omni.kit.app.get_app().update()
    raise RuntimeError('cooked collision enclosure did not converge')
