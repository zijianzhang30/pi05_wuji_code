from model import *
from dual_collision import *
import json
cand=json.loads((P/"runs/physical_v1/left_clearance_fast.json").read_text())[0];wide=json.loads((P/"runs/physical_v1/left_handoff_wide.json").read_text())[0]["q"];a=DualCollision(P/"runs/physical_v1/pick_tip_v18_gpu1/initial_scene.usda",cand["right_q"])
s=Usd.Stage.Open(str(P/"runs/physical_v1/pick_tip_v18_gpu1/initial_scene.usda"));pr=s.GetPrimAtPath("/World/TaskObjects/object/RealAsset/Model/visual");body=s.GetPrimAtPath("/World/TaskObjects/object");cache=UsdGeom.XformCache(0);m=UsdGeom.Mesh(pr);v=np.asarray(m.GetPointsAttr().Get());T=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(pr)).T;v=v@T[:3,:3].T+T[:3,3]
ids=list(m.GetFaceVertexIndicesAttr().Get());off=0;tri=[]
for n in m.GetFaceVertexCountsAttr().Get():tri.extend([[ids[off],ids[off+k],ids[off+k+1]] for k in range(1,n-1)]);off+=n
model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,np.asarray(tri,np.int32));model.endModel()
rows=json.loads((P/"runs/physical_v1/pick_tip_v18_gpu1/contacts.json").read_text());pose=np.array(rows[cand["handoff_frame"]]["object_pose"]);R=Rot.from_quat(pose[3:]).as_matrix();vv=v@R.T+pose[:3];ob=fcl.CollisionObject(model,fcl.Transform(R.copy(),pose[:3].copy()));a.right.append(("held_cube",ob,vv.min(0),vv.max(0)));print("WIDE",a.check(wide));print("CLOSED",a.check(cand["q"]))
