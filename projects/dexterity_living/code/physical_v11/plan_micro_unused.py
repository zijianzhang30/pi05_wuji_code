"""Recorded-frame geometry audit: robot/task, hand self-intersection and inter-arm intersections."""
import argparse,json,itertools,re
from pathlib import Path
import numpy as np,fcl
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--stride',type=int,default=1);ap.add_argument('--task-pairs',action='store_true');ap.add_argument('--start-frame',type=int,default=0);ap.add_argument('--end-frame',type=int);ap.add_argument('--output');a=ap.parse_args();out=Path(a.run);data=np.load(out/'trajectory.npz');paths=list(map(str,data['body_paths']));index={p:i for i,p in enumerate(paths)}
s=Usd.Stage.Open(str(out/'initial_scene.usda'));cache=UsdGeom.XformCache(0);parts=[]
cube_faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)
for pr in s.Traverse():
 path=str(pr.GetPath());robot=path.startswith('/World/Robot_') and '/visuals/' in path
 task=path.startswith(('/World/TaskObjects/','/World/TaskFood_','/World/TaskBin_','/World/TaskMicrowave/','/World/MusicRig/'))
 table=path in ['/World/V2Mat','/World/V2Tabletop']
 if not(robot or task or table) or not(pr.IsA(UsdGeom.Mesh) or pr.IsA(UsdGeom.Cube)) or UsdGeom.Imageable(pr).ComputeVisibility(0)=='invisible':continue
 if not robot and any(x in path.lower() for x in ['collision','collider','/material','/looks']):continue
 body=pr
 while body and str(body.GetPath()) not in index:body=body.GetParent()
 bp=str(body.GetPath()) if body else None;bi=index.get(bp,-1)
 W=np.asarray(cache.GetLocalToWorldTransform(pr)).T;B=np.asarray(cache.GetLocalToWorldTransform(body)).T if body else np.eye(4);T=np.linalg.inv(B)@W
 if pr.IsA(UsdGeom.Mesh):
  mesh=UsdGeom.Mesh(pr);v=np.asarray(mesh.GetPointsAttr().Get(),float);ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());tri=[];off=0
  for n in counts:tri.extend([[ids[off],ids[off+k],ids[off+k+1]] for k in range(1,n-1)]);off+=n
  tri=np.asarray(tri,np.int32)
 else:
  sz=float(UsdGeom.Cube(pr).GetSizeAttr().Get())/2;v=np.array(list(itertools.product([-sz,sz],repeat=3)));tri=cube_faces
 v=v@T[:3,:3].T+T[:3,3];bounds=np.array(list(itertools.product(*zip(v.min(0),v.max(0)))));model=fcl.BVHModel();model.beginModel(len(v),len(tri));model.addSubModel(v,tri);model.endModel()
 link=path.split('/')[3] if robot else '';match=re.search(r'finger([1-5])_(link([1-4])|tip_link)',link);finger=(int(match[1]),int(match[3]) if match[3] else 5) if match else None
 parts.append(dict(path=path,body=bp,index=bi,bounds=bounds,ob=fcl.CollisionObject(model),robot=robot,side='left' if '/Robot_left/' in path else 'right',finger=finger,link=link))
pairs=[]
for i,x in enumerate(parts):
 for j in range(i+1,len(parts)):
  y=parts[j]
  if not x['robot'] and not y['robot']:
   xp,yp=x['path'],y['path'];moving=lambda z:z.startswith(('/World/TaskFood_','/World/TaskBin_','/World/TaskObjects/'))
   table=lambda z:z in ['/World/V2Mat','/World/V2Tabletop']
   if a.task_pairs and x['body']!=y['body'] and ((moving(xp) and moving(yp)) or (moving(xp) and table(yp)) or (moving(yp) and table(xp))):pairs.append((i,j,'object_object_or_support'))
   continue
  if x['robot']!=y['robot']:pairs.append((i,j,'task'));continue
  if x['side']!=y['side']:pairs.append((i,j,'inter_arm'));continue
  if x['body']==y['body']:continue
  f,g=x['finger'],y['finger']
  if f and g and (f[0]!=g[0] or abs(f[1]-g[1])>1):pairs.append((i,j,'finger_self'))
  elif f and 'palm_link' in y['link'] and f[1]>=3:pairs.append((i,j,'finger_palm'))
  elif g and 'palm_link' in x['link'] and g[1]>=3:pairs.append((i,j,'finger_palm'))

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"physical_v10"))
from model import fk
contacts=json.loads((out/"contacts.json").read_text());report0=json.loads((out/"report.json").read_text());jns=report0["dof_names"]
selected=[(i,j,k) for i,j,k in pairs if k in ["task","finger_self","finger_palm"] and any(parts[t]["side"]=="right" and parts[t]["finger"] and parts[t]["finger"][0] in [4,5] for t in [i,j])]
used=set(t for i,j,k in selected for t in [i,j]);frames=list(range(0,230,15))+[175,180,200,210,225]
candidates=[]
for vals in itertools.product([0,.4,.8,1.],[-.15,0,.15],[0,.5,1.,1.4],[0,.4,.8]):
 hits=0
 for frame in frames:
  row=contacts[frame];q=dict(zip(jns,row["q"]))
  for fi in [4,5]:
   for ji,v in enumerate(vals,1):q[f"right_finger{fi}_joint{ji}"]=v
  boxes={}
  for k in used:
   x=parts[k]
   if x["robot"]:T=fk(x["link"],"right",q);R,t=T[:3,:3],T[:3,3]
   elif x["index"]>=0:
    pos=data["poses"][frame,x["index"]];R=Rotation.from_quat(pos[3:]).as_matrix();t=pos[:3].astype(float)
   else:R=np.eye(3);t=np.zeros(3)
   vs=x["bounds"]@R.T+t;boxes[k]=(vs.min(0),vs.max(0));x["ob"].setTransform(fcl.Transform(R,t))
  for i,j,kind in selected:
   if np.any(boxes[i][0]>boxes[j][1]) or np.any(boxes[j][0]>boxes[i][1]):continue
   r=fcl.CollisionResult();fcl.collide(parts[i]["ob"],parts[j]["ob"],fcl.CollisionRequest(),r)
   if r.is_collision:hits+=1;break
  if hits:break
 if not hits:candidates.append(dict(joints=vals,score=float(np.linalg.norm(np.array(vals)-[.8,0,1.,.4]))))
print("CANDIDATES",sorted(candidates,key=lambda x:x["score"])[:12],flush=True)
(out/"unused_posture_candidates.json").write_text(json.dumps(sorted(candidates,key=lambda x:x["score"]),indent=2))
