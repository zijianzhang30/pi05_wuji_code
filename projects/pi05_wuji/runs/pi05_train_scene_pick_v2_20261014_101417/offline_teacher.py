from pathlib import Path
import json,io,time
import numpy as np,pyarrow.parquet as pq
from PIL import Image
from multiprocessing.connection import Client
N=Path(__file__).resolve().parent
m=json.loads((N/'manifest.json').read_text());ready=json.loads((N/'policy_ready.json').read_text())
p=Path('/media/external/bowenjing/projects/act_wuji/datasets_v21/pick_place/data/chunk-000/episode_000000.parquet')
rows=pq.read_table(p,columns=['observation.state','action','observation.images.ego']).to_pydict()
q=np.asarray(rows['observation.state'],np.float32);a=np.asarray(rows['action'],np.float32)
names=json.loads((Path('/media/external/bowenjing/projects/pi05_wuji/runs/pi05_seven_resume10e_b32_20261014_001')/'data_audit.json').read_text())['joint_names']
frames=list(range(0,len(a),15));pred=[];ground=[];valid=[];timing=[]
for seed in m['seeds']:
 conn=Client(ready['socket'],family='AF_UNIX');conn.send(dict(task='pick_place',seed=seed,mode='pi05'));meta=conn.recv();assert meta['joint_names']==names
 one=[]
 for f in frames:
  im=np.asarray(Image.open(io.BytesIO(rows['observation.images.ego'][f]['bytes'])).convert('RGB'))
  began=time.monotonic();conn.send(dict(q=q[f].tolist(),rgb_bytes=im.tobytes(),control_step=f,jpeg_roundtrip=False));reply=conn.recv();timing.append(time.monotonic()-began);one.append(reply['actions'])
 pred.append(one);conn.send(None);conn.close()
for f in frames:
 ix=np.minimum(np.arange(f,f+30),len(a)-1);ground.append(a[ix]);valid.append(np.arange(f,f+30)<len(a))
pred=np.asarray(pred);ground=np.asarray(ground);valid=np.asarray(valid)
groups={'left_arm':list(range(6)),'left_hand':list(range(6,26)),'right_arm':list(range(26,32)),'right_hand':list(range(32,52))}
summary={}
for key,ix in groups.items():
 err=abs(pred[...,ix]-ground[None,...,ix]);sel=err[:,valid,:]
 summary[key]=dict(chunk_mae_rad=float(sel.mean()),chunk_p95_abs_rad=float(np.quantile(sel,.95)),first_action_mae_rad=float(err[:,:,0,:].mean()),first_action_p95_abs_rad=float(np.quantile(err[:,:,0,:],.95)))
perframe=[]
for i,f in enumerate(frames):
 d=dict(frame=f)
 for key,ix in groups.items():d[key+'_first_mae_rad']=float(abs(pred[:,i,0,:][:,ix]-a[f,ix]).mean())
 perframe.append(d)
np.savez_compressed(N/'offline/teacher_forced.npz',predictions=pred,ground_truth=ground,valid=valid,frames=frames,state=q[frames],joint_names=names,sampling_seeds=m['seeds'])
# Desired target FK difference, not simulated measured palm error.
import sys
sys.path.insert(0,'/media/external/bowenjing/projects/dexterity_living/code/physical_v10')
from model import fk
kinematic={}
for link in ['right_palm_link','right_finger1_tip_link','right_finger2_tip_link']:
 distances=[];first_distances=[]
 for si in range(pred.shape[0]):
  for fi,f in enumerate(frames):
   for k in range(30):
    if not valid[fi,k]:continue
    pd=dict(zip(names,pred[si,fi,k].tolist()));gd=dict(zip(names,ground[fi,k].tolist()))
    mm=float(np.linalg.norm(fk(link,'right',pd)[:3,3]-fk(link,'right',gd)[:3,3])*1000)
    distances.append(mm)
    if k==0:first_distances.append(mm)
 kinematic[link]=dict(chunk_mean_target_position_error_mm=float(np.mean(distances)),chunk_p95_target_position_error_mm=float(np.quantile(distances,.95)),first_mean_target_position_error_mm=float(np.mean(first_distances)),first_p95_target_position_error_mm=float(np.quantile(first_distances,.95)))
for fi,f in enumerate(frames):
 gt=fk('right_palm_link','right',dict(zip(names,a[f].tolist())))[:3,3]
 perframe[fi]['palm_first_target_error_mm_by_sampling_seed']=[float(np.linalg.norm(fk('right_palm_link','right',dict(zip(names,pred[si,fi,0].tolist())))[:3,3]-gt)*1000) for si in range(pred.shape[0])]

result=dict(kinematic_target_error=kinematic,expert_first_contact_frame=165,checkpoint_step=37000,task='pick_place',training_episode_index=0,reference=m['reference'],input='actual training Parquet RGB and actual qpos; no new image recompression',sampling_seeds=m['seeds'],num_observations=len(frames),num_inferences=len(timing),groups=summary,per_frame=perframe,mean_inference_seconds=float(np.mean(timing)),scope='teacher-forced train-set prediction; not closed-loop task success')
(N/'offline/summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary),flush=True)
