from pathlib import Path
import os,sys,json,ast,hashlib,io,time
import numpy as np,pyarrow.parquet as pq,h5py
from PIL import Image
R=Path('/media/external/bowenjing/projects/pi05_wuji')
A=Path("/media/external/bowenjing/projects/pi05_wuji/runs/pi05_code_audit_20261013_211237")
T=R/'runs/pi05_repair_clean_image_v4_20261013_001'
E=R/'runs/pi05_repair_clean_image_v4_dev_20261013_001'
D=R.parent/'act_wuji/datasets_v21'
sys.path.insert(0,str(E/'code'))
from adapter import Adapter
from policy_context import instruction
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared.normalize import NormStats
from openpi_client.image_tools import resize_with_pad
code=(T/'code/train_full_8gpu.py').read_text()
node=next(n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name=='Dataset')
namespace=dict(np=np,Image=Image,io=io,resize_with_pad=resize_with_pad,HORIZON=30,instruction=instruction)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(T/'code/train_full_8gpu.py'),'exec'),namespace)
data=namespace['Dataset'].__new__(namespace['Dataset'])
stats=json.loads((T/'normalization/norm_stats.json').read_text())['norm_stats']
data.stats={k:NormStats(**{n:np.asarray(v) for n,v in val.items()}) for k,val in stats.items()}
adapter=Adapter(E)
namespace['PROMPTS']=adapter.prompts
data.tokenizer=adapter.tokenizer
report=dict(scope='CPU contracts on two actual episodes per task; existing online failures; no policy success claim',created_at=time.time(),training_run=str(T),evaluation_run=str(E),checks=[],source_sha256={})
for f in [T/'code/train_full_8gpu.py',E/'code/adapter.py',E/'code/eval_environment.py',E/'code/policy_server.py',R/'openpi/src/openpi/models/pi0.py']:
 report['source_sha256'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for k in stats:
 for n in stats[k]:
  np.testing.assert_array_equal(stats[k][n],adapter.norm[k][n])
  assert len(stats[k][n])==52
 assert np.all(np.asarray(stats[k]['q99'])>np.asarray(stats[k]['q01']))
report['normalization_train_inference_identical']=True
online={x['task']:x for x in json.loads((E/'results.json').read_text())}
for task in adapter.prompts:
 sp=json.loads((D/task/'split_manifest.json').read_text())
 assert set(sp['train']).isdisjoint(sp['validation']) and len(sp['train'])==45 and len(sp['validation'])==5
 names=json.loads((D/task/'meta/info.json').read_text())['features']['action']['names']
 traj=np.load(Path(online[task]['run'])/'policy_rollout.npz')
 assert traj['joint_names'].tolist()==names
 for epindex in [sp['train'][0],sp['validation'][0]]:
  ep=next(x for x in sp['episodes'] if x['episode_index']==epindex)
  path=D/task/f'data/chunk-000/episode_{epindex:06d}.parquet'
  row=pq.read_table(path,columns=['observation.state','action','observation.images.ego','source_frame']).to_pydict()
  q=np.asarray(row['observation.state'],np.float32);actions=np.asarray(row['action'],np.float32)
  with h5py.File(Path(ep['source'])/'episode.hdf5') as hf:
   np.testing.assert_array_equal(q,hf['observations/qpos'][:-1])
   np.testing.assert_array_equal(actions,hf['action'][:])
   assert hf['metadata/joint_names'].asstr()[:].tolist()==names
   assert row['source_frame']==hf['source_frame'][:len(q)].tolist()
   raw=hf['action_substeps'][:]
   previous=np.concatenate([raw[:1,:1,:],raw[:-1,-1:,:]],axis=0)
   interp=previous+(raw[:,-1:,:]-previous)*np.arange(1,raw.shape[1]+1)[None,:,None]/raw.shape[1]
   window_error=np.max(abs(raw-interp),axis=(1,2))
  n=99 if task.startswith('gesture') else len(q)
  e=dict(q=q[:n],qn=data.normalize(q[:n],'state').clip(-1,1),a=actions[:n],
         an=data.normalize(actions[:n],'actions'),images=[x['bytes'] for x in row['observation.images.ego'][:n]])
  toks=[];masks=[];lengths=[]
  for f in range(n):
   ids=adapter.tokenizer._tokenizer.encode(data.full_prompt(task,e['qn'][f],f),add_bos=True)
   assert len(ids)<320
   tok,mask=adapter.tokenizer.tokenize(instruction(adapter.prompts[task],f),e['qn'][f])
   toks.append(tok.astype(np.int32));masks.append(mask);lengths.append(len(ids))
  e['tokens']=np.stack(toks);e['masks']=np.stack(masks)
  for f in [0,n//2,n-1]:
   rgb=np.asarray(Image.open(io.BytesIO(e['images'][f])).convert('RGB'))
   inp=adapter.input(task,e['q'][f],rgb,False,control_step=f)
   image,qn,tok,mask,an,valid=data.sample(task,e,f)
   np.testing.assert_array_equal(image,inp['image']['base_0_rgb'])
   np.testing.assert_array_equal(qn,inp['state'])
   np.testing.assert_array_equal(tok,inp['tokenized_prompt'])
   np.testing.assert_array_equal(mask,inp['tokenized_prompt_mask'])
   expected=np.minimum(np.arange(f,f+30),n-1)
   assert valid.sum()==min(30,n-f)
   np.testing.assert_allclose(adapter.output(an),e['a'][expected],atol=5e-7,rtol=1e-6)
  report['checks'].append(dict(task=task,episode=epindex,split=ep['split'],frames=n,passed=True,
    max_tokens=max(lengths),parquet_hdf5_exact=True,online_joint_order_exact=True,
    train_inference_input_exact=True,action_roundtrip=True,terminal_padding_mask=True,
    substep_endpoint_interpolation_error_max_rad=float(window_error.max()),
    substep_nonlinear_windows_over_1e6=int((window_error>1e-6).sum())))
 report.setdefault('online_results',[]).append({k:online[task].get(k) for k in ['task','seed','success','reason','frames','action_limit_clipped_frames']})
 contacts=json.loads((Path(online[task]['run'])/'contacts.json').read_text())
 report.setdefault('last_failure_evidence',{})[task]={k:v for k,v in contacts[-1].items() if k not in ['q','target','frame','t','collection_criteria']}
train_audit=json.loads((T/'data_audit.json').read_text())
report['all_training_token_max']=train_audit['max_actual_tokens']
report['original_randomized_head_tensors']=[x for x in json.loads((R/'runs/pi05_full52_en_b8_20261013_001/weight_load_audit.json').read_text())['tensors'] if x['source']=='new_52d_projection']
report['passed']=all(x['passed'] for x in report['checks'])
(A/'evidence/cpu_contract_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','last_failure_evidence']},indent=2))
print('FAILURE_EVIDENCE',json.dumps(report['last_failure_evidence']))
