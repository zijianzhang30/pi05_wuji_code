"""Frozen pi05 action-precision diagnostic; no optimizer updates or checkpoint selection."""
from pathlib import Path
import os,sys,json,time,fcntl,io
R=Path('/media/external/bowenjing/projects/pi05_wuji')
A=Path(__file__).resolve().parents[1]
E=R/'runs/pi05_repair_clean_image_v4_dev_20261013_001'
T=R/'runs/pi05_repair_clean_image_v4_20261013_001'
lock=open(R.parent/'dexterity_living/reports/process_manager/gpu0.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['OPENPI_DATA_HOME']=str(R/'cache/openpi')
sys.path.insert(0,str(E/'code'))
import numpy as np,jax,jax.numpy as jnp,pyarrow.parquet as pq
from PIL import Image
from openpi.models.pi0_config import Pi0Config
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from adapter import Adapter
manifest=json.loads((E/'manifest.json').read_text())
ck=Path(manifest['checkpoint'])
print('LOADING',ck,flush=True)
model=Pi0Config(pi05=True,action_dim=52,action_horizon=30,max_token_len=320).load(model_lib.restore_params(ck/'params'))
model.eval()
sample=nnx_utils.module_jit(model.sample_actions)
adapter=Adapter(E)
rows=[];first=True
for ti,task in enumerate(manifest['tasks']):
 sp=json.loads((R.parent/'act_wuji/datasets_v21'/task/'split_manifest.json').read_text())
 for epindex in [sp['train'][0],*sp['validation']]:
  path=R.parent/'act_wuji/datasets_v21'/task/f'data/chunk-000/episode_{epindex:06d}.parquet'
  data=pq.read_table(path,columns=['observation.state','action','observation.images.ego']).to_pydict()
  q=np.asarray(data['observation.state'],np.float32);actions=np.asarray(data['action'],np.float32)
  n=99 if task.startswith('gesture') else len(q)
  for frame in sorted(set([0,n//3,2*n//3,max(0,n-30),n-1])):
   rgb=np.asarray(Image.open(io.BytesIO(data['observation.images.ego'][frame]['bytes'])).convert('RGB'))
   inp=adapter.input(task,q[frame],rgb,False,control_step=frame)
   obs=model_lib.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],inp))
   key=jax.random.fold_in(jax.random.fold_in(jax.random.key(20261013),ti),epindex*10000+frame)
   begin=time.monotonic();predn=np.asarray(sample(key,obs,num_steps=10)[0]);pred=adapter.output(predn)
   assert pred.shape==(30,52) and np.isfinite(pred).all()
   if first:
    same=np.asarray(sample(key,obs,num_steps=10)[0]);np.testing.assert_array_equal(predn,same);first=False
   valid=np.arange(frame,frame+30)<n
   ref=actions[np.minimum(np.arange(frame,frame+30),n-1)]
   err=np.abs(pred[valid]-ref[valid])
   active=slice(0,26) if task=='microwave_open' else slice(26,52)
   inactive=slice(26,52) if task=='microwave_open' else slice(0,26)
   arm=slice(0,6) if task=='microwave_open' else slice(26,32)
   hand=slice(6,26) if task=='microwave_open' else slice(32,52)
   rows.append(dict(task=task,episode=epindex,split='train' if epindex in sp['train'] else 'validation',frame=frame,valid_steps=int(valid.sum()),
    active_mae_rad=float(err[:,active].mean()),active_p95_rad=float(np.quantile(err[:,active],.95)),
    arm_mae_rad=float(err[:,arm].mean()),hand_mae_rad=float(err[:,hand].mean()),inactive_mae_rad=float(err[:,inactive].mean()),
    first_arm_max_error_rad=float(abs(pred[0,arm]-ref[0,arm]).max()),first_hand_max_error_rad=float(abs(pred[0,hand]-ref[0,hand]).max()),
    seconds=time.monotonic()-begin))
  print('PROBED',task,epindex,flush=True)
summary={}
for task in manifest['tasks']:
 summary[task]={}
 for split in ['train','validation']:
  rs=[x for x in rows if x['task']==task and x['split']==split]
  summary[task][split]={key:float(np.mean([x[key] for x in rs])) for key in ['active_mae_rad','arm_mae_rad','hand_mae_rad','inactive_mae_rad','first_arm_max_error_rad','first_hand_max_error_rad']}
result=dict(checkpoint=str(ck),rows=rows,summary=summary,all_30x52_finite=True,same_seed_repeat_exact=True,
 scope='Seven tasks, first training episode and all five validation episodes; five actual observations per episode. Teacher-forced offline action precision, not closed-loop success.')
(A/'evidence/frozen_policy_precision.json').write_text(json.dumps(result,indent=2))
print('SUMMARY',json.dumps(summary),flush=True)
