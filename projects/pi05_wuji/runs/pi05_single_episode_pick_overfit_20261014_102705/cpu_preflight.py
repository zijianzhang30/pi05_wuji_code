from pathlib import Path
import os,sys,json,numpy as np
os.environ['JAX_PLATFORMS']='cpu'
os.environ['WUJI_RUN_DIR']='/media/external/bowenjing/projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705'
sys.path.insert(0,'/media/external/bowenjing/projects/pi05_wuji/runs/pi05_single_episode_pick_overfit_20261014_102705/code')
"""Single accepted training-episode diagnostic, full-parameter Pi05. Not held-out/generalization evidence."""
from pathlib import Path
import os,sys,json,time,hashlib,io,dataclasses,functools,traceback,fcntl,subprocess
from concurrent.futures import ThreadPoolExecutor
import math
ROOT=Path('/media/external/bowenjing/projects/pi05_wuji')
RUN=Path(os.environ['WUJI_RUN_DIR'])
RUN.mkdir(parents=True,exist_ok=True)
def save(name,d):
    p=RUN/name; p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def status(state,**kw):
    save('status.json',dict(state=state,pid=os.getpid(),updated_at=time.time(),**kw))
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import jax,jax.numpy as jnp
import flax.nnx as nnx
from flax import traverse_util
import optax
from openpi.models.pi0_config import Pi0Config
from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as config_lib,sharding,optimizer,checkpoints
from openpi.shared.normalize import NormStats
from openpi_client.image_tools import resize_with_pad
import importlib.util
spec=importlib.util.spec_from_file_location('upstream_train',ROOT/'openpi/scripts/train.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
BATCH=32;HORIZON=30;MAX_TOKENS=320;STEPS=int(os.environ.get('WUJI_REPAIR_STEPS','38000'));SEED=20261013
from policy_context import instruction
ORIGINAL_RUN=ROOT/'runs/pi05_full52_en_b8_20261013_001'
CONTINUE_RUN=Path('/media/external/bowenjing/projects/pi05_wuji/runs/pi05_train_scene_pick_v2_20261014_101417')
RESUME_STEP=37000;PRIOR_SAMPLES=0
TARGET_FRAME_EPOCHS=1000*BATCH/331;TRAIN_FRAME_DENOMINATOR=331
assert STEPS==RESUME_STEP+1000
TRIM={(x['task'],x['episode_index']):x['keep_end_exclusive'] for x in json.loads((RUN/'gesture_hold_manifest.json').read_text())['episodes']}
DATA=ROOT.parent/'act_wuji/datasets_v21'
TASKS=['pick_place']
PROMPTS=json.loads((RUN/'prompts.json').read_text())['instructions']
def load_episode(item):
    task,ep,info=item
    p=DATA/task/info['data_path'].format(episode_chunk=ep['episode_index']//info['chunks_size'],episode_index=ep['episode_index'])
    row=pq.read_table(p,columns=['observation.state','action','observation.images.ego','frame_index','episode_index','source_frame','source_time']).to_pydict()
    q=np.asarray(row['observation.state'],np.float32);a=np.asarray(row['action'],np.float32)
    n=ep['length'];assert q.shape==a.shape==(n,52);assert np.isfinite(q).all() and np.isfinite(a).all()
    assert row['frame_index']==list(range(n));assert set(row['episode_index'])=={ep['episode_index']}
    assert np.all(np.diff(row['source_frame'])==1)
    assert np.allclose(np.diff(row['source_time']),1/30,atol=1e-7,rtol=0)
    images=[x['bytes'] for x in row['observation.images.ego']]
    assert all(isinstance(b,bytes) for b in images)
    assert q.shape[0]==len(images)
    original_n=n
    n=TRIM.get((task,ep['episode_index']),n)
    q=q[:n];a=a[:n];images=images[:n]
    return dict(task=task,id=ep['episode_index'],split=ep['split'],q=q,a=a,images=images,
                provenance=dict(path=str(p),source=ep['source'],frames=n,original_frames=original_n,
                                numeric_sha256=hashlib.sha256(q.tobytes()+a.tobytes()).hexdigest(),
                                image_sha256=hashlib.sha256(b''.join(images)).hexdigest()))
class Dataset:
    def __init__(self):
        self.eps={t:{'train':[],'validation':[]} for t in TASKS};items=[];split_hashes={}
        joint_names=None
        for t in TASKS:
            info=json.loads((DATA/t/'meta/info.json').read_text())
            assert info['fps']==30 and info['total_episodes']==50
            names=info['features']['observation.state']['names']
            if joint_names is None:joint_names=names
            assert names==joint_names and names==info['features']['action']['names']
            sp=DATA/t/'split_manifest.json';s=json.loads(sp.read_text())
            assert set(s['train']).isdisjoint(s['validation']) and len(s['train'])==45 and len(s['validation'])==5
            assert all((e['episode_index'] in s[e['split']]) for e in s['episodes'])
            split_hashes[t]=hashlib.sha256(sp.read_bytes()).hexdigest()
            items.extend((t,e,info) for e in s['episodes'] if e['episode_index']==0 and e['split']=='train')
        with ThreadPoolExecutor(max_workers=8) as pool:
            for k,e in enumerate(pool.map(load_episode,items)):
                self.eps[e['task']][e['split']].append(e)
                if k%25==0:status('loading_dataset',episodes_loaded=k+1,total_episodes=1)
        train=[e for t in TASKS for e in self.eps[t]['train']]
        frozen=json.loads((ORIGINAL_RUN/'normalization/norm_stats.json').read_text())['norm_stats']
        self.stats={k:NormStats(**{name:np.asarray(value) for name,value in v.items()}) for k,v in frozen.items()}
        save('normalization_reuse.json',dict(source=str(ORIGINAL_RUN/'normalization/norm_stats.json'),sha256=hashlib.sha256((ORIGINAL_RUN/'normalization/norm_stats.json').read_bytes()).hexdigest(),unchanged_for_checkpoint_warmstart=True))
        self.tokenizer=PaligemmaTokenizer(max_len=MAX_TOKENS)
        tokenizer_lengths={}
        for t in TASKS:
            longest=0
            for subset in ['train','validation']:
                for e in self.eps[t][subset]:
                    e['qn']=self.normalize(e['q'],'state').clip(-1,1)
                    e['an']=self.normalize(e['a'],'actions')
                    toks=[];masks=[]
                    for frame,q in enumerate(e['qn']):
                        text=self.full_prompt(t,q,frame)
                        ids=self.tokenizer._tokenizer.encode(text,add_bos=True)
                        longest=max(longest,len(ids));assert len(ids)<=MAX_TOKENS,(t,len(ids))
                        token,mask=self.tokenizer.tokenize(instruction(PROMPTS[t],frame),q)
                        assert token[:len(ids)].tolist()==ids
                        toks.append(token.astype(np.int32));masks.append(mask)
                    e['tokens']=np.stack(toks);e['masks']=np.stack(masks)
            tokenizer_lengths[t]=longest
        self.frames={t:{s:sum(len(e['q']) for e in self.eps[t][s]) for s in ['train','validation']} for t in TASKS}
        self.total_train=sum(v['train'] for v in self.frames.values())
        assert self.total_train==331
        save('data_audit.json',dict(passed=True,total_train_frames=self.total_train,frames=self.frames,
            joint_names=joint_names,train_episodes=1,validation_episodes=0,split_sha256=split_hashes,
            normalization_train_only=True,state_clip=[-1,1],quantile_half_range_floor_rad=.04,
            original_source_unmodified=True,prompt_language='en',max_token_len=MAX_TOKENS,
            max_actual_tokens=tokenizer_lengths,action='52D absolute joint targets in radians',
            alignment='ego[t] + qpos[t] -> action[t:t+30]; no episode crossing; padded targets excluded from loss',
            episodes=[dict(task=t,split=s,episode_index=e['id'],**e['provenance']) for t in TASKS for s in ['train','validation'] for e in self.eps[t][s]],
            epoch_definition='new diagnostic updates *32/331; one training episode with replacement; fit checks are train-set checks'))
        from openpi.shared import normalize
        normalize.save(RUN/'normalization',self.stats)
        self.test_batch_contract()
    def full_prompt(self,t,q,frame):
        prompt=instruction(PROMPTS[t],frame).strip().replace('_',' ').replace('\n',' ')
        bins=np.digitize(q,bins=np.linspace(-1,1,257)[:-1])-1
        return f"Task: {prompt}, State: {' '.join(map(str,bins))};\nAction: "
    def normalize(self,x,key):
        n=self.stats[key];return (2*(x-n.q01)/(n.q99-n.q01)-1).astype(np.float32)
    def sample(self,t,e,frame):
        with Image.open(io.BytesIO(e['images'][frame])) as im:
            assert im.size==(640,480)
            image=resize_with_pad(np.asarray(im.convert('RGB')),224,224)
        indices=np.minimum(np.arange(frame,frame+HORIZON),len(e['a'])-1)
        valid=np.arange(frame,frame+HORIZON)<len(e['a'])
        return image,e['qn'][frame],e['tokens'][frame],e['masks'][frame],e['an'][indices],valid.astype(np.float32)
    def batch(self,step,task=None):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,step,0 if task is None else TASKS.index(task)+1]))
        samples=[]
        for j in range(BATCH):
            t=TASKS[int(rng.integers(len(TASKS)))] if task is None else task
            episodes=self.eps[t]['train'] # fixed diagnostic fit checks also use this training episode
            e=episodes[int(rng.integers(len(episodes)))] if task is None else episodes[j%len(episodes)]
            f=int(rng.integers(len(e['a'])))
            samples.append(self.sample(t,e,f))
        im,q,tk,tm,a,mask=[np.stack(x) for x in zip(*samples)]
        blank=np.zeros_like(im)
        obs=model_lib.Observation.from_dict(dict(image={'base_0_rgb':im,'left_wrist_0_rgb':blank,'right_wrist_0_rgb':blank},
            image_mask={'base_0_rgb':np.ones(BATCH,bool),'left_wrist_0_rgb':np.zeros(BATCH,bool),'right_wrist_0_rgb':np.zeros(BATCH,bool)},
            state=q,tokenized_prompt=tk,tokenized_prompt_mask=tm))
        return obs,a,mask
    def test_batch_contract(self):
        for t in TASKS:
            e=self.eps[t]['train'][0]
            _,q,tk,tm,a,valid=self.sample(t,e,len(e['a'])-1)
            assert valid.sum()==1 and a.shape==(30,52)
            np.testing.assert_allclose(a[0],self.normalize(e['a'][-1],'actions'),atol=1e-6)
            assert tk.dtype==np.int32 and len(tk)==MAX_TOKENS and tm.sum()<MAX_TOKENS
        obs,a,valid=self.batch(0)
        assert obs.images['base_0_rgb'].shape==(BATCH,224,224,3) and obs.state.shape==(BATCH,52)
        assert a.shape==(BATCH,30,52) and valid.shape==(BATCH,30)
        assert not obs.image_masks['left_wrist_0_rgb'].any()
        save('adapter_tests.json',dict(passed=True,last_frame_mask_count=1,batch_shape=[BATCH,30,52],missing_cameras_masked=True))
    def data_config(self):
        return config_lib.DataConfig(repo_id='local/wuji_seven',asset_id='wuji52',norm_stats=self.stats,use_quantile_norm=True)

dataset=Dataset()
assert TASKS==['pick_place'] and len(dataset.eps['pick_place']['train'])==1 and not dataset.eps['pick_place']['validation']
episode=dataset.eps['pick_place']['train'][0]
assert episode['id']==0 and episode['split']=='train' and len(episode['q'])==331
for frame in [0,165,330]:
 _,q,tokens,mask,action,valid=dataset.sample('pick_place',episode,frame)
 assert action.shape==(30,52) and np.isfinite(action).all() and valid.sum()==min(30,331-frame)
lo=dataset.stats['actions'].q01;hi=dataset.stats['actions'].q99
np.testing.assert_allclose((episode['an']+1)*.5*(hi-lo)+lo,episode['a'],atol=1e-6)
save('preflight/cpu_contract.json',dict(passed=True,train_episode_index=0,train_frames=331,train_episodes=1,validation_episodes=0,shape=[32,30,52],time_padding_masked=True,unchanged_normalization_roundtrip=True,source=episode['provenance']))
print(json.dumps(json.loads((RUN/'preflight/cpu_contract.json').read_text())))
