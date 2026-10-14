"""Full-parameter pi05, shared seven-task Wuji policy. All artifacts stay on PRO5000."""
from pathlib import Path
import os,sys,json,time,hashlib,io,dataclasses,functools,traceback,fcntl,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/media/external/bowenjing/projects/pi05_wuji')
RUN=Path(os.environ['WUJI_RUN_DIR'])
RUN.mkdir(parents=True,exist_ok=True)
def save(name,d):
    p=RUN/name; p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def status(state,**kw):
    save('status.json',dict(state=state,pid=os.getpid(),updated_at=time.time(),**kw))
locks=[]
for gpu in range(8):
    f=open(f'/media/external/bowenjing/projects/dexterity_living/reports/process_manager/gpu{gpu}.lock','a')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(line.split(',')[1])<1500 for line in memory.splitlines() if int(line.split(',')[0]) in range(8)),memory
status('importing')
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
BATCH=8;HORIZON=30;MAX_TOKENS=320;STEPS=int(os.environ.get('WUJI_REPAIR_STEPS','13000'));SEED=20261013
from policy_context import instruction
ORIGINAL_RUN=ROOT/'runs/pi05_full52_en_b8_20261013_001'
TRIM={(x['task'],x['episode_index']):x['keep_end_exclusive'] for x in json.loads((RUN/'gesture_hold_manifest.json').read_text())['episodes']}
DATA=ROOT.parent/'act_wuji/datasets_v21'
TASKS=sorted(json.loads((RUN/'prompts.json').read_text())['instructions'])
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
            items.extend((t,e,info) for e in s['episodes'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            for k,e in enumerate(pool.map(load_episode,items)):
                self.eps[e['task']][e['split']].append(e)
                if k%25==0:status('loading_dataset',episodes_loaded=k+1,total_episodes=350)
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
        assert self.total_train==107260
        save('data_audit.json',dict(passed=True,total_train_frames=self.total_train,frames=self.frames,
            joint_names=joint_names,train_episodes=315,validation_episodes=35,split_sha256=split_hashes,
            normalization_train_only=True,state_clip=[-1,1],quantile_half_range_floor_rad=.04,
            original_source_unmodified=True,prompt_language='en',max_token_len=MAX_TOKENS,
            max_actual_tokens=tokenizer_lengths,action='52D absolute joint targets in radians',
            alignment='ego[t] + qpos[t] -> action[t:t+30]; no episode crossing; padded targets excluded from loss',
            episodes=[dict(task=t,split=s,episode_index=e['id'],**e['provenance']) for t in TASKS for s in ['train','validation'] for e in self.eps[t][s]],
            epoch_definition='optimizer_steps * global_batch / total_train_frames; balanced sampling with replacement'))
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
            episodes=self.eps[t]['train' if task is None else 'validation']
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
        assert obs.images['base_0_rgb'].shape==(8,224,224,3) and obs.state.shape==(8,52)
        assert a.shape==(8,30,52) and valid.shape==(8,30)
        assert not obs.image_masks['left_wrist_0_rgb'].any()
        save('adapter_tests.json',dict(passed=True,last_frame_mask_count=1,batch_shape=[8,30,52],missing_cameras_masked=True))
    def data_config(self):
        return config_lib.DataConfig(repo_id='local/wuji_seven',asset_id='wuji52',norm_stats=self.stats,use_quantile_norm=True)
@dataclasses.dataclass(frozen=True)
class Weights:
    def load(self,reference):
        ck=ORIGINAL_RUN/'checkpoints/10000'
        source=model_lib.restore_params(ck/'params',restore_type=np.ndarray)
        flat=traverse_util.flatten_dict(source,sep='/');ref=traverse_util.flatten_dict(reference,sep='/')
        assert set(flat)==set(ref)
        out={}
        for k,v in ref.items():
            assert flat[k].shape==v.shape,(k,flat[k].shape,v.shape)
            out[k]=flat[k].astype(v.dtype)
        save('weight_load_audit.json',dict(passed=True,source=str(ck),all_tensors_warmstarted=True,tensors=len(out),random_projection_reset=False))
        return traverse_util.unflatten_dict(out,sep='/')
def step_fn(config,rng,state,batch):
    model=nnx.merge(state.model_def,state.params);model.train()
    obs,actions,valid=batch
    def loss_fn(m):
        losses=m.compute_loss(jax.random.fold_in(rng,state.step),obs,actions,train=True)
        return jnp.mean(jnp.sum(losses*valid,axis=-1)/jnp.maximum(valid.sum(-1),1))
    loss,grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,config.trainable_filter))(model)
    params=state.params.filter(config.trainable_filter)
    updates,new_opt=state.tx.update(grads,state.opt_state,params)
    nnx.update(model,optax.apply_updates(params,updates))
    new_state=dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=new_opt)
    flat=traverse_util.flatten_dict(grads.to_pure_dict(),sep='/')
    groups={'vision':[v for k,v in flat.items() if '/img/' in '/'+k],
            'language':[v for k,v in flat.items() if '/llm/' in '/'+k and '_1/' not in k],
            'action_expert':[v for k,v in flat.items() if '/llm/' in '/'+k and '_1/' in k],
            'projection':[v for k,v in flat.items() if k.startswith(('action_in_proj/','action_out_proj/'))]}
    metrics={'loss':loss,'grad_norm':optax.global_norm(grads)}
    metrics.update({f'grad_norm_{k}':optax.global_norm(v) for k,v in groups.items() if v})
    uflat=traverse_util.flatten_dict(updates.to_pure_dict(),sep='/')
    for key in ['vision','language','action_expert','projection']:
        def selected(k):
            return (('/img/' in '/'+k) if key=='vision' else
                ('/llm/' in '/'+k and '_1/' not in k) if key=='language' else
                ('/llm/' in '/'+k and '_1/' in k) if key=='action_expert' else
                k.startswith(('action_in_proj/','action_out_proj/')))
        vals=[v for k,v in uflat.items() if selected(k)]
        if vals:metrics['update_norm_'+key]=optax.global_norm(vals)
    return new_state,metrics
def eval_fn(state,rng,batch):
    m=nnx.merge(state.model_def,state.params);m.eval()
    obs,a,mask=batch
    loss=m.compute_loss(rng,obs,a,train=False)
    return jnp.mean((loss*mask).sum(-1)/mask.sum(-1))
def restore_resharded(manager,state,state_shard,step=None):
    from orbax.checkpoint import checkpoint_utils
    from openpi.shared import array_typing as at
    with at.disable_typechecking():
        target,params=checkpoints._split_params(state)
        target_shard,params_shard=checkpoints._split_params(state_shard)
        items={'train_state':target,'params':{'params':params}}
        shards={'train_state':target_shard,'params':{'params':params_shard}}
        kwargs={k:{'restore_args':checkpoint_utils.construct_restore_args(v,shards[k])} for k,v in items.items()}
        result=manager.restore(step,items=items,restore_kwargs=kwargs)
        restored=checkpoints._merge_params(result['train_state'],result['params'])
    jax.block_until_ready(restored)
    assert all(len(a.sharding.device_set)==8 for a in jax.tree.leaves(restored) if isinstance(a,jax.Array))
    return restored

def main():
    assert jax.device_count()==8,jax.devices()
    status('warming_up_collectives')
    warm=jax.pmap(lambda x:jax.lax.psum(x,'i'),axis_name='i')(jnp.arange(8,dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(warm),np.full(8,28.0))
    big=jnp.ones((8,8*1024*1024),dtype=jnp.float32)
    out=jax.pmap(lambda x:jax.lax.psum(x,'i'),axis_name='i')(big)
    out.block_until_ready()
    assert float(np.asarray(out[0,0]))==8.0
    del big,out
    save('collective_warmup_8gpu.json',dict(passed=True,nccl='2.28.9',gpu_count=8,large_collective_elements_per_gpu=8*1024*1024))
    status('loading_dataset')
    dataset=Dataset()
    config=config_lib.TrainConfig(name='pi05_wuji_full52_en',exp_name=RUN.name,
        model=Pi0Config(pi05=True,action_dim=52,action_horizon=30,max_token_len=MAX_TOKENS),
        freeze_filter=nnx.Nothing,weight_loader=Weights(),batch_size=8,num_train_steps=STEPS,seed=SEED,
        fsdp_devices=8,ema_decay=None,wandb_enabled=False,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1,peak_lr=1e-6,decay_steps=20000,decay_lr=1e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),save_interval=1000)
    save('config.json',dict(method='full_finetune',all_parameters_trainable=True,lora=False,global_batch=8,
        gradient_accumulation=1,fsdp_devices=8,gpu_ids=list(range(8)),max_steps=STEPS,learning_rate=1e-6,
        warmup_steps=1,action_dim=52,action_horizon=30,max_token_len=MAX_TOKENS,seed=SEED,
        language='en',ema=False,train_frames=dataset.total_train,repair='gesture_hold_trim_and_online_elapsed_step_context',weight_source=str(ORIGINAL_RUN/'checkpoints/10000'),optimizer='original step10000 AdamW moments and counts restored; constant 1e-6 continuation',normalization='unchanged original training statistics',
        additional_optimizer_steps=STEPS-10000,equivalent_additional_frame_epochs=(STEPS-10000)*BATCH/dataset.total_train,
        model_source_commit=subprocess.check_output(['git','-C',str(ROOT/'openpi'),'rev-parse','HEAD'],text=True).strip(),
        sampling='uniform task then uniform episode then uniform start frame; replacement',
        image_preprocessing='PIL bilinear resize+pad224, official training augmentations',
        quantile_half_range_floor_rad=.04,state_clip=[-1,1],action_clip=False))
    mesh=sharding.make_mesh(8)
    data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    rng=jax.random.key(SEED);rng,init_rng=jax.random.split(rng)
    manager,resuming=checkpoints.initialize_checkpoint_dir(RUN/'checkpoints',keep_period=20,overwrite=False,resume=True)
    status('initializing_full_model',resume=resuming)
    state,state_shard=upstream.init_train_state(config,init_rng,mesh,resume=True);jax.block_until_ready(state)
    if resuming:
        state=restore_resharded(manager,state,state_shard)
        save('resume_8gpu.json',dict(passed=True,step=int(state.step),gpu_count=8,optimizer_restored=True,all_arrays_on_8gpu_mesh=True))
    else:
        import orbax.checkpoint as ocp
        source_dir=ORIGINAL_RUN/'checkpoints'
        metadata=source_dir/'10000/_CHECKPOINT_METADATA'
        before=hashlib.sha256(metadata.read_bytes()).hexdigest()
        source=ocp.CheckpointManager(source_dir,item_handlers={
            'assets':checkpoints.CallbackHandler(),
            'train_state':ocp.PyTreeCheckpointHandler(),
            'params':ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(read_only=True,create=False))
        state=restore_resharded(source,state,state_shard,step=10000)
        source.close()
        assert int(state.step)==10000
        leaves=jax.tree.leaves(state.opt_state)
        scalar_counts=[int(np.asarray(x)) for x in leaves if hasattr(x,'shape') and x.shape==() and np.issubdtype(x.dtype,np.integer)]
        assert scalar_counts and all(x==10000 for x in scalar_counts),scalar_counts
        assert before==hashlib.sha256(metadata.read_bytes()).hexdigest()
        save('optimizer_continuation_audit.json',dict(passed=True,source=str(source_dir/'10000'),
            state_step=int(state.step),optimizer_counts=scalar_counts,source_metadata_unchanged=True,
            source_opened_read_only=True,all_arrays_on_8gpu_mesh=True,learning_rate=1e-6))
    allp=state.params.to_pure_dict();trainp=state.params.filter(config.trainable_filter).to_pure_dict()
    all_count=sum(x.size for x in jax.tree.leaves(allp));train_count=sum(x.size for x in jax.tree.leaves(trainp))
    assert all_count==train_count
    save('trainable_parameter_audit.json',dict(all_parameters=all_count,trainable_parameters=train_count,frozen_parameters=0))
    train=jax.jit(functools.partial(step_fn,config),in_shardings=(replicated,state_shard,data_shard),
        out_shardings=(state_shard,replicated),donate_argnums=(1,))
    evaluate=jax.jit(eval_fn,in_shardings=(state_shard,replicated,data_shard),out_shardings=replicated)
    start_step=int(state.step);started=time.monotonic()
    status('compiling_first_train_step',step=start_step,target_steps=STEPS)
    for i in range(start_step,STEPS):
        batch=jax.device_put(dataset.batch(i),data_shard)
        begin=time.monotonic()
        with sharding.set_mesh(mesh):state,metrics=train(rng,state,batch)
        values={k:float(v) for k,v in jax.device_get(metrics).items()}
        assert all(np.isfinite(v) for v in values.values()),values
        step=i+1;elapsed=time.monotonic()-started
        if step<=start_step+3 or step%10==0:
            mem=[dict(gpu=d.id,**(d.memory_stats() or {})) for d in jax.devices()]
            row=dict(step=step,target_steps=STEPS,global_batch=8,equivalent_frame_epochs=step*BATCH/dataset.total_train,
                elapsed_seconds=elapsed,last_step_seconds=time.monotonic()-begin,**values)
            with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            status('training',**row,gpu_memory=mem);print(json.dumps(row),flush=True)
        if step==start_step+3:
            assert values['update_norm_vision']>0 and values['update_norm_language']>0 and values['update_norm_projection']>0
            save('full_gradient_check_8gpu.json',dict(passed=True,step=step,metrics=values))
        if step in [start_step+20] or step%1000==0 or step==STEPS:
            status('saving_checkpoint',step=step,target_steps=STEPS)
            checkpoints.save_state(manager,state,dataset,step);manager.wait_until_finished()
            save('checkpoint_status.json',dict(step=step,path=str(RUN/'checkpoints'/str(step)),saved=True))
            if step==start_step+20:
                restored=restore_resharded(manager,state,state_shard,step=step);jax.block_until_ready(restored)
                np.testing.assert_allclose(np.asarray(restored.params['action_out_proj']['bias'].value),
                                           np.asarray(state.params['action_out_proj']['bias'].value),atol=0,rtol=0)
                assert int(restored.step)==step
                state=restored
                save('checkpoint_roundtrip_8gpu.json',dict(passed=True,step=step))
        if step in [start_step+20] or step%1000==0 or step==STEPS:
            val={}
            for t in TASKS:
                batch=jax.device_put(dataset.batch(777,task=t),data_shard)
                with sharding.set_mesh(mesh):v=evaluate(state,jax.random.key(991),batch)
                val[t]=float(v)
            assert all(np.isfinite(x) for x in val.values()),val
            save(f'validation/step_{step:06d}.json',dict(step=step,per_task_flow_loss=val,
                macro_mean=float(np.mean(list(val.values()))),scope='held-out fixed sample flow matching loss, not success rate'))
    manager.wait_until_finished();status('complete',step=STEPS,target_steps=STEPS)
if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc());raise
