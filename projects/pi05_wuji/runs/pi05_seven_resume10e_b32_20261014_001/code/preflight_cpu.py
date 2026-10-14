from pathlib import Path
import ast,os,json,time,hashlib
T=Path('/media/external/bowenjing/projects/pi05_wuji/runs/pi05_seven_resume10e_b32_20261014_001')
os.environ['WUJI_RUN_DIR']=str(T)
source=(T/'code/train_full_8gpu.py').read_text();tree=ast.parse(source)
allow={'ROOT','RUN','BATCH','HORIZON','MAX_TOKENS','STEPS','SEED','ORIGINAL_RUN','CONTINUE_RUN','RESUME_STEP','PRIOR_SAMPLES','TARGET_FRAME_EPOCHS','TRAIN_FRAME_DENOMINATOR','TRIM','DATA','TASKS','PROMPTS'}
nodes=[]
for node in tree.body:
 if isinstance(node,(ast.Import,ast.ImportFrom)):nodes.append(node)
 elif isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in allow for t in node.targets):nodes.append(node)
 elif isinstance(node,ast.FunctionDef) and node.name in ['save','status','load_episode']:nodes.append(node)
 elif isinstance(node,ast.ClassDef) and node.name=='Dataset':nodes.append(node)
g={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(T/'code/train_full_8gpu.py'),'exec'),g)
g['RUN']=T/'preflight/cpu';g['RUN'].mkdir(parents=True,exist_ok=True)
np=g['np'];ds=g['Dataset'].__new__(g['Dataset']);ds.eps={t:{'train':[],'validation':[]} for t in g['TASKS']}
raw=json.loads((T/'normalization/norm_stats.json').read_text())['norm_stats']
ds.stats={k:g['NormStats'](**{n:np.asarray(v) for n,v in value.items()}) for k,value in raw.items()}
ds.tokenizer=g['PaligemmaTokenizer'](max_len=320)
refs=[]
for t in g['TASKS']:
 split=json.loads((g['DATA']/t/'split_manifest.json').read_text());info=json.loads((g['DATA']/t/'meta/info.json').read_text())
 for idx in [split['train'][0],split['validation'][0]]:
  ref=next(x for x in split['episodes'] if x['episode_index']==idx);e=g['load_episode']((t,ref,info));ds.eps[t][e['split']].append(e)
  e['qn']=ds.normalize(e['q'],'state').clip(-1,1);e['an']=ds.normalize(e['a'],'actions')
  values=[ds.tokenizer.tokenize(g['instruction'](g['PROMPTS'][t],i),q) for i,q in enumerate(e['qn'])]
  e['tokens']=np.stack([x[0].astype(np.int32) for x in values]);e['masks']=np.stack([x[1] for x in values])
  refs.append(dict(task=t,episode_index=idx,split=ref['split'],frames=len(e['q'])))
ds.test_batch_contract();obs,a,mask=ds.batch(17000)
assert a.shape==(32,30,52) and obs.state.shape==(32,52) and mask.shape==(32,30)
assert np.isfinite(a).all() and np.isfinite(obs.state).all() and obs.image_masks['base_0_rgb'].all()
assert g['BATCH']%8==0
result=dict(passed=True,global_batch=32,per_gpu_batch=4,action_shape=list(a.shape),state_shape=list(obs.state.shape),invalid_terminal_actions_excluded=True,actual_episode_refs=refs,training_code_sha256=hashlib.sha256(source.encode()).hexdigest(),jax_backend=g['jax'].default_backend(),scope='Executed actual Dataset methods on14 real train/heldout episodes; CPU only, no GPU model or optimizer restore claimed',created_at=time.time())
(T/'preflight/cpu_contract.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
