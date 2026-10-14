from pathlib import Path
import json,os,subprocess,fcntl
root=Path('/media/external/bowenjing/projects/pi05_wuji')
run=Path(__file__).resolve().parent
lock=(run/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=run/'launch_receipt.json'
if receipt.exists():
 old=json.loads(receipt.read_text());pid=old['pid']
 cmd=Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and str(run/'code/train_full.py').encode() in cmd.read_bytes():
  raise SystemExit(f'Already running PID {pid}')
assert json.loads((root/'weights_integrity.json').read_text())['verified']
env=os.environ.copy()
for k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','JAX_PLATFORMS']:env.pop(k,None)
env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',WUJI_RUN_DIR=str(run),OPENPI_DATA_HOME=str(root/'cache/openpi'),XLA_PYTHON_CLIENT_MEM_FRACTION='0.85',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',JAX_COMPILATION_CACHE_DIR=str(root/'cache/jax'),PYTHONUNBUFFERED='1')
env.update(LD_PRELOAD=str(root/'runtime_overrides/nccl_2289/nvidia/nccl/lib/libnccl.so.2'))
with (run/'train.log').open('a') as log:
 proc=subprocess.Popen([str(root/'env/bin/python'),str(run/'code/train_full.py')],cwd=root/'openpi',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d={'pid':proc.pid,'run':str(run),'gpu_ids':[0,1,2,3]}
receipt.write_text(json.dumps(d,indent=2));(root/'multitask_full_ft/active_run.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d))
