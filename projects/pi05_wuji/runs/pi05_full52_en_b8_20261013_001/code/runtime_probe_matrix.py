from pathlib import Path
import os,subprocess,json,time,signal
p=Path('/media/external/bowenjing/projects/pi05_wuji');r=p/'runs/pi05_full52_en_b8_20261013_001'
base=os.environ.copy()
for k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','JAX_PLATFORMS','NCCL_P2P_DISABLE','NCCL_IB_DISABLE','NCCL_CUMEM_ENABLE','NCCL_CUMEM_HOST_ENABLE','NCCL_PROTO','XLA_FLAGS']:base.pop(k,None)
base.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4',NCCL_DEBUG='WARN')
cases=[('preallocate_default',{'XLA_PYTHON_CLIENT_PREALLOCATE':'true','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.85'}),('no_preallocate_no_xla_optimizations',{'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_FLAGS':'--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer='})]
results=[]
for name,options in cases:
 env=base|options
 log=(r/(name+'.log')).open('w')
 proc=subprocess.Popen([str(p/'env/bin/python'),str(r/'code/probe_imports_big.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=proc.wait(timeout=90)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
 log.close();text=(r/(name+'.log')).read_text()
 results.append({'case':name,'exit_code':code,'passed':'BIG_COLLECTIVE_PASSED' in text,'options':options})
 (r/'runtime_probe_results.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(results[-1]),flush=True)
