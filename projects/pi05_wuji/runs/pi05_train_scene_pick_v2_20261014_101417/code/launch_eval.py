"""Run one Isaac episode using the persistent frozen pi05 server."""
import argparse,subprocess,os,time,json,fcntl,shlex
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--attempt',type=int,default=1);ap.add_argument('--horizon',type=int,default=30);a=ap.parse_args();assert 1<=a.horizon<=30 and a.gpu in range(1,8)
S=Path(__file__).resolve().parents[1];R=S.parents[1];P=R.parent/'dexterity_living';code=S/'code'
name=f'pi05_step37000_five_tasks_x10_20261014_0925_{a.task}_{a.seed}_h30_a{a.attempt}';out=S/'episodes'/name;out.mkdir(parents=True,exist_ok=False)
ready=json.loads((S/'policy_ready.json').read_text());sock=ready['socket'];env=os.environ.copy()
for k in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','XLA_FLAGS']:env.pop(k,None)
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
gpu_lock=open(P/f'reports/process_manager/gpu{a.gpu}.lock','a')
for _ in range(1440):
 try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_UN);break
 except BlockingIOError:time.sleep(5)
else:raise RuntimeError('GPU lock wait timeout')
receipt={'task':a.task,'seed':a.seed,'gpu':a.gpu,'policy_pid':ready['pid'],'policy_gpu':0,'name':name,'run':str(P/'runs/physical_v1'/name),'started_at':time.time()}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2))
cmd='source '+shlex.quote(str(P/'scripts/pathtrace_env.sh'))+'\nexec '+shlex.join(['/home/bowenjing/envs/isaaclab3/bin/python',str(code/'run_eval.py'),'--task',a.task,'--seed',str(a.seed),'--gpu',str(a.gpu),'--name',name,'--socket',sock,'--horizon',str(a.horizon)])
with (out/'simulation.log').open('w') as log:
 sim=subprocess.Popen(['/bin/bash','-c',cmd],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=P)
 receipt['simulation_pid']=sim.pid;(out/'receipt.json').write_text(json.dumps(receipt,indent=2));started=time.monotonic()
 while sim.poll() is None:
  time.sleep(3);text=(out/'simulation.log').read_text(errors='replace')
  if '[crash]' in text or time.monotonic()-started>900 or (time.monotonic()-started>120 and not (P/'runs/physical_v1'/name/'eval_status.json').exists()):
   receipt['infrastructure_abort']='native_simulator_crash' if '[crash]' in text else 'simulator_timeout'
   (out/'receipt.json').write_text(json.dumps(receipt,indent=2));sim.terminate()
   try:sim.wait(timeout=5)
   except subprocess.TimeoutExpired:sim.kill();sim.wait()
   break
report=P/'runs/physical_v1'/name/'report.json'
result=json.loads(report.read_text()) if report.exists() else {'status':'infrastructure_error','success':None,'reason':'no_report'}
result['simulation_exit_code']=sim.returncode
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
