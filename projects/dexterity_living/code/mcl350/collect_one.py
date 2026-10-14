"""One isolated candidate through dynamics, recorded-frame mesh, RGB, schema gates."""
import argparse,json,subprocess,sys,os,signal,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parents[2];C=P/'code/mcl350'
ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--name',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--existing',action='store_true');a=ap.parse_args();run=P/'runs/physical_v1'/a.name
state=P/'reports/mcl350';state.mkdir(parents=True,exist_ok=True)
def status(stage,**kw):
 d=dict(task=a.task,seed=a.seed,name=a.name,gpu=a.gpu,stage=stage,time=time.time(),**kw);tmp=state/(a.name+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(state/(a.name+'.json'));print(json.dumps(d),flush=True)
def call(script,args,timeout=600):
 log=state/(a.name+'_'+Path(script).stem+'.log')
 with log.open('w') as h:
  p=subprocess.Popen([sys.executable,str(C/script),*map(str,args)],stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('timeout '+str(log))
 assert code==0,(script,code,str(log))
try:
 if not a.existing:
  status('simulating');call('run_seeded_episode.py',['--task',a.task,'--seed',a.seed,'--name',a.name,'--gpu',a.gpu],600)
 status('checking_dynamics');call('qualify_episode.py',[run]);gate=json.loads((run/'collection_gate.json').read_text())
 status('checking_geometry');call('audit_full_geometry.py',[run,'--task-pairs','--start-frame',gate['start_frame'],'--output','collection_geometry.json'],900)
 call('qualify_episode.py',[run,'--stage','geometry'])
 status('building_render_scene');call('build_solver_replay.py',[run,'--task',a.task,'--output',run/'render_scene'/a.task])
 status('rendering_rgb');call('render_rgb.py',[a.task,'--root',run/'render_scene','--solver-run',run,'--gpu',a.gpu,'--start',gate['start_frame'],'--end',gate['end_frame'],'--output-dir',run/'rgb','--width',640,'--height',480,'--spp',32,'--realtime'],3600)
 status('validating_and_exporting');call('qualify_episode.py',[run,'--stage','publish'],600)
 status('accepted',acceptance=json.loads((run/'dataset_acceptance.json').read_text()))
except Exception as e:
 status('rejected',error=str(e),traceback=traceback.format_exc());raise
