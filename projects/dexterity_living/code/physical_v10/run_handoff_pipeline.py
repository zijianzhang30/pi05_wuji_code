from pathlib import Path
import subprocess,json
P=Path(__file__).resolve().parents[2];D=P/'code/physical_v10';py='/home/bowenjing/envs/isaaclab3/bin/python'
for script,extra in [('plan_left_folded.py',['--direction','0','--orientation-weight','0','--tag','left_folded']),('plan_left_folded_wide.py',['--direction','0','--orientation-weight','0','--tag','left_folded_wide']),('plan_handoff_approach.py',[])]:
 print('STAGE',script,flush=True);subprocess.run([py,str(D/script),*extra],check=True)
jobs=[]
for gpu,force in [(7,2.5),(6,3.5)]:
 jobs.append(dict(id=f'handoff_v1_gpu{gpu}',gpu=gpu,script=str(D/'run_handoff.py'),args=['--name',f'handoff_v1_gpu{gpu}','--gpu',str(gpu),'--contact-stop','.3','--hold-force',str(force)],timeout_seconds=900))
f=P/'reports/handoff_physics_jobs.json';f.write_text(json.dumps(jobs,indent=2))
subprocess.run(['python3',str(P/'code/collection/process_manager.py'),'queue','--file',str(f)],check=True)
