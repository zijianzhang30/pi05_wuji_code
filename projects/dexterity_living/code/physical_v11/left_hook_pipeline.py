
import time,json,subprocess
from pathlib import Path
P=Path('/media/external/bowenjing/projects/dexterity_living')
files=[P/f'runs/physical_v1/micro_left_hook_plan_{i}/plan.json' for i in range(2)]
for _ in range(300):
 if all(p.exists() for p in files):break
 time.sleep(2)
else:raise RuntimeError('planner did not finish')
subprocess.run(['/home/bowenjing/envs/isaaclab3/bin/python',str(P/'code/physical_v11/audit_micro_left_hook_plan.py')],check=True)
jobs=[]
for i in range(2):
 p=files[i].parent;d=json.loads((p/'geometry_only/full_geometry_audit.json').read_text())
 print('GEOMETRY',i,d['frames_with_intersections'],d['intersection_counts'],flush=True)
 if d['frames_with_intersections']:continue
 for j,(force,lead) in enumerate([(4,.8),(7,1.2)]):
  gpu=i*4+j*2;name=f'micro_left_hook_v26_gpu{gpu}'
  jobs.append(dict(id=name,gpu=gpu,script=str(P/'code/physical_v11/run_micro_left_hook_arc.py'),args=['--name',name,'--plan',str(files[i]),'--gpu',str(gpu),'--force',str(force),'--lead',str(lead)],timeout_seconds=1200))
mf=P/'reports/v26_left_hook.json';mf.write_text(json.dumps(jobs,indent=2));subprocess.run(['python3',str(P/'code/collection/process_manager.py'),'queue','--file',str(mf)],check=True)
