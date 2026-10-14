from pathlib import Path
import sys,json,subprocess,concurrent.futures,numpy as np
p=Path('/media/external/bowenjing/projects/dexterity_living');py=sys.executable;run=Path(sys.argv[1]);n=len(np.load(run/'trajectory.npz')['poses']);workers=8
def chunk(i):
 name=f'geometry_chunk_{i}.json'
 with open(run/f'geometry_chunk_{i}.log','w') as log:
  subprocess.run([py,str(p/'code/physical_v11/audit_full_geometry.py'),str(run),'--task-pairs','--start-frame',str(n*i//workers),'--end-frame',str(n*(i+1)//workers),'--output',name],stdout=log,stderr=subprocess.STDOUT,check=True)
 return json.loads((run/name).read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:reports=list(pool.map(chunk,range(workers)))
failed=sorted([r for part in reports for r in part['failed_frames']],key=lambda r:r['frame']);counts={}
for part in reports:
 for key,v in part['intersection_counts'].items():counts[key]=counts.get(key,0)+v
result=dict(checked_frames=sum(r['checked_frames'] for r in reports),frames_with_intersections=len(failed),intersection_counts=counts,failed_frames=failed,scope='Recorded 30Hz frames: robot/task/table; inter-arm, finger self; free task objects against each other and table. Adjacent mechanical joints excluded; no continuous-time guarantee.')
assert result['checked_frames']==n
for name in ['full_geometry_audit.json','task_geometry_audit.json']:(run/name).write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!='failed_frames'},flush=True)
