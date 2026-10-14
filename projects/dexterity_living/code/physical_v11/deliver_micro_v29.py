import subprocess,json
from pathlib import Path
P=Path('/media/external/bowenjing/projects/dexterity_living');py='/home/bowenjing/envs/isaaclab3/bin/python';run=P/'runs/physical_v1/micro_left_hook_v29_gpu2'
# Full geometry audit already completed for the unchanged trajectory.
d=json.loads((run/'full_geometry_audit.json').read_text());assert d['frames_with_intersections']==0,d
subprocess.run([py,str(P/'code/collection/render_contact_suite.py'),'--tasks','microwave_open','--root',str(P/'runs/contact_suite_v29_20261011'),'--mapping-json',str(P/'reports/micro_v29_render_mapping.json'),'--gpus','0','1','2','3','4','5','6','7','--chunks','8'],check=True)
