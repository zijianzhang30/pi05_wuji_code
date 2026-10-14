import subprocess
from pathlib import Path
P=Path('/media/external/bowenjing/projects/dexterity_living');py='/home/bowenjing/envs/isaaclab3/bin/python'
subprocess.run([py,str(P/'code/physical_v1/build_solver_replay.py'),str(P/'runs/physical_v1/micro_left_hook_v27_gpu2'),'--task','microwave_open','--output',str(P/'runs/micro_debug_v27/microwave_open')],check=True)
subprocess.run([py,str(P/'code/collection/render_rgb.py'),'microwave_open','--root',str(P/'runs/micro_debug_v27'),'--solver-run',str(P/'runs/physical_v1/micro_left_hook_v27_gpu2'),'--gpu','1','--start','30','--end','330','--pilot','--width','1280','--height','960','--spp','64','--realtime'],check=True)
