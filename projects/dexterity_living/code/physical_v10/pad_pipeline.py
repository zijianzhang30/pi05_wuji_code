from pathlib import Path
import subprocess,sys
P=Path('/media/external/bowenjing/projects/dexterity_living');py='/home/bowenjing/envs/isaaclab3/bin/python';idx=int(sys.argv[1]);D=P/'code/physical_v8'
subprocess.run([py,str(D/f'plan_pad_approach_{idx}.py')],check=True)
subprocess.run([py,str(D/'run_pick_pad.py'),'--name',f'pick_pad_v13_gpu{idx}','--gpu',str(idx),'--candidate',str(idx),'--pinch','0','--move-seconds','3'],check=True)
