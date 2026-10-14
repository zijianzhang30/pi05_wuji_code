from model import *
from dual_collision import DualCollision
import json
allrows=[]
source=P/'runs/physical_v1/pick_tip_v18_gpu1/initial_scene.usda'
audit=None
for i in range(4):
 for r in json.loads((P/f'runs/physical_v1/left_handoff_{i}.json').read_text()):
  if max(r['errors_m'])>.003:continue
  if audit is None:audit=DualCollision(source,r['right_q'])
  r['dual_hits']=audit.check(r['q']);r['candidate']=i;allrows.append(r);print(i,r['errors_m'],r['dual_hits'],flush=True)
allrows.sort(key=lambda r:(len(r['dual_hits']),max(r['errors_m'])))
(P/'runs/physical_v1/handoff_ranked.json').write_text(json.dumps(allrows,indent=2))
