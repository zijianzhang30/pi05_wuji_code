from pathlib import Path
import sys,json,io,numpy as np,pyarrow.parquet as pq
from PIL import Image
S=Path(__file__).resolve().parents[1];R=S.parents[1];sys.path.insert(0,str(S/'code'))
from adapter import Adapter
adapter=Adapter(S);reports={}
for task in adapter.prompts:
 p=R.parent/'act_wuji/datasets_v21'/task/'data/chunk-000/episode_000045.parquet'
 row=pq.read_table(p,columns=['observation.state','action','observation.images.ego']).slice(0,1).to_pydict()
 q=np.asarray(row['observation.state'][0],np.float32);im=np.asarray(Image.open(io.BytesIO(row['observation.images.ego'][0]['bytes'])).convert('RGB'))
 obs=adapter.input(task,q,im,False)
 lo=np.asarray(adapter.norm['state']['q01']);hi=np.asarray(adapter.norm['state']['q99'])
 np.testing.assert_allclose(obs['state'],(2*(q-lo)/(hi-lo)-1).clip(-1,1),atol=1e-6)
 a=np.tile(np.asarray(row['action'][0],np.float32),(30,1));lo=np.asarray(adapter.norm['actions']['q01']);hi=np.asarray(adapter.norm['actions']['q99'])
 np.testing.assert_allclose(adapter.output(2*(a-lo)/(hi-lo)-1),a,atol=1e-6)
 assert obs['image']['base_0_rgb'].shape==(224,224,3) and obs['tokenized_prompt'].shape==(320,)
 assert obs['image_mask']['base_0_rgb'] and not obs['image_mask']['left_wrist_0_rgb']
 reports[task]={'tokens':int(obs['tokenized_prompt_mask'].sum()),'state_and_action_roundtrip':True}
out={'passed':True,'tasks':reports,'scope':'CPU preprocessing only; GPU checkpoint inference and simulation follow after training completes'}
(S/'preflight/adapter_cpu.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
