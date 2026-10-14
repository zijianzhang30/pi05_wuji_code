from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':7,'limit_cpu_threads':8})
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
try:
 omni.usd.get_context().open_stage('/media/external/bowenjing/projects/dexterity_living/runs/physical_v1/micro_tug_v23_gpu4/initial_scene.usda')
 w=World(physics_dt=1/960,rendering_dt=1/30,physics_prim_path='/World/PhysicsScene');m=w.scene.add(SingleArticulation('/World/TaskMicrowave/E_body_9',name='micro'));w.reset()
 print('RUNTIME_NAMES',m.dof_names,flush=True);print('RUNTIME_GAINS',m.get_articulation_controller().get_gains(),flush=True)
 v=m._articulation_view
 for name in ['get_max_efforts','get_friction_coefficients','get_joint_positions','get_joint_velocities','get_body_masses','get_body_inertias']:
  try: print('RUNTIME_'+name,getattr(v,name)(),flush=True)
  except Exception as e:print(name,type(e).__name__,str(e),flush=True)
except Exception:
 import traceback;traceback.print_exc()
finally:app.close()
