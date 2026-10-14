"""Collection-aligned produce goal and release/stability window.

Source: physical_v11/run_produce_surface_validated.py final goal;
mcl350/qualify_episode.py checks(): last15 release and position displacement.
Contact/penetration/drift gates and recorded-frame geometry remain separate.
"""
import numpy as np

def produce_collection_criteria(position_history,goals,peaks,contact_history):
 positions=np.asarray(position_history,dtype=float)[-15:,:,:3]
 contacts=np.asarray(contact_history,dtype=float)[-15:]
 final=positions[-1];goals=np.asarray(goals,dtype=float)
 goal=bool(np.all(np.linalg.norm(final[:,:2]-goals[:,:2],axis=1)<.065) and np.all((final[:,2]>.69)&(final[:,2]<.80)) and np.all(np.asarray(peaks)>.79))
 released=bool(len(contacts)==15 and np.all(contacts<.05))
 drift=float(np.linalg.norm(positions-final,axis=-1).max())
 stable=bool(len(positions)==15 and drift<.01)
 return {'success':goal and released and stable,'goal':goal,'released_last15':released,'settled_last15':stable,'max_last15_displacement_m':drift}
