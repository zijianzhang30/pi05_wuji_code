"""Fail-closed checks and publication of synchronized training episodes."""
import argparse,json,hashlib,shutil,io,fcntl
from pathlib import Path
import numpy as np,h5py
from PIL import Image
P=Path(__file__).resolve().parents[2]
def checks(run):
 r=json.loads((run/'report.json').read_text());m=json.loads((run/'collection_manifest.json').read_text());c=json.loads((run/'contacts.json').read_text());task=m['task']
 assert r['status'].startswith('passed'),r.get('reason')
 assert r['physics_stepped'] and r['object_pose_commands']==0 and r['object_attachments']==0
 assert len(c)==r['frames']
 with h5py.File(run/'episode.hdf5') as f:
  n=len(f['action']);offset=n-len(c);assert offset==(360 if task=='produce_sort' else 0),(task,n,len(c))
  q=f['observations/qpos'][:];act=f['action'][:];assert q.shape==(n+1,52) and act.shape==(n,52)
  active=slice(0,26) if task=='microwave_open' else slice(26,52)
  assert np.allclose(q[offset+1:,active],np.array([x['q'] for x in c]),atol=1e-6),'qpos off by one'
  logged=np.array([x['target'] for x in c]);actual_commands=act[offset:,active]
  different=np.where(np.max(abs(actual_commands-logged),axis=1)>1e-6)[0]
  for i in different:
   # These controllers update their *next* target after the last physics substep.
   # The logger records the command actually applied; retain it unchanged.
   post_contact_arm_lock=(task in ['pick_place','stack'] and c[i]['phase'] in ['grasp','settle_grasp'] and np.allclose(logged[i,:6],np.array(c[i]['q'])[:6],atol=1e-6) and np.allclose(logged[i,6:],actual_commands[i,6:],atol=1e-6))
   post_contact_descend_stop=(task=='produce_sort' and c[i]['phase']=='descend' and np.allclose(logged[i],c[i]['q'],atol=1e-6))
   assert post_contact_arm_lock or post_contact_descend_stop,('unexplained action log discrepancy',int(i))
  for k in ['observations/qpos','observations/qvel','action','action_substeps','gravity_feedforward_substeps','controller/gains_kp_kd']:
   assert np.isfinite(f[k][:]).all(),k
  pose=f['diagnostics/body_pose_xyzw'][:];names=f['metadata/body_paths'].asstr()[:].tolist()
  traj=np.load(run/'trajectory.npz');assert np.allclose(pose[offset+1:],traj['poses'],atol=1e-6),'body pose time alignment'
  digest=hashlib.sha256(np.round(q[offset+30:],5).tobytes()+np.round(act[offset+30:],5).tobytes()).hexdigest()
 if task in ['pick_place','stack']:
  assert r['precontact_max_drift_m']<=.0001
  assert all(x['object_contact_N']<.05 and np.linalg.norm(x['object_velocity'][:3])<.03 for x in c[-15:]),'not released and stable'
 elif task=='produce_sort':
  assert max(r['precontact_drift_m'])<=.0001
  assert all(max(x['food_contact_N'])<.05 for x in c[-15:]),'not released'
  positions=np.array([x['food_poses'] for x in c[-15:]])[:,:,:3]
  assert np.linalg.norm(positions-positions[-1],axis=-1).max()<.01,'food not settled within 1cm'
 elif task=='piano':
  assert [e['key'] for e in r['events']]==[5,5,6,6,7,7,6,4],r['events']
  assert all(e['force_N']>.05 and e['travel_m']>.0008 for e in r['events'])
 elif task.startswith('gesture'):
  assert r['max_joint_tracking_error_rad']<.2
  assert len([x for x in c if x['phase']=='gesture'])>=30,'gesture hold duration'
 elif task=='microwave_open':
  assert r['four_finger_max_consecutive_frames']>=5 and r['door_drive_max_force']==0
  assert c[-1]['door_angle_deg']>60 and abs(c[-1]['door_velocity'])<.08
 start=offset+30 # initialization/settling is retained in raw run, excluded from training.
 assert n>start
 return dict(task=task,seed=m['seed'],start_frame=start,end_frame=n,transitions=n-start,observations=n-start+1,raw_transitions=n,contact_offset=offset,trajectory_sha256=digest,physics_status=r['status'],post_step_target_update_frames=different.tolist())
def geometry(run,gate):
 g=json.loads((run/'collection_geometry.json').read_text());assert g['checked_frames']>=gate['observations']
 bad=[x for x in g['failed_frames'] if x['frame']>=gate['start_frame']]
 assert not bad,('visible_mesh_intersections',len(bad),bad[:2])
 return dict(checked_frames=g['checked_frames'],task_frames_with_intersections=0,initialization_frames_with_intersections=len(g['failed_frames']),scope=g['scope'])
def publish(run,gate):
 rgb=run/'rgb/rgb.hdf5';root=P/'datasets/wuji_mcl350_v1';root.mkdir(parents=True,exist_ok=True)
 lock=open(root/'.publish.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 accepted=[json.loads(x.read_text()) for x in (root/gate['task']).glob('episode_*/acceptance.json')]
 assert len(accepted)<50,'task quota already met'
 assert gate['trajectory_sha256'] not in [x['trajectory_sha256'] for x in accepted],'duplicate trajectory'
 dest=root/gate['task']/f"episode_{gate['seed']:06d}";dest.mkdir(parents=True,exist_ok=True)
 out=dest/'episode.partial.hdf5';start,n=gate['start_frame'],gate['end_frame'];m=json.loads((run/'collection_manifest.json').read_text())
 with h5py.File(run/'episode.hdf5') as src,h5py.File(rgb) as images,h5py.File(out,'w') as dst:
  frames=images['source_frame'][:];assert np.array_equal(frames,np.arange(start,n+1)),frames.shape
  assert len(images['ego'])==gate['observations']
  for i in range(len(images['ego'])):
   im=np.array(Image.open(io.BytesIO(images['ego'][i].tobytes())).convert('RGB'));assert im.shape==(480,640,3)
   if i==0:assert im.std()>5,'blank RGB'
  for k in ['observations/qpos','observations/qvel']:
   dst.create_dataset(k,data=src[k][start:],compression='gzip')
  dst.create_dataset('observations/time',data=np.arange(n-start+1)/30)
  for k in ['action','action_substeps','gravity_feedforward_substeps','controller/gains_kp_kd']:
   dst.create_dataset(k,data=src[k][start:],compression='gzip')
  src.copy('metadata',dst)
  images.copy('ego',dst.require_group('observations/images'));images.copy('intrinsics',dst.require_group('camera'));images.copy('camera_to_world',dst['camera'])
  dst['observations/images/ego'].attrs.update(encoding='JPEG',color_order='RGB',quality=95)
  dst.create_dataset('source_frame',data=frames)
  paths=src['metadata/body_paths'].asstr()[:].tolist();indices=[paths.index('/World/Robot_'+s+'/'+s+'_palm_link') for s in ['left','right']]
  dst.create_dataset('observations/palm_pose_world_xyzw',data=src['diagnostics/body_pose_xyzw'][start:][:,indices],compression='gzip')
  dst.create_dataset('terminated',data=np.r_[np.zeros(n-start-1,dtype=bool),True]);dst.create_dataset('success',data=True)
  dst.attrs.update(dict(src.attrs));dst.attrs.update(qualification='accepted',initialization_frames_excluded=start,rgb_alignment='RGB[i] and qpos[i] precede action[i]; final RGB/qpos terminal',instruction=m['instructions'])
 out.replace(dest/'episode.hdf5')
 for name in ['report.json','collection_manifest.json','collection_provenance.json','collection_geometry.json','contacts.json']:
  assert (run/name).exists(),name
  shutil.copy2(run/name,dest/name)
 movie=list((run/'rgb/chunks').glob('*.mp4'));assert len(movie)==1
 shutil.copy2(movie[0],dest/'ego.mp4')
 gate.update(accepted=True,source_run=str(run),dataset_path=str(dest),rgb_frames=gate['observations'],split='validation' if gate['seed']%10==0 else 'train',limitations=['Narrow initial-condition distribution; scripted feedback demonstrations; not human teleoperation','Geometry checks cover recorded frames, not a continuous-time mathematical guarantee'])
 if gate['task']=='microwave_open':gate['limitations'].append('Four-finger grip/unlatch followed by passive door inertia; no sustained four-finger contact claim')
 for final in [dest/'acceptance.json',run/'dataset_acceptance.json']:
  temp=final.with_suffix('.tmp');temp.write_text(json.dumps(gate,indent=2));temp.replace(final)
 lock.close();return gate
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--stage',choices=['pre','geometry','publish'],default='pre');a=ap.parse_args();run=Path(a.run)
 try:
  gate=checks(run)
  if a.stage!='pre':gate['geometry']=geometry(run,gate)
  if a.stage=='publish':gate=publish(run,gate)
  (run/'collection_gate.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate))
 except Exception as e:
  (run/'collection_rejection.json').write_text(json.dumps({'stage':a.stage,'reason':str(e)},indent=2));raise
