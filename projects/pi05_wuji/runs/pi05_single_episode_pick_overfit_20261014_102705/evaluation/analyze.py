from pathlib import Path
import json,numpy as np,h5py,datetime
from zoneinfo import ZoneInfo
N=Path(__file__).resolve().parent
m=json.loads((N/'manifest.json').read_text());ref=Path(m['reference']['source'])
with h5py.File(ref/'episode.hdf5') as f:
 q=f['observations/qpos'][:];a=f['action'][:];paths=f['metadata/body_paths'].asstr()[:].tolist();acceptance=json.loads((ref/'acceptance.json').read_text());recorded=np.load(Path(acceptance['source_run'])/'observation_trajectory.npz');poses=recorded['poses'][acceptance['start_frame']:acceptance['end_frame']+1]
groups={'left_arm':list(range(6)),'left_hand':list(range(6,26)),'right_arm':list(range(26,32)),'right_hand':list(range(32,52))}
out=dict(snapshot_beijing=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),checkpoint_step=38000,diagnostic_only=True,reference=m['reference'],teacher_forced=json.loads((N/'offline/summary.json').read_text()))
for mode in ['pi05','expert']:
 rr=json.loads((N/'results'/f'{mode}.json').read_text());run=Path(rr['run']);d=np.load(run/'policy_rollout.npz');tr=np.load(run/'trajectory.npz');c=json.loads((run/'contacts.json').read_text())
 n=min(len(a),len(d['actions']));firstcontact=next((x['frame'] for x in c if x['object_contact_N']>.01),None)
 counts=sum(x['object_contact_N']>.01 for x in c)
 data=dict(success=rr['success'],reason=rr['reason'],frames=rr['frames'],run=str(run),first_object_contact_frame=firstcontact,object_contact_frames=counts,peak_object_contact_N=max(x['object_contact_N'] for x in c),max_object_z_m=max(x['max_object_z_m'] for x in c),reset_alignment=rr['reset_alignment'],video=rr['video'],joint_error_vs_expert={})
 for key,ix in groups.items():
  diff=abs(d['actions'][:n,ix]-a[:n,ix]);state=abs(d['observations'][:n,ix]-q[:n,ix])
  data['joint_error_vs_expert'][key]=dict(command_mae_rad=float(diff.mean()),command_p95_abs_rad=float(np.quantile(diff,.95)),state_mae_rad=float(state.mean()),precontact_command_mae_rad=float(diff[:min(165,n)].mean()))
 pi=list(tr['body_paths']).index('/World/Robot_right/right_palm_link');ri=paths.index('/World/Robot_right/right_palm_link')
 mm=np.linalg.norm(tr['poses'][:n,pi,:3]-poses[1:n+1,ri,:3],axis=-1)*1000
 data['measured_palm_position_error_vs_expert_mm']=dict(mean=float(mm.mean()),p95=float(np.quantile(mm,.95)),precontact_mean=float(mm[:min(165,n)].mean()),at_frame_150=float(mm[150]) if n>150 else None,at_frame_165=float(mm[165]) if n>165 else None)
 data['geometry']=rr.get('geometry');data['simulation_infrastructure_errors']=rr.get('simulation_infrastructure_errors',[])
 out[mode]=data
(N/'ANALYSIS.json').write_text(json.dumps(out,indent=2))
tf=out['teacher_forced'];p=out['pi05'];e=out['expert']
lines=[f"# π0.5 training-scene diagnostic",f"Snapshot Beijing: {out['snapshot_beijing']}",f"Frozen step38000; pick_place train episode0 seed12001. This is a training-scene diagnostic, not a held-out success-rate estimate.","",f"Pi05: success={p['success']}, reason={p['reason']}, object-contact frames={p['object_contact_frames']}, frames={p['frames']}.",f"Expert endpoint comparator: success={e['success']}, reason={e['reason']}, object-contact frames={e['object_contact_frames']}, frames={e['frames']}.","","Actual training-input teacher-forced joint errors (four noise draws at23 recorded observation frames):"]
for k,v in tf['groups'].items():lines.append(f"- {k}: first-action MAE {v['first_action_mae_rad']:.6f}rad; full valid-chunk MAE {v['chunk_mae_rad']:.6f}rad.")
lines+=["",f"Teacher-forced target-FK position differences: {json.dumps(tf['kinematic_target_error'])}",f"Pi05 reset checks: {json.dumps(p['reset_alignment']['checks'])}",f"Expert reset checks: {json.dumps(e['reset_alignment']['checks'])}","",f"Pi05 measured palm positional difference versus demonstrated physical trajectory: {json.dumps(p['measured_palm_position_error_vs_expert_mm'])}","", "Interpretation limits: one training scene and one closed-loop noise draw; offline prediction uses actual recorded RGB/qpos, live policy uses rendered RGB and its own state. Expert endpoint playback is a control, never a model success. Image mismatch must be assessed before attributing closed-loop failure entirely to policy fit. Effective contact/geometry criteria were not relaxed."]
(N/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,indent=2))
