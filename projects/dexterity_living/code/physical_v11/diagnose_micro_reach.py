import sys,json
from types import SimpleNamespace
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'physical_v10'))
from model import *
a=SimpleNamespace(thumb_weight=.005,other_weight=.01)
run=P/'runs/physical_v1/micro_around_v21_gpu0'
r=json.loads((run/'report.json').read_text());c=next(x for x in json.loads((run/'contacts.json').read_text()) if x['phase']=='grip_hold')
lock_q=dict(zip(r['dof_names'],c['q']));hinge=np.array([-2.2972766,-3.4886971,.9657585])
feedback_names=arm_names('right')+[n for n in hand_names('right') if any(f'finger{i}_' in n for i in [1,2,3,4])];feedback_pull=[];previous=np.array([lock_q[n] for n in feedback_names]);tip_links=[f'right_finger{i}_tip_link' for i in [1,2,3,4,5]];offsets={l:np.array([0,0,-.0030825 if 'finger1_' in l else -.00755]) for l in tip_links};tips0=np.array([fk(l,'right',lock_q)[:3,3]+fk(l,'right',lock_q)[:3,:3]@offsets[l] for l in tip_links]);palmR0=fk('right_palm_link','right',lock_q)[:3,:3];midlinks=['right_finger2_link3','right_finger3_link3'];mid0=np.array([fk(l,'right',lock_q)[:3,3] for l in midlinks])
for theta in np.linspace(0,-np.deg2rad(45),91):
 R=Rot.from_euler('z',theta).as_matrix();targets=(tips0-hinge)@R.T+hinge;targets[0]+=R@np.array([.04,-.025,0])*min(1,abs(theta)/np.deg2rad(5))
 midbase=mid0-hinge;clearance=min(1,abs(theta)/np.deg2rad(5))
 def residual(x):
  qq=lock_q.copy();qq.update(zip(feedback_names,x));poses_fk=fk_many(tip_links+midlinks+['right_palm_link'],'right',qq);got=np.array([poses_fk[l][:3,3]+poses_fk[l][:3,:3]@offsets[l] for l in tip_links]);rotation=poses_fk['right_palm_link'][:3,:3];orientation=Rot.from_matrix((R@palmR0)@rotation.T).as_rotvec();mids=np.array([poses_fk[l][:3,3] for l in midlinks]);return np.r_[np.r_[((got[1:3]-targets[1:3])[:,:2]).ravel(),(got[1,2]-got[2,2])-(targets[1,2]-targets[2,2]),np.sign(np.mean((got[1:3]-targets[1:3])[:,2]))*max(0,abs(np.mean((got[1:3]-targets[1:3])[:,2]))-.020)],np.maximum(0,midbase[:,0]+.003*clearance-((mids-hinge)@R)[:,0]),np.maximum(0,((mids-hinge)@R)[:,1]-midbase[:,1]+.002*clearance),max(0,.028-(mids[0,2]-mids[1,2])),a.thumb_weight*(got[0]-targets[0]).ravel(),a.other_weight*(got[[3,4]]-targets[[3,4]]).ravel(),.001*orientation,.0005*(x-previous)]
 opt=least_squares(residual,np.clip(previous,[joints[n]['lo']+1e-6 for n in feedback_names],[joints[n]['hi']-1e-6 for n in feedback_names]),bounds=([joints[n]['lo']+1e-6 for n in feedback_names],[joints[n]['hi']-1e-6 for n in feedback_names]),max_nfev=250);
 if np.linalg.norm(residual(opt.x)[:11])>=.003:
  print('REACH_RESIDUAL',float(np.rad2deg(theta)),residual(opt.x)[:11].tolist(),[(n,round(float(x),3)) for n,x in zip(feedback_names,opt.x) if abs(x-joints[n]['lo'])<.002 or abs(x-joints[n]['hi'])<.002],flush=True)
 feedback_pull.append(opt.x.tolist());previous=opt.x