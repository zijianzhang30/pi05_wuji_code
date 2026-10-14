"""Record pre-action observations and actual motor commands, not replayed object motion."""
import json, hashlib
from pathlib import Path
import numpy as np

class Recorder:
    def __init__(self, out, task, seed, specification):
        self.out=Path(out); self.task=task; self.seed=seed; self.spec=specification
        self.robots={}; self.commands={}; self.efforts={}; self.observations=[]
        self.targets=[]; self.feedforward=[]; self.gains=[]; self.counter=0; self.view=None
        self.rng=np.random.default_rng(seed+70000); self.scene_changes=[]
    def install(self):
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        original_init=SingleArticulation.__init__; original_apply=SingleArticulation.apply_action
        original_reset=World.reset; original_step=World.step
        rec=self
        def init(robot,*args,**kwargs):
            original_init(robot,*args,**kwargs)
            path=str(args[0] if args else kwargs.get('prim_path',''))
            for side in ['left','right']:
                if path=='/World/Robot_'+side: rec.robots[side]=robot
        def apply(robot,action,*args,**kwargs):
            for side,r in rec.robots.items():
                if r is robot:
                    assert action.joint_indices is None, 'Partial actions need explicit joint mapping'
                    rec.commands[side]=np.asarray(action.joint_positions,dtype=np.float32).copy()
                    rec.efforts[side]=np.asarray(action.joint_efforts,dtype=np.float32).copy()
            return original_apply(robot,action,*args,**kwargs)
        def reset(world,*args,**kwargs):
            rec.prepare_scene(); result=original_reset(world,*args,**kwargs)
            rec.dt=float(world.get_physics_dt()); rec.decimation=round(1/(30*rec.dt))
            assert abs(rec.decimation*rec.dt-1/30)<1e-8
            rec.commands.clear(); return result
        def step(world,*args,**kwargs):
            recording=len(rec.commands)==2
            if recording:
                if rec.view is None: rec.initialize_view(world)
                if rec.counter==0: rec.observations.append(rec.observe())
                if rec.counter%rec.decimation==0:
                    rec.gains.append(np.concatenate([np.stack(rec.robots[s].get_articulation_controller().get_gains(),axis=0) for s in ['left','right']],axis=-1))
                rec.targets.append(np.concatenate([rec.commands[s] for s in ['left','right']]))
                rec.feedforward.append(np.concatenate([rec.efforts[s] for s in ['left','right']]))
            result=original_step(world,*args,**kwargs)
            if recording:
                rec.counter+=1
                if rec.counter%rec.decimation==0: rec.observations.append(rec.observe())
            return result
        SingleArticulation.__init__=init; SingleArticulation.apply_action=apply
        World.reset=reset; World.step=step
    def prepare_scene(self):
        import omni.usd
        from pxr import UsdGeom,Gf
        stage=omni.usd.get_context().get_stage()
        roots=[]
        if self.task in ['pick_place','stack']: roots=['/World/TaskObjects/object']
        if self.task=='produce_sort': roots=['/World/TaskFood_orange','/World/TaskFood_pepper']
        for path in roots:
            prim=stage.GetPrimAtPath(path); assert prim,path
            # Small object placement variation in the frozen workcell, only before reset.
            delta=np.r_[self.rng.uniform(-.003,.003,2),0.]
            xf=UsdGeom.Xformable(prim); T=np.array(xf.GetLocalTransformation()).T
            T[:3,3]+=delta; xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='collection_init').Set(Gf.Matrix4d(T.T.tolist()))
            self.scene_changes.append(dict(path=path,translation_m=delta.tolist()))
        stage.GetRootLayer().Save()
    def initialize_view(self,world):
        import omni.usd
        from pxr import UsdPhysics
        stage=omni.usd.get_context().get_stage()
        self.body_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.view=world.physics_sim_view.create_rigid_body_view(self.body_paths)
        assert list(self.view.prim_paths)==self.body_paths
        self.names={s:list(self.robots[s].dof_names) for s in ['left','right']}
    def observe(self):
        return dict(q=np.concatenate([self.robots[s].get_joint_positions() for s in ['left','right']]),
                    qvel=np.concatenate([self.robots[s].get_joint_velocities() for s in ['left','right']]),
                    body=self.view.get_transforms().copy())
    def finish(self):
        self.out.mkdir(parents=True,exist_ok=True)
        if not self.observations:
            (self.out/'recording_error.json').write_text(json.dumps({'error':'No control observations recorded'}));return
        import h5py
        assert self.counter%self.decimation==0, 'Partial control window'
        n=self.counter//self.decimation
        assert len(self.observations)==n+1
        q=np.stack([x['q'] for x in self.observations]); qv=np.stack([x['qvel'] for x in self.observations])
        poses=np.stack([x['body'] for x in self.observations]); commands=np.asarray(self.targets).reshape(n,self.decimation,-1)
        ff=np.asarray(self.feedforward).reshape(n,self.decimation,-1); gains=np.asarray(self.gains)
        assert q.shape[1]==52 and commands.shape[-1]==52
        assert all(np.isfinite(x).all() for x in [q,qv,commands,ff,gains,poses])
        instruction=self.spec['instructions'][self.task]
        with h5py.File(self.out/'episode.hdf5','w') as f:
            f.attrs.update(task=self.task,seed=self.seed,instruction=instruction,control_hz=30,physics_hz=round(1/self.dt),
                           state_order='left 26 joints then right 26 joints; explicit names stored',
                           action_semantics='absolute joint target radians, applied via recorded substep interpolation and gravity compensation',
                           alignment='observations[t] precedes action[t]; observations[T] is terminal; no duplicated terminal action',
                           qualification='candidate until dynamics, geometry, RGB and dataset gates pass')
            def put(k,v): f.create_dataset(k,data=v,compression='gzip' if np.asarray(v).ndim else None)
            put('observations/qpos',q);put('observations/qvel',qv);put('observations/time',np.arange(n+1)/30)
            put('action',commands[:,-1]);put('action_substeps',commands);put('gravity_feedforward_substeps',ff)
            put('controller/gains_kp_kd',gains);put('diagnostics/body_pose_xyzw',poses)
            f.create_dataset('metadata/joint_names',data=np.array(self.names['left']+self.names['right'],dtype=h5py.string_dtype()))
            f.create_dataset('metadata/body_paths',data=np.array(self.body_paths,dtype=h5py.string_dtype()))
            f.create_dataset('terminated',data=np.r_[np.zeros(n-1,dtype=bool),True])
        np.savez_compressed(self.out/'observation_trajectory.npz',poses=poses,body_paths=self.body_paths,time=np.arange(n+1)/30)
        manifest=dict(task=self.task,seed=self.seed,transitions=n,observations=n+1,joint_names=self.names,action_dim=52,
                      instructions=instruction,scene_changes=self.scene_changes,initial_robot_variation=self.spec['initial_robot_variation'],
                      controller=self.spec['controllers'][self.task],rgb_status='pending',qualification='candidate',
                      observation_alignment='pre-action; explicit terminal observation; final-substep absolute target plus all actual substep commands retained',
                      camera='approved EGO only; exact solver poses rendered offline',collection_source='scripted feedback-controller demonstrations, not human teleoperation',
                      narrow_distribution=True,split='validation' if self.seed%10==0 else 'train')
        (self.out/'collection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        print('RECORDED',self.task,self.seed,n,q.shape,flush=True)
