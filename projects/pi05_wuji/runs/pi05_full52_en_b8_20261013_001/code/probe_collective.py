from pathlib import Path
import fcntl,json,time
locks=[]
for i in range(4):
 f=open(f'/media/external/bowenjing/projects/dexterity_living/reports/process_manager/gpu{i}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
import jax,jax.numpy as jnp,numpy as np
print('DEVICES',jax.devices(),flush=True)
f=jax.pmap(lambda x:jax.lax.psum(x,'i'),axis_name='i')
start=time.monotonic()
out=np.asarray(f(jnp.arange(4,dtype=jnp.float32)))
assert np.array_equal(out,np.full(4,6.0)),out
print('COLLECTIVE_PASSED',json.dumps({'values':out.tolist(),'seconds':time.monotonic()-start}),flush=True)
