from pathlib import Path
import os,fcntl
locks=[]
for i in range(4):
 f=open(f'/media/external/bowenjing/projects/dexterity_living/reports/process_manager/gpu{i}.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
import torch
import jax,jax.numpy as jnp,numpy as np
print('TORCH_NCCL',torch.cuda.nccl.version(),flush=True)
print('LIBS',[l for l in Path('/proc/self/maps').read_text().splitlines() if 'libnccl' in l or 'libcudart' in l],flush=True)
print('DEVICES',jax.devices(),flush=True)
x=jax.pmap(lambda x:jax.lax.psum(x,'i'),axis_name='i')(jnp.arange(4,dtype=jnp.float32))
print('LIBS_AFTER',[l for l in Path('/proc/self/maps').read_text().splitlines() if 'libnccl' in l or 'libcudart' in l],flush=True)
print('RESULT',np.asarray(x),flush=True)

big=jnp.ones((4,8*1024*1024),dtype=jnp.float32)
big.block_until_ready()
print('BIG_ALLOCATION_READY',flush=True)
out=jax.pmap(lambda x:jax.lax.psum(x,'i'),axis_name='i')(big)
out.block_until_ready()
print('BIG_REDUCTION_READY',flush=True)
assert float(np.asarray(out[0,0]))==4.0
print('BIG_COLLECTIVE_PASSED',flush=True)
