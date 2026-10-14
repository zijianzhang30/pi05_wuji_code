from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
P=Path(__file__).resolve().parents[2]
u=ET.parse(P/'sources/20261011/robots/wuji/assembled.urdf').getroot()
joints={};parents={}
for j in u.findall('joint'):
 o=j.find('origin');T=np.eye(4)
 if o is not None:T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rot.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
 ax=j.find('axis');lim=j.find('limit');d=dict(name=j.get('name'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),T=T,axis=np.fromstring(ax.get('xyz'),sep=' ') if ax is not None else np.array([0,0,1]),kind=j.get('type'),lo=float(lim.get('lower',-3.14)) if lim is not None else 0,hi=float(lim.get('upper',3.14)) if lim is not None else 0,effort=float(lim.get('effort',1)) if lim is not None else 0)
 joints[d['name']]=d;parents[d['child']]=d
base={}
for side,x in [('left',-2.35),('right',-1.75)]:
 T=np.eye(4);T[:3,:3]=Rot.from_euler('z',90,degrees=True).as_matrix();T[:3,3]=[x,-3.995,.7215];base[side]=T

def links(side):
 pre='fl' if side=='left' else 'fr'
 return [pre+'_base_link']+[pre+'_link'+str(i) for i in range(1,7)]+[l.get('name') for l in u.findall('link') if l.get('name').startswith((side+'_palm',side+'_finger'))]
def chain(link,side):
 stop=('fl' if side=='left' else 'fr')+'_base_link';out=[]
 while link!=stop:
  j=parents[link];out.insert(0,j);link=j['parent']
 return out
def fk(link,side,q):
 T=base[side].copy()
 for j in chain(link,side):
  T=T@j['T']
  if j['kind']=='revolute':T[:3,:3]=T[:3,:3]@Rot.from_rotvec(j['axis']*q.get(j['name'],0)).as_matrix()
 return T
def arm_names(side):return [('fl' if side=='left' else 'fr')+'_joint'+str(i) for i in range(1,7)]
def hand_names(side):return [n for n,j in joints.items() if n.startswith(side+'_finger') and j['kind']=='revolute']
def open_q(side):return {n:float(np.clip(0,joints[n]['lo']+.002,joints[n]['hi']-.002)) for n in hand_names(side)}
def solve(side,target,seed=None):
 ns=arm_names(side);lo=np.array([joints[n]['lo'] for n in ns]);hi=np.array([joints[n]['hi'] for n in ns]);seed=np.array(seed if seed is not None else [.15,2.12,-.95,.17,-1.18,-.08])
 def fun(x):
  T=fk(side+'_palm_link',side,dict(zip(ns,x)));return np.r_[T[:3,3]-target[:3,3],.18*Rot.from_matrix(target[:3,:3]@T[:3,:3].T).as_rotvec(),.0001*(x-seed)]
 r=least_squares(fun,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo+1e-8,hi-1e-8),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 T=fk(side+'_palm_link',side,dict(zip(ns,r.x)));return r.x,float(np.linalg.norm(T[:3,3]-target[:3,3])),float(np.linalg.norm(Rot.from_matrix(target[:3,:3]@T[:3,:3].T).as_rotvec()))


def fk_many(requested, side, q):
 cache={(('fl' if side=='left' else 'fr')+'_base_link'):base[side].copy()}
 def get(link):
  if link not in cache:
   j=parents[link];T=get(j['parent'])@j['T']
   if j['kind']=='revolute':T[:3,:3]=T[:3,:3]@Rot.from_rotvec(j['axis']*q.get(j['name'],0)).as_matrix()
   cache[link]=T
  return cache[link]
 return {link:get(link) for link in requested}
