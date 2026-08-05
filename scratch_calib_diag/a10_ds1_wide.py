"""実測と同じ 5m グリッド (DS=1) で、探索範囲だけ広げた CCPSO2 を回して確認。"""
import numpy as np, sys, time
from pathlib import Path
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import simulate_flood_hd, iou_loss
from ccpso2 import CCPSO2
SC=Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d=np.load(SC/"data.npz"); dem,src,gtd=d["dem"],d["src"],d["gt_depth"]; gt=gtd>0
class Obj:
    def __init__(s_,K): s_.K=K; s_.n=0
    def __call__(s_,x):
        s_.n+=1
        return iou_loss(simulate_flood_hd(dem,src,float(x[0]),x[1:1+s_.K*s_.K].reshape(s_.K,s_.K),sigma=0.5), gt)
for K,wlo,whi,dlo,dhi in ((16,3.0,8.0,-2.0,2.0),(16,0.0,30.0,-8.0,8.0),(16,0.0,30.0,-20.0,20.0)):
    D=1+K*K; lb=np.empty(D); ub=np.empty(D); lb[0],ub[0]=wlo,whi; lb[1:],ub[1:]=dlo,dhi
    o=Obj(K); s=16; cycles=max(1,5000//(20*(D//s))); t=time.time()
    cc=CCPSO2(o,dim=D,n_particles=20,group_size=s,bounds=(lb,ub),p_cauchy=0.5,seed=0,verbose=False)
    r=cc.run(n_cycles=cycles); x=r["best_x"]
    iou=1-iou_loss(simulate_flood_hd(dem,src,float(x[0]),x[1:].reshape(K,K),sigma=0.5),gt)
    print(f"DS=1(5m) K={K} w[{wlo},{whi}] dh[{dlo},{dhi}] -> IoU={iou:.4f} (w*={x[0]:.2f}, evals={o.n}, {time.time()-t:.0f}s)", flush=True)
