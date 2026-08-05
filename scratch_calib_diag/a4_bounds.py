"""(E)/(B) 検証: 探索範囲 (w, dh の上下限) と K, 予算を振って CCPSO2 の到達 IoU を測る。
25m グリッド (DS=5) で高速に。元論文設定の baseline は 0.62 付近になるはず。"""
import numpy as np, sys, time, itertools, json
from pathlib import Path
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import simulate_flood_hd, iou_loss
from ccpso2 import CCPSO2
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data_ds5.npz"); dem, src, gtd = d["dem"], d["src"], d["gt_depth"]
gt = gtd > 0
SIGMA = 0.5

class Obj:
    def __init__(self,K): self.K=K; self.n=0; self.best=np.inf; self.log=[]
    def __call__(self,x):
        sim = simulate_flood_hd(dem, src, float(x[0]), x[1:1+self.K*self.K].reshape(self.K,self.K), sigma=SIGMA)
        c = iou_loss(sim, gt); self.n+=1
        if c<self.best: self.best=c;
        self.log.append((self.n,self.best)); return c

def run(K, wlo, whi, dlo, dhi, budget, seed=0, s=16):
    D=1+K*K
    lb=np.empty(D); ub=np.empty(D); lb[0],ub[0]=wlo,whi; lb[1:],ub[1:]=dlo,dhi
    o=Obj(K); cycles=max(1,budget//(20*(D//s)))
    t=time.time()
    cc=CCPSO2(o,dim=D,n_particles=20,group_size=s,bounds=(lb,ub),p_cauchy=0.5,seed=seed,verbose=False)
    r=cc.run(n_cycles=cycles)
    x=r["best_x"]; sim=simulate_flood_hd(dem,src,float(x[0]),x[1:].reshape(K,K),sigma=SIGMA)
    iou=1-iou_loss(sim,gt)
    wl = float(x[0]); dh=x[1:].reshape(K,K)
    print(f"  K={K:3d} w[{wlo},{whi}] dh[{dlo},{dhi}] budget={budget:6d} -> IoU={iou:.4f} "
          f"(w*={wl:.2f} dh:[{dh.min():.2f},{dh.max():.2f}] evals={o.n} {time.time()-t:.0f}s)"
          f"{'  [w at bound]' if abs(wl-whi)<0.05 or abs(wl-wlo)<0.05 else ''}")
    return iou, o.log

print("=== A. 現行の探索範囲 (w[3,8], dh[-2,2]) : 再現ベースライン ===")
for K in (8,16,32):
    run(K,3.0,8.0,-2.0,2.0,5000)
print("=== B. 範囲を広げる (w[0,30], dh[-8,8]) ===")
for K in (8,16,32):
    run(K,0.0,30.0,-8.0,8.0,5000)
print("=== C. さらに広い dh (w[0,30], dh[-20,20]) ===")
for K in (16,32):
    run(K,0.0,30.0,-20.0,20.0,5000)
print("=== D. 予算を増やす (範囲B, budget 30000) ===")
for K in (16,32):
    run(K,0.0,30.0,-8.0,8.0,30000)
print("=== E. 現行範囲のまま予算 30000 (収束不足の検証) ===")
run(16,3.0,8.0,-2.0,2.0,30000)
