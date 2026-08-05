"""現行定式化 (bilinear dh + 連結成分) の実質的な到達可能 IoU を、
oracle 初期値 + ブロック座標降下 (貪欲) で下から押し上げて測る。
= 「探索が十分うまければどこまで行くか」の実測。25m グリッド。"""
import numpy as np, sys, time
from pathlib import Path
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import simulate_flood_hd, iou_loss
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data_ds5.npz"); dem, src, gtd = d["dem"], d["src"], d["gt_depth"]
gt = gtd>0; H,W = dem.shape; NGT=int(gt.sum()); nan=np.isnan(dem)
SIGMA=0.5
LO,HI,STEP=-2.0,80.0,0.25; nb=int((HI-LO)/STEP)+2
bi=np.clip((((np.where(nan,1e4,dem))-LO)/STEP).astype(np.int32),0,nb-1); bi[nan]=nb-1
levels=LO+STEP*np.arange(nb)
def oracle_lev(K,lmax=None):
    by=np.minimum((np.arange(H)*K)//H,K-1); bx=np.minimum((np.arange(W)*K)//W,K-1)
    bid=(by[:,None]*K+bx[None,:]).astype(np.int64); nblk=K*K
    flat=(bid*nb+bi).ravel()
    hg=np.bincount(flat[gt.ravel()],minlength=nblk*nb).reshape(nblk,nb)
    hn=np.bincount(flat[(~gt).ravel()],minlength=nblk*nb).reshape(nblk,nb)
    TP=np.cumsum(hg,1); TP=np.concatenate([np.zeros((nblk,1),np.int64),TP[:,:-1]],1)
    FP=np.cumsum(hn,1); FP=np.concatenate([np.zeros((nblk,1),np.int64),FP[:,:-1]],1)
    ok=np.ones(nb,bool)
    if lmax is not None: ok&=(levels<=lmax)
    ok[0]=True
    TPo,FPo=TP[:,ok],FP[:,ok]; t=0.5
    for _ in range(60):
        k=np.argmax(TPo-t*FPo,axis=1); tp=TPo[np.arange(nblk),k].sum(); fp=FPo[np.arange(nblk),k].sum()
        t2=tp/(NGT+fp)
        if abs(t2-t)<1e-10: t=t2; break
        t=t2
    return t, levels[ok][k].reshape(K,K)

def ev(w,dh):
    return 1-iou_loss(simulate_flood_hd(dem,src,float(w),dh,sigma=SIGMA), gt)

def refine(K, w, dh, dlo, dhi, sweeps=3, tag=""):
    cur=ev(w,dh); n=1; t0=time.time()
    print(f"   [{tag}] init IoU={cur:.4f}")
    for s in range(sweeps):
        step=[2.0,1.0,0.5][min(s,2)]
        order=np.random.RandomState(s).permutation(K*K)
        for idx in order:
            i,j=divmod(idx,K)
            for dd in (step,-step):
                v=np.clip(dh[i,j]+dd,dlo,dhi)
                if v==dh[i,j]: continue
                old=dh[i,j]; dh[i,j]=v
                c=ev(w,dh); n+=1
                if c>cur+1e-6: cur=c
                else: dh[i,j]=old
        print(f"   [{tag}] sweep{s+1} step={step} IoU={cur:.4f} evals={n} {time.time()-t0:.0f}s")
    return cur, dh

for K in (16,32):
    print(f"=== K={K} (D={1+K*K}) ===")
    # 現行範囲
    t,lev=oracle_lev(K,lmax=10.0); w=8.0; dh=np.clip(lev-w,-2.0,2.0)
    print(f"   oracle(piecewise,<=10m) IoU={t:.4f}")
    c1,_=refine(K,w,dh.copy(),-2.0,2.0,3,tag=f"K{K} 現行範囲 dh[-2,2] w=8")
    # 広い範囲
    t,lev=oracle_lev(K); lev=np.where(lev<0,-50.0,lev)
    w=float(np.median(lev[lev>-40])); dh=np.clip(lev-w,-20,20)
    print(f"   oracle(piecewise,free) IoU={t:.4f}  w0={w:.2f}")
    c2,_=refine(K,w,dh.copy(),-20.0,20.0,3,tag=f"K{K} 広範囲 dh[-20,20]")
    print(f"   => K={K}: 現行範囲 {c1:.4f} / 広範囲 {c2:.4f}")
