"""free-level oracle が『物理的な水面』かどうかを検査。
隣接ブロック間の水位段差が大きい = 水の壁 = 物理的に不可能な当てはめ。
さらに滑らかさ制約 (隣接段差 <= g) を課したときの IoU を測る (投影付き座標降下)。"""
import numpy as np
from pathlib import Path
SC=Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d=np.load(SC/"data.npz"); dem,gtd=d["dem"],d["gt_depth"]
gt=gtd>0; nan=np.isnan(dem); H,W=dem.shape; NGT=int(gt.sum())
LO,HI,STEP=-2.0,80.0,0.25; nb=int((HI-LO)/STEP)+2
bi=np.clip((((np.where(nan,1e4,dem))-LO)/STEP).astype(np.int32),0,nb-1); bi[nan]=nb-1
levels=LO+STEP*np.arange(nb)
def tabs(K):
    by=np.minimum((np.arange(H)*K)//H,K-1); bx=np.minimum((np.arange(W)*K)//W,K-1)
    bid=(by[:,None]*K+bx[None,:]).astype(np.int64); nblk=K*K
    flat=(bid*nb+bi).ravel()
    hg=np.bincount(flat[gt.ravel()],minlength=nblk*nb).reshape(nblk,nb)
    hn=np.bincount(flat[(~gt).ravel()],minlength=nblk*nb).reshape(nblk,nb)
    TP=np.cumsum(hg,1); TP=np.concatenate([np.zeros((nblk,1),np.int64),TP[:,:-1]],1)
    FP=np.cumsum(hn,1); FP=np.concatenate([np.zeros((nblk,1),np.int64),FP[:,:-1]],1)
    return TP,FP
for K in (16,32,64):
    TP,FP=tabs(K); nblk=K*K; t=0.5
    for _ in range(60):
        k=np.argmax(TP-t*FP,axis=1); tp=TP[np.arange(nblk),k].sum(); fp=FP[np.arange(nblk),k].sum()
        t2=tp/(NGT+fp)
        if abs(t2-t)<1e-10: t=t2; break
        t=t2
    lev=levels[k].reshape(K,K)
    used=lev>LO+0.1   # 浸水させたブロック
    dy=np.abs(np.diff(lev,axis=0)); my=(used[:-1]&used[1:])
    dx=np.abs(np.diff(lev,axis=1)); mx=(used[:,:-1]&used[:,1:])
    j=np.concatenate([dy[my],dx[mx]])
    bl_m = (H/K*6.23+W/K*5.16)/2
    print(f"K={K:3d} (block~{bl_m:.0f}m) oracle IoU={t:.4f}  隣接ブロック水位段差: "
          f"median={np.median(j):.2f}m p90={np.percentile(j,90):.2f}m max={j.max():.1f}m  "
          f"段差>2m の割合={100*np.mean(j>2):.1f}%  → 勾配 {np.median(j)/bl_m*1000:.1f} m/km (実河川は ~1-3 m/km)")

print("\n滑らかさ制約付き oracle (隣接段差<=g m を満たす範囲で Dinkelbach 座標降下)")
def smooth_oracle(K,g,iters=30):
    TP,FP=tabs(K); nblk=K*K
    lev=np.full(nblk, float(np.median((dem+gtd)[gt&~nan])))
    t=0.4
    for it in range(iters):
        L=lev.reshape(K,K)
        for i in range(K):
            for jj in range(K):
                nb_=[]
                if i>0: nb_.append(L[i-1,jj])
                if i<K-1: nb_.append(L[i+1,jj])
                if jj>0: nb_.append(L[i,jj-1])
                if jj<K-1: nb_.append(L[i,jj+1])
                loA=max(n-g for n in nb_) if nb_ else LO
                hiA=min(n+g for n in nb_) if nb_ else HI
                ok=(levels>=loA)&(levels<=hiA)
                if not ok.any(): continue
                b=i*K+jj
                sc=TP[b][ok]-t*FP[b][ok]
                L[i,jj]=levels[ok][np.argmax(sc)]
        lev=L.ravel()
        kk=np.clip(((lev-LO)/STEP).astype(int),0,nb-1)
        tp=TP[np.arange(nblk),kk].sum(); fp=FP[np.arange(nblk),kk].sum()
        t=tp/(NGT+fp)
    return t,tp,fp
RES={}
for K,g in ((16,1.0),(16,2.0),(32,0.5),(32,1.0),(32,2.0),(64,0.5),(64,1.0),(64,2.0),(128,1.0)):
    t,tp,fp=smooth_oracle(K,g,iters=25)
    print(f"  K={K:3d} 隣接段差<= {g}m : IoU={t:.4f} (TP={tp} FP={fp} recall={tp/NGT:.3f})", flush=True)
