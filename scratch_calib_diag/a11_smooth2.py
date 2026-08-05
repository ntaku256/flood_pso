"""(A) 上限の本命: 「水面は g m/block 以下の勾配で滑らか」制約付き oracle。
free oracle 解を Lipschitz 射影して実行可能点にしてから座標降下 (今度は初期値が良い)。
さらに得られた水面を実モデル (5m, bilinear+連結成分) に入れて実 IoU を測る。"""
import numpy as np, sys
from pathlib import Path
from scipy.ndimage import label as nd_label
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import iou_loss, upsample_dh
SC=Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d=np.load(SC/"data.npz"); dem,src,gtd=d["dem"],d["src"],d["gt_depth"]
gt=gtd>0; nan=np.isnan(dem); H,W=dem.shape; NGT=int(gt.sum()); land=np.where(nan,1e4,dem)
LO,HI,STEP=-2.0,80.0,0.25; nb=int((HI-LO)/STEP)+2
bi=np.clip(((land-LO)/STEP).astype(np.int32),0,nb-1); bi[nan]=nb-1
levels=LO+STEP*np.arange(nb); st=np.ones((3,3),int)
def tabs(K):
    by=np.minimum((np.arange(H)*K)//H,K-1); bx=np.minimum((np.arange(W)*K)//W,K-1)
    bid=(by[:,None]*K+bx[None,:]).astype(np.int64); nblk=K*K
    flat=(bid*nb+bi).ravel()
    hg=np.bincount(flat[gt.ravel()],minlength=nblk*nb).reshape(nblk,nb)
    hn=np.bincount(flat[(~gt).ravel()],minlength=nblk*nb).reshape(nblk,nb)
    TP=np.cumsum(hg,1); TP=np.concatenate([np.zeros((nblk,1),np.int64),TP[:,:-1]],1)
    FP=np.cumsum(hn,1); FP=np.concatenate([np.zeros((nblk,1),np.int64),FP[:,:-1]],1)
    return TP,FP
def free_oracle(TP,FP,nblk):
    t=0.5
    for _ in range(80):
        k=np.argmax(TP-t*FP,axis=1); tp=TP[np.arange(nblk),k].sum(); fp=FP[np.arange(nblk),k].sum()
        t2=tp/(NGT+fp)
        if abs(t2-t)<1e-12: t=t2; break
        t=t2
    return t,k
def lip_project(L,g):
    """g-Lipschitz (4近傍) を満たす最大の場 (<=L) に射影"""
    L=L.copy()
    for _ in range(4*max(L.shape)):
        M=np.full_like(L,1e9)
        M[1:,:]=np.minimum(M[1:,:],L[:-1,:]+g); M[:-1,:]=np.minimum(M[:-1,:],L[1:,:]+g)
        M[:,1:]=np.minimum(M[:,1:],L[:,:-1]+g); M[:,:-1]=np.minimum(M[:,:-1],L[:,1:]+g)
        new=np.minimum(L,M)
        if np.allclose(new,L): break
        L=new
    return L
def eval_field(L,K,use_conn=True):
    Wf=upsample_dh(L,(H,W))[:H,:W]
    if Wf.shape!=(H,W): Wf=np.pad(Wf,((0,H-Wf.shape[0]),(0,W-Wf.shape[1])),mode="edge")
    cand=land<Wf
    if not use_conn:
        return int((cand&gt).sum()), int((cand&~gt).sum())
    lab,_=nd_label(cand,structure=st)
    keep=set(lab[src&cand].tolist()); keep.discard(0)
    fm=np.isin(lab,list(keep))
    return int((fm&gt).sum()), int((fm&~gt).sum())
print("K   g[m/block]  block[m]  IoU(piecewise,滑らか) | 実モデル(bilinear+連結) IoU")
for K in (16,32,64):
    TP,FP=tabs(K); nblk=K*K
    t0,k0=free_oracle(TP,FP,nblk); L0=levels[k0].reshape(K,K)
    # ブロックごとの最低標高 (=「何も浸さない」水位として使える正当な値)
    by=np.minimum((np.arange(H)*K)//H,K-1); bx=np.minimum((np.arange(W)*K)//W,K-1)
    bid=(by[:,None]*K+bx[None,:]).astype(np.int64)
    bmin=np.full(nblk,1e4)
    np.minimum.at(bmin, bid.ravel(), land.ravel())
    bmin=bmin.reshape(K,K)
    L0=np.where(L0<LO+0.1, bmin-0.1, L0)
    bl=(H/K*6.23+W/K*5.16)/2
    for g in (0.5,1.0,2.0,4.0,1e9):
        L=lip_project(L0,g) if g<1e8 else L0.copy()
        # 制約付き座標降下 (良い初期値から)
        t=0.6
        for it in range(12):
            for i in range(K):
                for j in range(K):
                    nbv=[]
                    if i>0: nbv.append(L[i-1,j])
                    if i<K-1: nbv.append(L[i+1,j])
                    if j>0: nbv.append(L[i,j-1])
                    if j<K-1: nbv.append(L[i,j+1])
                    lo_=max(v-g for v in nbv) if (nbv and g<1e8) else LO
                    hi_=min(v+g for v in nbv) if (nbv and g<1e8) else HI
                    ok=(levels>=lo_)&(levels<=hi_)
                    if not ok.any(): continue
                    b=i*K+j
                    L[i,j]=levels[ok][np.argmax(TP[b][ok]-t*FP[b][ok])]
            kk=np.clip(((L.ravel()-LO)/STEP).astype(int),0,nb-1)
            tp=TP[np.arange(nblk),kk].sum(); fp=FP[np.arange(nblk),kk].sum(); t=tp/(NGT+fp)
        tp2,fp2=eval_field(L,K)
        gg = "∞" if g>1e8 else f"{g:.1f}"
        print(f"{K:3d}  {gg:>9s}  {bl:7.0f}   IoU={t:.4f} (recall={tp/NGT:.3f})  |  {tp2/(NGT+fp2):.4f} "
              f"(recall={tp2/NGT:.3f} FP={fp2})", flush=True)
