"""oracle 由来の dh を実モデル (bilinear + 連結成分 + sigma) に入れて実 IoU を測る。
→ 「探索が完璧だったら現行定式化で幾らまで行くか」= 最適化の取りこぼし量を分離。"""
import numpy as np, sys
from pathlib import Path
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import simulate_flood_hd, iou_loss
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data.npz")
dem, src, gtd = d["dem"], d["src"], d["gt_depth"]
gt = gtd > 0
H, W = dem.shape
NGT = int(gt.sum())
nan = np.isnan(dem)
LO, HI, STEP = -2.0, 80.0, 0.25
nb = int((HI-LO)/STEP)+2
bin_idx = np.clip((((np.where(nan,1e4,dem))-LO)/STEP).astype(np.int32), 0, nb-1); bin_idx[nan]=nb-1
levels = LO + STEP*np.arange(nb)

def oracle_lev(K, lmax=None):
    by = np.minimum((np.arange(H)*K)//H, K-1); bx = np.minimum((np.arange(W)*K)//W, K-1)
    bid = (by[:,None]*K + bx[None,:]).astype(np.int64); nblk=K*K
    flat=(bid*nb+bin_idx).ravel()
    hg=np.bincount(flat[gt.ravel()],minlength=nblk*nb).reshape(nblk,nb)
    hn=np.bincount(flat[(~gt).ravel()],minlength=nblk*nb).reshape(nblk,nb)
    TP=np.cumsum(hg,1); TP=np.concatenate([np.zeros((nblk,1),np.int64),TP[:,:-1]],1)
    FP=np.cumsum(hn,1); FP=np.concatenate([np.zeros((nblk,1),np.int64),FP[:,:-1]],1)
    ok=np.ones(nb,bool)
    if lmax is not None: ok &= (levels<=lmax)
    ok[0]=True
    TPo,FPo=TP[:,ok],FP[:,ok]; t=0.5
    for _ in range(60):
        k=np.argmax(TPo-t*FPo,axis=1)
        tp=TPo[np.arange(nblk),k].sum(); fp=FPo[np.arange(nblk),k].sum()
        t2=tp/(NGT+fp)
        if abs(t2-t)<1e-10: t=t2; break
        t=t2
    return t, levels[ok][k].reshape(K,K)

print("baseline: 一様水位 + 連結成分 (K=1相当) を実モデルで sweep")
best=(0,None)
for w in np.arange(2.0, 14.01, 0.25):
    sim = simulate_flood_hd(dem, src, float(w), np.zeros((2,2)), sigma=0.5)
    i = 1-iou_loss(sim, gt)
    if i>best[0]: best=(i,w)
print(f"  best uniform-level IoU={best[0]:.4f} at w={best[1]:.2f}   (oracle w/o connectivity=0.5242)")

for K in (8,16,32,64):
    for lmax,tag in ((10.0,"bounded[<=10]"), (None,"free")):
        t, lev = oracle_lev(K, lmax=lmax)
        lev = np.where(lev < 0, -50.0, lev)      # 「浸水させない」ブロックは十分低く
        # 現行パラメータ化での表現: w=global, dh=lev-w (clip)
        if lmax is not None:
            w = 8.0; dh = np.clip(lev - w, -2.0, 2.0)
        else:
            w = float(np.median(lev[lev>-40])); dh = lev - w
        sim = simulate_flood_hd(dem, src, w, dh, sigma=0.5)
        iou = 1-iou_loss(sim, gt)
        sim0 = simulate_flood_hd(dem, src, w, dh, sigma=0.0)
        iou0 = 1-iou_loss(sim0, gt)
        # 連結性を外した (candidate だけ) の IoU
        field = w + __import__("flood_sim").upsample_dh(dh, dem.shape)[:H,:W]
        cand = np.where(nan, 1e4, dem) < field
        iouc = 1-iou_loss(cand.astype(np.float32), gt, sim_threshold=0.5)
        print(f"  K={K:3d} {tag:14s} oracle(piecewise)={t:.4f} | bilinear+conn sigma.5={iou:.4f} "
              f"sigma0={iou0:.4f} | bilinear no-conn={iouc:.4f}")
