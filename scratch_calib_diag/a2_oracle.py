"""ブロック単位で水位を自由に置いたときの IoU 上限 (oracle, 連結性制約なし)。
Dinkelbach 法で IoU = TP/(|GT|+FP) を厳密最大化。"""
import numpy as np
from pathlib import Path
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data.npz")
dem, gtd = d["dem"], d["gt_depth"]
gt = gtd > 0
H, W = dem.shape
nan = np.isnan(dem)
NGT = int(gt.sum())

LO, HI, STEP = -2.0, 80.0, 0.25
nb = int((HI-LO)/STEP)+2
demf = np.where(nan, 1e4, dem)
bin_idx = np.clip(((demf-LO)/STEP).astype(np.int32), 0, nb-1)   # nan -> last bin (never flooded)
bin_idx[nan] = nb-1
levels = LO + STEP*np.arange(nb)     # level such that bin<k <=> dem < levels[k]

def oracle(K, lmin=None, lmax=None, verbose=True):
    by = np.minimum((np.arange(H)*K)//H, K-1)
    bx = np.minimum((np.arange(W)*K)//W, K-1)
    bid = (by[:,None]*K + bx[None,:]).astype(np.int64)
    nblk = K*K
    flat = (bid*nb + bin_idx).ravel()
    hg = np.bincount(flat[gt.ravel()], minlength=nblk*nb).reshape(nblk, nb)
    hn = np.bincount(flat[(~gt).ravel()], minlength=nblk*nb).reshape(nblk, nb)
    TP = np.cumsum(hg, axis=1); TP = np.concatenate([np.zeros((nblk,1),np.int64), TP[:,:-1]], axis=1)
    FP = np.cumsum(hn, axis=1); FP = np.concatenate([np.zeros((nblk,1),np.int64), FP[:,:-1]], axis=1)
    ok = np.ones(nb, bool)
    if lmin is not None: ok &= (levels >= lmin)
    if lmax is not None: ok &= (levels <= lmax)
    ok[0] = True     # 「浸水させない」選択は常に許す
    TPo, FPo = TP[:, ok], FP[:, ok]
    t = 0.5
    for _ in range(60):
        score = TPo - t*FPo
        k = np.argmax(score, axis=1)
        tp = TPo[np.arange(nblk), k].sum(); fp = FPo[np.arange(nblk), k].sum()
        t2 = tp/(NGT+fp)
        if abs(t2-t) < 1e-10: t = t2; break
        t = t2
    lev = levels[ok][k]
    if verbose:
        print(f"  K={K:4d} blocks={nblk:6d}  IoU_max={t:.4f}  TP={tp:7d} FP={fp:7d} FN={NGT-tp:7d} "
              f"recall={tp/NGT:.3f} prec={tp/max(1,tp+fp):.3f}  level range=[{lev.min():.1f},{lev.max():.1f}]")
    return t, lev.reshape(K,K), (tp, fp)

print("=== 上限1: ブロック水位を完全自由 (境界なし・連結性なし) ===")
for K in (1,2,4,8,16,24,32,64,128,256):
    oracle(K)
print()
print("=== 上限2: 水位を現行探索範囲 [1,10] m に制限 (w in[3,8], dh in[-2,2]) ===")
for K in (1,8,16,32,64,128,256):
    oracle(K, lmin=None, lmax=10.0)
print()
print("=== 参考: セル単位自由 (K=H=W 相当) ===")
# 完全自由 = 各セルで level を選べる -> GT を完全再現 (連結性のみが制約)
print("  → 自明に IoU=1.0 (連結性制約を除けば)")
