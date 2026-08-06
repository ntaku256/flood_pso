"""(A) の本命: 「GT 自身が示す水面標高場」を無限解像度の理想 dh とみなし、
実モデル (dem < WSE -> 連結成分) を回して IoU を測る = バスタブ+連結成分の理論上限。
併せて FP の空間分布 (GT からの距離帯) と GT 連結成分・水源の関係を調べる。"""
import numpy as np, sys
from pathlib import Path
from scipy.ndimage import (label as nd_label, grey_dilation, uniform_filter,
                           distance_transform_edt, binary_dilation)
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import iou_loss
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data.npz"); dem, src, gtd = d["dem"], d["src"], d["gt_depth"]
gt = gtd>0; nan = np.isnan(dem); H,W = dem.shape; NGT = int(gt.sum()); CELL=5.7
land = np.where(nan, 1e4, dem)
st = np.ones((3,3),int)

def wse_field(B, smooth=3):
    wm = np.where(gt & ~nan, dem+gtd, np.nan)
    hb, wb = int(np.ceil(H/B)), int(np.ceil(W/B))
    pad = np.full((hb*B, wb*B), np.nan, np.float64); pad[:H,:W] = wm
    a = pad.reshape(hb,B,wb,B)
    with np.errstate(all="ignore"):
        med = np.nanmedian(a, axis=(1,3))
    f = med.copy()
    for _ in range(400):
        m = np.isnan(f)
        if not m.any(): break
        g = grey_dilation(np.where(np.isnan(f),-1e9,f), size=(3,3))
        f = np.where(m, np.where(g<-1e8, np.nan, g), f)
    f = np.nan_to_num(f, nan=float(np.nanmedian(med)))
    if smooth>1: f = uniform_filter(f, size=smooth, mode="nearest")
    return np.kron(f, np.ones((B,B)))[:H,:W], med

dist_gt = distance_transform_edt(~gt)*CELL     # GT からの距離 [m]

print("### A) GT 自身の水面標高場を与えたときの実モデル IoU (= バスタブ+CC の理論上限)")
print("     src = 現行の水源マスク / gtsrc = GT∩低標高を水源にした場合")
gtsrc = gt & (dem < 6.0) & ~nan
for B,label in ((9,"~50m"),(18,"~100m"),(35,"~200m"),(88,"~500m")):
    Wf, _ = wse_field(B)
    for bias in (0.0, -0.5, -1.0):
        cand = land < (Wf + bias)
        lab, _ = nd_label(cand, structure=st)
        for smask, tag in ((src,"src"), (gtsrc,"gtsrc")):
            sv = smask & cand
            keep = set(lab[sv].tolist()); keep.discard(0)
            fm = np.isin(lab, list(keep))
            tp = int((fm&gt).sum()); fp = int((fm&~gt).sum())
            print(f"   B={label:>6s} bias={bias:+.1f}m {tag:6s}: IoU={tp/(NGT+fp):.4f} "
                  f"recall={tp/NGT:.3f} prec={tp/max(1,tp+fp):.3f} TP={tp} FP={fp} "
                  f"(連結性で落ちた水面下セル={int((cand&~fm&~nan).sum())})")

print("\n### B) 『水面下なのに GT では非浸水』セルの GT からの距離分布 (B=~200m, bias 0)")
Wf,_ = wse_field(35)
under = (~gt)&(~nan)&(dem < Wf)
cand = land < Wf
lab,_ = nd_label(cand, structure=st)
keep = set(lab[src&cand].tolist()); keep.discard(0)
fm = np.isin(lab, list(keep))
for a,b in ((0,25),(25,50),(50,100),(100,250),(250,500),(500,1000),(1000,1e9)):
    m = under & (dist_gt>=a) & (dist_gt<b)
    mc = m & fm
    print(f"   距離 {a:5.0f}-{b if b<1e8 else 9999:5.0f} m: 水面下dry={int(m.sum()):7d}  "
          f"うち河川と連結(=実FP){int(mc.sum()):7d}")
print(f"   合計 水面下dry={int(under.sum())}  実FP={int((under&fm).sum())}")

print("\n### C) GT 側の性質: 連結成分と水源の関係")
lg, ng_ = nd_label(gt, structure=st)
sz = np.bincount(lg.ravel()); sz[0]=0
order = np.argsort(sz)[::-1]
print(f"   GT 連結成分数={ng_}  最大={sz[order[0]]} (全体の{100*sz[order[0]]/NGT:.1f}%)")
print(f"   上位5成分サイズ: {[int(sz[i]) for i in order[:5]]}")
touch = set(lg[src&gt].tolist()); touch.discard(0)
in_src = np.isin(lg, list(touch))
print(f"   水源bboxに触れる成分に属する GT セル: {int((in_src&gt).sum())} ({100*(in_src&gt).sum()/NGT:.1f}%)")
print(f"   小成分(<1000セル)の GT セル: {int(sum(s for s in sz if s<1000))}")

print("\n### D) ジオリファレンス整合: GT を (dy,dx) だけずらしたときの一様水位 best IoU")
def best_uniform(gm):
    best=(0,0)
    for w in np.arange(4.0,12.01,0.5):
        c = land<w
        l2,_ = nd_label(c, structure=st)
        k = set(l2[src&c].tolist()); k.discard(0)
        f2 = np.isin(l2, list(k))
        tp=int((f2&gm).sum()); fp=int((f2&~gm).sum()); n=int(gm.sum())
        i = tp/(n+fp)
        if i>best[0]: best=(i,w)
    return best
for dy,dx in ((0,0),(-4,0),(4,0),(0,-4),(0,4),(-9,0),(9,0),(0,-9),(0,9),(-9,-9),(9,9)):
    g2 = np.roll(np.roll(gt, dy, axis=0), dx, axis=1)
    i,w = best_uniform(g2)
    print(f"   shift dy={dy*CELL:+6.0f}m dx={dx*CELL:+6.0f}m : best uniform IoU={i:.4f} (w={w:.1f})")
