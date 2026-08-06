"""(A)(D) 検証: GT 浸水深から逆算した水面標高 WSE=dem+depth が
「バスタブ (局所的にほぼ平らな水面) 」と整合するかを測る。
矛盾セル数 -> バスタブ定式化の原理的な IoU 上限。"""
import numpy as np
from pathlib import Path
from scipy.ndimage import grey_dilation, maximum_filter, minimum_filter, uniform_filter
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data.npz"); dem, gtd = d["dem"], d["gt_depth"]
gt = gtd > 0
nan = np.isnan(dem); H,W = dem.shape
NGT = int(gt.sum())
CELL = 5.7   # m 平均セル寸法
# 深さランク区間 (m)
LOD = {0.25:(0.0,0.5), 1.75:(0.5,3.0), 4.0:(3.0,5.0), 7.5:(5.0,10.0), 15.0:(10.0,20.0), 25.0:(20.0,40.0)}
lo = np.zeros_like(gtd); hi = np.zeros_like(gtd)
for k,(a,b) in LOD.items():
    m = gtd==np.float32(k); lo[m]=a; hi[m]=b
wse_mid = np.where(gt & ~nan, dem + gtd, np.nan)

print("### 1) GT が示す水面標高 WSE=dem+depth の分布")
v = wse_mid[~np.isnan(wse_mid)]
print("   WSE percentiles:", np.round(np.percentile(v,[1,5,25,50,75,95,99]),2))
print(f"   → 領域内で水面標高が {v.min():.1f}..{v.max():.1f} m にわたる (河川縦断勾配)")

print("\n### 2) ブロック内での WSE のばらつき (バスタブなら小さいはず)")
for B,label in ((9,"~50m"),(18,"~100m"),(35,"~200m"),(88,"~500m")):
    hb, wb = H//B, W//B
    a = wse_mid[:hb*B,:wb*B].reshape(hb,B,wb,B)
    n = np.sum(~np.isnan(a),axis=(1,3))
    with np.errstate(all="ignore"):
        sd = np.nanstd(a,axis=(1,3))
        rng = np.nanmax(a,axis=(1,3)) - np.nanmin(a,axis=(1,3))
    ok = n>=20
    print(f"   block={label:>6s}: median std={np.nanmedian(sd[ok]):5.2f} m  "
          f"p90 std={np.nanpercentile(sd[ok],90):5.2f} m  median(max-min)={np.nanmedian(rng[ok]):5.2f} m")

print("\n### 3) ブロック水位の実現可能性 (max_GT dem  vs  min_dry dem)")
print("    Lb=max(dem over GT) : 全 GT を浸すのに必要な最低水位")
print("    Ud=min(dem over dry): 非浸水を保つ上限")
for B,label in ((9,"~50m"),(18,"~100m"),(35,"~200m"),(53,"~300m"),(88,"~500m")):
    hb, wb = H//B, W//B
    dg = np.where(gt & ~nan, dem, -1e4)[:hb*B,:wb*B].reshape(hb,B,wb,B).max(axis=(1,3))
    dd = np.where(~gt & ~nan, dem,  1e4)[:hb*B,:wb*B].reshape(hb,B,wb,B).min(axis=(1,3))
    ng = gt[:hb*B,:wb*B].reshape(hb,B,wb,B).sum(axis=(1,3))
    sel = (ng>0) & (dg>-1e3) & (dd<1e3)
    viol = sel & (dg > dd + 0.5)
    exc = np.where(viol, dg-dd, 0.0)
    print(f"   block={label:>6s}: GTを含むブロック {int(sel.sum()):6d}  矛盾 {int(viol.sum()):6d} "
          f"({100*viol.sum()/max(1,sel.sum()):5.1f}%)  矛盾ブロックの median 超過={np.median(exc[viol]) if viol.any() else 0:.2f} m")

print("\n### 4) 深さランクとの整合 (Lb > min(dem+hi) なら深さ分布もバスタブ非整合)")
for B,label in ((18,"~100m"),(35,"~200m")):
    hb, wb = H//B, W//B
    dg = np.where(gt & ~nan, dem, -1e4)[:hb*B,:wb*B].reshape(hb,B,wb,B).max(axis=(1,3))
    ub = np.where(gt & ~nan, dem+hi, 1e4)[:hb*B,:wb*B].reshape(hb,B,wb,B).min(axis=(1,3))
    ng = gt[:hb*B,:wb*B].reshape(hb,B,wb,B).sum(axis=(1,3))
    sel = (ng>=5) & (dg>-1e3)
    viol = sel & (dg > ub + 0.01)
    print(f"   block={label:>6s}: {int(viol.sum())}/{int(sel.sum())} ブロック ({100*viol.sum()/max(1,sel.sum()):.1f}%) で"
          f" 深さランクと平坦水面が両立しない (median 超過 {np.median((dg-ub)[viol]) if viol.any() else 0:.2f} m)")

print("\n### 5) GT 由来 WSE 場を内挿 -> バスタブ矛盾セル数と IoU 上限")
for B,label in ((18,"~100m"),(35,"~200m"),(88,"~500m")):
    hb, wb = H//B, W//B
    a = wse_mid[:hb*B,:wb*B].reshape(hb,B,wb,B)
    with np.errstate(all="ignore"):
        med = np.nanmedian(a,axis=(1,3))
    # GT の無いブロックは反復膨張で近傍から補完
    f = med.copy()
    for _ in range(200):
        m = np.isnan(f)
        if not m.any(): break
        g = grey_dilation(np.where(np.isnan(f),-1e9,f), size=(3,3))
        f = np.where(m, np.where(g<-1e8, np.nan, g), f)
    f = np.nan_to_num(f, nan=float(np.nanmedian(med)))
    fs = uniform_filter(f, size=3, mode="nearest")
    Wf = np.kron(fs, np.ones((B,B)))[:H,:W]
    if Wf.shape != (H,W):
        Wf = np.pad(Wf, ((0,H-Wf.shape[0]),(0,W-Wf.shape[1])), mode="edge")
    for tol in (0.0, 0.5, 1.0):
        fp = int(np.sum((~gt) & (~nan) & (dem + tol < Wf)))
        fn = int(np.sum(gt & (~nan) & (dem > Wf + tol)))
        tp = NGT - fn
        print(f"   WSE場 block={label:>6s} tol={tol:.1f}m: "
              f"『水面下なのに非浸水』={fp:7d} 『水面上なのに浸水』={fn:7d} "
              f"→ IoU上限≈{tp/(NGT+fp):.3f}")

print("\n### 6) 近傍半径 r 内の標高逆転 (連続水面を仮定した矛盾) ")
for r_m in (25, 50, 100, 200, 400):
    r = max(1,int(round(r_m/CELL)))
    hi_gt = maximum_filter(np.where(gt&~nan, dem, -1e4), size=2*r+1)
    lo_dry = minimum_filter(np.where(~gt&~nan, dem, 1e4), size=2*r+1)
    # 非浸水セルで「半径 r 内に自分より高い GT セルがある」= 必ず FP になる
    fp = int(np.sum((~gt)&(~nan)&(hi_gt > dem + 0.5)))
    fn = int(np.sum(gt&(~nan)&(dem > lo_dry + 0.5)))
    print(f"   r={r_m:4d} m: 強制FP={fp:7d} ({100*fp/((~gt&~nan).sum()):4.1f}% of dry)  "
          f"強制FN候補={fn:7d} ({100*fn/NGT:4.1f}% of GT)  "
          f"→ 全再現時 IoU上限≈{NGT/(NGT+fp):.3f} / 全FN切り捨て時≈{(NGT-fn)/NGT:.3f}")
