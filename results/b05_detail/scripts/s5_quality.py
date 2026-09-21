"""S5 水面品質: 「想定図をそのまま拡大する」のに対して変換器が何を良くするのか。

A) 公式 25m 想定図の素朴拡大 (ランク代表深を 5m 格子に最近傍で載せる) — 面 IoU の最強ベースライン
B) HAND ブロックオラクル K=32 (到達しうる連続水面形の上限)
C) 実測解 (HAND + depth∈[-4,20]、実河道水源、λ=0、K=16、CCPSO2 5000評価 × seed 0-2)

指標 (s5_vizquality.py の定義):
  面 IoU / 1m 丸めで階段になる隣接ペアの割合 / 取り得る水深値の数 /
  水面<地形 の違反数 / 水面の連結成分数
  ※「隣接段差 非ゼロ率」は「区分定数か連続か」を測っていただけなので出さない。

出力: results/b05_detail/s5_quality.json  +  キャッシュに surfaces.npz (図用)
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_detail import (CACHE, OUT, ST8, K_OPT, BUDGET, SEEDS, DMAX,
                           load_grids, viz_metrics, HandObj, run_ccpso2)

t0 = time.time()
g = load_grids()
land, valid, gt = g["land"], g["valid"], g["gt"]
gt_depth, LS, land_eff, zd = g["gt_depth"], g["LS"], g["land_eff"], g["zd"]
H, W = land.shape
res = {}

# ── A) 公式 25m 図を素朴に最近傍拡大 ──────────────────────────────
f = 5
hh, ww = H // f, W // f
blk = lambda a: a[:hh * f, :ww * f].reshape(hh, f, ww, f)
wcnt = blk(gt.astype(np.float64)).sum(axis=(1, 3))
nval = blk(valid.astype(np.float64)).sum(axis=(1, 3))
wet_c = (wcnt > 0) & (np.divide(wcnt, np.maximum(nval, 1)) >= 0.5)
dsum = np.nan_to_num(blk(np.where(gt, np.maximum(gt_depth, 0.0), 0.0))).sum(axis=(1, 3))
drep = np.divide(dsum, np.maximum(wcnt, 1e-9))
mA = np.zeros((H, W), bool)
dA = np.zeros((H, W))
mA[:hh * f, :ww * f] = np.repeat(np.repeat(wet_c, f, axis=0), f, axis=1)
dA[:hh * f, :ww * f] = np.repeat(np.repeat(drep, f, axis=0), f, axis=1)
mA &= valid
wseA = land + dA
res["A_naive25m"] = viz_metrics("A) 公式25m想定図の素朴拡大", mA, wseA, land, valid, gt)

# ── B) HAND ブロックオラクル K=32 ────────────────────────────────
from scipy.ndimage import label as nd_label
hand = land_eff - zd
K = 32
ys = np.linspace(0, H, K + 1).astype(int)
xs = np.linspace(0, W, K + 1).astype(int)
thr = np.zeros((H, W))
for i in range(K):
    for jj in range(K):
        sl = (slice(ys[i], ys[i + 1]), slice(xs[jj], xs[jj + 1]))
        fv = hand[sl].ravel()
        gg = gt[sl].ravel()
        if fv.size == 0:
            continue
        o = np.argsort(fv, kind="stable")
        fs, gs = fv[o], gg[o]
        tot = gs.sum()
        err = np.concatenate(([tot], np.cumsum(~gs) + (tot - np.cumsum(gs))))
        k = int(np.argmin(err))
        t = (fs[0] - 1e-6) if k == 0 else (fs[k - 1] + 1e-9)
        thr[sl] = min(max(t, 0.0), 10.0)
wseB = thr + zd
cand = LS < wseB
lab, nl = nd_label(cand, structure=ST8)
sv = g["src_bbox"] & cand
u = np.unique(lab[sv]); u = u[u > 0]
lut = np.zeros(nl + 1, bool); lut[u] = True
mB = lut[lab] & ((wseB - land) > 0.05)
res["B_hand_oracle_K32"] = viz_metrics("B) HANDブロックオラクル K=32", mB, wseB, land, valid, gt)

# ── C) 実測解 (実河道水源・λ=0) ─────────────────────────────────
per_seed = []
best = None
for sd in SEEDS:
    o = HandObj(g, K_OPT, g["src_drain"], lam=0.0)
    x = run_ccpso2(o, K=K_OPT, seed=sd, budget=BUDGET)
    mC, wseC = o.expand(x, g)
    r = viz_metrics(f"C) 実測解 seed={sd}", mC, wseC, land, valid, gt)
    # 参考: src_reg.py が使っていた「HAND 水深場 tf 上の 1m 階段」(WSE ではない)
    mm, tf = o.mask(x)
    tfl = np.full((H, W), np.nan)
    tfl[o.r0:o.r1, o.c0:o.c1] = tf
    Wb = np.where(mC, np.floor(tfl), np.nan)
    py = mC[:-1] & mC[1:]; px = mC[:, :-1] & mC[:, 1:]
    st = int(np.count_nonzero((np.abs(np.diff(Wb, axis=0)) > 0.5) & py)
             + np.count_nonzero((np.abs(np.diff(Wb, axis=1)) > 0.5) & px))
    pr = int(np.count_nonzero(py) + np.count_nonzero(px))
    r["stair_rate_on_depthfield"] = st / pr if pr else 0.0
    r["seed"] = sd
    r["x"] = np.asarray(x).tolist()
    per_seed.append(r)
    print(f"   (seed {sd}: {time.time()-t0:.0f}s, 参考 depth場上の1m階段 "
          f"{100*r['stair_rate_on_depthfield']:.2f}%)", flush=True)
    if best is None or r["iou"] > best[0]:
        best = (r["iou"], sd, mC.copy(), wseC.copy())

agg = {}
for k in ("iou", "stair_rate", "ncomp", "nvals", "bad", "cells", "steps", "pairs",
          "stair_rate_on_depthfield"):
    a = np.array([r[k] for r in per_seed], dtype=float)
    agg[k] = float(a.mean())
    agg[k + "_sd"] = float(a.std(ddof=1))
agg["name"] = "C) 実測解 (実河道水源・λ=0, CCPSO2 5000評価, seed 0-2 平均)"
agg["per_seed"] = per_seed
res["C_solution"] = agg

os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/s5_quality.json", "w"), ensure_ascii=False, indent=1, default=float)
np.savez_compressed(f"{CACHE}/surfaces.npz",
                    mA=mA, wseA=wseA, mB=mB, wseB=wseB,
                    mC=best[2], wseC=best[3], best_seed=best[1])
print(f"\nDONE {time.time()-t0:.0f}s  -> {OUT}/s5_quality.json", flush=True)
