"""
calibrate_hidaka.py
実データ校正（Phase1）: 日高川 洪水浸水想定区域（想定最大規模, 国土地理院タイル由来GT）
に対し、バスタブ+連結成分シミュの高次元パラメータ
  x[0]      : 大局水位 water_level_global
  x[1:1+K²] : K×K ブロック水位補正 dh_map
を CCPSO2 / 標準PSO で校正し、両者を比較する。

合成GTベンチ（benchmark.py）の実データ版。
GT は hazard_gt.load_hazard_gt（GSIタイル→DEM格子）で取得。
真の dh_map が存在しないため Δh RMSE は算出せず、loss と IoU、および
「校正後シミュ vs 実ハザードマップ」の可視化で評価する。

使い方:
  .venv/bin/python src/calibrate_hidaka.py
  FLOOD_PSO_K=8 FLOOD_PSO_BUDGET=2000 FLOOD_PSO_LOSS=depth .venv/bin/python src/calibrate_hidaka.py
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source, simulate_flood_hd, iou_loss, depth_loss
from hazard_gt import load_hazard_gt
from ccpso2 import CCPSO2
from pyswarms.single import GlobalBestPSO

REPO = Path(__file__).resolve().parent.parent
DEM_DIR = os.environ.get(
    "FLOOD_PSO_DEM_DIR",
    str(REPO.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-DEM5A-20250620"),
)
OUT = REPO / "results" / "hidaka"
OUT.mkdir(parents=True, exist_ok=True)

K = int(os.environ.get("FLOOD_PSO_K", "16"))
BUDGET = int(os.environ.get("FLOOD_PSO_BUDGET", "3000"))
SEED = int(os.environ.get("FLOOD_PSO_SEED", "0"))
LOSS = os.environ.get("FLOOD_PSO_LOSS", "iou")          # iou | depth
ZOOM = int(os.environ.get("FLOOD_PSO_HAZARD_ZOOM", "15"))
SIGMA = 0.5
HIDAKA_RIVER_BBOX = {"lat_min": 33.855, "lat_max": 33.905,
                     "lon_min": 135.145, "lon_max": 135.215}


def loss_fn(sim, gt_depth, gt_mask):
    return iou_loss(sim, gt_mask) if LOSS == "iou" else depth_loss(sim, gt_depth)


def choose_group_size(D):
    return 4 if D <= 20 else (8 if D <= 80 else 16)


class ObjPSO:
    """pyswarms 用（バッチ評価）。評価回数と best を記録。"""
    def __init__(self, dem, src, gt_depth, gt_mask, K, sigma):
        self.dem, self.src = dem, src
        self.gtd, self.gtm = gt_depth, gt_mask
        self.K, self.sigma = K, sigma
        self.n = 0
        self.log = []
        self.best = np.inf

    def eval_one(self, x):
        sim = simulate_flood_hd(self.dem, self.src, float(x[0]),
                                x[1:1 + self.K * self.K].reshape(self.K, self.K),
                                sigma=self.sigma)
        return loss_fn(sim, self.gtd, self.gtm)

    def __call__(self, X):
        out = np.array([self.eval_one(x) for x in X])
        self.n += len(X)
        self.best = min(self.best, float(out.min()))
        self.log.append((self.n, self.best))
        return out


class ObjCC:
    """CCPSO2 用（単一評価）。同じカウンタ・ログ構造。"""
    def __init__(self, dem, src, gt_depth, gt_mask, K, sigma):
        self.base = ObjPSO(dem, src, gt_depth, gt_mask, K, sigma)

    def __call__(self, x):
        c = self.base.eval_one(x)
        self.base.n += 1
        self.base.best = min(self.base.best, c)
        self.base.log.append((self.base.n, self.base.best))
        return c


def save_viz(dem, gt_depth, gt_mask, sim_pso, sim_cc, obj_pso, obj_cc,
             pso_iou, cc_iou, pso_loss, cc_loss, path):
    th = 0.05
    demv = np.where(np.isnan(dem), np.nan, dem)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def panel(a, mask_or_depth, title, depth=True):
        a.imshow(demv, cmap="terrain", origin="upper")
        layer = np.where(mask_or_depth > th, mask_or_depth, np.nan)
        a.imshow(layer, cmap="Blues", origin="upper", vmin=0, vmax=15, alpha=0.75)
        a.set_title(title)
        a.set_xticks([]); a.set_yticks([])

    panel(ax[0, 0], gt_depth, f"GT (real hazard, L2 max-scale)  {int(gt_mask.sum())} cells")
    panel(ax[0, 1], sim_cc, f"CCPSO2 calibrated  IoU={cc_iou:.3f}")
    panel(ax[0, 2], sim_pso, f"Standard PSO calibrated  IoU={pso_iou:.3f}")

    # 差分（CCPSO2）
    diff = np.zeros_like(dem)
    sm = sim_cc > th
    diff[sm & gt_mask] = 1     # hit
    diff[sm & ~gt_mask] = 2    # 過大(false positive)
    diff[~sm & gt_mask] = 3    # 見逃し(false negative)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#00000000", "#2ca02c", "#ff7f0e", "#d62728"])
    ax[1, 0].imshow(demv, cmap="gray", origin="upper")
    ax[1, 0].imshow(np.where(diff > 0, diff, np.nan), cmap=cmap, vmin=1, vmax=3, origin="upper")
    ax[1, 0].set_title("CCPSO2 vs GT  green=hit  orange=over  red=miss")
    ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])

    # 収束
    ax[1, 1].plot(*zip(*obj_pso.log), color="C0", label=f"PSO (final {pso_loss:.4f})")
    ax[1, 1].plot(*zip(*obj_cc.base.log), color="C1", label=f"CCPSO2 (final {cc_loss:.4f})")
    ax[1, 1].set_xlabel("function evaluations")
    ax[1, 1].set_ylabel(f"best loss ({LOSS})")
    ax[1, 1].set_title(f"Convergence  K={K} D={1+K*K}")
    ax[1, 1].grid(True, alpha=0.3); ax[1, 1].legend()

    ax[1, 2].axis("off")
    txt = (f"Real-data calibration (Hidaka R., L2 max-scale)\n\n"
           f"K = {K}   D = {1 + K*K}\nbudget = {BUDGET}   loss = {LOSS}\n\n"
           f"           loss      IoU\n"
           f"PSO     {pso_loss:7.4f}  {pso_iou:6.3f}\n"
           f"CCPSO2  {cc_loss:7.4f}  {cc_iou:6.3f}\n\n"
           f"GT inundation: {int(gt_mask.sum())} cells\n"
           f"DEM grid: {dem.shape}")
    ax[1, 2].text(0.02, 0.95, txt, va="top", family="monospace", fontsize=12)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    t_all = time.time()
    print("[setup] DEM 読み込み (25m)...")
    info = downsample(mosaic_tiles(DEM_DIR), 5)
    dem = info["dem"]
    src = make_river_source(dem, lat_max=info["lat_max"], res_lat=info["res_lat"],
                            lon_min=info["lon_min"], res_lon=info["res_lon"],
                            river_bbox=HIDAKA_RIVER_BBOX, elev_max=5.0)
    print("[setup] GT 実ハザード (GSIタイル)...")
    gt_depth, gt_mask = load_hazard_gt(info, zoom=ZOOM)
    D = 1 + K * K
    print(f"\nK={K}  D={D}  budget={BUDGET}  loss={LOSS}  seed={SEED}")
    print(f"DEM={dem.shape}  src={int(src.sum())}cells  GT浸水={int(gt_mask.sum())}cells "
          f"({100*gt_mask.mean():.1f}%)\n")

    lb = np.empty(D); ub = np.empty(D)
    lb[0], ub[0] = 3.0, 8.0
    lb[1:], ub[1:] = -2.0, 2.0

    # ── 標準 PSO ──
    obj_pso = ObjPSO(dem, src, gt_depth, gt_mask, K, SIGMA)
    npart = 30; iters = max(1, BUDGET // npart)
    np.random.seed(SEED)
    opt = GlobalBestPSO(n_particles=npart, dimensions=D,
                        options={"c1": 1.5, "c2": 1.5, "w": 0.7},
                        bounds=(lb, ub), ftol=-np.inf)
    t = time.time()
    pso_loss, pp = opt.optimize(obj_pso, iters=iters, verbose=False)
    pso_t = time.time() - t
    sim_pso = simulate_flood_hd(dem, src, float(pp[0]), pp[1:].reshape(K, K), sigma=SIGMA)
    pso_iou = 1.0 - iou_loss(sim_pso, gt_mask)
    print(f"PSO     loss={pso_loss:.4f}  IoU={pso_iou:.4f}  evals={obj_pso.n}  t={pso_t:.0f}s")

    # ── CCPSO2 ──
    obj_cc = ObjCC(dem, src, gt_depth, gt_mask, K, SIGMA)
    s = choose_group_size(D); N = 20; cycles = max(1, BUDGET // (N * (D // s)))
    cc = CCPSO2(obj_cc, dim=D, n_particles=N, group_size=s, bounds=(lb, ub),
                p_cauchy=0.5, seed=SEED, verbose=False)
    r = cc.run(n_cycles=cycles)
    cc_loss = float(r["best_cost"])
    cw = float(r["best_x"][0]); cdh = r["best_x"][1:].reshape(K, K)
    sim_cc = simulate_flood_hd(dem, src, cw, cdh, sigma=SIGMA)
    cc_iou = 1.0 - iou_loss(sim_cc, gt_mask)
    print(f"CCPSO2  loss={cc_loss:.4f}  IoU={cc_iou:.4f}  evals={obj_cc.base.n}  "
          f"t={r['elapsed_s']:.0f}s  s={s}  cycles={cycles}")

    winner = "CCPSO2" if cc_loss < pso_loss else "PSO"
    print(f"\n→ winner: {winner}  (Δloss={abs(cc_loss-pso_loss):.4f})")

    tag = f"K{K}_{LOSS}_seed{SEED}"
    out_json = {
        "K": K, "D": D, "budget": BUDGET, "seed": SEED, "loss_kind": LOSS, "zoom": ZOOM,
        "gt_inundation_cells": int(gt_mask.sum()),
        "pso": {"loss": float(pso_loss), "iou": float(pso_iou), "best_w": float(pp[0]),
                "evals": obj_pso.n, "t_s": pso_t, "log": obj_pso.log},
        "ccpso2": {"loss": cc_loss, "iou": float(cc_iou), "best_w": cw, "s": s, "cycles": cycles,
                   "evals": obj_cc.base.n, "t_s": r["elapsed_s"], "log": obj_cc.base.log},
        "winner": winner,
    }
    (OUT / f"calib_{tag}.json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False))
    save_viz(dem, gt_depth, gt_mask, sim_pso, sim_cc, obj_pso, obj_cc,
             pso_iou, cc_iou, pso_loss, cc_loss, OUT / f"calib_{tag}.png")
    print(f"\nSaved {OUT}/calib_{tag}.json , calib_{tag}.png   (total {time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
