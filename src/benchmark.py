"""
benchmark.py
標準PSO vs CCPSO2 の比較ベンチマーク。

各 K に対して:
  - 合成 ground truth を共有（同じシード）
  - 評価回数バジェットを揃えて両手法を実行
  - best_cost の収束履歴を「評価回数」軸でプロット
  - Δh RMSE / IoU / 計算時間 を表に出力
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source, simulate_flood_hd, iou_loss
from pso_calibrate_hd import make_synthetic_ground_truth
from flood_sim import depth_loss
from ccpso2 import CCPSO2

import pyswarms as ps  # noqa
from pyswarms.single import GlobalBestPSO


# ─────────────────────────────────────────────────────────────
# 損失関数の選択（mask=IoU、depth=浸水深MAE）
# ─────────────────────────────────────────────────────────────
LOSS_KIND = os.environ.get("FLOOD_PSO_LOSS", "depth")  # "depth" or "iou"


def _make_loss_fn(gt_inundation, gt_mask):
    if LOSS_KIND == "iou":
        return lambda sim: iou_loss(sim, gt_mask)
    return lambda sim: depth_loss(sim, gt_inundation)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEM_DIR = REPO_ROOT.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-DEM5A-20250620"
DEM_DIR  = Path(os.environ.get("FLOOD_PSO_DEM_DIR", str(DEFAULT_DEM_DIR)))
OUT_DIR  = REPO_ROOT / "results" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIDAKA_RIVER_BBOX = {
    "lat_min": 33.855, "lat_max": 33.905,
    "lon_min": 135.145, "lon_max": 135.215,
}
RIVER_ELEV_MAX = 5.0
DS_FACTOR = 5
WATER_TRUE = 5.0
DH_AMP = 1.5
SIGMA = 0.5

# ─────────────────────────────────────────────────────────────
# 各 K のグループサイズと粒子数を選定
# ─────────────────────────────────────────────────────────────
# D = 1 + K*K
# s は D の約数を選ぶ（あるいは D を s で割り切れるよう微調整）
# ここでは D を s で割り切れるように s を選択
def choose_group_size(D: int) -> int:
    """D に応じてグループサイズを選択（CCPSO2 推奨値の精神に沿った発見的選択）。
    s が小さいほど分割統治の粒度が細かくなる。素数Dでも可（最終グループが小さくなるだけ）。"""
    if D <= 20:
        return 4
    elif D <= 80:
        return 8
    elif D <= 200:
        return 16
    else:
        return 16


# ─────────────────────────────────────────────────────────────
# 評価カウンタ付き PSO 目的関数
# ─────────────────────────────────────────────────────────────

class EvalCountedObjective:
    """
    pyswarms 用ラッパ。各イテレーション後の評価累計と best_cost を記録する。
    損失は LOSS_KIND（depth or iou）に応じて切替。
    """
    def __init__(self, dem, source, gt_inundation, gt_mask, K, sigma):
        self.dem = dem; self.source = source
        self.gt_inundation = gt_inundation; self.gt_mask = gt_mask
        self.K = K; self.sigma = sigma
        self._loss = _make_loss_fn(gt_inundation, gt_mask)
        self.n_evals = 0
        self.eval_log: list[tuple[int, float]] = []
        self._best = np.inf

    def __call__(self, X):
        n = X.shape[0]
        losses = np.empty(n, dtype=np.float64)
        for i in range(n):
            x = X[i]
            water = float(x[0])
            dh = x[1:1 + self.K * self.K].reshape(self.K, self.K)
            sim = simulate_flood_hd(self.dem, self.source,
                                    water_level_global=water,
                                    dh_map=dh, sigma=self.sigma)
            losses[i] = self._loss(sim)
        self.n_evals += n
        m = float(np.min(losses))
        if m < self._best:
            self._best = m
        self.eval_log.append((self.n_evals, self._best))
        return losses


# ─────────────────────────────────────────────────────────────
# CCPSO2 を高次元 flood 問題用ラッパに包む
# ─────────────────────────────────────────────────────────────

class CCPSO2EvalLogger:
    """CCPSO2 の objective_full をラップして評価ログを取る。"""
    def __init__(self, dem, source, gt_inundation, gt_mask, K, sigma):
        self.dem = dem; self.source = source
        self.gt_inundation = gt_inundation; self.gt_mask = gt_mask
        self.K = K; self.sigma = sigma
        self._loss = _make_loss_fn(gt_inundation, gt_mask)
        self.n_evals = 0
        self.eval_log: list[tuple[int, float]] = []
        self._best = np.inf

    def __call__(self, x: np.ndarray) -> float:
        water = float(x[0])
        dh = x[1:1 + self.K * self.K].reshape(self.K, self.K)
        sim = simulate_flood_hd(self.dem, self.source,
                                water_level_global=water,
                                dh_map=dh, sigma=self.sigma)
        c = self._loss(sim)
        self.n_evals += 1
        if c < self._best:
            self._best = c
        self.eval_log.append((self.n_evals, self._best))
        return c


# ─────────────────────────────────────────────────────────────
# 1 ケース (K, seed) を両手法で実行
# ─────────────────────────────────────────────────────────────

def run_one_case(dem, source, K: int, budget: int, seed: int) -> dict:
    print(f"\n--- K={K}  D={1+K*K}  budget={budget}  seed={seed} ---")
    D = 1 + K * K

    # 合成 ground truth（K だけに依存し、全手法・全シードで共通）
    gt = make_synthetic_ground_truth(
        dem, source, K=K,
        water_level_true=WATER_TRUE,
        dh_amp=DH_AMP, sigma=SIGMA, seed=42,
    )

    # 探索範囲
    lb = np.empty(D); ub = np.empty(D)
    lb[0] = 3.0; ub[0] = 8.0
    lb[1:] = -2.0; ub[1:] = 2.0

    # ── 1) 標準 PSO ───────────────────────────────────────
    n_p_pso = 30
    n_iter_pso = max(1, budget // n_p_pso)
    obj_pso = EvalCountedObjective(dem, source, gt["gt_inundation"], gt["gt_mask"], K, SIGMA)

    np.random.seed(seed)
    optimizer = GlobalBestPSO(
        n_particles=n_p_pso,
        dimensions=D,
        options={"c1": 1.5, "c2": 1.5, "w": 0.7},
        bounds=(lb, ub),
        ftol=-np.inf,  # 早期停止しない（バジェット使い切る）
    )
    t0 = time.time()
    best_cost_pso, best_pos_pso = optimizer.optimize(obj_pso, iters=n_iter_pso, verbose=False)
    elapsed_pso = time.time() - t0

    pso_log = obj_pso.eval_log
    pso_cum_evals  = [e[0] for e in pso_log]
    pso_best_curve = np.minimum.accumulate([e[1] for e in pso_log])

    pso_dh = best_pos_pso[1:1 + K*K].reshape(K, K)
    pso_w  = float(best_pos_pso[0])
    pso_dh_rmse = float(np.linalg.norm(pso_dh - gt["dh_true"]) / np.sqrt(K * K))
    # 補助指標としてマスク IoU も計算
    sim_pso = simulate_flood_hd(dem, source, pso_w, pso_dh, sigma=SIGMA)
    pso_iou = 1.0 - iou_loss(sim_pso, gt["gt_mask"])
    print(f"  PSO    : loss={best_cost_pso:.4f}  IoU={pso_iou:.4f}  Δh_RMSE={pso_dh_rmse:.3f}  "
          f"evals={obj_pso.n_evals}  t={elapsed_pso:.1f}s")

    # ── 2) CCPSO2 ────────────────────────────────────────
    s = choose_group_size(D)
    N_cc = 20
    cycles = max(1, budget // (N_cc * (D // s)))
    obj_cc = CCPSO2EvalLogger(dem, source, gt["gt_inundation"], gt["gt_mask"], K, SIGMA)

    cc = CCPSO2(obj_cc, dim=D, n_particles=N_cc, group_size=s,
                bounds=(lb, ub), p_cauchy=0.5, seed=seed, verbose=False)
    res_cc = cc.run(n_cycles=cycles)

    cc_cum_evals  = [e[0] for e in obj_cc.eval_log]
    cc_best_curve = [e[1] for e in obj_cc.eval_log]

    cc_dh = res_cc["best_x"][1:1 + K*K].reshape(K, K)
    cc_w  = float(res_cc["best_x"][0])
    cc_dh_rmse = float(np.linalg.norm(cc_dh - gt["dh_true"]) / np.sqrt(K * K))
    sim_cc = simulate_flood_hd(dem, source, cc_w, cc_dh, sigma=SIGMA)
    cc_iou = 1.0 - iou_loss(sim_cc, gt["gt_mask"])
    print(f"  CCPSO2 : loss={res_cc['best_cost']:.4f}  IoU={cc_iou:.4f}  Δh_RMSE={cc_dh_rmse:.3f}  "
          f"evals={obj_cc.n_evals}  t={res_cc['elapsed_s']:.1f}s  s={s}  cycles={cycles}")

    return {
        "K": K, "D": D, "seed": seed, "budget": budget,
        "loss_kind": LOSS_KIND,
        "pso": {
            "n_p": n_p_pso, "n_iter": n_iter_pso,
            "loss": float(best_cost_pso), "iou": float(pso_iou),
            "best_w": pso_w, "dh_rmse": pso_dh_rmse,
            "n_evals": obj_pso.n_evals, "elapsed_s": elapsed_pso,
            "cum_evals": pso_cum_evals, "best_curve": list(map(float, pso_best_curve)),
            "best_dh": pso_dh.tolist(),
        },
        "ccpso2": {
            "s": s, "N": N_cc, "cycles": cycles,
            "loss": float(res_cc["best_cost"]), "iou": float(cc_iou),
            "best_w": cc_w, "dh_rmse": cc_dh_rmse,
            "n_evals": obj_cc.n_evals, "elapsed_s": res_cc["elapsed_s"],
            "cum_evals": cc_cum_evals, "best_curve": list(map(float, cc_best_curve)),
            "best_dh": cc_dh.tolist(),
        },
        "gt": {
            "water_true": WATER_TRUE,
            "dh_true": gt["dh_true"].tolist(),
        },
    }


# ─────────────────────────────────────────────────────────────
# プロット
# ─────────────────────────────────────────────────────────────

def plot_convergence(case: dict, save_path: Path):
    K = case["K"]; D = case["D"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(case["pso"]["cum_evals"], case["pso"]["best_curve"],
            "-", lw=1.4, label=f"Standard PSO (n_p={case['pso']['n_p']})", color="C0")
    ax.plot(case["ccpso2"]["cum_evals"], case["ccpso2"]["best_curve"],
            "-", lw=1.4, label=f"CCPSO2 (s={case['ccpso2']['s']}, N={case['ccpso2']['N']})", color="C1")
    ax.set_xlabel("function evaluations")
    ax.set_ylabel(f"best cost ({case['loss_kind']})")
    ax.set_title(f"K={K}, D={D}  seed={case['seed']}  budget={case['budget']}  loss={case['loss_kind']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_dh_compare(case: dict, save_path: Path):
    K = case["K"]
    dh_true = np.array(case["gt"]["dh_true"])
    pso_dh  = np.array(case["pso"]["best_dh"])
    cc_dh   = np.array(case["ccpso2"]["best_dh"])
    vmax = max(np.abs(dh_true).max(), np.abs(pso_dh).max(), np.abs(cc_dh).max())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(dh_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"true (K={K})")
    axes[1].imshow(pso_dh, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"PSO  RMSE={case['pso']['dh_rmse']:.3f}")
    im = axes[2].imshow(cc_dh, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"CCPSO2  RMSE={case['ccpso2']['dh_rmse']:.3f}")
    plt.colorbar(im, ax=axes, fraction=0.04)
    plt.savefig(save_path, dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("Standard PSO vs CCPSO2  on flood-PSO HD calibration")
    print("=" * 66)

    print("\n[setup] Loading DEM...")
    dem_info = downsample(mosaic_tiles(str(DEM_DIR)), DS_FACTOR)
    dem = dem_info["dem"]
    source = make_river_source(
        dem,
        lat_max=dem_info["lat_max"], res_lat=dem_info["res_lat"],
        lon_min=dem_info["lon_min"], res_lon=dem_info["res_lon"],
        river_bbox=HIDAKA_RIVER_BBOX, elev_max=RIVER_ELEV_MAX,
    )
    print(f"  DEM shape={dem.shape}  src cells={int(np.sum(source))}")

    K_VALUES = [4, 8, 16]
    BUDGET   = 5000
    SEEDS    = [0, 1, 2]

    rows_per_K: dict = {K: [] for K in K_VALUES}
    for K in K_VALUES:
        cases_for_K = []
        for sd in SEEDS:
            case = run_one_case(dem, source, K=K, budget=BUDGET, seed=sd)
            cases_for_K.append(case)
            rows_per_K[K].append(case)
            (OUT_DIR / f"case_K{K}_seed{sd}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False))
        # 代表 (seed=0) のグラフ
        plot_convergence(cases_for_K[0], OUT_DIR / f"conv_K{K}.png")
        plot_dh_compare(cases_for_K[0],  OUT_DIR / f"dh_K{K}.png")

    # 集計
    rows = []
    for K in K_VALUES:
        cases = rows_per_K[K]
        D = cases[0]["D"]
        def stat(field_method, field):
            vals = [c[field_method][field] for c in cases]
            return float(np.mean(vals)), float(np.std(vals))
        pso_loss_m, pso_loss_s = stat("pso", "loss")
        cc_loss_m,  cc_loss_s  = stat("ccpso2", "loss")
        pso_iou_m,  _ = stat("pso", "iou")
        cc_iou_m,   _ = stat("ccpso2", "iou")
        pso_rmse_m, _ = stat("pso", "dh_rmse")
        cc_rmse_m,  _ = stat("ccpso2", "dh_rmse")
        pso_t_m,    _ = stat("pso", "elapsed_s")
        cc_t_m,     _ = stat("ccpso2", "elapsed_s")
        rows.append({
            "K": K, "D": D, "n_seeds": len(SEEDS),
            "PSO_loss_mean": pso_loss_m, "PSO_loss_std": pso_loss_s,
            "PSO_iou_mean": pso_iou_m, "PSO_dhRMSE_mean": pso_rmse_m,
            "PSO_t_mean": pso_t_m,
            "CC_loss_mean": cc_loss_m, "CC_loss_std": cc_loss_s,
            "CC_iou_mean": cc_iou_m, "CC_dhRMSE_mean": cc_rmse_m,
            "CC_t_mean": cc_t_m,
            "CC_s": cases[0]["ccpso2"]["s"],
        })

    # サマリー表
    print("\n" + "=" * 84)
    print(f"Summary  loss_kind={LOSS_KIND}  n_seeds={len(SEEDS)}  budget={BUDGET}")
    print("=" * 84)
    print(f"{'K':>3} {'D':>4} | {'PSO_loss(±)':>16} {'PSO_IoU':>8} {'PSO_RMSE':>9} | "
          f"{'CC_loss(±)':>16} {'CC_IoU':>8} {'CC_RMSE':>9} {'CC_s':>4}")
    for r in rows:
        print(f"{r['K']:>3} {r['D']:>4} | "
              f"{r['PSO_loss_mean']:>7.4f}±{r['PSO_loss_std']:<6.4f} {r['PSO_iou_mean']:>8.4f} {r['PSO_dhRMSE_mean']:>9.3f} | "
              f"{r['CC_loss_mean']:>7.4f}±{r['CC_loss_std']:<6.4f} {r['CC_iou_mean']:>8.4f} {r['CC_dhRMSE_mean']:>9.3f} {r['CC_s']:>4d}")

    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    # サマリープロット: loss vs D
    Ds = [r["D"] for r in rows]
    pso_m = [r["PSO_loss_mean"] for r in rows]
    pso_s = [r["PSO_loss_std"]  for r in rows]
    cc_m  = [r["CC_loss_mean"]  for r in rows]
    cc_s  = [r["CC_loss_std"]   for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(Ds, pso_m, yerr=pso_s, fmt="-o", capsize=4, label="Standard PSO", color="C0")
    ax.errorbar(Ds, cc_m,  yerr=cc_s,  fmt="-s", capsize=4, label="CCPSO2",       color="C1")
    ax.set_xlabel("dimension D = 1 + K^2")
    ax.set_ylabel(f"final loss ({LOSS_KIND}, mean ± std)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Standard PSO vs CCPSO2  (n_seeds={len(SEEDS)}, budget={BUDGET})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_loss_vs_D.png", dpi=120)
    plt.close()

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
