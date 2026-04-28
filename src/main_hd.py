"""
main_hd.py
高次元 PSO 校正実験のエントリーポイント（標準 PSO 単独版）。

CCPSO2 との比較は benchmark.py で行う。
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source, simulate_flood_hd, iou_loss
from pso_calibrate_hd import make_synthetic_ground_truth, run_pso_hd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEM_DIR = REPO_ROOT.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-DEM5A-20250620"
DEM_DIR  = Path(os.environ.get("FLOOD_PSO_DEM_DIR", str(DEFAULT_DEM_DIR)))
OUT_DIR  = REPO_ROOT / "results" / "hd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 河川領域
HIDAKA_RIVER_BBOX = {
    "lat_min": 33.855, "lat_max": 33.905,
    "lon_min": 135.145, "lon_max": 135.215,
}
RIVER_ELEV_MAX = 5.0
DS_FACTOR = 5

# 高次元実験設定
K = 8                  # K×K = 64 ブロック → D = 1+64 = 65
WATER_TRUE = 5.0       # 真の global water_level
DH_AMP = 1.5           # 真の Δh の振幅 [m]
SIGMA = 0.5            # 平滑化（全体共通）

# PSO 設定
N_PARTICLES = 30
N_ITER = 100


def main():
    print("=" * 60)
    print(f"高次元 PSO 校正  K={K} (D={1+K*K})")
    print("=" * 60)

    # 1. DEM ロード（ダウンサンプル版）
    print("\n[1/4] Loading DEM and downsampling...")
    dem_info_5m = mosaic_tiles(str(DEM_DIR))
    dem_info = downsample(dem_info_5m, DS_FACTOR)
    dem = dem_info["dem"]
    print(f"  DS shape: {dem.shape}")

    # 2. 水源マスク
    source = make_river_source(
        dem,
        lat_max=dem_info["lat_max"], res_lat=dem_info["res_lat"],
        lon_min=dem_info["lon_min"], res_lon=dem_info["res_lon"],
        river_bbox=HIDAKA_RIVER_BBOX, elev_max=RIVER_ELEV_MAX,
    )
    print(f"  Source cells: {np.sum(source)}")

    # 3. 合成 ground truth
    print(f"\n[2/4] Building synthetic ground truth (K={K}, water_true={WATER_TRUE})...")
    gt = make_synthetic_ground_truth(dem, source, K=K,
                                     water_level_true=WATER_TRUE,
                                     dh_amp=DH_AMP, sigma=SIGMA, seed=42)
    print(f"  Ground-truth flood cells: {int(np.sum(gt['gt_mask']))}")

    # 速度測定
    t0 = time.time()
    _ = simulate_flood_hd(dem, source, WATER_TRUE,
                          gt["dh_true"], sigma=SIGMA)
    t1 = time.time()
    est = (t1 - t0) * N_PARTICLES * N_ITER / 60
    print(f"  1 sim: {(t1-t0)*1000:.1f}ms  Est. PSO time: ~{est:.1f} min")

    # 4. 標準 PSO 実行
    print(f"\n[3/4] Running standard PSO ({N_PARTICLES}p × {N_ITER}iter)...")
    result = run_pso_hd(dem, source, gt["gt_mask"], K=K,
                        sigma=SIGMA,
                        n_particles=N_PARTICLES, n_iter=N_ITER,
                        verbose=False)

    dh_err = float(np.linalg.norm(result["best_dh"] - gt["dh_true"]) / np.sqrt(K*K))
    print(f"\n  best water: {result['best_water']:.3f} (true {WATER_TRUE})")
    print(f"  best_iou:   {result['best_iou']:.4f}")
    print(f"  Δh RMSE:    {dh_err:.3f} m")
    print(f"  elapsed:    {result['elapsed_s']:.1f} s")

    # 5. 保存
    print("\n[4/4] Saving results...")
    np.savez(OUT_DIR / f"pso_hd_K{K}.npz",
             best_water=result["best_water"],
             best_dh=result["best_dh"],
             dh_true=gt["dh_true"],
             gt_mask=gt["gt_mask"],
             history=np.array(result["history"]))

    # 収束履歴
    plt.figure(figsize=(7, 4))
    plt.plot(result["history"], "-o", ms=3, lw=1, label="Standard PSO")
    plt.xlabel("iteration"); plt.ylabel("best cost (1 - IoU)")
    plt.title(f"Standard PSO (D={1+K*K}, K={K})")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"convergence_pso_K{K}.png", dpi=120)
    plt.close()

    # Δh の推定 vs 真値
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax = max(np.abs(gt["dh_true"]).max(), np.abs(result["best_dh"]).max())
    axes[0].imshow(gt["dh_true"], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Δh true (K×K)")
    im = axes[1].imshow(result["best_dh"], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Δh estimated  (RMSE={dh_err:.3f})")
    plt.colorbar(im, ax=axes, fraction=0.046)
    plt.savefig(OUT_DIR / f"dh_estimation_K{K}.png", dpi=120)
    plt.close()

    summary = {
        "K": K, "D": 1 + K*K,
        "method": "standard_pso",
        "n_particles": N_PARTICLES, "n_iter": N_ITER,
        "water_true": WATER_TRUE, "best_water": result["best_water"],
        "best_iou": result["best_iou"], "dh_rmse": dh_err,
        "elapsed_s": result["elapsed_s"],
    }
    (OUT_DIR / f"summary_pso_K{K}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
