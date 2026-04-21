"""
main.py
御坊市 洪水PSO校正シミュレーション エントリーポイント

処理フロー:
  1. DEM 5m タイルをモザイク読み込み
  2. PSO 校正は 25m ダウンサンプル DEM で実施（速度優先）
  3. 最適パラメータを 5m フル解像度 DEM に適用して最終浸水マップ生成
  4. 結果を results/ に保存

実行:
    cd flood_pso/src
    python main.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source, simulate_flood, make_reference_mask, iou_loss
from pso_calibrate import run_pso
from visualize import plot_dem, plot_flood_overlay, plot_pso_history, plot_inundation_depth_histogram

# ─────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────
DATA_ROOT = Path(r"C:\Users\moriken\Documents\ntaku\特別実験\資料\地形データ")
DEM_DIR   = DATA_ROOT / "FG-GML-503561-DEM5A-20250620"
OUT_DIR   = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 日高川 水源領域 (緯度経度)
# 御坊市内を流れる日高川の中下流域
# ─────────────────────────────────────────────
HIDAKA_RIVER_BBOX = {
    "lat_min": 33.855,
    "lat_max": 33.905,
    "lon_min": 135.145,
    "lon_max": 135.215,
}
RIVER_ELEV_MAX = 5.0   # 水源セルとして扱う最大標高 [m]

# ─────────────────────────────────────────────
# 参照浸水域の標高閾値 [m]
# 日高川計画洪水位に基づく目安（御坊市の氾濫原は概ね5m以下）
# ─────────────────────────────────────────────
REF_ELEV_THRESHOLD = 5.0  # m

# ─────────────────────────────────────────────
# ダウンサンプリング倍率（PSO 校正用）
# 5 → 5m × 5 = 25m 解像度
# ─────────────────────────────────────────────
DS_FACTOR = 5

# ─────────────────────────────────────────────
# PSO ハイパーパラメータ
# ─────────────────────────────────────────────
PSO_N_PARTICLES = 20
PSO_N_ITER      = 60


def build_source_and_ref(dem_info, river_bbox, river_elev_max, ref_elev_thresh):
    dem = dem_info["dem"]
    source = make_river_source(
        dem,
        lat_max=dem_info["lat_max"],
        res_lat=dem_info["res_lat"],
        lon_min=dem_info["lon_min"],
        res_lon=dem_info["res_lon"],
        river_bbox=river_bbox,
        elev_max=river_elev_max,
    )
    ref_mask = make_reference_mask(dem, ref_elev_thresh)
    return source, ref_mask


def main():
    print("=" * 60)
    print("御坊市 洪水PSO校正シミュレーション")
    print("=" * 60)

    # ── 1. DEM 読み込み ──────────────────────────────────
    print("\n[1/6] Loading DEM tiles (5m)...")
    dem_info_5m = mosaic_tiles(str(DEM_DIR))
    dem_5m = dem_info_5m["dem"]

    plot_dem(dem_5m, title="御坊市周辺 DEM [m] (5mメッシュ)",
             save_path=str(OUT_DIR / "01_dem_5m.png"))
    plt.close("all")
    print(f"  Shape: {dem_5m.shape}")

    # ── 2. ダウンサンプル (25m) ──────────────────────────
    print(f"\n[2/6] Downsampling to {DS_FACTOR*5}m resolution...")
    dem_info_ds = downsample(dem_info_5m, DS_FACTOR)
    dem_ds = dem_info_ds["dem"]
    print(f"  Downsampled shape: {dem_ds.shape}")

    plot_dem(dem_ds, title=f"御坊市周辺 DEM [m] ({DS_FACTOR*5}mダウンサンプル)",
             save_path=str(OUT_DIR / "02_dem_downsampled.png"))
    plt.close("all")

    # ── 3. 水源・参照マスク (ダウンサンプル) ────────────
    print("\n[3/6] Building source and reference masks (downsampled)...")
    source_ds, ref_mask_ds = build_source_and_ref(
        dem_info_ds, HIDAKA_RIVER_BBOX, RIVER_ELEV_MAX, REF_ELEV_THRESHOLD
    )
    print(f"  Source cells: {np.sum(source_ds)}")
    print(f"  Reference cells: {np.sum(ref_mask_ds)}")

    if np.sum(source_ds) == 0:
        print("  WARNING: No source cells found! Adjust HIDAKA_RIVER_BBOX or RIVER_ELEV_MAX.")

    # 速度目安
    import time
    t0 = time.time()
    _ = simulate_flood(dem_ds, source_ds, water_level=6.0, sigma=0.5)
    t1 = time.time()
    est_total = (t1 - t0) * PSO_N_PARTICLES * PSO_N_ITER / 60
    print(f"  1 sim: {t1-t0:.3f}s  Est. PSO time: ~{est_total:.1f} min")

    # ── 4. PSO 実行 ──────────────────────────────────────
    print(f"\n[4/6] Running PSO ({PSO_N_PARTICLES} particles × {PSO_N_ITER} iters)...")
    best_params, best_cost, _, history = run_pso(
        dem_ds, source_ds, ref_mask_ds,
        n_particles=PSO_N_PARTICLES,
        n_iter=PSO_N_ITER,
        verbose=True,
    )
    best_iou_ds = 1.0 - best_cost
    print(f"\n  Best params: {best_params}")
    print(f"  IoU (downsampled): {best_iou_ds:.4f}")

    plot_pso_history(history,
                     save_path=str(OUT_DIR / "03_pso_convergence.png"))
    plt.close("all")

    # ── 5. フル解像度 (5m) で最終シミュレーション ────────
    print("\n[5/6] Applying best params to full-resolution 5m DEM...")
    source_5m, ref_mask_5m = build_source_and_ref(
        dem_info_5m, HIDAKA_RIVER_BBOX, RIVER_ELEV_MAX, REF_ELEV_THRESHOLD
    )
    best_inundation_5m = simulate_flood(
        dem_5m, source_5m,
        water_level=best_params["water_level"],
        sigma=best_params["sigma"],
    )
    final_iou = 1.0 - iou_loss(best_inundation_5m, ref_mask_5m)
    print(f"  IoU (full-res): {final_iou:.4f}")

    # ── 6. 結果保存 ──────────────────────────────────────
    print("\n[6/6] Saving results...")

    plot_flood_overlay(
        dem_5m, best_inundation_5m,
        source_mask=source_5m,
        ref_mask=ref_mask_5m,
        title=f"洪水浸水域 (IoU={final_iou:.3f}, WL={best_params['water_level']:.1f}m)",
        save_path=str(OUT_DIR / "04_flood_overlay_5m.png"),
    )
    plt.close("all")

    plot_inundation_depth_histogram(best_inundation_5m,
                                    save_path=str(OUT_DIR / "05_depth_histogram.png"))
    plt.close("all")

    # サマリー
    flooded_cells = int(np.sum(best_inundation_5m > 0.05))
    res_m = dem_info_5m["res_lat"] * 111320
    flooded_km2 = flooded_cells * res_m ** 2 / 1e6

    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"  最適 water_level  : {best_params['water_level']:.2f} m")
    print(f"  最適 sigma        : {best_params['sigma']:.4f} cells")
    print(f"  IoU (5m DEM)      : {final_iou:.4f}")
    print(f"  浸水セル数        : {flooded_cells:,}")
    print(f"  推定浸水面積      : {flooded_km2:.2f} km2")
    print(f"\n  結果画像: {OUT_DIR}/")


if __name__ == "__main__":
    main()
