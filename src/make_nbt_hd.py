"""
make_nbt_hd.py
高次元シミュレーション結果（標準PSO / CCPSO2 / Ground Truth）を NBT 化する。

入力: results/benchmark/case_K{K}_seed{seed}.json （benchmark.py の出力）
出力: results/nbt/hd/gobo_hd_K{K}_{method}.nbt （flood_pso_meta コンパウンド付き）

実行例:
    python make_nbt_hd.py --K 16 --seed 0
    python make_nbt_hd.py --K 16 --seed 0 --preset md_5m
    python make_nbt_hd.py --K 16 --seed 0 --preset huge_5m
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles
from flood_sim import make_river_source, simulate_flood_hd
from nbt_export import export_to_nbt, estimate_size

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEM_DIR = REPO_ROOT.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-DEM5A-20250620"
DEM_DIR  = os.environ.get("FLOOD_PSO_DEM_DIR", str(DEFAULT_DEM_DIR))
BENCH_DIR = REPO_ROOT / "results" / "benchmark"
OUT_DIR  = REPO_ROOT / "results" / "nbt" / "hd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 御坊市中心
LAT_CENTER = 33.875
LON_CENTER = 135.168

RIVER_BBOX = {
    "lat_min": 33.855, "lat_max": 33.905,
    "lon_min": 135.145, "lon_max": 135.215,
}
RIVER_ELEV_MAX = 5.0
SIGMA = 0.5  # benchmark.py と同じ

PRESETS = {
    "xs_overview":  (2000, 2000, 10, 1, 3),
    "sm_5m":        (2500, 2500,  5, 1, 3),
    "md_5m":        (5000, 5000,  5, 1, 2),
    "lg_10m":      (10000,10000, 10, 1, 2),
    "xl_5m":       (10000,10000,  5, 1, 2),
    "huge_5m":     (15000,15000,  5, 1, 2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="md_5m", choices=list(PRESETS.keys()))
    ap.add_argument("--methods", default="pso,ccpso2,gt",
                    help="comma-separated subset of {pso,ccpso2,gt}")
    args = ap.parse_args()

    case_path = BENCH_DIR / f"case_K{args.K}_seed{args.seed}.json"
    if not case_path.exists():
        sys.exit(f"benchmark JSON not found: {case_path}\n"
                 f"  → run `.venv/bin/python src/benchmark.py` first")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    K = case["K"]

    # DEM 読み込み（NBT 化はフル解像度 5m DEM を使う）
    print("Loading DEM (5m, full resolution)...")
    dem_info = mosaic_tiles(DEM_DIR)
    dem = dem_info["dem"]
    source = make_river_source(
        dem,
        lat_max=dem_info["lat_max"], res_lat=dem_info["res_lat"],
        lon_min=dem_info["lon_min"], res_lon=dem_info["res_lon"],
        river_bbox=RIVER_BBOX, elev_max=RIVER_ELEV_MAX,
    )
    print(f"  DEM={dem.shape}  src cells={int(np.sum(source))}")

    width_m, depth_m, h_res, v_res, v_exag = PRESETS[args.preset]
    est = estimate_size(dem_info, LAT_CENTER, LON_CENTER,
                        width_m, depth_m, h_res=h_res, v_res=v_res, v_exag=v_exag)
    print(f"  preset={args.preset}  ~{est['estimated_nbt_MB']} MB/file  "
          f"({est['nx (East-West blocks)']}×{est['nz (North-South blocks)']} blocks)")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # 各手法の (water, dh_map) を抽出（ground truth は case["gt"] から）
    runs = {}
    if "pso" in methods:
        runs["pso"] = {
            "water": float(case["pso"]["best_w"]),
            "dh":    np.array(case["pso"]["best_dh"], dtype=np.float64),
            "loss":  float(case["pso"]["loss"]),
            "iou":   float(case["pso"]["iou"]),
            "dh_rmse": float(case["pso"]["dh_rmse"]),
            "n_evals": int(case["pso"]["n_evals"]),
            "elapsed_s": float(case["pso"]["elapsed_s"]),
            "method_long": "Standard Global-Best PSO (pyswarms)",
        }
    if "ccpso2" in methods:
        runs["ccpso2"] = {
            "water": float(case["ccpso2"]["best_w"]),
            "dh":    np.array(case["ccpso2"]["best_dh"], dtype=np.float64),
            "loss":  float(case["ccpso2"]["loss"]),
            "iou":   float(case["ccpso2"]["iou"]),
            "dh_rmse": float(case["ccpso2"]["dh_rmse"]),
            "n_evals": int(case["ccpso2"]["n_evals"]),
            "elapsed_s": float(case["ccpso2"]["elapsed_s"]),
            "method_long": f"CCPSO2 (s={case['ccpso2']['s']}, custom impl)",
        }
    if "gt" in methods:
        runs["gt"] = {
            "water": float(case["gt"]["water_true"]),
            "dh":    np.array(case["gt"]["dh_true"], dtype=np.float64),
            "loss":  0.0,
            "iou":   1.0,
            "dh_rmse": 0.0,
            "n_evals": 0,
            "elapsed_s": 0.0,
            "method_long": "Synthetic ground truth (target for inverse problem)",
        }

    for tag, r in runs.items():
        print(f"\n--- Generating NBT: {tag} (water={r['water']:.3f}, IoU={r['iou']:.3f}) ---")
        # 5m フル解像度 DEM 上でシミュレーションを再実行
        inundation = simulate_flood_hd(
            dem, source,
            water_level_global=r["water"],
            dh_map=r["dh"],
            sigma=SIGMA,
        )
        flooded = int(np.sum(inundation > 0.05))
        print(f"  full-res flooded cells: {flooded:,}")

        out = OUT_DIR / f"gobo_hd_K{K}_seed{args.seed}_{args.preset}_{tag}.nbt"
        meta = {
            "experiment": "flood_pso_HD_benchmark",
            "method": tag,
            "method_long": r["method_long"],
            "loss_kind": case.get("loss_kind", "depth"),
            "K": K, "D": case["D"],
            "seed": args.seed, "budget": case.get("budget"),
            "water_level_global_m": r["water"],
            "dh_amp_m": 1.5,
            "dh_bounds_m": [-2.0, 2.0],
            "dh_map":   r["dh"],          # ndarray → Float List + _shape
            "sigma":    float(SIGMA),
            "loss":     r["loss"],
            "iou":      r["iou"],
            "dh_rmse":  r["dh_rmse"],
            "n_evals":  r["n_evals"],
            "elapsed_s": r["elapsed_s"],
            "river_bbox": [RIVER_BBOX["lat_min"], RIVER_BBOX["lat_max"],
                            RIVER_BBOX["lon_min"], RIVER_BBOX["lon_max"]],
            "river_elev_max_m": float(RIVER_ELEV_MAX),
            "dem_source":    "FG-GML-503561-DEM5A-20250620 (国土地理院 5m DEM)",
            "study_area":    "Gobo city / Hidaka river, Wakayama, Japan",
            "preset": args.preset,
            "ref_doc": "flood_pso/docs/05_ベンチマーク結果.md",
        }
        export_to_nbt(
            dem_info, inundation,
            lat_center=LAT_CENTER, lon_center=LON_CENTER,
            width_m=width_m, depth_m=depth_m,
            h_res=h_res, v_res=v_res, v_exag=v_exag,
            out_path=str(out), meta=meta,
        )

    print(f"\nAll done. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
