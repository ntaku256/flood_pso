"""
make_nbt.py
スケールを変えながら NBT を試作するスクリプト。

実行:
    python make_nbt.py
    → いくつかのスケールパターンを順番に試す

カスタム実行:
    python make_nbt.py --width 3000 --h_res 5 --v_exag 3
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source, simulate_flood, make_reference_mask
from nbt_export import estimate_size, export_to_nbt

# ─────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────
DEM_DIR = r"C:\Users\moriken\Documents\ntaku\特別実験\資料\地形データ\FG-GML-503561-DEM5A-20250620"
OUT_DIR = Path(__file__).parent.parent / "results" / "nbt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 御坊市中心 (日高川河口付近)
LAT_CENTER = 33.875
LON_CENTER = 135.168

# 洪水パラメータ (main.py で得た最適値)
WATER_LEVEL = 5.06
SIGMA       = 0.09

RIVER_BBOX = {
    "lat_min": 33.855, "lat_max": 33.905,
    "lon_min": 135.145, "lon_max": 135.215,
}

# 試すスケールパターン
PRESETS = [
    # name,    width_m, depth_m, h_res, v_res, v_exag
    ("xs_overview",  2000,  2000,   10,    1,    3),   # 小さめ全体俯瞰
    ("sm_5m",        2500,  2500,    5,    1,    3),   # 5m/block 2.5km四方
    ("md_5m",        5000,  5000,    5,    1,    2),   # 5m/block 5km四方
    ("lg_10m",      10000, 10000,   10,    1,    2),   # 10m/block 10km四方
]


def load_dem_and_flood():
    print("Loading DEM (5m)...")
    dem_info = mosaic_tiles(DEM_DIR)

    print("Running flood simulation...")
    dem = dem_info["dem"]
    source = make_river_source(
        dem, dem_info["lat_max"], dem_info["res_lat"],
        dem_info["lon_min"], dem_info["res_lon"],
        RIVER_BBOX, elev_max=5.0,
    )
    inundation = simulate_flood(dem, source,
                                 water_level=WATER_LEVEL, sigma=SIGMA)
    return dem_info, inundation


def show_estimates(dem_info):
    print("\n" + "=" * 65)
    print("スケール別サイズ見積もり")
    print("=" * 65)
    print(f"{'プリセット':<18} {'ブロック(X×Z)':<16} {'高さY':<8} {'ブロック数':<14} {'推定MB'}")
    print("-" * 65)
    for name, w, d, hr, vr, ve in PRESETS:
        est = estimate_size(dem_info, LAT_CENTER, LON_CENTER,
                            w, d, h_res=hr, v_res=vr, v_exag=ve)
        nx = est["nx (East-West blocks)"]
        nz = est["nz (North-South blocks)"]
        ny = est["ny (Vertical blocks)"]
        nb = est["total_blocks (estimate)"]
        mb = est["estimated_nbt_MB"]
        print(f"  {name:<16} {nx}×{nz:<10}   {ny:<8} {nb:<14,} {mb} MB")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=None,
                        choices=[p[0] for p in PRESETS] + ["all"],
                        help="実行するプリセット名 (デフォルト: 見積もり表示のみ)")
    parser.add_argument("--width",  type=float, default=None)
    parser.add_argument("--depth",  type=float, default=None)
    parser.add_argument("--h_res",  type=float, default=5.0,
                        help="1ブロックの水平サイズ [m]")
    parser.add_argument("--v_res",  type=float, default=1.0,
                        help="1ブロックの垂直サイズ [m]")
    parser.add_argument("--v_exag", type=float, default=2.0,
                        help="垂直誇張倍率")
    args = parser.parse_args()

    dem_info, inundation = load_dem_and_flood()
    show_estimates(dem_info)

    if args.preset is None and args.width is None:
        print("ヒント: --preset sm_5m  など指定して変換を実行してください")
        print("例:   python make_nbt.py --preset sm_5m")
        print("例:   python make_nbt.py --width 3000 --depth 3000 --h_res 5 --v_exag 3")
        return

    # プリセット実行
    if args.preset == "all":
        targets = PRESETS
    elif args.preset is not None:
        targets = [p for p in PRESETS if p[0] == args.preset]
    else:
        targets = [("custom",
                    args.width or 3000, args.depth or 3000,
                    args.h_res, args.v_res, args.v_exag)]

    for name, w, d, hr, vr, ve in targets:
        print(f"\n--- 変換中: {name} ({w}m×{d}m, {hr}m/block, v×{ve}) ---")
        out = str(OUT_DIR / f"gobo_{name}.nbt")
        size, n_blocks = export_to_nbt(
            dem_info, inundation,
            lat_center=LAT_CENTER, lon_center=LON_CENTER,
            width_m=w, depth_m=d,
            h_res=hr, v_res=vr, v_exag=ve,
            out_path=out,
        )
        print(f"  完了: {out}")


if __name__ == "__main__":
    main()
