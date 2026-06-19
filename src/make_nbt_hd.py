"""
make_nbt_hd.py
高次元シミュレーション結果（標準PSO / CCPSO2 / Ground Truth）を NBT 化する。

入力: results/benchmark/case_K{K}[_ks{ks}]_seed{seed}.json （benchmark.py の出力）
出力: results/nbt/hd/gobo_hd_K{K}[_ks{ks}]_seed{seed}_{preset}_{method}.nbt
       （flood_pso_meta コンパウンド付き）

実行例:
    python make_nbt_hd.py --K 16 --seed 0
    python make_nbt_hd.py --K 16 --seed 0 --preset md_5m
    python make_nbt_hd.py --K 16 --seed 0 --preset huge_5m
    # Phase1 EX2: sigma_map 付き（benchmark を FLOOD_PSO_SIGMA_MAP_KS=8 で回した結果）
    python make_nbt_hd.py --K 8 --ks 8 --seed 0 --preset md_5m
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
# FG-GML ベクタ（建物 BldA / 道路 RdEdg）。--use-fgd の既定ソース。
FGD_ALL_DIR = REPO_ROOT.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-ALL-20251001"
DEFAULT_BLD_XML   = str(FGD_ALL_DIR / "FG-GML-503561-BldA-20251001-0001.xml")
DEFAULT_RDEDG_XML = str(FGD_ALL_DIR / "FG-GML-503561-RdEdg-20251001-0001.xml")
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
    "amada_200m":   (200,   200,  1, 1, 1.5),  # 天田橋周辺の局所詳細
    "amada_300m_5m": (300,   300,  5, 1, 1.5),  # 天田橋周辺 300m × 300m を 5m/block で（建物・道路含む）
    "amada_500m_5m": (500,   500,  5, 1, 1.5),  # 同 500m × 500m
    "amada_500m_1m": (500,   500,  1, 1, 1.5),  # 同 500m × 500m × 1m/block（建物・道路を高精細に）
    "huge_5m":     (15000,15000,  5, 1, 2),
    # 歩行用：真スケール v_exag=1（崖だらけにならない）、1m/block、御坊市街地 1km²。
    # --use-fgd で建物・道路を載せると「歩ける町」になる。
    "gobo_walk_1km": (1000, 1000, 1, 1, 1.0),
    "gobo_walk_2km": (2000, 2000, 1, 1, 1.0),
}

# preset 既定の中心座標（--center-lat/lon 未指定時）。歩行用は市街地中心へ。
PRESET_CENTERS = {
    "gobo_walk_1km": (33.8875, 135.1515),
    "gobo_walk_2km": (33.8875, 135.1515),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--ks", type=int, default=0,
                    help="K_s (sigma_map size); 0 = scalar sigma (既存)。"
                         "  >0 で benchmark_ks{ks}.json を読み sigma_map も埋め込む")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="md_5m", choices=list(PRESETS.keys()))
    ap.add_argument("--methods", default="pso,ccpso2,gt",
                    help="comma-separated subset of {pso,ccpso2,gt}")
    ap.add_argument("--quality", default="enhanced", choices=["enhanced", "legacy"],
                    help="terrain rendering quality: enhanced (Tellus 風改善, default) または legacy")
    ap.add_argument("--sea-level", type=float, default=0.0,
                    help="海面標高 [m]（enhanced のみ、御坊海岸は 0.0）")
    ap.add_argument("--terrain-source", default="gsi", choices=["gsi", "mapzen", "tellus_world"],
                    help="表示用 DEM のソース。gsi=国土地理院 5m DEM (default、校正と同じ)、"
                         "mapzen=Tellus が使う AWS Mapzen Joerd 全球 DEM "
                         "（inundation を bilinear 再投影して上書き）、"
                         "tellus_world=Tellus mod が生成した Anvil world 全部 "
                         "（地形・地表ブロック共に Tellus そのまま、inundation だけ overlay）")
    ap.add_argument("--tellus-world-dir", default=None,
                    help="terrain_source=tellus_world のとき必須。level.dat のあるディレクトリ。")
    ap.add_argument("--tellus-world-scale", type=float, default=1.0,
                    help="Tellus 世界生成時の world_scale（既定 1 = 1 block/m, real-Earth scale）")
    ap.add_argument("--tellus-sea-level-y", type=int, default=0,
                    help="Tellus 世界の海面 y。dem (m) = block_y - sea_level_y（既定 0）")
    ap.add_argument("--mapzen-zoom", type=int, default=15,
                    help="Mapzen タイル zoom (14≈9.5m, 15≈4.8m, 16≈2.4m)")
    ap.add_argument("--use-esa", action="store_true",
                    help="ESA WorldCover 2021 の土地被覆別ブロック割当を有効化（rasterio 必須）")
    ap.add_argument("--use-osm", action="store_true",
                    help="OpenStreetMap の建物 polygon と道路 polyline を Overpass API から取得して "
                         "地表に重ねる（建物=stone 立体、道路=gravel 上書き）")
    ap.add_argument("--wakayama-grd", default=None,
                    help="和歌山県 LiDAR グラウンド点群テキスト（_grd.txt）を真の1m DEM として使う。"
                         "指定時は GSI 5m DEM の代わりにこれを読む（系VI→緯度経度・1mグリッド化）")
    ap.add_argument("--use-fgd", action="store_true",
                    help="国土地理院 FG-GML の建物(BldA)・道路(RdEdg)をローカルから取得して "
                         "地表に重ねる（建物=stone 立体、道路=gravel 上書き、API不要・高精度）")
    ap.add_argument("--fgd-bld", default=DEFAULT_BLD_XML,
                    help="--use-fgd の建物 BldA GML パス")
    ap.add_argument("--fgd-rdedg", default=DEFAULT_RDEDG_XML,
                    help="--use-fgd の道路 RdEdg GML パス")
    ap.add_argument("--building-height", type=float, default=6.0,
                    help="建物の高さ [m]（既定 6m ≒ 2 階建て）")
    ap.add_argument("--v-exag", type=float, default=None,
                    help="陸の垂直誇張倍率を上書き（プリセットの v_exag を override）")
    ap.add_argument("--smooth-sigma", type=float, default=1.0,
                    help="cliff-aware smoothing の sigma [cells]（既定 1.0）")
    ap.add_argument("--cliff-threshold", type=float, default=0.4,
                    help="急斜面とみなす slope 閾値 [m/m]（既定 0.4 ≒ 22°）")
    ap.add_argument("--center-lat", type=float, default=None,
                    help="出力エリア中心の緯度（デフォルト 33.875 = 御坊市中心）")
    ap.add_argument("--center-lon", type=float, default=None,
                    help="出力エリア中心の経度（デフォルト 135.168）")
    ap.add_argument("--width", type=float, default=None,
                    help="東西幅 [m] を上書き（プリセット値を override）")
    ap.add_argument("--depth", type=float, default=None,
                    help="南北幅 [m] を上書き")
    ap.add_argument("--h-res", type=float, default=None,
                    help="水平解像度 [m/block] を上書き（小さいほど詳細・重い）")
    ap.add_argument("--tag-suffix", type=str, default="",
                    help="出力ファイル名の追加サフィックス（例: --tag-suffix amada）")
    args = ap.parse_args()

    suffix = f"_ks{args.ks}" if args.ks > 0 else ""
    case_path = BENCH_DIR / f"case_K{args.K}{suffix}_seed{args.seed}.json"
    if not case_path.exists():
        sys.exit(f"benchmark JSON not found: {case_path}\n"
                 f"  → run `.venv/bin/python src/benchmark.py` first"
                 + (f"  (with FLOOD_PSO_SIGMA_MAP_KS={args.ks})" if args.ks > 0 else ""))
    case = json.loads(case_path.read_text(encoding="utf-8"))
    K = case["K"]
    ks = int(case.get("ks", 0) or 0)
    if ks != args.ks:
        print(f"  warning: case file ks={ks} does not match --ks={args.ks}")

    # DEM 読み込み。--wakayama-grd 指定時は真の1m LiDAR、未指定は GSI 5m DEM。
    if args.wakayama_grd:
        from wakayama_pcd import load_wakayama_dem
        print(f"Loading Wakayama LiDAR DEM (true 1m): {args.wakayama_grd}")
        dem_info = load_wakayama_dem(args.wakayama_grd)
    else:
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
    if args.width  is not None: width_m = args.width
    if args.depth  is not None: depth_m = args.depth
    if args.h_res  is not None: h_res   = args.h_res
    _def_lat, _def_lon = PRESET_CENTERS.get(args.preset, (LAT_CENTER, LON_CENTER))
    if args.wakayama_grd:
        # LiDAR タイルの被覆中心を既定中心にする（タイルは市街地の一部のみ）
        _def_lat = 0.5 * (dem_info["lat_min"] + dem_info["lat_max"])
        _def_lon = 0.5 * (dem_info["lon_min"] + dem_info["lon_max"])
    lat_c = args.center_lat if args.center_lat is not None else _def_lat
    lon_c = args.center_lon if args.center_lon is not None else _def_lon
    est = estimate_size(dem_info, lat_c, lon_c,
                        width_m, depth_m, h_res=h_res, v_res=v_res, v_exag=v_exag)
    print(f"  preset={args.preset}  center=({lat_c:.6f},{lon_c:.6f})  "
          f"{width_m}×{depth_m}m  h_res={h_res}m  ~{est['estimated_nbt_MB']} MB/file  "
          f"({est['nx (East-West blocks)']}×{est['nz (North-South blocks)']} blocks)")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # 各手法の (water, dh_map[, sigma_map]) を抽出（ground truth は case["gt"] から）
    def _sigma_map_from(method_block):
        sm = method_block.get("best_sigma_map")
        if ks > 0 and sm is not None:
            return np.array(sm, dtype=np.float64)
        return None

    runs = {}
    if "pso" in methods:
        runs["pso"] = {
            "water": float(case["pso"]["best_w"]),
            "dh":    np.array(case["pso"]["best_dh"], dtype=np.float64),
            "sigma_map": _sigma_map_from(case["pso"]),
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
            "sigma_map": _sigma_map_from(case["ccpso2"]),
            "loss":  float(case["ccpso2"]["loss"]),
            "iou":   float(case["ccpso2"]["iou"]),
            "dh_rmse": float(case["ccpso2"]["dh_rmse"]),
            "n_evals": int(case["ccpso2"]["n_evals"]),
            "elapsed_s": float(case["ccpso2"]["elapsed_s"]),
            "method_long": f"CCPSO2 (s={case['ccpso2']['s']}, custom impl)",
        }
    if "gt" in methods:
        # 合成 GT は scalar SIGMA で生成されているため sigma_map_true は持たない。
        # ks>0 でも GT は scalar SIGMA で simulate（docs/12 §10.6 の ill-posed 注記参照）。
        runs["gt"] = {
            "water": float(case["gt"]["water_true"]),
            "dh":    np.array(case["gt"]["dh_true"], dtype=np.float64),
            "sigma_map": None,
            "loss":  0.0,
            "iou":   1.0,
            "dh_rmse": 0.0,
            "n_evals": 0,
            "elapsed_s": 0.0,
            "method_long": "Synthetic ground truth (target for inverse problem)",
        }

    for tag, r in runs.items():
        sm_note = f", sigma_map K_s={ks}" if r["sigma_map"] is not None else ""
        print(f"\n--- Generating NBT: {tag} "
              f"(water={r['water']:.3f}, IoU={r['iou']:.3f}{sm_note}) ---")
        # 5m フル解像度 DEM 上でシミュレーションを再実行
        if r["sigma_map"] is not None:
            inundation = simulate_flood_hd(
                dem, source,
                water_level_global=r["water"],
                dh_map=r["dh"],
                sigma_map=r["sigma_map"],
            )
        else:
            inundation = simulate_flood_hd(
                dem, source,
                water_level_global=r["water"],
                dh_map=r["dh"],
                sigma=SIGMA,
            )
        flooded = int(np.sum(inundation > 0.05))
        print(f"  full-res flooded cells: {flooded:,}")

        qsuffix = "" if args.quality == "enhanced" else f"_{args.quality}"
        tsuffix = "" if args.terrain_source == "gsi" else f"_{args.terrain_source}"
        if args.use_esa: tsuffix += "_esa"
        if args.use_osm: tsuffix += "_osm"
        if args.use_fgd: tsuffix += "_fgd"
        usuffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
        out = OUT_DIR / f"gobo_hd_K{K}{suffix}_seed{args.seed}_{args.preset}_{tag}{tsuffix}{qsuffix}{usuffix}.nbt"
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
        if ks > 0:
            meta["K_s"] = ks
            meta["sigma_bounds_m"] = [0.0, 3.0]
            meta["sigma_levels_m"] = [0.0, 0.5, 1.0, 2.0, 4.0]  # flood_sim と整合
            if r["sigma_map"] is not None:
                meta["sigma_map"] = r["sigma_map"]   # ndarray → Float List + _shape
        eff_v_exag = args.v_exag if args.v_exag is not None else v_exag
        export_to_nbt(
            dem_info, inundation,
            lat_center=lat_c, lon_center=lon_c,
            width_m=width_m, depth_m=depth_m,
            h_res=h_res, v_res=v_res, v_exag=eff_v_exag,
            out_path=str(out), meta=meta,
            terrain_quality=args.quality,
            sea_level_m=args.sea_level,
            smooth_sigma_cells=args.smooth_sigma,
            cliff_threshold_m_per_m=args.cliff_threshold,
            terrain_source=args.terrain_source,
            mapzen_zoom=args.mapzen_zoom,
            use_esa=args.use_esa,
            use_osm=args.use_osm,
            use_fgd=args.use_fgd,
            fgd_bld_xml=args.fgd_bld,
            fgd_rdedg_xml=args.fgd_rdedg,
            building_height_m=args.building_height,
            tellus_world_dir=args.tellus_world_dir,
            tellus_world_scale=args.tellus_world_scale,
            tellus_sea_level_y=args.tellus_sea_level_y,
        )

    print(f"\nAll done. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
