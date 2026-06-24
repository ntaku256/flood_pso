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
DEFAULT_WA_XML    = (str(FGD_ALL_DIR / "FG-GML-503561-WA-20251001-0001.xml") + "," +
                     str(FGD_ALL_DIR / "FG-GML-503561-WStrA-20251001-0001.xml"))
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
    ap.add_argument("--wakayama-org", default=None,
                    help="和歌山県 LiDAR オリジナル点群（_org.txt, DSM）の明示パス。建物実高さに使う。"
                         "未指定なら --wakayama-grd の _grd.txt を _org.txt に置換して探す")
    ap.add_argument("--use-fgd", action="store_true",
                    help="国土地理院 FG-GML の建物(BldA)・道路(RdEdg)をローカルから取得して "
                         "地表に重ねる（建物=stone 立体、道路=gravel 上書き、API不要・高精度）")
    ap.add_argument("--fgd-bld", default=DEFAULT_BLD_XML,
                    help="--use-fgd の建物 BldA GML パス。カンマ区切りで複数メッシュ可"
                         "（タイルが境界を跨ぐとき union, 例 503551,503561）")
    ap.add_argument("--fgd-rdedg", default=DEFAULT_RDEDG_XML,
                    help="--use-fgd の道路 RdEdg GML パス。カンマ区切りで複数メッシュ可")
    ap.add_argument("--fgd-wa", default=DEFAULT_WA_XML,
                    help="--use-fgd の水域 WA/WStrA GML パス（河川・池を水面に）。"
                         "カンマ区切りで複数可。空文字で水域無効")
    ap.add_argument("--plateau-bld", default=None,
                    help="PLATEAU CityGML 建物ディレクトリ（udx/bldg）。指定時は建物を PLATEAU の "
                         "正確な footprint+実測高さ(measuredHeight) から生成（高精度版）")
    ap.add_argument("--osm-bld", action="store_true",
                    help="OSM(Overpass) の建物 footprint + LiDAR DSM 高さで建物を生成（いつも通り版）")
    ap.add_argument("--plateau-lod2", action="store_true",
                    help="PLATEAU LOD2 の屋根形状を建物高さに反映（城など。--plateau-bld と併用、やや重い）")
    ap.add_argument("--surface-ortho", action="store_true",
                    help="GSI シームレス空中写真から地表色を決める（viewer 既知のバニラブロックへ"
                         "色マッチ：草/砂/砂利/石/水/岩盤）。傾斜分類より写真寄りの見た目に")
    ap.add_argument("--ortho-zoom", type=int, default=18,
                    help="空中写真タイル zoom（18≈0.6m/px, 17≈1.2m/px）")
    ap.add_argument("--ortho-saturation", type=float, default=1.4,
                    help="空中写真の彩度ブースト（自動レベル後。1=彩度補正なし, 1.4既定, 2+で過飽和）")
    ap.add_argument("--no-litematic", action="store_true",
                    help="既定で併せて出力する .litematic を抑止（.nbt のみ）")
    ap.add_argument("--building-height", type=float, default=6.0,
                    help="建物の高さ [m]（DSM が無いとき/--no-building-heights 時の一律値）")
    ap.add_argument("--no-building-heights", action="store_true",
                    help="和歌山 LiDAR DSM(_org) からの建物実高さ推定を使わず一律高さにする")
    ap.add_argument("--solid-buildings", dest="hollow_buildings", action="store_false",
                    help="建物を空洞化せず中身を詰めた旧来のソリッド建物にする（既定は空洞＝"
                         "ガラス窓・階ごとの床・内部照明・出入口つき）")
    ap.add_argument("--legend-layer", action="store_true",
                    help="地下に土地利用の解釈を色付きガラスで層化して埋め込む（光源なし、コマンド応用向け）。"
                         "重なる洪水・樹木は間隔をあけて別の高さに分離（地形は上に退避）：y=0土地利用(建物=赤/"
                         "道路=黒/海=青/河川=水色/橋=橙/地表=白), y=2洪水(浸水=水色), y=4樹木(緑)")
    ap.add_argument("--no-flood-barrier", action="store_true",
                    help="洪水計算で建物を浸水バリアにしない（従来どおり地形のみで浸水）")
    ap.add_argument("--trees", action="store_true",
                    help="LiDAR class3(植生)から樹冠高マップを作り、陸セルに幹+葉の樹木を立てる"
                         "（建物・道路・水域・海は除外）")
    ap.add_argument("--tree-mode", default="canopy", choices=["canopy", "sparse"],
                    help="樹木配置法: canopy=class3セル毎に幹+葉(密な森) / sparse=間引いた個別樹木(球状樹冠)")
    ap.add_argument("--no-veg-filter", action="store_true",
                    help="建物 DSM(_org) から LiDAR 植生クラス(class 3)を除外しない。"
                         "既定は除外して樹木混入の建物高さ（御坊で建物の24%が影響）を浄化")
    ap.add_argument("--bridges-json", type=str,
                    default=str(REPO_ROOT / "data_cache" / "osm" / "gobo_bridges_geom.json"),
                    help="OSM 橋(bridge=yes highway)の Overpass geom JSON。存在すれば道路が"
                         "水域を渡る箇所に桁+坂+橋脚を立体化（FG-GMLに橋情報が無いため）。"
                         "空文字で無効化")
    ap.add_argument("--evac", action="store_true",
                    help="国土数値情報 P20 避難施設を緑柱+発光マーカーで配置（パーソナル防災ナビ用）")
    ap.add_argument("--evac-xml", type=str,
                    default=str(REPO_ROOT / "data_cache" / "ksj" / "P20-12_30.xml"),
                    help="--evac の P20 避難施設 GML パス（既定: 和歌山県 P20-12）")
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
    ap.add_argument("--scale", type=float, default=1.0,
                    help="1ブロックを細かくして全体を拡大する倍率。1.3 で 1block≈0.77m、"
                         "ブロック数は約1.69倍。LiDAR もこの解像度で再グリッドする")
    ap.add_argument("--tiles", type=str, default=None,
                    help="出力を重なりなくグリッド分割（大スケール時の OOM 回避）。"
                         "'COLSxROWS'（例 4x1=東西4分割, 2x2=四分割）または整数N=Nx1。"
                         "col0=西/row0=北。ファイル名に _c{col}/_r{row}c{col} を付す。"
                         "DEM は1回だけロードしタイルごとに書き出すので省メモリ。")
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

    # --scale 1.3 → LiDAR/ブロックを 1/1.3≈0.77m 解像度に（細かく＝拡大）
    lidar_res = 1.0 / args.scale if args.scale and args.scale > 0 else 1.0

    # DEM 読み込み。--wakayama-grd 指定時は真の1m LiDAR、未指定は GSI 5m DEM。
    if args.wakayama_grd:
        from wakayama_pcd import load_wakayama_dem
        print(f"Loading Wakayama LiDAR DEM (res={lidar_res:.3f}m): {args.wakayama_grd}")
        dem_info = load_wakayama_dem(args.wakayama_grd, res_m=lidar_res)
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

    # 建物高さグリッド（DSM 由来）：和歌山 LiDAR の _org（DSM）があれば DSM-DEM を建物実高に使う。
    building_height_grid = None
    if args.wakayama_grd and not args.no_building_heights:
        org_csv = args.wakayama_org if args.wakayama_org \
            else str(args.wakayama_grd).replace("_grd.txt", "_org.txt")
        if Path(org_csv.split(",")[0].strip()).exists():
            from wakayama_pcd import load_wakayama_dem
            from tellus_data import reproject_to_grid
            veg_classes = None if args.no_veg_filter else (3,)
            print(f"Loading Wakayama LiDAR DSM (building heights)"
                  + ("" if veg_classes is None else f"  [veg-filter: exclude class {veg_classes}]"))
            dsm_info = load_wakayama_dem(org_csv, res_m=lidar_res,
                                         exclude_classes=veg_classes)
            dsm_on_dem = reproject_to_grid(dsm_info["dem"], dsm_info, dem_info, fill_value=np.nan)
            building_height_grid = np.clip(dsm_on_dem - dem, 0, None).astype(np.float32)
            print(f"  obj-height: median={np.nanmedian(building_height_grid):.2f}m "
                  f"99%={np.nanpercentile(building_height_grid,99):.1f}m")

    # 樹冠高グリッド（LiDAR class3=植生のみの DSM − DEM）。--trees で樹木を立てる。
    tree_height_grid = None
    if args.wakayama_grd and args.trees:
        org_csv = args.wakayama_org if args.wakayama_org \
            else str(args.wakayama_grd).replace("_grd.txt", "_org.txt")
        if Path(org_csv.split(",")[0].strip()).exists():
            from wakayama_pcd import load_wakayama_dem
            from tellus_data import reproject_to_grid
            print(f"Loading Wakayama LiDAR canopy (class3)")
            canopy = load_wakayama_dem(org_csv, res_m=lidar_res, keep_classes=(3,))
            canopy_on_dem = reproject_to_grid(canopy["dem"], canopy, dem_info, fill_value=np.nan)
            tree_height_grid = np.clip(canopy_on_dem - dem, 0, None).astype(np.float32)
            n_tree = int(np.sum(tree_height_grid > 2.0))
            print(f"  canopy: median={np.nanmedian(tree_height_grid[tree_height_grid>0]):.1f}m "
                  f"樹木セル(>2m)={n_tree:,}")

    # 洪水バリア: 建物footprintの高さを地形に加えた DEM で浸水計算（水が建物を避け道路を流れる）。
    # --no-flood-barrier で従来どおり地形のみ。地形描画は dem（建物は別レイヤ）のまま。
    dem_flood = dem
    if (not args.no_flood_barrier) and building_height_grid is not None \
            and building_height_grid.shape == dem.shape:
        dem_flood = dem + np.nan_to_num(building_height_grid, nan=0.0)
        print(f"  flood-barrier: 建物 {int(np.sum(building_height_grid > 0.5)):,} セルを浸水バリア化")

    width_m, depth_m, h_res, v_res, v_exag = PRESETS[args.preset]
    if args.width  is not None: width_m = args.width
    if args.depth  is not None: depth_m = args.depth
    if args.h_res  is not None: h_res   = args.h_res
    if args.scale and args.scale != 1.0:
        # 1ブロックを細かく（h/v 両方）＝全体を args.scale 倍に拡大
        h_res = h_res / args.scale
        v_res = v_res / args.scale
    _def_lat, _def_lon = PRESET_CENTERS.get(args.preset, (LAT_CENTER, LON_CENTER))
    if args.wakayama_grd:
        # LiDAR タイルの被覆中心を既定中心にする（タイルは市街地の一部のみ）
        _def_lat = 0.5 * (dem_info["lat_min"] + dem_info["lat_max"])
        _def_lon = 0.5 * (dem_info["lon_min"] + dem_info["lon_max"])
    lat_c = args.center_lat if args.center_lat is not None else _def_lat
    lon_c = args.center_lon if args.center_lon is not None else _def_lon
    # LiDAR タイル指定で size/center を明示していなければ、タイル全域をカバーする
    # （preset の 1km² だとタイル 2km×1.5km の一部しか出ないため）。
    if (args.wakayama_grd and args.width is None and args.depth is None
            and args.center_lat is None and args.center_lon is None):
        import math as _math
        # マージン無し（×1.0）でタイル公称全域を出す。隣接タイルが実データの重なり分で
        # 密着し、タイル間に隙間が出ない（旧 ×0.985 は各タイルを縮めて隙間の原因だった）。
        width_m = (dem_info["lon_max"] - dem_info["lon_min"]) * 111320.0 \
            * _math.cos(_math.radians(lat_c))
        depth_m = (dem_info["lat_max"] - dem_info["lat_min"]) * 111320.0
        print(f"  [wakayama] タイル全域: {width_m:.0f}×{depth_m:.0f}m")
    est = estimate_size(dem_info, lat_c, lon_c,
                        width_m, depth_m, h_res=h_res, v_res=v_res, v_exag=v_exag)
    print(f"  preset={args.preset}  center=({lat_c:.6f},{lon_c:.6f})  "
          f"{width_m}×{depth_m}m  h_res={h_res}m  ~{est['estimated_nbt_MB']} MB/file  "
          f"({est['nx (East-West blocks)']}×{est['nz (North-South blocks)']} blocks)")

    # 建物リスト（PLATEAU 高精度 / OSM いつも通り）を世界範囲で一括読み込み。
    # build_building_maps が各パッチ範囲で描画する（FG-GML の代替。LiDAR DSM が高さを補完）。
    building_list = None
    if args.plateau_bld or args.osm_bld:
        import math as _m
        _hlat = (depth_m / 2 + 120) / 111320.0
        _hlon = (width_m / 2 + 120) / (111320.0 * _m.cos(_m.radians(lat_c)))
        _wb = (lat_c - _hlat, lat_c + _hlat, lon_c - _hlon, lon_c + _hlon)
        if args.plateau_bld:
            from plateau import load_plateau_buildings
            building_list = load_plateau_buildings(args.plateau_bld,
                                                   lat_min=_wb[0], lat_max=_wb[1], lon_min=_wb[2], lon_max=_wb[3],
                                                   lod2=args.plateau_lod2)
        else:
            from tellus_data import fetch_osm_buildings_roads
            _osm = fetch_osm_buildings_roads(_wb[0], _wb[1], _wb[2], _wb[3])
            building_list = [{"coords": b["coords"], "holes": [],
                              "tags": {"fgd_type": "普通建物", "height_m": None}}
                             for b in _osm["buildings"] if len(b.get("coords", [])) >= 4]
            print(f"  [osm-bld] 建物 {len(building_list)} 棟（OSM footprint + LiDAR高さ）")

    # ── タイル分割（--tiles）: 全域を重なりなく COLS×ROWS に割り、各タイルを個別に書き出す。
    #    DEM/inundation は全域で1回だけ計算し、export_to_nbt が中心+幅でクロップする（省メモリ）。
    if args.tiles:
        _s = args.tiles.lower().replace(" ", "")
        n_cols, n_rows = (int(v) for v in _s.split("x")) if "x" in _s else (int(_s), 1)
    else:
        n_cols, n_rows = 1, 1
    import math as _m2
    _lon_per_m = 1.0 / (111320.0 * _m2.cos(_m2.radians(lat_c)))
    _lat_per_m = 1.0 / 111320.0
    _tw, _td = width_m / n_cols, depth_m / n_rows
    tile_specs = []  # (ttag, tile_lat_c, tile_lon_c, tile_w, tile_d)
    for ri in range(n_rows):
        for ci in range(n_cols):
            t_lon = lon_c + (ci - (n_cols - 1) / 2.0) * _tw * _lon_per_m  # col0=西
            t_lat = lat_c + ((n_rows - 1) / 2.0 - ri) * _td * _lat_per_m  # row0=北
            if n_cols == 1 and n_rows == 1:
                ttag = ""
            elif n_rows == 1:
                ttag = f"_c{ci}"
            elif n_cols == 1:
                ttag = f"_r{ri}"
            else:
                ttag = f"_r{ri}c{ci}"
            tile_specs.append((ttag, t_lat, t_lon, _tw, _td))
    if args.tiles:
        print(f"  tiles={n_cols}×{n_rows}  各 {_tw:.0f}×{_td:.0f}m  "
              f"(~{int(_tw/h_res)}×{int(_td/h_res)} blocks/tile)")

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
                dem_flood, source,
                water_level_global=r["water"],
                dh_map=r["dh"],
                sigma_map=r["sigma_map"],
            )
        else:
            inundation = simulate_flood_hd(
                dem_flood, source,
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
        if args.surface_ortho: tsuffix += "_ortho"
        if not args.hollow_buildings: tsuffix += "_solid"
        if args.legend_layer: tsuffix += "_legend"
        usuffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
        base_name = (f"gobo_hd_K{K}{suffix}_seed{args.seed}_{args.preset}_"
                     f"{tag}{tsuffix}{qsuffix}{usuffix}")
        eff_v_exag = args.v_exag if args.v_exag is not None else v_exag

        # タイルごとに書き出す（--tiles 未指定なら tile_specs は ttag="" の単一要素）。
        for ttag, t_lat, t_lon, t_w, t_d in tile_specs:
            out = OUT_DIR / f"{base_name}{ttag}.nbt"
            if ttag:
                print(f"\n  -- tile {ttag}: center=({t_lat:.6f},{t_lon:.6f})  "
                      f"{t_w:.0f}×{t_d:.0f}m → {out.name} --")
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
                # 再現用の幾何パラメータ（タイル単位で記録）
                "scale": float(args.scale),
                "lat_center": float(t_lat), "lon_center": float(t_lon),
                "width_m": float(t_w), "depth_m": float(t_d),
                "h_res_m": float(h_res), "v_res_m": float(v_res),
                "v_exag": float(eff_v_exag),
                "tile": ttag or "full", "tile_grid": f"{n_cols}x{n_rows}",
                "ref_doc": "flood_pso/docs/05_ベンチマーク結果.md",
            }
            if ks > 0:
                meta["K_s"] = ks
                meta["sigma_bounds_m"] = [0.0, 3.0]
                meta["sigma_levels_m"] = [0.0, 0.5, 1.0, 2.0, 4.0]  # flood_sim と整合
                if r["sigma_map"] is not None:
                    meta["sigma_map"] = r["sigma_map"]   # ndarray → Float List + _shape
            export_to_nbt(
                dem_info, inundation,
                lat_center=t_lat, lon_center=t_lon,
                width_m=t_w, depth_m=t_d,
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
                fgd_wa_xml=(args.fgd_wa or None),
                building_list=building_list,
                surface_ortho=args.surface_ortho,
                ortho_zoom=args.ortho_zoom,
                ortho_saturation=args.ortho_saturation,
                building_height_m=args.building_height,
                building_height_grid=building_height_grid,
                tree_height_grid=tree_height_grid,
                tree_mode=args.tree_mode,
                tellus_world_dir=args.tellus_world_dir,
                tellus_world_scale=args.tellus_world_scale,
                tellus_sea_level_y=args.tellus_sea_level_y,
                bridges_json=(args.bridges_json or None),
                evac_xml=(args.evac_xml if args.evac else None),
                hollow_buildings=args.hollow_buildings,
                legend_layer=args.legend_layer,
            )

            # 既定で Litematica (.litematic) も併せて出力（redtact / Litematica mod 用）
            if not args.no_litematic:
                from nbt_to_litematic import structure_nbt_to_litematic
                structure_nbt_to_litematic(str(out))

    print(f"\nAll done. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
