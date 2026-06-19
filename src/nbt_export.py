"""
nbt_export.py
DEM + 洪水シミュレーション結果 → Minecraft Structure NBT (.nbt) 変換

【スケール設計】
  h_res  : 水平解像度 [m/block]  例: 5 → 5m = 1block
  v_res  : 垂直解像度 [m/block]  例: 1 → 1m = 1block
  v_exag : 垂直誇張倍率          例: 3 → 実際の3倍の高さで表現

【ブロック構成】
  石 (stone)        : 地表より下の地盤
  草 (grass_block)  : 地表面
  砂 (sand)         : 海岸・低標高地
  水 (water)        : 洪水域
  空気 (air)        : それ以外（省略 = デフォルト）

【構造体サイズ推奨】
  NBTビューアーで快適に動作するおおよその目安:
    〜 500×500 ブロック : 軽量 (~10MB)
    500〜1000×1000      : 普通 (~50MB)
    1000〜2000×2000     : 重い (~200MB、要スペック)
"""

import gzip
import struct
import datetime as _dt
import subprocess as _sp
import numpy as np
from pathlib import Path
import nbtlib


# ─────────────────────────────────────────────────────────────
# ブロックパレット定義
# ─────────────────────────────────────────────────────────────

# パレットは block_palette.py（単一真実源, ~80 バニラブロック）から生成。
# water/blue_ice はアニメーションテクスチャ回避のため stained_glass で代替（block_palette 内で定義）。
from block_palette import (BLOCKS as _BLOCKS, PALETTE_KEYS as _PALETTE_KEYS,
                           block_state_properties as _block_state_properties)


def _palette_compound(key: str) -> nbtlib.Compound:
    name = _BLOCKS[key][0]
    props = _block_state_properties(name)
    if props:
        return nbtlib.Compound({
            "Name": nbtlib.String(name),
            "Properties": nbtlib.Compound({k: nbtlib.String(v) for k, v in props.items()}),
        })
    return nbtlib.Compound({"Name": nbtlib.String(name)})


PALETTE = {k: _palette_compound(k) for k in _PALETTE_KEYS}

PALETTE_LIST_KEYS = list(PALETTE.keys())
PALETTE_INDEX = {k: i for i, k in enumerate(PALETTE_LIST_KEYS)}


def block_id(name: str) -> nbtlib.Int:
    return nbtlib.Int(PALETTE_INDEX[name])


def surface_block(elev_m: float) -> str:
    """標高から地表ブロック種別を決定。"""
    if elev_m < 0.5:
        return "sand"
    elif elev_m < 2.0:
        return "gravel"
    else:
        return "grass"


# ─────────────────────────────────────────────────────────────
# サイズ見積もり
# ─────────────────────────────────────────────────────────────

def estimate_size(dem_info: dict, lat_center: float, lon_center: float,
                  width_m: float, depth_m: float,
                  h_res: float = 5.0, v_res: float = 1.0, v_exag: float = 1.0) -> dict:
    """
    指定パラメータでのブロック数とファイルサイズを見積もる。
    実際に変換する前に確認するために使う。
    """
    nx = int(width_m / h_res)
    nz = int(depth_m / h_res)

    # 対象エリアのDEM統計（近似）
    dem = dem_info["dem"]
    row_c = round((dem_info["lat_max"] - lat_center) / dem_info["res_lat"])
    col_c = round((lon_center - dem_info["lon_min"]) / dem_info["res_lon"])
    h_cells = int(width_m / dem_info["res_lat"] / 111320)
    d_cells = int(depth_m / dem_info["res_lon"] / 111320 / np.cos(np.radians(lat_center)))
    r0 = max(0, row_c - d_cells // 2)
    r1 = min(dem.shape[0], row_c + d_cells // 2)
    c0 = max(0, col_c - h_cells // 2)
    c1 = min(dem.shape[1], col_c + h_cells // 2)

    patch = dem[r0:r1, c0:c1]
    valid = patch[~np.isnan(patch)]
    max_elev = float(valid.max()) if len(valid) > 0 else 100.0
    min_elev = float(valid.min()) if len(valid) > 0 else 0.0

    ny = int(max_elev / v_res * v_exag) + 10

    # 表面ブロック数 ≈ nx × nz (地表1層 + 水ブロック)
    surface_blocks = nx * nz * 2   # 地表 + 可能な水
    # 地盤柱 (各セルの下3ブロックを埋める場合)
    solid_blocks = nx * nz * 3

    total_blocks = surface_blocks + solid_blocks
    est_mb = total_blocks * 40 / 1e6  # 40 bytes/block 目安

    return {
        "nx (East-West blocks)": nx,
        "ny (Vertical blocks)": ny,
        "nz (North-South blocks)": nz,
        "total_blocks (estimate)": total_blocks,
        "estimated_nbt_MB": round(est_mb, 1),
        "elev_range_m": f"{min_elev:.1f} ~ {max_elev:.1f}",
        "area_km2": round(width_m * depth_m / 1e6, 2),
    }


# ─────────────────────────────────────────────────────────────
# DEM → ブロック配列変換
# ─────────────────────────────────────────────────────────────

def dem_to_blocks(dem_patch: np.ndarray,
                  inundation_patch: np.ndarray,
                  h_res_dem: float, h_res_block: float,
                  v_res: float = 1.0, v_exag: float = 1.0,
                  flood_threshold: float = 0.05) -> list:
    """
    DEM パッチとfloat浸水深マップから NBT ブロックエントリリストを生成。
    numpy ベクトル化により高速処理。

    Returns
    -------
    blocks_list : list of nbtlib.Compound (pos + state)
    size        : [nx, ny, nz]
    """
    # ─── ダウンサンプル (ブロック平均) ───
    factor = max(1, round(h_res_block / h_res_dem))
    H, W = dem_patch.shape
    nz = H // factor
    nx = W // factor

    # reshape して nanmean / nanmax
    d = dem_patch[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
    i = inundation_patch[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dem_ds = np.nanmean(d, axis=(1, 3))  # (nz, nx)
        idn_ds = np.nanmax(i,  axis=(1, 3))  # (nz, nx)

    # ─── y座標変換 ───
    scale = v_exag / v_res
    y_surf = np.where(np.isnan(dem_ds), 0,
                      np.maximum(1, (dem_ds * scale).astype(int)))
    y_flood_top = np.where(idn_ds > flood_threshold,
                           ((dem_ds + idn_ds) * scale).astype(int),
                           y_surf)

    valid_elevs = dem_ds[~np.isnan(dem_ds)]
    if len(valid_elevs) == 0:
        return [], [nx, 1, nz]
    max_y = min(int(valid_elevs.max() * scale) + 5, 500)

    # ─── bx, bz の座標グリッド ───
    BZ, BX = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")  # (nz, nx)

    blocks = []

    # --- 海・NoData セル → 水ブロック y=0 ---
    sea_mask = np.isnan(dem_ds)
    for bz_v, bx_v in zip(BZ[sea_mask].tolist(), BX[sea_mask].tolist()):
        blocks.append(nbtlib.Compound({
            "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(0), nbtlib.Int(bz_v)]),
            "state": block_id("water"),
        }))

    land_mask = ~sea_mask

    # --- 地盤柱 (地表の3ブロック下まで stone) ---
    ys  = y_surf[land_mask]
    bx_ = BX[land_mask]
    bz_ = BZ[land_mask]
    for bx_v, bz_v, y_top in zip(bx_.tolist(), bz_.tolist(), ys.tolist()):
        for dy in range(max(0, y_top - 3), y_top):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(dy), nbtlib.Int(bz_v)]),
                "state": block_id("stone"),
            }))

    # --- 地表ブロック ---
    elevs = dem_ds[land_mask]
    for bx_v, bz_v, y_top, elev in zip(bx_.tolist(), bz_.tolist(),
                                         ys.tolist(), elevs.tolist()):
        surf = surface_block(elev)
        blocks.append(nbtlib.Compound({
            "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(y_top), nbtlib.Int(bz_v)]),
            "state": block_id(surf),
        }))

    # --- 浸水ブロック ---
    flood_mask = land_mask & (idn_ds > flood_threshold)
    yf_top = y_flood_top[flood_mask]
    ys_f   = y_surf[flood_mask]
    bx_f   = BX[flood_mask]
    bz_f   = BZ[flood_mask]
    for bx_v, bz_v, y_s, y_ft in zip(bx_f.tolist(), bz_f.tolist(),
                                       ys_f.tolist(), yf_top.tolist()):
        for fy in range(y_s + 1, min(y_ft + 1, y_s + 30)):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                "state": block_id("water"),
            }))

    return blocks, [nx, max_y + 1, nz]


# ─────────────────────────────────────────────────────────────
# NBT Structure ファイル書き出し
# ─────────────────────────────────────────────────────────────

def _git_revision() -> str:
    try:
        out = _sp.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_meta_compound(meta: dict) -> nbtlib.Compound:
    """
    Python dict → nbtlib.Compound 変換（flood_pso_meta 用）。
    対応型: str, int, float, bool, list (numeric/str), dict (再帰), np.ndarray (Float List 化)。
    """
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = build_meta_compound(v)
        elif isinstance(v, np.ndarray):
            arr = v.astype(np.float64).flatten().tolist()
            out[k] = nbtlib.List[nbtlib.Float]([nbtlib.Float(x) for x in arr])
            out[k + "_shape"] = nbtlib.List[nbtlib.Int]([nbtlib.Int(s) for s in v.shape])
        elif isinstance(v, bool):
            out[k] = nbtlib.Byte(1 if v else 0)
        elif isinstance(v, int):
            out[k] = nbtlib.Int(v)
        elif isinstance(v, float):
            out[k] = nbtlib.Double(v)
        elif isinstance(v, str):
            out[k] = nbtlib.String(v)
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                out[k] = nbtlib.List[nbtlib.String]([])
            elif all(isinstance(x, str) for x in v):
                out[k] = nbtlib.List[nbtlib.String]([nbtlib.String(x) for x in v])
            elif all(isinstance(x, (int, np.integer)) for x in v):
                out[k] = nbtlib.List[nbtlib.Int]([nbtlib.Int(int(x)) for x in v])
            else:
                out[k] = nbtlib.List[nbtlib.Double]([nbtlib.Double(float(x)) for x in v])
        else:
            out[k] = nbtlib.String(str(v))
    return nbtlib.Compound(out)


def write_nbt_structure(blocks: list, size: list, out_path: str,
                        meta: dict | None = None):
    """
    Minecraft Structure NBT 形式でファイルを書き出す。
    Minecraft 1.17+ の Structure Block 形式。

    meta が与えられたら ``flood_pso_meta`` キーとして埋め込む（任意のkey-value辞書）。
    """
    palette_list = nbtlib.List[nbtlib.Compound]([PALETTE[k] for k in PALETTE_LIST_KEYS])
    blocks_list  = nbtlib.List[nbtlib.Compound](blocks)
    size_list    = nbtlib.List[nbtlib.Int]([nbtlib.Int(s) for s in size])

    root = {
        "DataVersion": nbtlib.Int(4671),   # Minecraft 1.21.4相当
        "author":      nbtlib.String("flood_pso"),
        "size":        size_list,
        "palette":     palette_list,
        "blocks":      blocks_list,
        "entities":    nbtlib.List[nbtlib.Compound]([]),
    }

    if meta is not None:
        full_meta = dict(meta)
        full_meta.setdefault("schema_version", 1)
        full_meta.setdefault("generator", "flood_pso/nbt_export.py")
        full_meta.setdefault("timestamp_utc",
                             _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z")
        full_meta.setdefault("git_revision", _git_revision())
        root["flood_pso_meta"] = build_meta_compound(full_meta)

    nbt_file = nbtlib.File(nbtlib.Compound(root))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbt_file.save(str(out_path), gzipped=True)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB, {len(blocks):,} blocks)"
          + (" [+meta]" if meta is not None else ""))


# ─────────────────────────────────────────────────────────────
# メイン変換関数
# ─────────────────────────────────────────────────────────────

def export_to_nbt(dem_info: dict, inundation: np.ndarray,
                  lat_center: float, lon_center: float,
                  width_m: float = 5000.0, depth_m: float = 5000.0,
                  h_res: float = 5.0,
                  v_res: float = 1.0, v_exag: float = 2.0,
                  out_path: str = "output.nbt",
                  meta: dict | None = None,
                  terrain_quality: str = "enhanced",
                  sea_level_m: float = 0.0,
                  ocean_max_depth_m: float = 8.0,
                  smooth_sigma_cells: float = 1.0,
                  cliff_threshold_m_per_m: float = 0.4,
                  v_exag_sea: float | None = None,
                  deep_ground: int = 8,
                  terrain_source: str = "gsi",
                  mapzen_zoom: int = 15,
                  use_esa: bool = False,
                  use_osm: bool = False,
                  use_fgd: bool = False,
                  fgd_bld_xml: str | None = None,
                  fgd_rdedg_xml: str | None = None,
                  surface_ortho: bool = False,
                  ortho_zoom: int = 18,
                  ortho_saturation: float = 1.4,
                  building_height_m: float = 6.0,
                  building_height_grid: np.ndarray | None = None,
                  tellus_world_dir: str | None = None,
                  tellus_world_scale: float = 1.0,
                  tellus_sea_level_y: int = 0,
                  bridges_json: str | None = None):
    """
    DEMと浸水マップの指定範囲をMinecraft NBT Structureに変換する。

    Parameters
    ----------
    dem_info   : mosaic_tiles() の返り値
    inundation : simulate_flood() の返り値 (dem と同形状)
    lat_center, lon_center : 出力エリアの中心座標
    width_m    : 東西幅 [m]
    depth_m    : 南北幅 [m]
    h_res      : 1ブロックの水平サイズ [m] (DEMの整数倍推奨)
    v_res      : 1ブロックの垂直サイズ [m]
    v_exag     : 垂直誇張倍率（terrain_quality="enhanced" では陸用、海は v_exag_sea）
    out_path   : 出力 .nbt ファイルパス
    meta       : NBT に埋め込むメタデータ辞書（任意）

    terrain_quality : "enhanced" (default, Tellus 風の改善) | "legacy" (旧 dem_to_blocks)
    sea_level_m     : 海面標高 [m]（enhanced のみ）
    ocean_max_depth_m : 沖合の最大水深 [m]（enhanced のみ、200m 沖でこの値に到達）
    smooth_sigma_cells: cliff-aware smoothing の sigma [cells]（enhanced のみ）
    cliff_threshold_m_per_m: 急斜面判定の slope 閾値 [m/m]（enhanced のみ）
    v_exag_sea      : 海中の垂直誇張倍率。None なら v_exag * 0.33（enhanced のみ）
    deep_ground     : 陸の地盤柱の深さ [block]（enhanced のみ、既定 8）

    terrain_source  : "gsi" (default, 国土地理院 5m DEM をそのまま使う)
                      "mapzen" (Tellus が使う AWS Mapzen Joerd 全球 DEM を取得して
                                 表示用地形を差し替え。inundation は bilinear で再投影)
                      "tellus_world" (Tellus mod が生成済の Anvil world フォルダを読み、
                                       高さも地表ブロックも Tellus と同一にする。
                                       --tellus-world-dir 必須)
    mapzen_zoom     : Mapzen タイルの zoom（14 ≈ 9.5m, 15 ≈ 4.8m, 16 ≈ 2.4m）
    use_esa         : ESA WorldCover 2021 v200 を取得して土地被覆別ブロック割当を有効化
                      （rasterio 必須、Tellus.MountainSurfaceRules ロジック相当）

    tellus_world_dir   : terrain_source="tellus_world" のとき必須。level.dat のあるパス
    tellus_world_scale : Tellus 世界生成時の world_scale（既定 1 = real-Earth scale）
    tellus_sea_level_y : Tellus 世界の海面 y。dem (m) = block_y - tellus_sea_level_y
    """
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * np.cos(np.radians(lat_center)))

    # ─── terrain_source 分岐：表示用 dem を Mapzen / Tellus world で差し替え ───
    cover_patch = None
    tellus_surface_grid = None      # tellus_world のときのみセット (object dtype, palette key)
    tellus_inundation_grid = None   # tellus_world のときに使う再投影済 inundation
    if terrain_source == "tellus_world":
        if tellus_world_dir is None:
            raise ValueError("terrain_source='tellus_world' requires tellus_world_dir")
        from anvil_loader import (
            TellusWorld, lon_to_blockX, lat_to_blockZ, blockX_to_lon, blockZ_to_lat,
        )
        tw = TellusWorld(tellus_world_dir, world_scale=tellus_world_scale)

        # 中心 (lat,lon) を Tellus 投影 block 座標に
        bx_c = lon_to_blockX(lon_center, tellus_world_scale)
        bz_c = lat_to_blockZ(lat_center, tellus_world_scale)
        # h_res に整列した block 範囲（h_res>1 でも block 単位で取得して後段の reshape に任せる）
        half_w = (width_m / 2.0) / tellus_world_scale
        half_d = (depth_m / 2.0) / tellus_world_scale
        bx_min = int(np.floor(bx_c - half_w))
        bx_max = bx_min + int(np.ceil(2 * half_w)) - 1
        bz_min = int(np.floor(bz_c - half_d))
        bz_max = bz_min + int(np.ceil(2 * half_d)) - 1

        print(f"[tellus_world] center (lat,lon)=({lat_center:.6f},{lon_center:.6f}) "
              f"→ block (X,Z)=({bx_c:.1f},{bz_c:.1f})  "
              f"region ({int(bx_c)>>9},{int(bz_c)>>9})")
        dem_blocks_y, surf_grid, stats = tw.fetch_grid(bx_min, bx_max, bz_min, bz_max)
        print(f"[tellus_world] grid shape={stats['shape']}  "
              f"loaded {stats['n_loaded_cells']}/{stats['n_cells']} cells  "
              f"chunks {stats['n_chunks_loaded']}/{stats['n_chunks_total']}")
        if stats['n_loaded_cells'] == 0:
            raise RuntimeError(
                f"[tellus_world] no chunks generated within bbox "
                f"(bx={bx_min}..{bx_max}, bz={bz_min}..{bz_max}). "
                f"Tellus 側で先にこの座標を訪れて chunk を生成して下さい。"
            )

        # block_y → 標高 m に変換（world_scale=1 で 1 block = 1 m）
        dem_render = (dem_blocks_y - tellus_sea_level_y) * tellus_world_scale
        # NaN は np.nan のまま（dem_blocks_y 側で NaN）

        # inundation を Tellus grid に bilinear 再投影
        # Tellus grid 各セルの (lat, lon) を計算 → src GSI grid の (row,col) に変換
        H, W = dem_render.shape
        bx_grid = bx_min + np.arange(W)             # (W,)
        bz_grid = bz_min + np.arange(H)             # (H,)
        lon_grid = bx_grid * tellus_world_scale / 111319.49166666667
        # lat は Mercator 逆投影
        mY = -bz_grid * tellus_world_scale          # (H,)
        lat_grid = np.degrees(np.arctan(np.sinh(mY / 6378137.0)))   # (H,)
        # GSI dem_info の (row,col) インデックスへ
        src_lat_max = dem_info['lat_max']
        src_lon_min = dem_info['lon_min']
        src_res_lat = dem_info['res_lat']
        src_res_lon = dem_info['res_lon']
        idn_src = inundation
        src_rows = (src_lat_max - lat_grid) / src_res_lat   # (H,)
        src_cols = (lon_grid - src_lon_min) / src_res_lon   # (W,)
        # メッシュ → map_coordinates
        from scipy.ndimage import map_coordinates
        rr, cc = np.meshgrid(src_rows, src_cols, indexing='ij')   # (H,W)
        tellus_inundation_grid = map_coordinates(
            np.nan_to_num(idn_src, nan=0.0), [rr, cc], order=1, mode='constant', cval=0.0,
        ).astype(np.float32)
        # 範囲外（src の dem 範囲外）はゼロのまま

        # dem_info を Tellus grid に書き換える（後段の patch 抽出ロジックを通すため疑似的に）
        # ここで「render 側の patch を生成する」段は既に終わっているので、
        # 代わりに dem_info_render を作り、エリア抽出をスキップさせるフラグ的に使う。
        dem_info_render = {
            'dem': dem_render,
            'lat_max': float(lat_grid[0]),
            'lat_min': float(lat_grid[-1]),
            'lon_min': float(lon_grid[0]),
            'lon_max': float(lon_grid[-1]),
            'res_lat': float((lat_grid[0] - lat_grid[-1]) / max(1, H - 1)),
            'res_lon': float((lon_grid[-1] - lon_grid[0]) / max(1, W - 1)),
        }
        inundation_render = tellus_inundation_grid
        tellus_surface_grid = surf_grid
        print(f"[tellus_world] elevation range (m): "
              f"min={float(np.nanmin(dem_render)):.1f} median={float(np.nanmedian(dem_render)):.1f} "
              f"max={float(np.nanmax(dem_render)):.1f}  "
              f"flood max in patch={float(tellus_inundation_grid.max()):.2f}m")

    elif terrain_source == "mapzen":
        from tellus_data import fetch_mapzen_dem, reproject_to_grid
        # 描画 BBOX（やや広めに取って端の補間を安定化）
        half_lat = (depth_m / 2) * lat_per_m
        half_lon = (width_m / 2) * lon_per_m
        margin = max(half_lat, half_lon) * 0.05
        mapzen_info = fetch_mapzen_dem(
            lat_min=lat_center - half_lat - margin,
            lat_max=lat_center + half_lat + margin,
            lon_min=lon_center - half_lon - margin,
            lon_max=lon_center + half_lon + margin,
            zoom=mapzen_zoom,
        )
        # inundation を Mapzen grid に bilinear 再投影
        inundation_render = reproject_to_grid(
            inundation, src_meta=dem_info, dst_meta=mapzen_info, fill_value=0.0,
        )
        dem_info_render = mapzen_info
        if use_esa:
            from tellus_data import fetch_esa_worldcover
            esa = fetch_esa_worldcover(
                lat_min=lat_center - half_lat - margin,
                lat_max=lat_center + half_lat + margin,
                lon_min=lon_center - half_lon - margin,
                lon_max=lon_center + half_lon + margin,
            )
            # ESA を Mapzen grid に最近傍で reproject（class 値はカテゴリなので nearest）
            cover_patch_full = reproject_to_grid(esa["cover"].astype(np.float32),
                                                   src_meta=esa,
                                                   dst_meta=mapzen_info,
                                                   fill_value=0.0).astype(np.uint8)
        print(f"[mapzen] dem grid {dem_info_render['dem'].shape}  "
              f"reprojected inundation max={float(inundation_render.max()):.2f}m")
    elif terrain_source == "gsi":
        dem_info_render = dem_info
        inundation_render = inundation
    else:
        raise ValueError(f"unknown terrain_source: {terrain_source} (use 'gsi' or 'mapzen')")

    dem = dem_info_render["dem"]
    lat_max = dem_info_render["lat_max"]
    lon_min = dem_info_render["lon_min"]
    res_lat = dem_info_render["res_lat"]
    res_lon = dem_info_render["res_lon"]
    bh_patch = None   # 建物高さ[m] パッチ（DSM 由来, 任意）

    if terrain_source == "tellus_world":
        # tellus_world では fetch_grid 段階で既に target bbox を切り出してあるので
        # 全部使う（h_res は world_scale=1 で 1 m/block 固定）。
        dem_patch = dem
        idn_patch = inundation_render
        patch_bbox_latlon = (
            float(dem_info_render["lat_min"]),
            float(dem_info_render["lat_max"]),
            float(dem_info_render["lon_min"]),
            float(dem_info_render["lon_max"]),
        )
    else:
        # 中心ピクセル
        row_c = round((lat_max - lat_center) / res_lat)
        col_c = round((lon_center - lon_min) / res_lon)

        # エリアを DEMセル数で計算
        half_rows = int((depth_m / 2) * lat_per_m / res_lat)
        half_cols = int((width_m / 2) * lon_per_m / res_lon)

        r0 = max(0, row_c - half_rows)
        r1 = min(dem.shape[0], row_c + half_rows)
        c0 = max(0, col_c - half_cols)
        c1 = min(dem.shape[1], col_c + half_cols)

        dem_patch = dem[r0:r1, c0:c1]
        idn_patch = inundation_render[r0:r1, c0:c1]
        if building_height_grid is not None and building_height_grid.shape == dem.shape:
            bh_patch = building_height_grid[r0:r1, c0:c1]
        if terrain_source == "mapzen" and use_esa:
            cover_patch = cover_patch_full[r0:r1, c0:c1]

        # patch の経緯度 bbox（OSM 取得 + grid 変換に使用）
        patch_bbox_latlon = (
            lat_max - r1 * res_lat,   # lat_min（南端）
            lat_max - r0 * res_lat,   # lat_max（北端）
            lon_min + c0 * res_lon,   # lon_min
            lon_min + c1 * res_lon,   # lon_max
        )

    print(f"DEM patch: {dem_patch.shape} cells = {dem_patch.shape[1]*res_lon/lon_per_m:.0f}m W x {dem_patch.shape[0]*res_lat/lat_per_m:.0f}m N "
          f"[source={terrain_source}]")

    h_res_dem = res_lat / lat_per_m   # DEMセル = 何m か

    # ブロック解像度が DEM 解像度より細かい場合、bilinear で upsample
    # （例: 200m 局所詳細で h_res=1m, GSI 5m DEM → 5x upsample）
    if h_res > 0 and h_res < h_res_dem * 0.95:
        from scipy.ndimage import zoom as nd_zoom
        up_factor = h_res_dem / h_res
        dem_patch = nd_zoom(dem_patch, up_factor, order=1, mode="nearest")
        idn_patch = nd_zoom(idn_patch, up_factor, order=1, mode="nearest")
        if cover_patch is not None:
            cover_patch = nd_zoom(cover_patch, up_factor, order=0, mode="nearest")
        if bh_patch is not None:
            bh_patch = nd_zoom(bh_patch, up_factor, order=1, mode="nearest")
        print(f"  [upsample] {up_factor:.2f}× source DEM → patch shape {dem_patch.shape}")
        h_res_dem = h_res

    if terrain_quality == "enhanced":
        from terrain_render import dem_to_blocks_enhanced, build_osm_masks
        v_es = v_exag_sea if v_exag_sea is not None else v_exag * 0.33

        # OSM 取得 + ブロック grid 上の建物・道路 mask を事前生成
        building_mask = road_mask = None
        building_height_block = None   # P1: per-building 集約のフラット高さ（FG-GML 経路）
        building_id_grid = None        # P2: 建物ごとの整数ラベル
        building_wall_keys = None      # P2: 建物 id → 壁ブロックキー
        building_roof_keys = None      # P2: 建物 id → 屋根ブロックキー(fallback)
        if use_osm:
            from tellus_data import fetch_osm_buildings_roads
            osm = fetch_osm_buildings_roads(
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )
            # ダウンサンプル後の grid サイズ（dem_to_blocks_enhanced と同じ計算式）
            factor = max(1, round(h_res / h_res_dem))
            nz_g = dem_patch.shape[0] // factor
            nx_g = dem_patch.shape[1] // factor
            building_mask, road_mask = build_osm_masks(
                osm, patch_bbox_latlon,
                grid_h=nz_g, grid_w=nx_g, h_res_block_m=h_res,
            )
            print(f"  [osm] buildings={osm['n_buildings']}  roads={osm['n_roads']}  "
                  f"→ mask cells: building={int(building_mask.sum())}  road={int(road_mask.sum())}")

        # FG-GML（国土地理院ローカルベクタ）建物・道路 mask。OSM と併用時は union。
        if use_fgd:
            import warnings as _warnings
            from fgd_vector import load_fgd_buildings_roads
            from terrain_render import build_building_maps
            fgd = load_fgd_buildings_roads(
                fgd_bld_xml, fgd_rdedg_xml,
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )
            factor = max(1, round(h_res / h_res_dem))
            nz_g = dem_patch.shape[0] // factor
            nx_g = dem_patch.shape[1] // factor
            # 道路だけ従来 mask（建物は per-building 集約で別途生成）
            _, rm_f = build_osm_masks(
                {"roads": fgd["roads"]}, patch_bbox_latlon,
                grid_h=nz_g, grid_w=nx_g, h_res_block_m=h_res,
            )
            # P1: 建物高さ[m] を block grid にダウンサンプル → 各 footprint で p75 集約しフラット化
            # P2: 同時に建物 id / type 別の壁・屋根キーも生成
            dsm_h_block = None
            if bh_patch is not None:
                bpf = bh_patch[:nz_g*factor, :nx_g*factor].reshape(nz_g, factor, nx_g, factor)
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", RuntimeWarning)
                    dsm_h_block = np.nanmean(bpf, axis=(1, 3))
            bmaps = build_building_maps(
                fgd["buildings"], dsm_h_block, patch_bbox_latlon, nz_g, nx_g,
            )
            bm_f = bmaps["mask"]
            building_mask = bm_f if building_mask is None else (building_mask | bm_f)
            road_mask = rm_f if road_mask is None else (road_mask | rm_f)
            building_height_block = bmaps["height"]
            building_id_grid = bmaps["id"]
            building_wall_keys = bmaps["wall_keys"]
            building_roof_keys = bmaps["roof_keys"]
            _bh_in = building_height_block[np.isfinite(building_height_block)]
            _med = float(np.median(_bh_in)) if _bh_in.size else 0.0
            print(f"  [fgd] buildings={fgd['n_buildings']}  roads={fgd['n_roads']}  "
                  f"→ mask cells: building={int(bm_f.sum())}  road={int(rm_f.sum())}  "
                  f"per-building flat-height median={_med:.1f}m  n_bld={len(building_wall_keys)}")

        # 地表色を空中写真から（最優先の surface_grid_override に流す）
        surface_override = tellus_surface_grid
        if surface_ortho:
            from ortho_surface import ortho_surface_grid
            ph, pw = dem_patch.shape
            dst_meta = {
                "lat_min": patch_bbox_latlon[0], "lat_max": patch_bbox_latlon[1],
                "lon_min": patch_bbox_latlon[2], "lon_max": patch_bbox_latlon[3],
                "res_lat": (patch_bbox_latlon[1] - patch_bbox_latlon[0]) / ph,
                "res_lon": (patch_bbox_latlon[3] - patch_bbox_latlon[2]) / pw,
                "shape": (ph, pw),
            }
            surface_override = ortho_surface_grid(dst_meta, zoom=ortho_zoom,
                                                  saturation=ortho_saturation)

        # OSM 橋（bridge=yes + layer）を読み、patch 範囲に交差するものを立体化対象に
        bridges_render = None
        if bridges_json:
            from bridge_osm import load_bridges
            bridges_render = load_bridges(
                bridges_json,
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )
            print(f"  [bridge] OSM 橋 {len(bridges_render)} 本を patch 内に配置"
                  + (f"（例: {', '.join(b['name'] for b in bridges_render if b['name'])[:60]}）"
                     if any(b['name'] for b in bridges_render) else ""))

        print(f"Converting to blocks [enhanced] "
              f"(h_res={h_res}m/block, v_res={v_res}m/block, "
              f"v_exag_land={v_exag}, v_exag_sea={v_es:.2f}, "
              f"sea_level={sea_level_m}m, smooth_sigma={smooth_sigma_cells}, "
              f"cliff_thr={cliff_threshold_m_per_m}"
              + (f", esa_cover ✓" if cover_patch is not None else "")
              + (f", osm ✓" if use_osm else "")
              + (f", ortho ✓" if surface_ortho else "")
              + ")...")
        blocks, size = dem_to_blocks_enhanced(
            dem_patch, idn_patch, h_res_dem, h_res,
            v_res_land=v_res,
            v_exag_land=v_exag,
            v_exag_sea=v_es,
            sea_level_m=sea_level_m,
            ocean_max_depth_m=ocean_max_depth_m,
            smooth_sigma_cells=smooth_sigma_cells,
            cliff_threshold_m_per_m=cliff_threshold_m_per_m,
            deep_ground=deep_ground,
            cover_patch=cover_patch,
            building_mask=building_mask,
            road_mask=road_mask,
            building_height_m=building_height_m,
            building_height_patch=bh_patch,
            building_height_block=building_height_block,
            building_id=building_id_grid,
            building_wall_keys=building_wall_keys,
            building_roof_keys=building_roof_keys,
            color_building_roofs=surface_ortho,
            surface_grid_override=surface_override,
            bridges=bridges_render,
            patch_bbox_latlon=patch_bbox_latlon,
        )
    elif terrain_quality == "legacy":
        print(f"Converting to blocks [legacy] "
              f"(h_res={h_res}m/block, v_res={v_res}m/block, v_exag×{v_exag})...")
        blocks, size = dem_to_blocks(dem_patch, idn_patch, h_res_dem, h_res,
                                      v_res=v_res, v_exag=v_exag)
    else:
        raise ValueError(f"unknown terrain_quality: {terrain_quality} (use 'enhanced' or 'legacy')")
    print(f"Structure size: {size[0]} x {size[1]} x {size[2]} blocks ({len(blocks):,} block entries)")

    # メタデータに描画範囲・解像度情報を補足
    full_meta = None
    if meta is not None:
        full_meta = dict(meta)
        full_meta.setdefault("center_lat", float(lat_center))
        full_meta.setdefault("center_lon", float(lon_center))
        full_meta.setdefault("width_m", float(width_m))
        full_meta.setdefault("depth_m", float(depth_m))
        full_meta.setdefault("h_res_m_per_block", float(h_res))
        full_meta.setdefault("v_res_m_per_block", float(v_res))
        full_meta.setdefault("v_exag", float(v_exag))
        full_meta.setdefault("structure_size_xyz", [int(size[0]), int(size[1]), int(size[2])])
        full_meta.setdefault("n_block_entries", int(len(blocks)))
        full_meta.setdefault("terrain_quality", str(terrain_quality))
        full_meta.setdefault("terrain_source", str(terrain_source))
        if terrain_source == "mapzen":
            full_meta.setdefault("mapzen_zoom", int(mapzen_zoom))
            full_meta.setdefault("use_esa_worldcover", bool(use_esa))
        if terrain_source == "tellus_world":
            full_meta.setdefault("tellus_world_dir", str(tellus_world_dir))
            full_meta.setdefault("tellus_world_scale", float(tellus_world_scale))
            full_meta.setdefault("tellus_sea_level_y", int(tellus_sea_level_y))
        if use_osm:
            full_meta.setdefault("use_osm", True)
            full_meta.setdefault("building_height_m", float(building_height_m))
        if use_fgd:
            full_meta.setdefault("use_fgd", True)
            full_meta.setdefault("building_height_m", float(building_height_m))
        if surface_ortho:
            full_meta.setdefault("surface_ortho", True)
            full_meta.setdefault("ortho_zoom", int(ortho_zoom))
        if terrain_quality == "enhanced":
            full_meta.setdefault("sea_level_m", float(sea_level_m))
            full_meta.setdefault("ocean_max_depth_m", float(ocean_max_depth_m))
            full_meta.setdefault("smooth_sigma_cells", float(smooth_sigma_cells))
            full_meta.setdefault("cliff_threshold_m_per_m", float(cliff_threshold_m_per_m))
            full_meta.setdefault("v_exag_sea",
                                  float(v_exag_sea if v_exag_sea is not None else v_exag * 0.33))
            full_meta.setdefault("deep_ground", int(deep_ground))

    write_nbt_structure(blocks, size, out_path, meta=full_meta)
    return size, len(blocks)
