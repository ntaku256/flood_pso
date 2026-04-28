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

PALETTE = {
    "air":         nbtlib.Compound({"Name": nbtlib.String("minecraft:air")}),
    "stone":       nbtlib.Compound({"Name": nbtlib.String("minecraft:stone")}),
    "grass":       nbtlib.Compound({"Name": nbtlib.String("minecraft:grass_block"),
                                    "Properties": nbtlib.Compound({"snowy": nbtlib.String("false")})}),
    "sand":        nbtlib.Compound({"Name": nbtlib.String("minecraft:sand")}),
    "gravel":      nbtlib.Compound({"Name": nbtlib.String("minecraft:gravel")}),
    # water/blue_ice はアニメーションテクスチャでブラウザ描画エラーになるため
    # blue_stained_glass / cyan_stained_glass で代替
    "water":       nbtlib.Compound({"Name": nbtlib.String("minecraft:blue_stained_glass")}),
    "blue_ice":    nbtlib.Compound({"Name": nbtlib.String("minecraft:cyan_stained_glass")}),
    "bedrock":     nbtlib.Compound({"Name": nbtlib.String("minecraft:bedrock")}),
}

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
                  meta: dict | None = None):
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
    v_exag     : 垂直誇張倍率
    out_path   : 出力 .nbt ファイルパス
    meta       : NBT に埋め込むメタデータ辞書（任意）。
                 method, K, D, water_level, sigma, dh_map(np.ndarray), loss, iou, seed,
                 dem_source, bbox, n_evals, elapsed_s 等を入れる想定。
    """
    dem = dem_info["dem"]
    lat_max = dem_info["lat_max"]
    lon_min = dem_info["lon_min"]
    res_lat = dem_info["res_lat"]
    res_lon = dem_info["res_lon"]

    # 中心ピクセル
    row_c = round((lat_max - lat_center) / res_lat)
    col_c = round((lon_center - lon_min) / res_lon)

    # エリアを DEMセル数で計算
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * np.cos(np.radians(lat_center)))
    half_rows = int((depth_m / 2) * lat_per_m / res_lat)
    half_cols = int((width_m / 2) * lon_per_m / res_lon)

    r0 = max(0, row_c - half_rows)
    r1 = min(dem.shape[0], row_c + half_rows)
    c0 = max(0, col_c - half_cols)
    c1 = min(dem.shape[1], col_c + half_cols)

    dem_patch = dem[r0:r1, c0:c1]
    idn_patch = inundation[r0:r1, c0:c1]

    print(f"DEM patch: {dem_patch.shape} cells = {dem_patch.shape[1]*res_lon/lon_per_m:.0f}m W x {dem_patch.shape[0]*res_lat/lat_per_m:.0f}m N")

    h_res_dem = res_lat / lat_per_m   # DEMセル = 何m か

    print(f"Converting to blocks (h_res={h_res}m/block, v_res={v_res}m/block, v_exag×{v_exag})...")
    blocks, size = dem_to_blocks(dem_patch, idn_patch, h_res_dem, h_res,
                                  v_res=v_res, v_exag=v_exag)
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

    write_nbt_structure(blocks, size, out_path, meta=full_meta)
    return size, len(blocks)
