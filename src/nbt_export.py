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
                           block_state_properties_for_key as _block_state_properties_for_key)


def _palette_compound(key: str) -> nbtlib.Compound:
    name = _BLOCKS[key][0]
    props = _block_state_properties_for_key(key)
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


def _finalize_meta(meta: dict | None):
    """flood_pso_meta 用の最終 dict を nbtlib.Compound 化（両書き出し経路で共有）。"""
    if meta is None:
        return None
    full_meta = dict(meta)
    full_meta.setdefault("schema_version", 1)
    full_meta.setdefault("generator", "flood_pso/nbt_export.py")
    full_meta.setdefault("timestamp_utc",
                         _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z")
    full_meta.setdefault("git_revision", _git_revision())
    return build_meta_compound(full_meta)


# 1 ボクセル = Compound{pos:List<Int>[x,y,z], state:Int} の NBT ペイロード(固定36byte)。
# 可変部は x(11..15) y(15..19) z(19..23) state(31..35) の big-endian int32 のみ。
_VOXEL_TMPL = (b"\x09\x00\x03pos\x03\x00\x00\x00\x03" + b"\x00" * 12
               + b"\x03\x00\x05state" + b"\x00" * 4 + b"\x00")
assert len(_VOXEL_TMPL) == 36


def _write_nbt_dense(arr: np.ndarray, size: list, out_path,
                     meta_compound=None, compresslevel: int = 6) -> int:
    """密3D配列 ``arr[y,z,x]``（uint16, 0=air, 値=palette index）を Structure NBT へ
    **ストリーミング書き出し**する（施策③）。

    非air ボクセルを numpy でベクトル encode し gzip ストリームへ直書きするため、
    中間 nbtlib.Compound 群を一切作らず、メモリは密配列＋数百MBのバッファに収まる
    （docs/06 の 8-12GB 問題の根治）。出力は標準 NBT バイナリで、既存の nbtlib／
    nbt_preview._parse_fast の双方で読め、従来 write_nbt_structure と等価。
    返り値: 書き込んだ非air ブロック数。

    compresslevel : gzip 圧縮レベル 0-9（既定 6, 旧既定は gzip の 9）。実測:
      御坊 400m crop タイル(raw 66.0MB)  L9=8.86s/5.712MB → L6=0.73s/5.113MB
      御坊 1km 本番タイル (raw 60.8MB)  L9=6.01s/5.291MB → L6=0.67s/4.589MB
      本データは L9 の lazy matching がかえって不利で、**L6 は 9〜12倍速かつ
      サイズも 11〜13% 小さい**（L1/L4/L5 は速いがサイズが増える）。
      0 は gzip level 0（deflate stored）。**gzip ストリームとしては正当で
      Minecraft も nbtlib も普通に読める**（magic 1f8b08 を実測確認）。
      圧縮しない分サイズが数倍に膨らむだけなので、書き出し時間だけを詰めたい
      検証用途に使う。
    """
    ys, zs, xs = np.nonzero(arr)            # air(=0) を除く非air(Y,Z,X 昇順)
    states = arr[ys, zs, xs]
    n = int(xs.shape[0])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _named(tag_id: int, name: bytes) -> bytes:
        return bytes([tag_id]) + struct.pack(">H", len(name)) + name

    rec = np.frombuffer(_VOXEL_TMPL, dtype=np.uint8)
    with gzip.open(str(out_path), "wb", compresslevel=int(compresslevel)) as f:
        f.write(b"\x0a\x00\x00")                                   # root TAG_Compound, name ""
        f.write(_named(3, b"DataVersion") + struct.pack(">i", 4671))
        au = b"flood_pso"
        f.write(_named(8, b"author") + struct.pack(">H", len(au)) + au)
        f.write(_named(9, b"size") + b"\x03" + struct.pack(">i", 3)
                + struct.pack(">iii", int(size[0]), int(size[1]), int(size[2])))
        f.write(_named(9, b"palette") + b"\x0a" + struct.pack(">i", len(PALETTE_LIST_KEYS)))
        for k in PALETTE_LIST_KEYS:
            PALETTE[k].write(f)                                    # compound payload+end
        f.write(_named(9, b"blocks") + b"\x0a" + struct.pack(">i", n))
        CH = 2_000_000
        for s0 in range(0, n, CH):
            s1 = min(n, s0 + CH); m = s1 - s0
            buf = np.broadcast_to(rec, (m, 36)).copy()
            buf[:, 11:15] = xs[s0:s1].astype(">i4").view(np.uint8).reshape(-1, 4)
            buf[:, 15:19] = ys[s0:s1].astype(">i4").view(np.uint8).reshape(-1, 4)
            buf[:, 19:23] = zs[s0:s1].astype(">i4").view(np.uint8).reshape(-1, 4)
            buf[:, 31:35] = states[s0:s1].astype(">i4").view(np.uint8).reshape(-1, 4)
            f.write(buf.tobytes())
        f.write(_named(9, b"entities") + b"\x0a" + struct.pack(">i", 0))   # 空 List<Compound>
        if meta_compound is not None:
            f.write(_named(10, b"flood_pso_meta"))
            meta_compound.write(f)
        f.write(b"\x00")                                           # root TAG_End
    return n


def write_nbt_structure(blocks, size: list, out_path: str,
                        meta: dict | None = None,
                        compresslevel: int = 6):
    """
    Minecraft Structure NBT 形式（1.17+ Structure Block）でファイルを書き出す。

    blocks が **np.ndarray**（密3D配列 ``arr[y,z,x]``, 0=air）なら _write_nbt_dense で
    ストリーミング省メモリ書き出し（施策③）。**list[Compound]**（legacy）なら従来の
    nbtlib 直列化。meta が与えられたら ``flood_pso_meta`` を埋め込む。

    compresslevel : gzip 圧縮レベル 0-9（既定 6, 密配列パスのみ有効）。
                    legacy(list[Compound]) パスは nbtlib.save の既定に従う。
    """
    meta_compound = _finalize_meta(meta)
    out_path = Path(out_path)

    if isinstance(blocks, np.ndarray):
        n = _write_nbt_dense(blocks, size, out_path, meta_compound,
                             compresslevel=compresslevel)
    else:
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
        if meta_compound is not None:
            root["flood_pso_meta"] = meta_compound
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nbtlib.File(nbtlib.Compound(root)).save(str(out_path), gzipped=True)
        n = len(blocks)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB, {n:,} blocks)"
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
                  underfill_cap: int | None = None,
                  tunnel_core_always_covered: bool = False,
                  tunnel_core_cover_slack: int | None = None,
                  tunnel_cover_close_blocks: int | None = None,
                  power_clip_spans_to_grid: bool = True,
                  global_anchors: bool = True,
                  terrain_source: str = "gsi",
                  mapzen_zoom: int = 15,
                  use_esa: bool = False,
                  use_osm: bool = False,
                  use_fgd: bool = False,
                  road_curb_use_osm: bool = True,   # 道路境界線の交差点偽枠線をOSM回廊で抑制(既定ON)
                  fgd_bld_xml: str | None = None,
                  fgd_rdedg_xml: str | None = None,
                  fgd_wa_xml: str | None = None,
                  fgd_rail_xml: str | None = None,
                  building_list: list | None = None,
                  remove_bld_polys: list | None = None,  # 重心がこの[lat,lon]環内のFGD建物を除去
                  add_bld_list: list | None = None,       # FGD建物に追加する新設建物dict
                  terrain_skirt_cells: int = 0,           # >0: ワールド外周を斜面化し境界の崖を無くす
                  surface_ortho: bool = False,
                  ortho_zoom: int = 18,
                  ortho_saturation: float = 1.4,
                  ortho_layer: str = "seamlessphoto",
                  building_height_m: float = 6.0,
                  building_height_grid: np.ndarray | None = None,
                  tree_height_grid: np.ndarray | None = None,
                  tree_mode: str = "canopy",
                  tellus_world_dir: str | None = None,
                  tellus_world_scale: float = 1.0,
                  tellus_sea_level_y: int = 0,
                  bridges_json: str | None = None,
                  tunnels_json: str | None = None,
                  power_json: str | None = None,
                  parking_json: str | None = None,
                  evac_xml: str | None = None,
                  hollow_buildings: bool = True,
                  legend_layer: bool = False,
                  tile_crop: tuple | None = None,
                  anvil_out: str | None = None,
                  anvil_offset: tuple | None = None,
                  anvil_merge: bool = False,
                  anvil_level_name: str = "flood_pso",
                  anvil_level_template: str | None = None,
                  world_base_y: int = 0,
                  nbt_compresslevel: int = 6,
                  write_intermediate_nbt: bool = True,
                  strict_osm_json: bool = False):
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
    deep_ground     : 陸の地盤柱の深さ [block]（enhanced のみ、既定 8）。自動アンダー
                      フィル（underfill_cap=None）では上限の**下限**としてしか効かない。
                      実際に効いた上限は full_meta["underfill_cap_effective"] に記録される。

    旧挙動エスケープハッチ（既定はすべて新挙動。make_nbt_hd の同名 CLI から到達可能）:
    underfill_cap   : int でアンダーフィル深さを一律クランプ（旧挙動。8 で HEAD 相当）
    tunnel_core_always_covered : True で OSM トンネル way 本体を無条件密閉（旧挙動）
    tunnel_core_cover_slack    : コア被覆判定の緩さ [block]（None=terrain_render 既定）
    tunnel_cover_close_blocks  : 被覆判定の station 方向 closing 長 [block]（0 で無効）
    power_clip_spans_to_grid   : False で端点がタイル外の径間を丸ごと捨てる（旧挙動）
    global_anchors  : False で送電線/トンネルの全域DEMアンカー（タイル継ぎ目の段差対策）を
                      使わずタイルローカル走査に戻す（旧挙動）。橋のアンカーは常に有効。

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

    nbt_compresslevel  : 中間 structure NBT の gzip 圧縮レベル 0-9（既定 6）
    write_intermediate_nbt : False で中間 .nbt を書かない（anvil_out 指定時のみ許可）。
                         Anvil ワールドだけ欲しいとき、72タイルで 3.2GB になる中間 NBT を省ける。
    strict_osm_json    : True で bridges/tunnels/power/parking の JSON パスが指定済なのに
                         存在しない場合 FileNotFoundError（既定 False は警告のみ＝従来動作）。
    """
    if not write_intermediate_nbt and anvil_out is None:
        raise ValueError("write_intermediate_nbt=False は anvil_out 指定時のみ有効"
                         "（両方無しでは出力が何も残らない）")
    for _lbl, _p in (("--bridges-json", bridges_json), ("--tunnels-json", tunnels_json),
                     ("--power-json", power_json), ("--parking-json", parking_json)):
        if _p and not Path(_p).exists():
            _msg = f"{_lbl} に指定された OSM JSON が存在しません: {_p}"
            if strict_osm_json:
                raise FileNotFoundError(
                    _msg + "  → 無言で0本のワールドが出来るのを防ぐため停止しました"
                           "（意図的に無効化するなら空文字を渡す）")
            print(f"  [warn] {_msg} → このフィーチャは0件になります")
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * np.cos(np.radians(lat_center)))

    # ─── terrain_source 分岐：表示用 dem を Mapzen / Tellus world で差し替え ───
    cover_patch = None
    cover_patch_full = None         # ESA WorldCover（use_esa 時のみ取得, dem grid 整合）
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
    tree_patch = None  # 樹冠高[m] パッチ（LiDAR class3 由来, 任意）
    _tile_core = None  # 施策④halo: (top,left,core_rows,core_cols) ブロック単位の切り戻し窓
    r0 = c0 = 0        # パッチの DEM セル原点（軸6-2 ディザの世界座標基準。tellus 経路は 0）

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
        if tile_crop is not None:
            # 施策④: 呼び出し側(make_nbt_hd)が全域を整数でタイル分割した DEMセル範囲を
            # そのまま使う（タイル毎の独立丸めを排し、隣接タイルが境界セルを共有して密着）。
            tr0, tr1, tc0, tc1 = tile_crop
            r0 = max(0, int(tr0)); r1 = min(dem.shape[0], int(tr1))
            c0 = max(0, int(tc0)); c1 = min(dem.shape[1], int(tc1))
            # 施策④halo: 継ぎ目で建物/道路のエッジ効果（寄棟屋根の distance_transform、
            # 壁周のラスタ縁、道路バッファの打ち切り）がグリッド端に誤って出るのを防ぐため、
            # クロップを halo セル分広げてレンダし、出力ブロック配列をコアに切り戻す。
            # リサンプル無し（factor=1, ブロック=DEMセル 1:1）かつ enhanced 時のみ厳密に
            # 切り戻せるので適用（=wakayama LiDAR タイル運用）。それ以外は halo=0 で従来どおり。
            _hrd0 = res_lat / lat_per_m
            _no_resample = (not (h_res > 0 and h_res < _hrd0 * 0.95)
                            and max(1, round(h_res / _hrd0)) == 1)
            _halo = 16 if (_no_resample and terrain_quality == "enhanced") else 0
            if _halo > 0:
                er0 = max(0, r0 - _halo); er1 = min(dem.shape[0], r1 + _halo)
                ec0 = max(0, c0 - _halo); ec1 = min(dem.shape[1], c1 + _halo)
                # 切り戻し窓（拡張パッチ内のコア位置, factor=1 なのでブロック=セル）
                _tile_core = (r0 - er0, c0 - ec0, r1 - r0, c1 - c0)
                r0, r1, c0, c1 = er0, er1, ec0, ec1
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
        if tree_height_grid is not None and tree_height_grid.shape == dem.shape:
            tree_patch = tree_height_grid[r0:r1, c0:c1]
        if use_esa and cover_patch_full is not None:
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
    print(f"  [patch_bbox] lat[{patch_bbox_latlon[0]:.7f},{patch_bbox_latlon[1]:.7f}] "
          f"lon[{patch_bbox_latlon[2]:.7f},{patch_bbox_latlon[3]:.7f}]")

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
        if tree_patch is not None:
            tree_patch = nd_zoom(tree_patch, up_factor, order=1, mode="nearest")
        print(f"  [upsample] {up_factor:.2f}× source DEM → patch shape {dem_patch.shape}")
        h_res_dem = h_res

    if terrain_quality == "enhanced":
        from terrain_render import dem_to_blocks_enhanced, build_osm_masks
        v_es = v_exag_sea if v_exag_sea is not None else v_exag * 0.33

        # OSM 取得 + ブロック grid 上の建物・道路 mask を事前生成
        building_mask = road_mask = water_mask = road_major_mask = None
        building_height_block = None   # P1: per-building 集約のフラット高さ（FG-GML 経路）
        building_id_grid = None        # P2: 建物ごとの整数ラベル
        building_wall_keys = None      # P2: 建物 id → 壁ブロックキー
        building_roof_keys = None      # P2: 建物 id → 屋根ブロックキー(fallback)
        building_roof_solid = None     # 建物 id → 屋根を型単色化しオルソ焼込無効(新設建物)
        building_style_keys = None     # 建物 id → スタイル(wood_house/apartment/shop/rc/...)
        building_facade_by_id = None   # 建物 id → 外壁装飾スペック(アーキタイプ由来)
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
            building_mask, road_mask, road_major_mask = build_osm_masks(
                osm, patch_bbox_latlon,
                grid_h=nz_g, grid_w=nx_g, h_res_block_m=h_res,
            )
            print(f"  [osm] buildings={osm['n_buildings']}  roads={osm['n_roads']}  "
                  f"→ mask cells: building={int(building_mask.sum())}  road={int(road_mask.sum())}")

        # FG-GML（国土地理院ローカルベクタ）建物・道路 mask。OSM と併用時は union。
        if use_fgd or building_list is not None:
            import warnings as _warnings
            from terrain_render import build_building_maps
            if building_list is not None:
                # PLATEAU / OSM などで事前読み込みした建物リストを使う（道路は別経路=OSM）
                _blds = building_list; _roads = []
            else:
                from fgd_vector import load_fgd_buildings_roads
                _fgd = load_fgd_buildings_roads(
                    fgd_bld_xml, fgd_rdedg_xml,
                    lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                    lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
                )
                _blds = _fgd["buildings"]; _roads = _fgd["roads"]
            # 現況補正: 解体済み建物を除去 → 新設建物を追加（FGD/building_list どちらにも適用）
            if remove_bld_polys:
                from shapely.geometry import Polygon as _Poly
                _rm = [_Poly([(lo, la) for la, lo in ring]) for ring in remove_bld_polys]
                def _in_rm(coords):
                    if len(coords) < 3:
                        return False
                    c = _Poly([(lo, la) for la, lo in coords]).centroid
                    return any(c.within(p) for p in _rm)
                _n0 = len(_blds)
                _blds = [b for b in _blds if not _in_rm(b.get("coords", []))]
                print(f"  [bld-fix] 除去: {_n0} -> {len(_blds)} 棟（解体済み {_n0 - len(_blds)} 棟を削除）")
            if add_bld_list:
                _blds = _blds + add_bld_list
                print(f"  [bld-fix] 追加: +{len(add_bld_list)} 棟（新設）→ 計 {len(_blds)} 棟")
            factor = max(1, round(h_res / h_res_dem))
            nz_g = dem_patch.shape[0] // factor
            nx_g = dem_patch.shape[1] // factor
            # 道路だけ従来 mask（建物は per-building 集約で別途生成）
            _, rm_f, rmaj_f = build_osm_masks(
                {"roads": _roads}, patch_bbox_latlon,
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
                _blds, dsm_h_block, patch_bbox_latlon, nz_g, nx_g,
            )
            bm_f = bmaps["mask"]
            building_mask = bm_f if building_mask is None else (building_mask | bm_f)
            road_mask = rm_f if road_mask is None else (road_mask | rm_f)
            road_major_mask = rmaj_f if road_major_mask is None else (road_major_mask | rmaj_f)
            # FG-GML 水域(WA/WStrA: 河川・池) → 地表を水面に。fgd_wa_xml はカンマ区切り複数可
            if fgd_wa_xml:
                from fgd_vector import load_water
                from terrain_render import polygon_mask_from_latlon
                wm = np.zeros((nz_g, nx_g), dtype=bool)
                _nw = 0
                for wx in str(fgd_wa_xml).split(","):
                    wx = wx.strip()
                    if not wx:
                        continue
                    for w in load_water(wx, lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                                        lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3]):
                        wm |= polygon_mask_from_latlon(w["coords"], patch_bbox_latlon, nz_g, nx_g)
                        _nw += 1
                water_mask = wm if water_mask is None else (water_mask | wm)
                print(f"  [fgd-water] WA/WStrA {_nw}面 → 水面mask cells={int(wm.sum())}")
            building_height_block = bmaps["height"]
            building_id_grid = bmaps["id"]
            building_wall_keys = bmaps["wall_keys"]
            building_roof_keys = bmaps["roof_keys"]
            building_roof_solid = bmaps.get("roof_solid")
            building_style_keys = bmaps.get("style_keys")
            building_facade_by_id = bmaps.get("facade")
            _bh_in = building_height_block[np.isfinite(building_height_block)]
            _med = float(np.median(_bh_in)) if _bh_in.size else 0.0
            _src = "plateau/osm" if building_list is not None else "fgd"
            print(f"  [{_src}] buildings={len(_blds)}  roads={len(_roads)}  "
                  f"→ mask cells: building={int(bm_f.sum())}  road={int(rm_f.sum())}  "
                  f"per-building flat-height median={_med:.1f}m  n_bld={len(building_wall_keys)}")

        # 地表色を空中写真から（最優先の surface_grid_override に流す）
        surface_override = tellus_surface_grid
        ortho_rgb = None
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
            surface_override, ortho_rgb = ortho_surface_grid(
                dst_meta, zoom=ortho_zoom, saturation=ortho_saturation,
                layer=ortho_layer, return_rgb=True)

        # OSM 橋（bridge=yes + layer）を読み、patch 範囲に交差するものを立体化対象に
        bridges_render = None
        if bridges_json:
            from bridge_osm import load_bridges
            import math as _math
            _ctx_m = 650.0
            _mid_lat = 0.5 * (patch_bbox_latlon[0] + patch_bbox_latlon[1])
            _ctx_lat = _ctx_m / 111320.0
            _ctx_lon = _ctx_m / (111320.0 * max(0.2, _math.cos(_math.radians(_mid_lat))))
            _bridge_bbox = (
                max(float(dem_info["lat_min"]), patch_bbox_latlon[0] - _ctx_lat),
                min(float(dem_info["lat_max"]), patch_bbox_latlon[1] + _ctx_lat),
                max(float(dem_info["lon_min"]), patch_bbox_latlon[2] - _ctx_lon),
                min(float(dem_info["lon_max"]), patch_bbox_latlon[3] + _ctx_lon),
            )
            bridges_render = load_bridges(
                bridges_json,
                lat_min=_bridge_bbox[0], lat_max=_bridge_bbox[1],
                lon_min=_bridge_bbox[2], lon_max=_bridge_bbox[3],
            )
            print(f"  [bridge] OSM 橋 {len(bridges_render)} 本を patch 周辺({_ctx_m:.0f}m)に配置"
                  + (f"（例: {', '.join(b['name'] for b in bridges_render if b['name'])[:60]}）"
                     if any(b['name'] for b in bridges_render) else ""))
            # 端アンカー高を全域DEMで事前計算（--tiles 分割で橋端点がタイル外に出てもデッキが
            # 地表へ降下しないよう、全タイルが同一の高さを参照＝高架が一貫して連続平坦飛行する）。
            from terrain_render import assign_global_bridge_anchors
            assign_global_bridge_anchors(
                bridges_render, dem_info["dem"],
                dem_info["lat_max"], dem_info["lon_min"],
                dem_info["res_lat"], dem_info["res_lon"],
                h_res_block_m=h_res, scale_land=(v_exag / max(v_res, 1e-6)),
                lift=(6 if legend_layer else 1), sea_level_m=sea_level_m)

        # OSM トンネル（tunnel=yes）を patch 範囲で読む（橋と同じ Overpass geom JSON 形式）
        tunnels_render = None
        if tunnels_json:
            from bridge_osm import load_bridges as _load_ways
            tunnels_render = _load_ways(
                tunnels_json,
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )
            print(f"  [tunnel] OSM トンネル {len(tunnels_render)} 本を patch 内で検出")
            # 坑口床高を全域DEMで事前計算（--tiles 分割でトンネルが複数タイルにまたがっても
            # 床が同一勾配になる＝タイル境界で床高が段差にならない）。橋と同じ手当て。
            if global_anchors and tunnels_render:
                from terrain_render import assign_global_tunnel_anchors
                assign_global_tunnel_anchors(
                    tunnels_render, dem, lat_max, lon_min, res_lat, res_lon,
                    h_res_block_m=h_res, scale_land=(v_exag / max(v_res, 1e-6)),
                    lift=(6 if legend_layer else 1))

        # OSM 送電線/鉄塔（power=line/tower）を patch 範囲で読む
        power_lines = power_towers = None
        if power_json:
            from power_osm import load_power
            _pw = load_power(
                power_json,
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )
            power_lines, power_towers = _pw["lines"], _pw["towers"]
            print(f"  [power] OSM 送電線 {len(power_lines)} 本 / 鉄塔・電柱 {len(power_towers)} 基を配置")
            # 径間端点（鉄塔）の地表Yを全域DEMで事前計算（--tiles 分割で端点がタイル外に
            # 出ても全タイルが同一の架線直線を引く＝継ぎ目の段差と対地クリアランス破綻を防ぐ）。
            if global_anchors and power_lines:
                from terrain_render import assign_global_power_anchors
                assign_global_power_anchors(
                    power_lines, dem, lat_max, lon_min, res_lat, res_lon,
                    scale_land=(v_exag / max(v_res, 1e-6)),
                    lift=(6 if legend_layer else 1))

        # FG-GML 鉄道中心線（RailCL）を patch 範囲で読む
        rail_render = None
        if fgd_rail_xml:
            from fgd_vector import load_rail
            rail_render = []
            for rx in str(fgd_rail_xml).split(","):
                rx = rx.strip()
                if rx and Path(rx).exists():
                    rail_render += load_rail(
                        rx, lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                        lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
                    )
            print(f"  [rail] FG-GML RailCL {len(rail_render)} 本を敷設")

        # OSM 駐車場（amenity=parking）を patch 範囲で読む
        parking_render = None
        if parking_json:
            from parking_osm import load_parking
            parking_render = load_parking(
                parking_json,
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )
            print(f"  [parking] OSM 駐車場 {len(parking_render)} 面を地表に配置")

        evac_render = None
        if evac_xml:
            from ksj_evac import load_evac_facilities
            evac_render = load_evac_facilities(
                evac_xml,
                lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
            )

        # 道路境界線(curb)の交差点偽枠線対策: OSM道路センターラインを塗りつぶし回廊にした mask を
        # 用意して dem_to_blocks_enhanced に渡す（centerline は交差点を連続して貫くので「同一道路」
        # 判定に使える）。オフライン等で取得失敗時は None=極小穴埋めのみにフォールバック。
        road_curb_osm_mask = None
        if road_mask is not None and road_curb_use_osm:
            try:
                from tellus_data import fetch_osm_buildings_roads as _fetch_osm
                _osm_c = _fetch_osm(
                    lat_min=patch_bbox_latlon[0], lat_max=patch_bbox_latlon[1],
                    lon_min=patch_bbox_latlon[2], lon_max=patch_bbox_latlon[3],
                    verbose=False,
                )
                _factor = max(1, round(h_res / h_res_dem))
                _nzg = dem_patch.shape[0] // _factor
                _nxg = dem_patch.shape[1] // _factor
                _, road_curb_osm_mask, _ = build_osm_masks(
                    _osm_c, patch_bbox_latlon, grid_h=_nzg, grid_w=_nxg, h_res_block_m=h_res,
                )
                print(f"  [road-curb] OSM道路回廊 {int(road_curb_osm_mask.sum())} cells "
                      f"(roads={_osm_c.get('n_roads')}) → 交差点の偽枠線を抑制")
            except Exception as _e:
                print(f"  [road-curb] OSM回廊取得不可 ({_e}); 極小穴埋めのみで対応")
                road_curb_osm_mask = None

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
            underfill_cap=underfill_cap,
            tunnel_core_always_covered=tunnel_core_always_covered,
            tunnel_core_cover_slack=tunnel_core_cover_slack,
            tunnel_cover_close_blocks=tunnel_cover_close_blocks,
            power_clip_spans_to_grid=power_clip_spans_to_grid,
            cover_patch=cover_patch,
            building_mask=building_mask,
            road_mask=road_mask,
            building_height_m=building_height_m,
            building_height_patch=bh_patch,
            tree_height_patch=tree_patch,
            tree_mode=tree_mode,
            building_height_block=building_height_block,
            building_id=building_id_grid,
            building_wall_keys=building_wall_keys,
            building_roof_keys=building_roof_keys,
            building_roof_solid=building_roof_solid,
            building_style_keys=building_style_keys,
            building_facade_by_id=building_facade_by_id,
            hollow_buildings=hollow_buildings,
            legend_layer=legend_layer,
            color_building_roofs=surface_ortho,
            terrain_skirt_cells=terrain_skirt_cells,
            surface_grid_override=surface_override,
            bridges=bridges_render,
            tunnels=tunnels_render,
            powerlines=power_lines,
            power_towers=power_towers,
            rails=rail_render,
            parkings=parking_render,
            ortho_rgb=ortho_rgb,
            evac_facilities=evac_render,
            patch_bbox_latlon=patch_bbox_latlon,
            water_mask=water_mask,
            road_major_mask=road_major_mask,
            road_curb_osm_mask=road_curb_osm_mask,
            cell_offset=(c0, r0),   # 軸6-2: 地表ディザの世界座標基準（タイル間整合）
        )
    elif terrain_quality == "legacy":
        print(f"Converting to blocks [legacy] "
              f"(h_res={h_res}m/block, v_res={v_res}m/block, v_exag×{v_exag})...")
        blocks, size = dem_to_blocks(dem_patch, idn_patch, h_res_dem, h_res,
                                      v_res=v_res, v_exag=v_exag)
    else:
        raise ValueError(f"unknown terrain_quality: {terrain_quality} (use 'enhanced' or 'legacy')")

    # 施策④halo: 拡張パッチでレンダした密ブロック配列を、コア [c0,c1)×[r0,r1) に切り戻す。
    # 軸順は (Y, Z, X)=(高さ, 南北, 東西)。size=[nx, ny, nz]。Y は絶対標高基準で不変。
    # これで建物/道路/橋のエッジ効果は halo 域に出て破棄され、隣接タイルのコアが密着する。
    if _tile_core is not None and isinstance(blocks, np.ndarray):
        _top, _left, _crz, _crx = _tile_core
        blocks = np.ascontiguousarray(blocks[:, _top:_top + _crz, _left:_left + _crx])
        size = [int(_crx), int(size[1]), int(_crz)]
        print(f"  [halo] 拡張パッチ {dem_patch.shape[1]}×{dem_patch.shape[0]} → "
              f"コア {_crx}×{_crz} に切り戻し（継ぎ目のエッジ効果を破棄）")

    n_entries = int(np.count_nonzero(blocks)) if isinstance(blocks, np.ndarray) else len(blocks)
    print(f"Structure size: {size[0]} x {size[1]} x {size[2]} blocks ({n_entries:,} block entries)")

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
        full_meta.setdefault("n_block_entries", int(n_entries))
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
            # アンダーフィル: 自動モードでは deep_ground は「上限の下限」としてしか効かず
            # (_cap = max(UNDERFILL_MIN, UNDERFILL_HARD_CAP, deep_ground))、既定 8 は無視される。
            # 生成物から実際に効いた上限が読めるよう実効値とモードを記録する。
            from terrain_render import (UNDERFILL_MIN, UNDERFILL_EXTRA, UNDERFILL_HARD_CAP)
            if underfill_cap is None:
                full_meta.setdefault("underfill_cap_mode", "auto")
                full_meta.setdefault("underfill_cap_effective",
                                     int(max(UNDERFILL_MIN, UNDERFILL_HARD_CAP,
                                             int(deep_ground))))
                full_meta.setdefault("underfill_min", int(UNDERFILL_MIN))
                full_meta.setdefault("underfill_extra", int(UNDERFILL_EXTRA))
            else:
                full_meta.setdefault("underfill_cap_mode", "fixed")
                full_meta.setdefault("underfill_cap_effective",
                                     int(max(2, int(underfill_cap))))
                full_meta.setdefault("underfill_min", 2)
                full_meta.setdefault("underfill_extra", 1)
            full_meta.setdefault("tunnel_core_always_covered", bool(tunnel_core_always_covered))
            full_meta.setdefault("power_clip_spans_to_grid", bool(power_clip_spans_to_grid))
            full_meta.setdefault("global_anchors", bool(global_anchors))

    if write_intermediate_nbt:
        write_nbt_structure(blocks, size, out_path, meta=full_meta,
                            compresslevel=nbt_compresslevel)
    else:
        print(f"  [nbt] 中間 structure NBT をスキップ（--no-intermediate-nbt）: {out_path}")

    # 施策⑤: native Anvil world(.mca)も出力（密配列のときのみ＝enhanced）。
    if anvil_out is not None:
        if isinstance(blocks, np.ndarray):
            from anvil_export import write_anvil_world
            ox, oz = (int(anvil_offset[0]), int(anvil_offset[1])) if anvil_offset else (0, 0)
            write_anvil_world(blocks, size, anvil_out, x_offset=ox, z_offset=oz,
                              y_offset=int(world_base_y),
                              merge=bool(anvil_merge), level_name=anvil_level_name,
                              level_template=anvil_level_template)
        elif not write_intermediate_nbt:
            raise RuntimeError(
                "[anvil] dense 配列でないため Anvil を書けず、write_intermediate_nbt=False で "
                "中間 NBT も書かないため出力が空になります（terrain_quality='enhanced' にするか "
                "--no-intermediate-nbt を外して下さい）")
        else:
            print("  [anvil] スキップ: dense 配列でない（terrain_quality=enhanced が必要）")

    return size, len(blocks)
