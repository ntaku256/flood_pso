"""gap_fill.py
LiDAR（3次元点群）が無い領域（図郭外＝DEM が NaN）を外部データで補完する。

- 地形: mapzen terrarium DEM を reproject して NaN セルを埋める（点群が無くても陸地になる）。
- 建物高さ: OSM(building:levels/height) をラスタ化し、欠落域の building_height_grid を埋める。
  FGD footprint はそのまま使い、点群が無い所だけ OSM 由来の高さを与える方針。

make_nbt_hd の --fill-gap-osm から呼ぶ。"""
from __future__ import annotations

import numpy as np


def _grid_geo(dem_info):
    H, W = dem_info["dem"].shape
    lat_max = float(dem_info["lat_max"]); lon_min = float(dem_info["lon_min"])
    rl = float(dem_info["res_lat"]); ro = float(dem_info["res_lon"])
    lat_min = lat_max - rl * H; lon_max = lon_min + ro * W
    return lat_min, lat_max, lon_min, lon_max, rl, ro, H, W


def fill_terrain_gap_mapzen(dem, dem_info, gap_mask, *, zoom: int = 14, verbose: bool = True) -> int:
    """dem(=dem_info['dem']) の gap(NaN) セルを mapzen DEM で **in-place** 補完。補完セル数を返す。"""
    from tellus_data import fetch_mapzen_dem, reproject_to_grid
    lat_min, lat_max, lon_min, lon_max, rl, ro, H, W = _grid_geo(dem_info)
    try:
        mz = fetch_mapzen_dem(lat_min, lat_max, lon_min, lon_max, zoom=zoom, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  [gap-fill] mapzen 取得失敗→地形補完スキップ: {e}")
        return 0
    dst = {"lat_max": lat_max, "lon_min": lon_min, "res_lat": rl, "res_lon": ro,
           "dem": np.zeros((H, W), np.float32)}
    mz_on = reproject_to_grid(mz["dem"], mz, dst, fill_value=np.nan)
    fillable = gap_mask & np.isfinite(mz_on)
    dem[fillable] = mz_on[fillable].astype(dem.dtype)
    if verbose:
        print(f"  [gap-fill] 地形: mapzen(z{zoom}) で {int(fillable.sum()):,} セル補完 "
              f"(gap {int(gap_mask.sum()):,} / 図郭外)")
    return int(fillable.sum())


def fill_terrain_gap_nearest(dem, dem_info, gap_mask, *, smooth: float = 1.2,
                             sea_level_m: float = 0.0, preserve_sea_voids: bool = True,
                             sea_void_margin_m: float = 3.0, verbose: bool = True) -> int:
    """dem の gap(NaN) セルを「最近傍の有効 LiDAR 値」で **in-place** 補完。補完セル数を返す。

    mapzen terrarium(=表層 DSM, z14≈4.8m) で埋めると、沿岸の点群欠落域に発電所等の構造物高
    (建物/タンク/煙突)が地形へ焼き込まれ、平坦であるべき埋立地が max 数十 m の凸塊になる
    (LiDAR-mapzen は mean -2.3m / std 5.4m, 標高基準も T.P. と不一致)。
    本関数は外部 DEM を使わず、欠落域を周囲の実測 LiDAR から最近傍補間する:
      - 海(標高≤海面)に近いセルは海値を、陸(埋立地)に近いセルは陸値を継承
        → sea_mask(標高ベース) と整合し、海岸線も自然に継承される
      - gap セルのみガウシアン平滑し、最近傍補間の方向縞/境界段差を緩和(実測 LiDAR は不変)
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, label
    fin = np.isfinite(dem)
    if not fin.any() or not gap_mask.any():
        return 0
    idx = distance_transform_edt(~fin, return_distances=False, return_indices=True)
    nn = dem[tuple(idx)]                       # 各セル ← 最近傍の有効値(海/陸を継承)
    # 施策(海): GSI DEM は外洋を NaN で返す。これを陸値で埋めると海が「標高≈1mの陸」になり、
    # 暗い海の航空写真から black_concrete 等に塗られる。画像端に連結する広い void で、周囲の
    # 最近傍実測が海面近く(≤sea_level+margin)のもの＝外洋 とみなし、埋めずに海面下値へ落として
    # make_sea_mask(標高≤海面 or NaN → 水)で水に分類させる。内陸の小欠損(周囲が高い陸)は従来通り埋める。
    sea_void = np.zeros_like(gap_mask, dtype=bool)
    if preserve_sea_voids:
        lbl, nlab = label(gap_mask)
        edge_ids = set(int(v) for v in np.unique(
            np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])) if v)
        min_sea = max(2000, int(0.003 * dem.size))     # これ以上の端連結成分のみ「外洋」
        for c in edge_ids:
            comp = (lbl == c)
            if int(comp.sum()) < min_sea:
                continue
            if float(np.nanmedian(nn[comp])) <= sea_level_m + sea_void_margin_m:
                sea_void |= comp
    fill_mask = gap_mask & ~sea_void          # 実際に陸値で埋める(=内陸欠損)のみ
    out = dem.copy()
    out[fill_mask] = nn[fill_mask]
    if smooth and smooth > 0 and fill_mask.any():
        base = np.where(np.isfinite(out), out, sea_level_m - 0.5).astype(np.float32)
        sm = gaussian_filter(base, smooth)
        out[fill_mask] = sm[fill_mask]        # gap のみ平滑。境界の実測値はブレンドに寄与し不変
    dem[fill_mask] = out[fill_mask].astype(dem.dtype)
    if sea_void.any():
        dem[sea_void] = np.asarray(sea_level_m - 0.5, dtype=dem.dtype)   # 海面下=水へ分類
    n_land = int(fill_mask.sum()); n_sea = int(sea_void.sum())
    if verbose:
        v = dem[fill_mask] if n_land else np.array([0.0], dtype=np.float32)
        print(f"  [gap-fill] 地形: 陸欠損 {n_land:,} を近傍補間(mean={float(np.mean(v)):.2f} "
              f"max={float(np.max(v)):.2f}) / 外洋void {n_sea:,} を水(海面下)へ保持")
    return n_land + n_sea


def _osm_height_m(tags: dict, default_levels: float, m_per_level: float) -> float:
    """OSM tags → 建物高さ[m]。height > building:levels×3 > 既定(2階)の順。"""
    h = tags.get("height")
    if h is not None:
        try:
            return max(float(str(h).split(";")[0].split()[0].replace("m", "")), 2.5)
        except Exception:
            pass
    lv = tags.get("building:levels") or tags.get("levels")
    if lv is not None:
        try:
            return max(float(str(lv).split(";")[0]) * m_per_level, 2.5)
        except Exception:
            pass
    return default_levels * m_per_level


def fill_building_heights_gap_osm(bh_grid, dem_info, gap_mask, *, default_levels: float = 2.0,
                                  m_per_level: float = 3.0, verbose: bool = True) -> int:
    """欠落域(gap)の building_height_grid(=DSM-DEM) を OSM 建物高さでラスタ埋め（in-place）。
    建物ごとにローカル窓だけ判定するので全グリッド path 判定より高速。補完セル数を返す。"""
    import matplotlib.path as mpath
    from tellus_data import fetch_osm_buildings_roads
    lat_min, lat_max, lon_min, lon_max, rl, ro, H, W = _grid_geo(dem_info)
    try:
        osm = fetch_osm_buildings_roads(lat_min, lat_max, lon_min, lon_max, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  [gap-fill] OSM 取得失敗→建物高さ補完スキップ: {e}")
        return 0
    blds = osm.get("buildings", [])
    n_fill = 0
    n_bld = 0
    for b in blds:
        coords = b.get("coords") or []
        if len(coords) < 4:
            continue
        las = [c[0] for c in coords]; los = [c[1] for c in coords]
        r_lo = max(0, int(np.floor((lat_max - max(las)) / rl)))
        r_hi = min(H, int(np.ceil((lat_max - min(las)) / rl)) + 1)
        c_lo = max(0, int(np.floor((min(los) - lon_min) / ro)))
        c_hi = min(W, int(np.ceil((max(los) - lon_min) / ro)) + 1)
        if r_hi <= r_lo or c_hi <= c_lo:
            continue
        win_gap = gap_mask[r_lo:r_hi, c_lo:c_hi]
        if not win_gap.any():
            continue
        sh, sw = r_hi - r_lo, c_hi - c_lo
        yy, xx = np.mgrid[0:sh, 0:sw]
        lat_c = (lat_max - (r_lo + yy + 0.5) * rl).ravel()
        lon_c = (lon_min + (c_lo + xx + 0.5) * ro).ravel()
        poly = np.array([[lo, la] for la, lo in coords])  # (lon, lat)
        inside = mpath.Path(poly).contains_points(np.column_stack([lon_c, lat_c])).reshape(sh, sw)
        win_bh = bh_grid[r_lo:r_hi, c_lo:c_hi]
        sel = inside & win_gap & ~np.isfinite(win_bh)
        if sel.any():
            win_bh[sel] = np.float32(_osm_height_m(b.get("tags", {}) or {}, default_levels, m_per_level))
            n_fill += int(sel.sum()); n_bld += 1
    if verbose:
        print(f"  [gap-fill] 建物高さ: OSM {n_bld} 棟を欠落域にラスタ化 {n_fill:,} セル")
    return n_fill
