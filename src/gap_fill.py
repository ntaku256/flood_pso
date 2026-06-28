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
