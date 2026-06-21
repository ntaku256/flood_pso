"""
wakayama_pcd.py
和歌山県 航空レーザ測量 3次元点群オープンデータ（平面直角座標系 第VI系 / JGD2011）
のテキストを読み込み、1m 緯度経度グリッドの DEM に変換して
`dem_parser.mosaic_tiles` 互換の dem_info dict を返す。

これにより、GSI 5m DEM の代わりに「真の 1m」地形を
make_nbt_hd.py / calibrate 系へ FLOOD_PSO_DEM_DIR 相当で差し込める。

入力形式（和歌山県オープンデータ）:
  グラウンド grd: CSV 4列  id, easting, northing, Z   （地物除去＝地形。これを使う）
  オリジナル org: CSV 5列  id, easting, northing, Z, class  （建物/樹木含む DSM）
  easting/northing は平面直角座標系 第VI系（EPSG:6674, JGD2011）[m]。CRLF 改行。

使い方:
  from wakayama_pcd import load_wakayama_dem
  info = load_wakayama_dem("data_cache/wakayama_lidar/06RC802_grd.txt")
  # info: {dem, lat_min, lat_max, lon_min, lon_max, res_lat, res_lon}

  # CLI（キャッシュ生成 + 範囲表示）
  .venv/bin/python src/wakayama_pcd.py data_cache/wakayama_lidar/06RC802_grd.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 平面直角座標系 第VI系（和歌山）JGD2011
EPSG_ZONE6 = "EPSG:6674"
EPSG_WGS84 = "EPSG:4326"
M_PER_DEG_LAT = 111320.0


def _read_xyz(path: str, want_class: bool = False):
    """grd/org テキストから easting, northing, Z（+ want_class なら class）を読む。
    列: grd=4列[id,E,N,Z], org=5列[id,E,N,Z,class]。先頭3数値列[1,2,3]=E,N,Z, [4]=class。
    want_class=True は org（5列）専用。grd には class 列が無いので呼ばないこと。"""
    import pandas as pd
    if want_class:
        # class も float64 で読む（int 固定だと一部タイルで "Integer column has NA"
        # になる＝複数列同時読みで稀に NA 化する行があるため）。NaN class は呼び出し側で
        # 「未分類（除外しない）」として扱う。
        df = pd.read_csv(path, header=None, usecols=[1, 2, 3, 4],
                         names=["e", "n", "z", "c"], dtype=np.float64,
                         engine="c", memory_map=True)
        return (df["e"].to_numpy(), df["n"].to_numpy(),
                df["z"].to_numpy(), df["c"].to_numpy())
    df = pd.read_csv(path, header=None, usecols=[1, 2, 3],
                     names=["e", "n", "z"], dtype=np.float64,
                     engine="c", memory_map=True)
    return df["e"].to_numpy(), df["n"].to_numpy(), df["z"].to_numpy()


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """内部の NaN セル（ビン化の取りこぼし）を最近傍の有効値で埋める。
    タイルは陸/海岸の連続矩形で、水面も LiDAR では Z≈0 の実値を持つため、
    NaN は基本「点が落ちなかった隙間」。穴を残すと enhanced 描画が海と誤認するので埋める。"""
    nan = np.isnan(grid)
    if not nan.any():
        return grid
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def load_wakayama_dem(grd_path: str, res_m: float = 1.0,
                      cache: bool = True, fill_gaps: bool = True,
                      verbose: bool = True,
                      exclude_classes: tuple | None = None,
                      keep_classes: tuple | None = None) -> dict:
    """
    和歌山県点群（グラウンド推奨）を res_m[m] の緯度経度グリッド DEM にして
    dem_parser 互換 dict を返す。

    res_m=1.0 で「真の 1m」DEM。結果は同ディレクトリに .npz キャッシュ。

    exclude_classes : org（DSM, 5列）でこの LiDAR 分類コードの点を除外してからグリッド化。
      例 (3,) で植生（低木）を除いた「地面＋建物」DSM になり、建物高さの樹木混入を防ぐ。
      grd（地形, 4列）には class 列が無いので指定しないこと。除外版は別キャッシュに保存。
    """
    # grd_path はカンマ区切りで複数図郭を mosaic 可（範囲がタイル境界を跨ぐとき）
    paths = [p.strip() for p in str(grd_path).split(",") if p.strip()]
    multi = len(paths) > 1
    exc = tuple(sorted(set(int(c) for c in exclude_classes))) if exclude_classes else ()
    kep = tuple(sorted(set(int(c) for c in keep_classes))) if keep_classes else ()
    tag = ("_exc" + "".join(str(c) for c in exc)) if exc else ""
    tag += ("_keep" + "".join(str(c) for c in kep)) if kep else ""
    if multi:
        stems = "+".join(sorted(Path(p).stem for p in paths))
        cache_path = Path(paths[0]).parent / f"{stems}.grid{res_m:g}m{tag}.npz"
    else:
        cache_path = Path(paths[0]).with_suffix(f".grid{res_m:g}m{tag}.npz")
    if cache and cache_path.exists():
        if verbose:
            print(f"[wakayama] load cache {cache_path.name}")
        z = np.load(cache_path)
        return {k: (float(z[k]) if z[k].ndim == 0 else z[k]) for k in z.files}

    Es, Ns, Zs = [], [], []
    for p in paths:
        if verbose:
            print(f"[wakayama] reading {Path(p).name} ...")
        if exc or kep:
            e, n, zv, cv = _read_xyz(p, want_class=True)
            finite = np.isfinite(e) & np.isfinite(n) & np.isfinite(zv)
            if not finite.all():
                e, n, zv, cv = e[finite], n[finite], zv[finite], cv[finite]
            m = np.isin(cv, kep) if kep else ~np.isin(cv, exc)
            if verbose:
                print(f"[wakayama]   classes {'keep'+str(kep) if kep else 'exc'+str(exc)}: "
                      f"{int(m.sum()):,}/{len(m):,} pts")
            e, n, zv = e[m], n[m], zv[m]
        else:
            e, n, zv = _read_xyz(p)
        Es.append(e); Ns.append(n); Zs.append(zv)
    e = np.concatenate(Es); n = np.concatenate(Ns); zv = np.concatenate(Zs)
    if multi and verbose:
        print(f"[wakayama] mosaic {len(paths)} 図郭 → {len(e):,} pts")

    from pyproj import Transformer
    tr = Transformer.from_crs(EPSG_ZONE6, EPSG_WGS84, always_xy=True)
    lon, lat = tr.transform(e, n)
    lon = np.asarray(lon); lat = np.asarray(lat)

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    mid_lat = 0.5 * (lat_min + lat_max)
    res_lat = res_m / M_PER_DEG_LAT
    res_lon = res_m / (M_PER_DEG_LAT * np.cos(np.radians(mid_lat)))

    H = int(round((lat_max - lat_min) / res_lat)) + 1
    W = int(round((lon_max - lon_min) / res_lon)) + 1

    # row0 = 北端(lat_max)。dem_parser と同じ向き（北→南, 西→東）。
    row = np.round((lat_max - lat) / res_lat).astype(np.int64)
    col = np.round((lon - lon_min) / res_lon).astype(np.int64)
    np.clip(row, 0, H - 1, out=row)
    np.clip(col, 0, W - 1, out=col)
    flat = row * W + col

    ssum = np.bincount(flat, weights=zv, minlength=H * W)
    cnt = np.bincount(flat, minlength=H * W)
    with np.errstate(invalid="ignore"):
        dem = (ssum / cnt).reshape(H, W).astype(np.float32)
    dem[cnt.reshape(H, W) == 0] = np.nan

    n_gap = int(np.isnan(dem).sum())
    if fill_gaps:
        dem = _fill_nan_nearest(dem)

    if verbose:
        valid = dem[~np.isnan(dem)]
        print(f"[wakayama] grid {dem.shape} @ {res_m}m  "
              f"lat[{lat_min:.5f},{lat_max:.5f}] lon[{lon_min:.5f},{lon_max:.5f}]  "
              f"Z[{valid.min():.1f},{valid.max():.1f}]m  gaps_filled={n_gap}  "
              f"points={len(zv):,}")

    info = {
        "dem": dem,
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "res_lat": res_lat, "res_lon": res_lon,
    }
    if cache:
        np.savez_compressed(
            cache_path, dem=dem,
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
            res_lat=res_lat, res_lon=res_lon,
        )
        if verbose:
            print(f"[wakayama] cached {cache_path.name}")
    return info


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path(__file__).resolve().parent.parent /
            "data_cache" / "wakayama_lidar" / "06RC802_grd.txt")
    res = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    info = load_wakayama_dem(p, res_m=res)
    print(f"dem shape={info['dem'].shape}  "
          f"res_lat={info['res_lat']*M_PER_DEG_LAT:.2f}m  "
          f"res_lon={info['res_lon']*M_PER_DEG_LAT*np.cos(np.radians(info['lat_min'])):.2f}m")
