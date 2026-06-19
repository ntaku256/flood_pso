"""
tellus_data.py
`web-app/Tellus/` mod が内部で使う「Earth-scale データソース」を Python から直接取得する。

Tellus 本体（Minecraft Fabric mod）を起動せずに、**同じデータソース・同じ精度** で
flood_pso 用の Tellus 風地形を作るためのモジュール。

提供するデータ：

  1. Mapzen Joerd Terrain Tiles（AWS Open Data Registry）
     URL  : https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png
     形式 : PNG (R, G, B) → elevation_m = (R*256 + G + B/256) - 32768
     解像度: zoom 15 で約 4.8 m/pixel（御坊周辺）
     ライセンス: 各国 DEM の出典クレジット要（README に記載済み）

  2. ESA WorldCover 2021 v200（10m 土地被覆ラスタ）  ※rasterio 必須
     URL  : https://esa-worldcover.s3.amazonaws.com/v200/2021/map/
            ESA_WorldCover_10m_2021_v200_N33E135.tif  （3°×3° タイル）
     値   : 10=tree, 20=shrubland, 30=grassland, 40=cropland, 50=built,
            60=bare, 70=snow, 80=water, 90=wetland, 95=mangrove, 100=moss
     ライセンス: CC BY 4.0

キャッシュ: `flood_pso/data_cache/{mapzen,esa}/` 配下にローカル保存。再ダウンロードを避ける。
"""

from __future__ import annotations

import io
import math
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data_cache"
MAPZEN_BASE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium"
GSI_ORTHO_BASE_URL = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto"  # 全国シームレス空中写真(JPG)
ESA_BASE_URL    = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"
OVERPASS_URL    = "https://overpass-api.de/api/interpreter"

# OSM highway → 道路幅 [m]（典型値、両車線合計）
OSM_HIGHWAY_WIDTH_M = {
    "motorway":     14, "motorway_link": 8,
    "trunk":        12, "trunk_link":    7,
    "primary":       9, "primary_link":  6,
    "secondary":     7, "secondary_link":5,
    "tertiary":      6, "tertiary_link": 4,
    "residential":   5, "unclassified":  5, "living_street": 4,
    "service":       3, "track":         3,
    "footway":       2, "path":          2, "cycleway":      2,
    "pedestrian":    3, "steps":         2,
}


# ─────────────────────────────────────────────────────────────
# Slippy Map タイル座標計算
# ─────────────────────────────────────────────────────────────

def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """経緯度 → タイル番号 (x, y)（標準 slippy map / Web Mercator）"""
    n = 2 ** zoom
    xt = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(max(-85.0511, min(85.0511, lat)))
    yt = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xt, yt


def tile_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    """タイル左上 (north-west) の経緯度"""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return lon, math.degrees(lat_rad)


def tiles_for_bbox(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    zoom: int,
) -> list[tuple[int, int]]:
    """BBOX をカバーするタイル (x, y) のリスト（包含的）"""
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)  # 北西
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)  # 南東
    return [(x, y) for y in range(min(y0, y1), max(y0, y1) + 1)
                   for x in range(min(x0, x1), max(x0, x1) + 1)]


# ─────────────────────────────────────────────────────────────
# Mapzen Joerd Terrain Tiles 取得
# ─────────────────────────────────────────────────────────────

def _http_get_with_retry(url: str, timeout: float = 20.0,
                         tries: int = 3, backoff_s: float = 1.5) -> bytes:
    """単純な GET + 指数バックオフ。429/5xx は再試行。404 は即座に raise。"""
    last_err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flood_pso/tellus_data"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        time.sleep(backoff_s ** attempt)
    raise RuntimeError(f"GET failed after {tries} tries: {url} -- {last_err}")


def fetch_mapzen_tile(z: int, x: int, y: int,
                       cache_dir: Path = DEFAULT_CACHE_DIR) -> np.ndarray:
    """1 タイル（256×256, terrarium 形式）を取得して elevation [m] を返す。
    キャッシュがあれば使う。"""
    tile_dir = cache_dir / "mapzen" / str(z) / str(x)
    tile_path = tile_dir / f"{y}.png"
    if not tile_path.exists():
        url = f"{MAPZEN_BASE_URL}/{z}/{x}/{y}.png"
        data = _http_get_with_retry(url)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path.write_bytes(data)

    img = Image.open(tile_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)  # (H, W, 3)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    elev = (r * 256.0 + g + b / 256.0) - 32768.0
    return elev.astype(np.float32)


def fetch_mapzen_dem(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int = 15,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> dict:
    """
    BBOX をカバーする Mapzen タイルをダウンロードしてモザイク化する。

    Returns
    -------
    dict with keys:
      'dem'     : np.ndarray (H, W)  標高 [m]、海面下も負値で連続
      'lat_min', 'lat_max', 'lon_min', 'lon_max' : float（モザイク全体の bbox）
      'res_lat', 'res_lon' : float（ピクセル解像度 [度/pixel]）
      'source'  : "mapzen_terrarium"
      'zoom'    : zoom
    """
    tiles = tiles_for_bbox(lon_min, lat_min, lon_max, lat_max, zoom)
    if not tiles:
        raise ValueError(f"empty bbox: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")

    # タイル毎の lat/lon 範囲
    xs = sorted({x for x, _ in tiles})
    ys = sorted({y for _, y in tiles})
    nw_lon, nw_lat = tile_to_lonlat(xs[0],     ys[0],     zoom)
    se_lon, se_lat = tile_to_lonlat(xs[-1] + 1, ys[-1] + 1, zoom)

    tile_size = 256
    H = (ys[-1] - ys[0] + 1) * tile_size
    W = (xs[-1] - xs[0] + 1) * tile_size
    mosaic = np.full((H, W), np.nan, dtype=np.float32)

    if verbose:
        print(f"[mapzen] zoom={zoom}  tiles={len(tiles)}  → mosaic {H}×{W}")
    for i, (x, y) in enumerate(tiles, 1):
        try:
            tile = fetch_mapzen_tile(zoom, x, y, cache_dir=cache_dir)
        except Exception as e:
            if verbose:
                print(f"  [warn] tile ({x},{y}) failed: {e}")
            continue
        r0 = (y - ys[0]) * tile_size
        c0 = (x - xs[0]) * tile_size
        mosaic[r0:r0 + tile_size, c0:c0 + tile_size] = tile
        if verbose and (i % 8 == 0 or i == len(tiles)):
            print(f"  [{i}/{len(tiles)}] tiles fetched")

    res_lat = (nw_lat - se_lat) / H   # 1 pixel あたり何度
    res_lon = (se_lon - nw_lon) / W

    return {
        "dem": mosaic,
        "lat_min": se_lat, "lat_max": nw_lat,
        "lon_min": nw_lon, "lon_max": se_lon,
        "res_lat": res_lat, "res_lon": res_lon,
        "source": "mapzen_terrarium",
        "zoom":   zoom,
    }


# ─────────────────────────────────────────────────────────────
# GSI シームレス空中写真（オルソ RGB）取得
# ─────────────────────────────────────────────────────────────

def fetch_gsi_ortho_tile(z: int, x: int, y: int,
                          cache_dir: Path = DEFAULT_CACHE_DIR) -> np.ndarray:
    """1 タイル（256×256 RGB）を取得。キャッシュあれば使う。"""
    tile_dir = cache_dir / "gsi_ortho" / str(z) / str(x)
    tile_path = tile_dir / f"{y}.jpg"
    if not tile_path.exists():
        url = f"{GSI_ORTHO_BASE_URL}/{z}/{x}/{y}.jpg"
        data = _http_get_with_retry(url)
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path.write_bytes(data)
    img = Image.open(tile_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)  # (256, 256, 3)


def fetch_gsi_ortho(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    zoom: int = 18,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> dict:
    """
    BBOX をカバーする GSI 空中写真タイルをモザイク化して RGB grid を返す。

    Returns dict:
      'rgb'    : np.ndarray (H, W, 3) uint8
      'lat_min','lat_max','lon_min','lon_max','res_lat','res_lon'  (mosaic 全体)
      'source' : "gsi_seamlessphoto", 'zoom'
    （fetch_mapzen_dem と同じ bbox/res 規約。reproject_to_grid にそのまま渡せる）
    """
    tiles = tiles_for_bbox(lon_min, lat_min, lon_max, lat_max, zoom)
    if not tiles:
        raise ValueError(f"empty bbox: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")
    xs = sorted({x for x, _ in tiles})
    ys = sorted({y for _, y in tiles})
    nw_lon, nw_lat = tile_to_lonlat(xs[0],     ys[0],     zoom)
    se_lon, se_lat = tile_to_lonlat(xs[-1] + 1, ys[-1] + 1, zoom)

    ts = 256
    H = (ys[-1] - ys[0] + 1) * ts
    W = (xs[-1] - xs[0] + 1) * ts
    mosaic = np.zeros((H, W, 3), dtype=np.uint8)
    if verbose:
        print(f"[gsi_ortho] zoom={zoom}  tiles={len(tiles)}  → mosaic {H}×{W}")
    for i, (x, y) in enumerate(tiles, 1):
        try:
            tile = fetch_gsi_ortho_tile(zoom, x, y, cache_dir=cache_dir)
        except Exception as e:
            if verbose:
                print(f"  [warn] ortho tile ({x},{y}) failed: {e}")
            continue
        r0 = (y - ys[0]) * ts
        c0 = (x - xs[0]) * ts
        mosaic[r0:r0 + ts, c0:c0 + ts] = tile
        if verbose and (i % 16 == 0 or i == len(tiles)):
            print(f"  [{i}/{len(tiles)}] ortho tiles fetched")

    return {
        "rgb": mosaic,
        "lat_min": se_lat, "lat_max": nw_lat,
        "lon_min": nw_lon, "lon_max": se_lon,
        "res_lat": (nw_lat - se_lat) / H,
        "res_lon": (se_lon - nw_lon) / W,
        "source": "gsi_seamlessphoto",
        "zoom":   zoom,
    }


# ─────────────────────────────────────────────────────────────
# ESA WorldCover 2021 取得（rasterio が要る）
# ─────────────────────────────────────────────────────────────

# 11 クラス（10/20/.../100）の意味
ESA_COVER_CLASSES = {
    10:  "tree_cover",
    20:  "shrubland",
    30:  "grassland",
    40:  "cropland",
    50:  "built",
    60:  "bare",
    70:  "snow_ice",
    80:  "water",
    90:  "wetland",
    95:  "mangrove",
    100: "moss_lichen",
}


def _esa_tile_id(lat: float, lon: float) -> str:
    """3°×3° タイル ID。例: lat=33.875, lon=135.168 → 'N33E135'"""
    # タイル左下（南西）の度
    tlat = int(math.floor(lat / 3.0)) * 3
    tlon = int(math.floor(lon / 3.0)) * 3
    ns = "N" if tlat >= 0 else "S"
    ew = "E" if tlon >= 0 else "W"
    return f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"


def fetch_esa_worldcover_tile(tile_id: str,
                                cache_dir: Path = DEFAULT_CACHE_DIR,
                                verbose: bool = True) -> Path:
    """3°×3° の WorldCover GeoTIFF をローカルにダウンロード。Path を返す。"""
    out = cache_dir / "esa" / f"ESA_WorldCover_10m_2021_v200_{tile_id}.tif"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ESA_BASE_URL}/ESA_WorldCover_10m_2021_v200_{tile_id}.tif"
    if verbose:
        print(f"[esa] downloading {tile_id} ...")
    data = _http_get_with_retry(url, timeout=180.0)
    out.write_bytes(data)
    if verbose:
        print(f"  saved {out} ({len(data)/1e6:.1f} MB)")
    return out


def fetch_esa_worldcover(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> dict:
    """
    BBOX に必要な ESA WorldCover タイルをダウンロード + crop してモザイク化。
    rasterio が必須（ESA は GeoTIFF 形式）。

    Returns
    -------
    dict:
      'cover'   : np.ndarray (H, W) uint8  ESA cover class
      'lat_min', 'lat_max', 'lon_min', 'lon_max' : float
      'res_lat', 'res_lon' : float
      'source'  : "esa_worldcover_2021_v200"
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.merge import merge
    except ImportError as e:
        raise RuntimeError(
            "ESA WorldCover の読み込みには rasterio が必要です。"
            "  .venv/bin/pip install rasterio"
        ) from e

    # 必要なタイル ID を列挙
    needed_tiles: list[str] = []
    lat = math.floor(lat_min / 3) * 3
    while lat < lat_max:
        lon = math.floor(lon_min / 3) * 3
        while lon < lon_max:
            needed_tiles.append(_esa_tile_id(lat + 0.1, lon + 0.1))
            lon += 3
        lat += 3
    needed_tiles = sorted(set(needed_tiles))

    paths = [fetch_esa_worldcover_tile(t, cache_dir=cache_dir, verbose=verbose)
             for t in needed_tiles]

    # 単一タイルなら crop だけ、複数なら merge
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
            cover = src.read(1, window=window)
            tx = src.window_transform(window)
            res_lon, _, lon0, _, res_lat, lat_max_act = (
                tx.a, tx.b, tx.c, tx.d, tx.e, tx.f)
    else:
        srcs = [rasterio.open(p) for p in paths]
        try:
            merged, tx = merge(srcs, bounds=(lon_min, lat_min, lon_max, lat_max))
            cover = merged[0]
            res_lon, _, lon0, _, res_lat, lat_max_act = (
                tx.a, tx.b, tx.c, tx.d, tx.e, tx.f)
        finally:
            for s in srcs:
                s.close()

    H, W = cover.shape
    return {
        "cover": cover.astype(np.uint8),
        "lat_min": lat_max_act + res_lat * H,  # res_lat は負
        "lat_max": lat_max_act,
        "lon_min": lon0,
        "lon_max": lon0 + res_lon * W,
        "res_lat": -res_lat,                    # 正の値に正規化
        "res_lon": res_lon,
        "source": "esa_worldcover_2021_v200",
        "tiles":  needed_tiles,
    }


# ─────────────────────────────────────────────────────────────
# 経緯度ベースの bilinear 再投影
# ─────────────────────────────────────────────────────────────

def reproject_to_grid(
    src_array: np.ndarray,
    src_meta: dict,
    dst_meta: dict,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    src_array (src_meta が記述する経緯度 grid 上) を dst_meta の grid に bilinear 再投影。

    src_meta / dst_meta は dem_parser.mosaic_tiles 互換：
      'lat_min', 'lat_max', 'lon_min', 'lon_max', 'res_lat', 'res_lon'

    NaN はそのまま伝播させたい場合は呼び出し側で 0 に置き換えてから渡し、
    マスクを別途持つこと。本関数は単純な scipy.ndimage.map_coordinates ベース。
    """
    from scipy.ndimage import map_coordinates

    src_lat_max = src_meta["lat_max"]
    src_lon_min = src_meta["lon_min"]
    src_res_lat = src_meta["res_lat"]
    src_res_lon = src_meta["res_lon"]

    dst_lat_max = dst_meta["lat_max"]
    dst_lon_min = dst_meta["lon_min"]
    dst_res_lat = dst_meta["res_lat"]
    dst_res_lon = dst_meta["res_lon"]
    H_dst = src_array.shape[0] if "dem" not in dst_meta else dst_meta["dem"].shape[0]
    W_dst = src_array.shape[1] if "dem" not in dst_meta else dst_meta["dem"].shape[1]
    if "dem" in dst_meta:
        H_dst, W_dst = dst_meta["dem"].shape
    elif "cover" in dst_meta:
        H_dst, W_dst = dst_meta["cover"].shape

    # dst の各 (row, col) に対応する経緯度を計算 → src の (row, col) に逆変換
    rows = np.arange(H_dst)
    cols = np.arange(W_dst)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    lats = dst_lat_max - R * dst_res_lat
    lons = dst_lon_min + C * dst_res_lon
    src_rows = (src_lat_max - lats) / src_res_lat
    src_cols = (lons - src_lon_min) / src_res_lon

    arr = np.where(np.isnan(src_array), fill_value, src_array).astype(np.float64)
    out = map_coordinates(arr, [src_rows, src_cols], order=1,
                           mode="constant", cval=fill_value)
    return out.astype(src_array.dtype)


# ─────────────────────────────────────────────────────────────
# ESA cover_class → flood_pso パレット
# ─────────────────────────────────────────────────────────────

def cover_class_to_block(cover: int, *, slope_steep: bool = False,
                          high_alpine: bool = False) -> str:
    """
    ESA WorldCover クラス → flood_pso パレットのブロック種別。

    Tellus.MountainSurfaceRules の判定を圧縮（red sand などは無し、
    flood_pso の `nbt_export.PALETTE` 8 種に縮約）。
    """
    if cover == 80:                # water
        return "water"
    if cover == 70:                # snow/ice
        return "blue_ice"
    if cover == 50:                # built
        return "stone"
    if cover == 60:                # bare
        return "gravel"
    if cover == 90:                # wetland
        return "gravel"
    if cover == 95:                # mangrove
        return "grass"
    if high_alpine or slope_steep:
        return "stone"
    if cover == 10:                # tree
        return "grass"
    if cover == 20 or cover == 30:  # shrub / grass
        return "grass"
    if cover == 40:                # cropland
        return "grass"
    if cover == 100:               # moss/lichen
        return "gravel"
    return "grass"                 # default


# ─────────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# OSM (OpenStreetMap) 建物 + 道路 — Overpass API
# ─────────────────────────────────────────────────────────────

def fetch_osm_buildings_roads(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> dict:
    """
    Overpass API で BBOX 内の建物 (way[building]) と道路 (way[highway]) を取得。
    結果を JSON でキャッシュする（再ダウンロード回避、Overpass の負荷を減らすため必須）。

    Returns
    -------
    {
      "buildings": [{"coords": [[lat,lon],...], "tags": {...}}, ...],
      "roads":     [{"coords": [...], "tags": {...}, "width_m": float}, ...],
      "bbox":      [lat_min, lat_max, lon_min, lon_max],
      "n_buildings": int, "n_roads": int,
    }
    """
    import json as _json

    key = f"osm_{lat_min:.5f}_{lat_max:.5f}_{lon_min:.5f}_{lon_max:.5f}.json"
    cache_path = cache_dir / "osm" / key
    if cache_path.exists():
        return _json.loads(cache_path.read_text(encoding="utf-8"))

    query = (
        "[out:json][timeout:60];("
        f'way["building"]({lat_min},{lon_min},{lat_max},{lon_max});'
        f'way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});'
        f'relation["building"]({lat_min},{lon_min},{lat_max},{lon_max});'
        ");out geom;"
    )
    if verbose:
        print(f"[osm] Overpass query bbox lat[{lat_min:.4f},{lat_max:.4f}] lon[{lon_min:.4f},{lon_max:.4f}]")

    import urllib.parse as _up
    body = _up.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL, data=body,
        headers={
            "User-Agent": "flood_pso/tellus_data",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Overpass API failed: HTTP {e.code} {e.reason}") from e
    response = _json.loads(raw.decode("utf-8"))

    buildings: list = []
    roads: list = []
    for el in response.get("elements", []):
        tags = el.get("tags", {}) or {}
        # way: geometry に [{lat, lon}, ...] が入る
        if el.get("type") == "way" and "geometry" in el:
            coords = [[g["lat"], g["lon"]] for g in el["geometry"]]
            if "building" in tags:
                buildings.append({"coords": coords, "tags": tags})
            elif "highway" in tags:
                ht = tags.get("highway", "")
                roads.append({
                    "coords": coords, "tags": tags,
                    "width_m": float(OSM_HIGHWAY_WIDTH_M.get(ht, 4)),
                })
        # relation (multipolygon building) は outer ring を抽出
        elif el.get("type") == "relation" and "members" in el and "building" in tags:
            for m in el["members"]:
                if m.get("type") == "way" and m.get("role") == "outer" and "geometry" in m:
                    coords = [[g["lat"], g["lon"]] for g in m["geometry"]]
                    buildings.append({"coords": coords, "tags": tags})

    out = {
        "buildings": buildings, "roads": roads,
        "bbox": [lat_min, lat_max, lon_min, lon_max],
        "n_buildings": len(buildings), "n_roads": len(roads),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(_json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[osm] cached {cache_path.name}  buildings={len(buildings)}  roads={len(roads)}")
    return out


if __name__ == "__main__":
    # 御坊エリアでスモークテスト
    BBOX = dict(lat_min=33.83, lat_max=33.92, lon_min=135.12, lon_max=135.25)
    print("=== Mapzen smoke test (Gobo bbox, zoom=14) ===")
    d = fetch_mapzen_dem(zoom=14, **BBOX, verbose=True)
    dem = d["dem"]
    print(f"dem shape: {dem.shape}")
    print(f"valid cells: {(~np.isnan(dem)).sum()} / {dem.size}")
    valid = dem[~np.isnan(dem)]
    if len(valid) > 0:
        print(f"elevation: min={valid.min():.1f} mean={valid.mean():.1f} max={valid.max():.1f}")
    print(f"res_lat={d['res_lat']*111320:.2f} m  res_lon={d['res_lon']*111320*math.cos(math.radians(33.875)):.2f} m")
