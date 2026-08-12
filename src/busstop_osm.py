"""
busstop_osm.py
OSM のバス停（highway=bus_stop ノード）を読み、terrain_render.add_busstop_blocks 用の
点リストに変換する。御坊 bbox 実測（2026-08-12）: 16 ノード（御坊警察署前・体育館前・
日高病院前 等）。KSJ P11(68件)が上位だが未取得なので OSM を使う（要DL不要）。
橋/電力/信号と同じ osm_cache.fetch_overpass_geom 共用。
"""
from __future__ import annotations

import json
from pathlib import Path


def load_busstops(json_path: str | None = None, *,
                  lat_min: float | None = None, lat_max: float | None = None,
                  lon_min: float | None = None, lon_max: float | None = None,
                  fetch_if_missing: bool = True, verbose: bool = True) -> list[dict]:
    """→ [{"lat","lon","name"}]。取得失敗は空（任意機能）。"""
    elements: list = []
    p = Path(json_path) if json_path else None
    if p is not None and p.exists():
        try:
            elements = json.loads(p.read_text(encoding="utf-8")).get("elements", [])
        except Exception:
            return []
    elif fetch_if_missing and None not in (lat_min, lat_max, lon_min, lon_max):
        try:
            from osm_cache import fetch_overpass_geom
            elements = fetch_overpass_geom("busstop", lat_min, lat_max, lon_min, lon_max,
                                           verbose=verbose).get("elements", [])
        except Exception as e:                       # OfflineError は BaseException で伝播
            if verbose:
                print(f"  [busstop] OSM 取得スキップ: {e}")
            return []
    else:
        return []
    out = []
    for e in elements:
        if e.get("type") == "node" and "lat" in e and "lon" in e:
            la, lo = e["lat"], e["lon"]
            if None in (lat_min, lat_max, lon_min, lon_max) or \
               (lat_min <= la <= lat_max and lon_min <= lo <= lon_max):
                out.append({"lat": la, "lon": lo,
                            "name": e.get("tags", {}).get("name", "")})
    return out
