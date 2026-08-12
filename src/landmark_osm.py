"""
landmark_osm.py
OSM の POI（社寺 place_of_worship / 駅 station / 学校・役所・図書館・病院 amenity /
銭湯 public_bath）を読み、FG-GML 建物を「ランドマーク」として特定するための点リストに
変換する。terrain_render.build_building_maps がこの点を建物 footprint と点内包で突き合わせ、
専用外装(朱塗り社寺 / 列柱civic / 銭湯)を割り当てる。osm_cache 共用（量子化キャッシュ）。
"""
from __future__ import annotations

import json
from pathlib import Path


def _classify(tags: dict) -> str | None:
    if tags.get("amenity") == "place_of_worship":
        return "shrine"                                  # 神社・寺（朱塗り系）
    if tags.get("railway") == "station" or tags.get("building") == "train_station":
        return "civic"                                   # 駅舎（civic 系）
    a = tags.get("amenity")
    if a == "public_bath":
        return "sento"
    if a in ("school", "townhall", "hospital", "library", "community_centre"):
        return "civic"
    return None


def _centroid(e: dict):
    if e.get("type") == "node" and "lat" in e:
        return e["lat"], e["lon"]
    g = e.get("geometry")
    if g:
        return sum(p["lat"] for p in g) / len(g), sum(p["lon"] for p in g) / len(g)
    c = e.get("center")
    if c:
        return c["lat"], c["lon"]
    return None


def load_landmarks(json_path: str | None = None, *,
                   lat_min: float | None = None, lat_max: float | None = None,
                   lon_min: float | None = None, lon_max: float | None = None,
                   fetch_if_missing: bool = True, verbose: bool = True) -> list[dict]:
    """POI → [{"lat","lon","type"(shrine/civic/sento),"name"}]。取得失敗は空（任意機能）。"""
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
            elements = fetch_overpass_geom("landmark", lat_min, lat_max, lon_min, lon_max,
                                           verbose=verbose).get("elements", [])
        except Exception as e:
            if verbose:
                print(f"  [landmark] OSM 取得スキップ: {e}")
            return []
    else:
        return []
    out = []
    for e in elements:
        k = _classify(e.get("tags", {}))
        if not k:
            continue
        c = _centroid(e)
        if c is None:
            continue
        la, lo = c
        if None in (lat_min, lat_max, lon_min, lon_max) or \
           (lat_min <= la <= lat_max and lon_min <= lo <= lon_max):
            out.append({"lat": la, "lon": lo, "type": k, "name": e.get("tags", {}).get("name", "")})
    return out
