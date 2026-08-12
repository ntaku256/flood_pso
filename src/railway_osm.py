"""
railway_osm.py
OSM の鉄道付帯物（踏切 level_crossing / ホーム platform / 駅 station）を読み、
terrain_render.add_railway_blocks 用の dict に変換する。FG-GML RailCL（--fgd-rail）が
線路を敷くのに対し、こちらは「御坊らしさ」を出す踏切・ホーム・駅を足す。

御坊 bbox 実測（2026-08-12）: level_crossing 39 / platform 7 / station 7
（紀州鉄道 西御坊・紀伊御坊・市役所前・学門・御坊 と JR紀勢本線 御坊・道成寺）。
橋/電力/信号と同じ osm_cache.fetch_overpass_geom（bbox量子化キャッシュ・オフラインガード）を共用。
"""
from __future__ import annotations

import json
from pathlib import Path


def load_railway(json_path: str | None = None, *,
                 lat_min: float | None = None, lat_max: float | None = None,
                 lon_min: float | None = None, lon_max: float | None = None,
                 fetch_if_missing: bool = True, verbose: bool = True) -> dict:
    """→ {"crossings":[{"lat","lon"}], "platforms":[{"coords":[[lat,lon],...]}],
          "stations":[{"lat","lon","name"}]}。取得失敗は空（任意機能）。"""
    empty = {"crossings": [], "platforms": [], "stations": []}
    elements: list = []
    p = Path(json_path) if json_path else None
    if p is not None and p.exists():
        try:
            elements = json.loads(p.read_text(encoding="utf-8")).get("elements", [])
        except Exception:
            return empty
    elif fetch_if_missing and None not in (lat_min, lat_max, lon_min, lon_max):
        try:
            from osm_cache import fetch_overpass_geom
            elements = fetch_overpass_geom("railway", lat_min, lat_max, lon_min, lon_max,
                                           verbose=verbose).get("elements", [])
        except Exception as e:                       # OfflineError は BaseException で伝播
            if verbose:
                print(f"  [railway] OSM 取得スキップ: {e}")
            return empty
    else:
        return empty

    def _hit(la, lo):
        return None in (lat_min, lat_max, lon_min, lon_max) or \
            (lat_min <= la <= lat_max and lon_min <= lo <= lon_max)

    crossings, platforms, stations = [], [], []
    for e in elements:
        rw = (e.get("tags", {}) or {}).get("railway", "")
        if e.get("type") == "node" and "lat" in e and "lon" in e:
            la, lo = e["lat"], e["lon"]
            if not _hit(la, lo):
                continue
            if rw == "level_crossing":
                crossings.append({"lat": la, "lon": lo})
            elif rw == "station":
                stations.append({"lat": la, "lon": lo,
                                 "name": e.get("tags", {}).get("name", "")})
        elif e.get("type") == "way" and rw == "platform" and "geometry" in e:
            coords = [[g["lat"], g["lon"]] for g in e["geometry"] if "lat" in g]
            if len(coords) >= 2 and any(_hit(la, lo) for la, lo in coords):
                platforms.append({"coords": coords})
    return {"crossings": crossings, "platforms": platforms, "stations": stations}
