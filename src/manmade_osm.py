"""
manmade_osm.py
OSM の人工構造物（man_made）を読み、terrain_render.add_manmade_blocks 用に
「タンク / 煙突 / 堤防」へ分類する。FG-GML WStrL（防波堤/砂防ダム等）と補完関係で、
こちらはタンク・煙突など WStrL に無い立体物を足す。

御坊 bbox 実測（2026-08-12）: storage_tank 4 / embankment 4 / breakwater 2 /
chimney 2 / pier 1。橋/電力と同じ osm_cache 共用。
"""
from __future__ import annotations

import json
from pathlib import Path

TANK = ("storage_tank", "silo", "water_tower", "gasometer")
CHIMNEY = ("chimney",)
BANK = ("embankment", "breakwater", "pier")
WORKS = ("works",)   # 工場/発電所の敷地・建屋ポリゴン（man_made=works or power=plant/generator）


def _centroid(geom):
    la = sum(g["lat"] for g in geom) / len(geom)
    lo = sum(g["lon"] for g in geom) / len(geom)
    return la, lo


def load_manmade(json_path: str | None = None, *,
                 lat_min: float | None = None, lat_max: float | None = None,
                 lon_min: float | None = None, lon_max: float | None = None,
                 fetch_if_missing: bool = True, verbose: bool = True) -> dict:
    """→ {"tanks":[{"lat","lon","coords"?}], "chimneys":[{"lat","lon"}],
          "banks":[{"coords":[[lat,lon],...],"kind"}]}。取得失敗は空。"""
    empty = {"tanks": [], "chimneys": [], "banks": [], "works": []}
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
            elements = fetch_overpass_geom("manmade", lat_min, lat_max, lon_min, lon_max,
                                           verbose=verbose).get("elements", [])
        except Exception as e:                        # OfflineError は BaseException で伝播
            if verbose:
                print(f"  [manmade] OSM 取得スキップ: {e}")
            return empty
    else:
        return empty

    def _hit(la, lo):
        return None in (lat_min, lat_max, lon_min, lon_max) or \
            (lat_min <= la <= lat_max and lon_min <= lo <= lon_max)

    tanks, chimneys, banks, works = [], [], [], []
    for e in elements:
        _tags = e.get("tags", {}) or {}
        mm = _tags.get("man_made", "")
        pw = _tags.get("power", "")
        if e.get("type") == "node" and "lat" in e:
            la, lo = e["lat"], e["lon"]
            if not _hit(la, lo):
                continue
            if mm in TANK:
                tanks.append({"lat": la, "lon": lo, "coords": None})
            elif mm in CHIMNEY:
                chimneys.append({"lat": la, "lon": lo})
        elif e.get("type") == "way" and "geometry" in e and e["geometry"]:
            geom = e["geometry"]
            coords = [[g["lat"], g["lon"]] for g in geom if "lat" in g]
            la, lo = _centroid(geom)
            if not _hit(la, lo):
                continue
            if mm in TANK:
                tanks.append({"lat": la, "lon": lo, "coords": coords})
            elif mm in CHIMNEY:
                chimneys.append({"lat": la, "lon": lo})
            elif mm in BANK and len(coords) >= 2:
                banks.append({"coords": coords, "kind": mm})
            elif (mm in WORKS or pw in ("plant", "generator")) and len(coords) >= 3:
                works.append({"coords": coords, "kind": mm or pw})
    return {"tanks": tanks, "chimneys": chimneys, "banks": banks, "works": works}
