"""
barrier_osm.py
OSM の barrier ライン（擁壁 retaining_wall / 塀 wall / 柵 fence / 生垣 hedge /
ガードレール guard_rail）を読み、terrain_render.add_barrier_blocks 用に変換する。
道路と宅地の段差・境界を表し、地味だが「日本の郊外」感が出る。

御坊 bbox 実測（2026-08-12）: retaining_wall 24 / wall 10 / hedge 5 / fence 3 /
guard_rail 1（kerb 7 は低すぎ・道路と重なるので対象外）。橋/電力と同じ osm_cache 共用。
"""
from __future__ import annotations

import json
from pathlib import Path

KINDS = ("retaining_wall", "wall", "fence", "hedge", "guard_rail")


def load_barriers(json_path: str | None = None, *,
                  lat_min: float | None = None, lat_max: float | None = None,
                  lon_min: float | None = None, lon_max: float | None = None,
                  fetch_if_missing: bool = True, verbose: bool = True) -> list[dict]:
    """→ [{"coords":[[lat,lon],...], "kind":str}]。取得失敗は空（任意機能）。"""
    elements: list = []
    p = Path(json_path) if json_path else None
    if p is not None and p.exists():
        try:
            elements = json.loads(p.read_text(encoding="utf-8")).get("elements", [])
        except Exception:
            return []
    elif fetch_if_missing and None not in (lat_min, lat_max, lon_min, lon_max):
        try:
            from osm_cache import fetch_overpass_geom, bbox_intersects
            elements = fetch_overpass_geom("barrier", lat_min, lat_max, lon_min, lon_max,
                                           verbose=verbose).get("elements", [])
        except Exception as e:                        # OfflineError は BaseException で伝播
            if verbose:
                print(f"  [barrier] OSM 取得スキップ: {e}")
            return []
    else:
        return []
    bbox = (lat_min, lat_max, lon_min, lon_max) if None not in (lat_min, lat_max, lon_min, lon_max) else None
    out = []
    for e in elements:
        if e.get("type") != "way" or "geometry" not in e:
            continue
        kind = (e.get("tags", {}) or {}).get("barrier", "")
        if kind not in KINDS:
            continue
        coords = [[g["lat"], g["lon"]] for g in e["geometry"] if "lat" in g and "lon" in g]
        if len(coords) < 2:
            continue
        if bbox is not None:
            las = [c[0] for c in coords]; los = [c[1] for c in coords]
            if max(las) < bbox[0] or min(las) > bbox[1] or max(los) < bbox[2] or min(los) > bbox[3]:
                continue
        out.append({"coords": coords, "kind": kind})
    return out
