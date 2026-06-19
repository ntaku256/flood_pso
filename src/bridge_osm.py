"""
bridge_osm.py
OSM の橋（way: bridge=yes + highway）ジオメトリ JSON を読み、
橋レンダラ（terrain_render.add_bridge_blocks）用の dict リストに変換する。

FG-GML には橋情報が無いため、OSM の bridge / layer タグを Tellus と同じ入力として使う。
キャッシュ生成（御坊周辺、要ネット）:
  curl -G https://overpass-api.de/api/interpreter --data-urlencode \\
    'data=[out:json][timeout:120];way["bridge"]["highway"](S,W,N,E);out geom tags;' \\
    > data_cache/osm/gobo_bridges_geom.json
"""
from __future__ import annotations

import json
from pathlib import Path

# highway → 道路クラス（幅・橋脚スタイルに反映）
_HIGHWAY_CLASS = {
    "motorway": "main", "trunk": "main", "primary": "main",
    "motorway_link": "main", "trunk_link": "main", "primary_link": "main",
    "secondary": "normal", "tertiary": "normal", "secondary_link": "normal",
    "tertiary_link": "normal", "residential": "normal", "unclassified": "normal",
    "living_street": "normal",
    "service": "dirt", "footway": "dirt", "path": "dirt",
    "cycleway": "dirt", "pedestrian": "dirt", "track": "dirt",
}
_DEFAULT_WIDTH_M = {"main": 9.0, "normal": 5.5, "dirt": 3.0}


def _parse_width(s, fallback: float) -> float:
    try:
        return float(str(s).split()[0])
    except Exception:
        return fallback


def load_bridges(json_path: str, *,
                 lat_min: float | None = None, lat_max: float | None = None,
                 lon_min: float | None = None, lon_max: float | None = None) -> list[dict]:
    """OSM Overpass 'out geom tags' JSON → 橋 dict のリスト。

    返り値: [{"coords":[[lat,lon],...], "layer":int, "road_class":str,
             "width_m":float, "name":str, "highway":str}]
    bbox を与えると、その範囲に交差する橋のみ返す。
    """
    p = Path(json_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for e in data.get("elements", []):
        geom = e.get("geometry") or []
        if len(geom) < 2:
            continue
        coords = [[g["lat"], g["lon"]] for g in geom if "lat" in g and "lon" in g]
        if len(coords) < 2:
            continue
        if lat_min is not None:
            las = [c[0] for c in coords]; los = [c[1] for c in coords]
            if (max(las) < lat_min or min(las) > lat_max
                    or max(los) < lon_min or min(los) > lon_max):
                continue
        t = e.get("tags", {})
        hw = t.get("highway", "")
        rc = _HIGHWAY_CLASS.get(hw, "normal")
        try:
            layer = max(0, int(float(t.get("layer", "1"))))
        except Exception:
            layer = 1
        width = _parse_width(t.get("width"), _DEFAULT_WIDTH_M[rc])
        out.append({"coords": coords, "layer": layer, "road_class": rc,
                    "width_m": width, "name": t.get("name", ""), "highway": hw})
    return out


if __name__ == "__main__":
    import sys
    jp = sys.argv[1] if len(sys.argv) > 1 else "data_cache/osm/gobo_bridges_geom.json"
    bs = load_bridges(jp)
    print(f"{len(bs)} bridges")
    for b in bs[:15]:
        print(f"  {b['road_class']:6} layer={b['layer']} w={b['width_m']:.1f}m "
              f"pts={len(b['coords'])} {b['name']}")
