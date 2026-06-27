"""
power_osm.py
OSM の送電線（power=line/minor_line/cable）+ 鉄塔/電柱（power=tower/pole）の
Overpass ジオメトリ JSON を読み、電力レンダラ（terrain_render.add_power_blocks）用に変換する。

FG-GML には電力設備のレイヤが無いため、OSM の power タグを入力に使う（arnis 同様）。
キャッシュ生成（御坊周辺, 要ネット）:
  curl -G https://overpass-api.de/api/interpreter --data-urlencode \\
    'data=[out:json][timeout:120];(way["power"~"^(line|minor_line|cable)$"](S,W,N,E);
     node["power"~"^(tower|pole)$"](S,W,N,E););out geom tags;' \\
    > data_cache/osm/gobo_power_geom.json
"""
from __future__ import annotations

import json
from pathlib import Path


def _max_voltage(s) -> int:
    """'500000;77000' 等 → 最大電圧[V]。不明は 0。"""
    if s is None:
        return 0
    v = 0
    for part in str(s).replace(";", " ").replace(",", " ").split():
        try:
            v = max(v, int(float(part)))
        except Exception:
            pass
    return v


def _bbox_hit(las, los, bbox) -> bool:
    if bbox is None:
        return True
    lat_min, lat_max, lon_min, lon_max = bbox
    return not (max(las) < lat_min or min(las) > lat_max
                or max(los) < lon_min or min(los) > lon_max)


def load_power(json_path: str, *,
               lat_min: float | None = None, lat_max: float | None = None,
               lon_min: float | None = None, lon_max: float | None = None) -> dict:
    """OSM Overpass 'out geom tags' JSON → {'lines':[...], 'towers':[...]}。

    lines:  [{"coords":[[lat,lon],...], "voltage":int, "layer":int, "kind":str}]
    towers: [{"lat":float, "lon":float, "kind":"tower"|"pole"}]
    bbox を与えると、その範囲に交差する要素のみ返す（線は bbox 重なり判定）。
    """
    p = Path(json_path)
    if not p.exists():
        return {"lines": [], "towers": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"lines": [], "towers": []}
    bbox = (lat_min, lat_max, lon_min, lon_max) if lat_min is not None else None
    lines, towers = [], []
    for e in data.get("elements", []):
        t = e.get("tags", {})
        pw = t.get("power", "")
        if e.get("type") == "way" and pw in ("line", "minor_line", "cable"):
            geom = e.get("geometry") or []
            coords = [[g["lat"], g["lon"]] for g in geom if "lat" in g and "lon" in g]
            if len(coords) < 2:
                continue
            las = [c[0] for c in coords]; los = [c[1] for c in coords]
            if not _bbox_hit(las, los, bbox):
                continue
            try:
                layer = int(float(t.get("layer", "0")))
            except Exception:
                layer = 0
            lines.append({"coords": coords, "voltage": _max_voltage(t.get("voltage")),
                          "layer": layer, "kind": pw})
        elif e.get("type") == "node" and pw in ("tower", "pole"):
            if "lat" not in e or "lon" not in e:
                continue
            if not _bbox_hit([e["lat"]], [e["lon"]], bbox):
                continue
            towers.append({"lat": e["lat"], "lon": e["lon"], "kind": pw})
    return {"lines": lines, "towers": towers}


if __name__ == "__main__":
    import sys
    jp = sys.argv[1] if len(sys.argv) > 1 else "data_cache/osm/gobo_power_geom.json"
    r = load_power(jp)
    print(f"{len(r['lines'])} lines / {len(r['towers'])} towers")
    for L in r["lines"][:12]:
        print(f"  {L['kind']:10} {L['voltage']:>7}V layer={L['layer']} pts={len(L['coords'])}")
