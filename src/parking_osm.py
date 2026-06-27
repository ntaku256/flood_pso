"""
parking_osm.py
OSM の駐車場（amenity=parking）ポリゴンの Overpass ジオメトリ JSON を読み、
駐車場レンダラ（terrain_render の surf_block 上書き）用の dict リストに変換する。

FG-GML には駐車場レイヤが無いため OSM を入力に使う（arnis 同様）。
キャッシュ生成: way/relation["amenity"="parking"](S,W,N,E); out geom tags;
"""
from __future__ import annotations

import json
from pathlib import Path


def load_parking(json_path: str, *,
                 lat_min: float | None = None, lat_max: float | None = None,
                 lon_min: float | None = None, lon_max: float | None = None) -> list[dict]:
    """OSM 'out geom tags' JSON → [{"coords":[[lat,lon],...], "name":str}]（閉リング）。

    bbox を与えると、その範囲に交差する駐車場のみ返す。
    relation（マルチポリゴン）は geometry を持たない場合スキップ（way が主）。
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
        if e.get("tags", {}).get("amenity") != "parking":
            continue
        geom = e.get("geometry") or []
        coords = [[g["lat"], g["lon"]] for g in geom if "lat" in g and "lon" in g]
        if len(coords) < 3:
            continue
        if lat_min is not None:
            las = [c[0] for c in coords]; los = [c[1] for c in coords]
            if (max(las) < lat_min or min(las) > lat_max
                    or max(los) < lon_min or min(los) > lon_max):
                continue
        out.append({"coords": coords, "name": e.get("tags", {}).get("name", "")})
    return out


if __name__ == "__main__":
    import sys
    jp = sys.argv[1] if len(sys.argv) > 1 else "data_cache/osm/gobo_parking_geom.json"
    ps = load_parking(jp)
    print(f"{len(ps)} parking")
    for pk in ps[:12]:
        print(f"  pts={len(pk['coords'])} {pk['name']}")
