"""
bridge_osm.py
OSM の橋（way: bridge=yes + highway）ジオメトリ JSON を読み、
橋レンダラ（terrain_render.add_bridge_blocks）用の dict リストに変換する。

FG-GML には橋情報が無いため、OSM の bridge / layer タグを Tellus と同じ入力として使う。
キャッシュ生成（御坊周辺、要ネット）:
  curl -G https://overpass-api.de/api/interpreter --data-urlencode \\
    'data=[out:json][timeout:120];way["bridge"]["highway"](S,W,N,E);out geom tags;' \\
    > data_cache/osm/gobo_bridges_geom.json

JSON を手で用意しなくても、bbox を渡して `fetch_if_missing=True` にすれば
osm_cache 経由で Overpass から取得できる（bbox 量子化キャッシュ + オフラインガード付き）。
既定は False なので、従来どおり「JSON が無ければ空リスト」の挙動は変わらない。
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


def _require_bbox(lat_min, lat_max, lon_min, lon_max, *, what: str) -> bool:
    """bbox が 4 要素そろっていれば True / 全 None なら False / 部分指定は ValueError。

    osm_cache が import できない環境（単体コピー等）でも壊れないようフォールバックを持つ。
    """
    try:
        try:
            from osm_cache import require_bbox
        except ImportError:
            from .osm_cache import require_bbox
    except ImportError:                      # osm_cache 無し＝最低限の自前判定
        vals = {"lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max}
        missing = [k for k, v in vals.items() if v is None]
        if not missing:
            return True
        if len(missing) == 4:
            return False
        raise ValueError(f"{what}: bbox が不完全です（{', '.join(missing)} が None）")
    return require_bbox(lat_min, lat_max, lon_min, lon_max, what=what)


def _load_elements(json_path, *, bbox, fetch_if_missing: bool, kind: str,
                   verbose: bool, cache_dir=None) -> list:
    """ローカル JSON → Overpass の順で `out geom tags` の elements を得る。

    json_path が無く fetch_if_missing=True かつ bbox が **4 要素そろって** いる場合のみ
    ネットワークへ。bbox が部分指定なら osm_cache.require_bbox が
    「どの要素が None か」を示す ValueError を出す（従来は quantize_bbox の中で
    素の TypeError になっていた）。
    cache_dir=None なら osm_cache の既定（リポジトリの data_cache/）。
    （FLOOD_PSO_OFFLINE=1 なら osm_cache が OfflineError を送出する。）
    """
    p = Path(json_path) if json_path else None
    if p is not None and p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("elements", [])
        except Exception:
            return []
    if not fetch_if_missing:
        return []
    try:
        from osm_cache import DEFAULT_CACHE_DIR, fetch_overpass_geom, require_bbox
    except ImportError:
        from .osm_cache import DEFAULT_CACHE_DIR, fetch_overpass_geom, require_bbox
    bb = bbox if bbox is not None else (None, None, None, None)
    if not require_bbox(*bb, what=f"load_bridges(kind={kind!r}, fetch_if_missing=True)"):
        return []                      # bbox 未指定＝従来どおり取得しない
    data = fetch_overpass_geom(kind, bb[0], bb[1], bb[2], bb[3],
                               cache_dir=(DEFAULT_CACHE_DIR if cache_dir is None
                                          else cache_dir),
                               verbose=verbose)
    return data.get("elements", [])


def load_bridges(json_path: str, *,
                 lat_min: float | None = None, lat_max: float | None = None,
                 lon_min: float | None = None, lon_max: float | None = None,
                 fetch_if_missing: bool = False, kind: str = "bridge",
                 verbose: bool = True, cache_dir=None) -> list[dict]:
    """OSM Overpass 'out geom tags' JSON → 橋 dict のリスト。

    返り値: [{"coords":[[lat,lon],...], "layer":int, "road_class":str,
             "width_m":float, "name":str, "highway":str}]
    bbox を与えると、その範囲に交差する橋のみ返す。
    fetch_if_missing=True かつ JSON が無い場合は Overpass から取得する
    （kind="bridge" / "tunnel"。既定 False＝従来どおり空リスト）。
    cache_dir を渡すと Overpass geom キャッシュもそのディレクトリ配下に置く
    （既定 None＝osm_cache.DEFAULT_CACHE_DIR）。
    bbox は 4 要素すべて指定するか 4 要素とも省略する（部分指定は ValueError）。
    """
    _require_bbox(lat_min, lat_max, lon_min, lon_max, what="load_bridges")
    elements = _load_elements(
        json_path, bbox=(lat_min, lat_max, lon_min, lon_max),
        fetch_if_missing=fetch_if_missing, kind=kind, verbose=verbose,
        cache_dir=cache_dir)
    out = []
    for e in elements:
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
