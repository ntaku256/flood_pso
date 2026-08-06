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

JSON を手で用意しなくても、bbox を渡して `fetch_if_missing=True` にすれば
osm_cache 経由で同じクエリを Overpass へ投げて取得できる（bbox 量子化キャッシュ +
オフラインガード付き）。既定は False なので従来の「JSON 無し＝空」挙動は変わらない。
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


def _bbox_hit(las, los, bbox) -> bool:
    if bbox is None:
        return True
    lat_min, lat_max, lon_min, lon_max = bbox
    return not (max(las) < lat_min or min(las) > lat_max
                or max(los) < lon_min or min(los) > lon_max)


def load_power(json_path: str, *,
               lat_min: float | None = None, lat_max: float | None = None,
               lon_min: float | None = None, lon_max: float | None = None,
               fetch_if_missing: bool = False, verbose: bool = True,
               cache_dir=None) -> dict:
    """OSM Overpass 'out geom tags' JSON → {'lines':[...], 'towers':[...]}。

    lines:  [{"coords":[[lat,lon],...], "voltage":int, "layer":int, "kind":str}]
    towers: [{"lat":float, "lon":float, "kind":"tower"|"pole"}]
    bbox を与えると、その範囲に交差する要素のみ返す（線は bbox 重なり判定）。
    bbox は 4 要素すべて指定するか 4 要素とも省略する（部分指定は分かりやすい ValueError。
    従来は quantize_bbox の内部で素の TypeError になっていた）。
    fetch_if_missing=True かつ JSON が無く bbox がある場合のみ Overpass から取得
    （既定 False＝従来どおり空を返す）。
    cache_dir を渡すと Overpass geom キャッシュもそのディレクトリ配下に置く
    （既定 None＝osm_cache.DEFAULT_CACHE_DIR）。
    """
    has_bbox = _require_bbox(lat_min, lat_max, lon_min, lon_max, what="load_power")

    elements: list = []
    p = Path(json_path) if json_path else None
    if p is not None and p.exists():
        try:
            elements = json.loads(p.read_text(encoding="utf-8")).get("elements", [])
        except Exception:
            return {"lines": [], "towers": []}
    elif fetch_if_missing and has_bbox:
        try:
            from osm_cache import DEFAULT_CACHE_DIR, fetch_overpass_geom
        except ImportError:
            from .osm_cache import DEFAULT_CACHE_DIR, fetch_overpass_geom
        elements = fetch_overpass_geom(
            "power", lat_min, lat_max, lon_min, lon_max,
            cache_dir=(DEFAULT_CACHE_DIR if cache_dir is None else cache_dir),
            verbose=verbose).get("elements", [])
    else:
        return {"lines": [], "towers": []}
    bbox = (lat_min, lat_max, lon_min, lon_max) if has_bbox else None
    lines, towers = [], []
    for e in elements:
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
