"""
waterway_osm.py
OSM の水路（waterway=stream/drain/ditch/canal/river）ラインの Overpass ジオメトリ JSON を読み、
水路レンダラ（nbt_export で water_mask へ OR）用の dict リストに変換する。

FG-GML の WA/WStrA は**面**として取れる河川・池だけで、御坊のような水田地帯を実際に
特徴づけている幅 1〜2 m の用水路・排水路は面が無く落ちる。実測（2026-08-12, Overpass,
bbox 33.855,135.125-33.915,135.205）では stream 87 / drain 68 / river 58 / canal 2 本で、
このうち面で拾えているのは river の一部だけだった。

既定で river を含めないのは、河川は FG-GML WA/WStrA の面が既に水面にしているためで、
線を重ねると両岸の外へ水が 1 セルはみ出す。面が無い範囲を線で補いたいときだけ
include_river=True にする。

キャッシュ生成: way["waterway"~"^(stream|drain|ditch|canal|river)$"](S,W,N,E); out geom tags;
"""
from __future__ import annotations

import json
from pathlib import Path

#: waterway 値 → 既定の水路幅[m]。OSM の width/est_width があればそちらを優先する。
#: 用水路・排水路は圃場整備の三面張りで概ね 1〜2 m、canal はそれより広い。
DEFAULT_WIDTH_M: dict[str, float] = {
    "ditch":  1.0,
    "drain":  1.5,
    "stream": 2.0,
    "canal":  4.0,
    "river":  8.0,
}
#: river を明示的に足さない限り扱う種別（上記の理由で river は既定外）。
DEFAULT_KINDS: tuple[str, ...] = ("stream", "drain", "ditch", "canal")


def _require_bbox(lat_min, lat_max, lon_min, lon_max, what: str) -> bool:
    """bbox は 4 要素そろっているか全部 None か。部分指定は分かりやすく落とす
    （power_osm._require_bbox と同じ検査。osm_cache 無しでも動くようフォールバック）。"""
    try:
        try:
            from osm_cache import require_bbox
        except ImportError:
            from .osm_cache import require_bbox
    except ImportError:
        vals = {"lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max}
        missing = [k for k, v in vals.items() if v is None]
        if not missing:
            return True
        if len(missing) == 4:
            return False
        raise ValueError(f"{what}: bbox が不完全です（{', '.join(missing)} が None）")
    return require_bbox(lat_min, lat_max, lon_min, lon_max, what=what)


def _width_m(tags: dict, kind: str) -> float:
    """OSM の width / est_width（"1.5", "2 m", "1,5" 等）→ m。無ければ種別の既定。"""
    for key in ("width", "est_width"):
        raw = tags.get(key)
        if raw is None:
            continue
        s = str(raw).strip().replace(",", ".")
        for suffix in (" m", "m"):
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
        try:
            w = float(s)
        except ValueError:
            continue
        if w > 0:
            return w
    return DEFAULT_WIDTH_M.get(kind, 2.0)


def load_waterways(json_path: str, *,
                   lat_min: float | None = None, lat_max: float | None = None,
                   lon_min: float | None = None, lon_max: float | None = None,
                   kinds: tuple[str, ...] | None = None,
                   include_river: bool = False,
                   fetch_if_missing: bool = False, verbose: bool = True,
                   cache_dir=None) -> list[dict]:
    """OSM 'out geom tags' JSON → [{"coords":[[lat,lon],...], "kind":str, "width_m":float}]。

    bbox を与えると、その範囲に交差する水路のみ返す（線分の bbox 重なり判定）。
    bbox は 4 要素すべて指定するか 4 要素とも省略する。
    fetch_if_missing=True かつ JSON が無く bbox がある場合のみ Overpass から取得
    （既定 False＝従来どおり空を返す）。
    cache_dir を渡すと Overpass geom キャッシュもそのディレクトリ配下に置く。

    暗渠（tunnel/culvert）は地表に水面を作らないので除外する。
    """
    has_bbox = _require_bbox(lat_min, lat_max, lon_min, lon_max, what="load_waterways")

    if kinds is None:
        kinds = DEFAULT_KINDS + (("river",) if include_river else ())
    elif include_river and "river" not in kinds:
        kinds = tuple(kinds) + ("river",)
    kinds = tuple(kinds)

    elements: list = []
    p = Path(json_path) if json_path else None
    if p is not None and p.exists():
        try:
            elements = json.loads(p.read_text(encoding="utf-8")).get("elements", [])
        except Exception:
            return []
    elif fetch_if_missing and has_bbox:
        try:
            from osm_cache import DEFAULT_CACHE_DIR, fetch_overpass_geom
        except ImportError:
            from .osm_cache import DEFAULT_CACHE_DIR, fetch_overpass_geom
        elements = fetch_overpass_geom(
            "waterway", lat_min, lat_max, lon_min, lon_max,
            cache_dir=(DEFAULT_CACHE_DIR if cache_dir is None else cache_dir),
            verbose=verbose).get("elements", [])
    else:
        return []

    out = []
    for e in elements:
        if e.get("type") != "way":
            continue
        tags = e.get("tags", {}) or {}
        kind = tags.get("waterway", "")
        if kind not in kinds:
            continue
        # 暗渠は地上に見えない。tunnel=culvert / covered=yes を落とす。
        if tags.get("tunnel") or str(tags.get("covered", "")).lower() == "yes":
            continue
        geom = e.get("geometry") or []
        coords = [[g["lat"], g["lon"]] for g in geom if "lat" in g and "lon" in g]
        if len(coords) < 2:
            continue
        if has_bbox:
            las = [c[0] for c in coords]
            los = [c[1] for c in coords]
            if (max(las) < lat_min or min(las) > lat_max
                    or max(los) < lon_min or min(los) > lon_max):
                continue
        out.append({"coords": coords, "kind": kind, "width_m": _width_m(tags, kind),
                    "name": tags.get("name", "")})
    return out


if __name__ == "__main__":
    import sys
    jp = sys.argv[1] if len(sys.argv) > 1 else "data_cache/osm/gobo_waterways_geom.json"
    ws = load_waterways(jp)
    print(f"{len(ws)} waterways")
    by_kind: dict[str, int] = {}
    for w in ws:
        by_kind[w["kind"]] = by_kind.get(w["kind"], 0) + 1
    for k, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8} {n:>4}")
    for w in ws[:8]:
        print(f"  {w['kind']:8} w={w['width_m']:.1f}m pts={len(w['coords'])} {w['name']}")
