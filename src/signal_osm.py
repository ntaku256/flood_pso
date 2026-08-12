"""
signal_osm.py
OSM の交通信号（highway=traffic_signals ノード）を読み、信号レンダラ
（terrain_render.add_signal_blocks）用の点リストに変換する。

御坊 bbox 実測（2026-08-12）: traffic_signals 30ノード。OSM のこれは交差点の
**停止線ノード**であって灯器の物理位置ではないが、1m/block の粒度では見た目の
差は出ない。橋/電力/駐車場と同じ osm_cache.fetch_overpass_geom（bbox量子化キャッシュ・
オフラインガード）を共用する。
"""
from __future__ import annotations

import json
from pathlib import Path


def load_signals(json_path: str | None = None, *,
                 lat_min: float | None = None, lat_max: float | None = None,
                 lon_min: float | None = None, lon_max: float | None = None,
                 fetch_if_missing: bool = True, verbose: bool = True) -> list[dict]:
    """交通信号ノード → [{"lat","lon"}]。

    json_path があればそれを（Overpass 生 JSON）読む。無ければ bbox で osm_cache 経由取得
    （既定 fetch_if_missing=True）。取得失敗（オフライン等）は空リスト（信号は任意機能）。
    """
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
            elements = fetch_overpass_geom("signal", lat_min, lat_max, lon_min, lon_max,
                                           verbose=verbose).get("elements", [])
        except Exception as e:                # OfflineError は BaseException なので伝播
            if verbose:
                print(f"  [signals] OSM 取得スキップ: {e}")
            return []
    else:
        return []
    out = []
    for e in elements:
        if e.get("type") == "node" and "lat" in e and "lon" in e:
            la, lo = e["lat"], e["lon"]
            if None in (lat_min, lat_max, lon_min, lon_max) or \
               (lat_min <= la <= lat_max and lon_min <= lo <= lon_max):
                out.append({"lat": la, "lon": lo})
    return out
