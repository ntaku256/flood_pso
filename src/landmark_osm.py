"""
landmark_osm.py
OSM のランドマーク（駅/役所/学校/病院/消防/警察/寺社）を **点** で取得し、
category + name 付きで返す。terrain_render.add_landmark_markers が category 色の
発光柱で強調する（専用3Dモデルが無い御坊で「ランドマークを目立たせる」ための層）。

橋/電力/駐車場と同じ osm_cache.fetch_overpass_geom（bbox量子化キャッシュ・オフライン
ガード・Overpass ミラー）を共用。way はジオメトリ重心を代表点にする。
"""
from __future__ import annotations

from osm_cache import fetch_overpass_geom

# category → 柱ブロック（遠望で識別できる色分け）。頂部は共通で sea_lantern+glowstone。
LANDMARK_STYLE = {
    "station":      "orange_concrete",   # 駅
    "townhall":     "cyan_concrete",     # 役所
    "school":       "yellow_concrete",   # 学校・大学
    "hospital":     "white_concrete",    # 病院
    "fire_station": "red_concrete",      # 消防
    "police":       "blue_concrete",     # 警察
    "worship":      "purple_concrete",   # 寺社・教会
}
DEFAULT_LANDMARK_BLOCK = "light_gray_concrete"

# 日本語ラベル（ログ用）
CATEGORY_JA = {
    "station": "駅", "townhall": "役所", "school": "学校",
    "hospital": "病院", "fire_station": "消防", "police": "警察", "worship": "寺社",
}


def _category(tags: dict) -> str | None:
    """OSM タグ → ランドマーク category（該当しなければ None）。"""
    if (tags.get("railway") == "station" or tags.get("building") == "train_station"
            or tags.get("public_transport") == "station"):
        return "station"
    am = tags.get("amenity", "")
    if am == "townhall":
        return "townhall"
    if am in ("school", "university", "college"):
        return "school"
    if am == "hospital":
        return "hospital"
    if am == "fire_station":
        return "fire_station"
    if am == "police":
        return "police"
    if am == "place_of_worship":
        return "worship"
    return None


def load_landmarks(*, lat_min, lat_max, lon_min, lon_max, verbose: bool = True) -> list[dict]:
    """bbox 内の OSM ランドマークを [{"lat","lon","category","name"}] で返す。
    取得失敗（オフライン等）は空リスト（強調は任意機能なので劣化継続）。
    同一 (category, name) は 1 件に統合（node と way の重複を除去）。"""
    try:
        data = fetch_overpass_geom("landmark", lat_min, lat_max, lon_min, lon_max,
                                   verbose=verbose)
    except Exception as e:               # OfflineError は BaseException なので伝播（意図通り）
        if verbose:
            print(f"  [landmark] OSM 取得スキップ: {e}")
        return []
    out: list[dict] = []
    seen: set = set()
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        cat = _category(tags)
        if cat is None:
            continue
        if el.get("type") == "node":
            la, lo = el.get("lat"), el.get("lon")
        elif "geometry" in el and el["geometry"]:
            g = el["geometry"]
            la = sum(p["lat"] for p in g) / len(g)
            lo = sum(p["lon"] for p in g) / len(g)
        else:
            continue
        if la is None or lo is None:
            continue
        if not (lat_min <= la <= lat_max and lon_min <= lo <= lon_max):
            continue
        name = tags.get("name", "")
        key = (cat, name) if name else (cat, round(la, 5), round(lo, 5))
        if key in seen:
            continue
        seen.add(key)
        out.append({"lat": la, "lon": lo, "category": cat, "name": name})
    return out
