"""
fgd_vector.py
国土地理院 基盤地図情報（FG-GML）のベクタレイヤを読み込み、
`tellus_data.fetch_osm_buildings_roads` と同じ dict 形式に変換する。

これにより、既存の OSM 配置パイプライン
  terrain_render.build_osm_masks → dem_to_blocks_enhanced(building_mask, road_mask)
を一切変更せず、建物・道路ソースだけ FG-GML（ローカル・高精度・API不要）に
差し替えられる。

対応レイヤ:
  BldA  建築物の外周線（面）  → buildings（ポリゴン → 立体化）
  RdEdg 道路縁              → roads（ポリライン → 幅バッファで砂利）
  WA    水域（面）          → water（ポリゴン、任意）

GML ジオメトリパス:
  BldA  : area/Surface/patches/PolygonPatch/exterior/Ring/curveMember/Curve/segments/LineStringSegment/posList
  RdEdg : loc/Curve/segments/LineStringSegment/posList
  posList は "lat lon lat lon ..." の空白区切り（JGD2024 緯度経度）

使い方:
  from fgd_vector import load_fgd_buildings_roads
  d = load_fgd_buildings_roads(bld_xml, rdedg_xml,
                               lat_min, lat_max, lon_min, lon_max)
  # d は {"buildings":[{"coords":[[lat,lon],...],"tags":{...}}],
  #       "roads":[{"coords":[...],"width_m":w,"tags":{...}}], ...}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from lxml import etree

GML = "http://www.opengis.net/gml/3.2"
FGD = "http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema"
POSLIST = f"{{{GML}}}posList"

# 建物 type → 高さ[m]（おおよその階数 × 3m）
BUILDING_HEIGHT_BY_TYPE = {
    "普通建物": 6.0,        # 木造2階相当
    "堅ろう建物": 12.0,     # RC 中層
    "普通無壁舎": 4.0,      # 倉庫・車庫など
    "堅ろう無壁舎": 4.0,
    "高層建物": 24.0,
}
DEFAULT_BUILDING_HEIGHT = 6.0

# 道路 type → 幅[m]
ROAD_WIDTH_BY_TYPE = {
    "真幅道路": 5.0,
    "軽車道": 3.0,
    "徒歩道": 2.0,
    "庭園路等": 2.0,
    "トンネル内の道路": 5.0,
    "建物内の道路": 4.0,
}
DEFAULT_ROAD_WIDTH = 4.0


def _parse_poslist(text: str) -> list[list[float]]:
    """'lat lon lat lon ...' → [[lat,lon], ...]"""
    vals = text.split()
    coords = []
    for i in range(0, len(vals) - 1, 2):
        coords.append([float(vals[i]), float(vals[i + 1])])
    return coords


def _coords_intersect_bbox(coords, lat_min, lat_max, lon_min, lon_max) -> bool:
    """ポリゴン/ラインの bbox が対象 bbox と交差するか（高速プレフィルタ）。"""
    arr = np.asarray(coords)
    if arr.size == 0:
        return False
    la, lo = arr[:, 0], arr[:, 1]
    return not (la.max() < lat_min or la.min() > lat_max
                or lo.max() < lon_min or lo.min() > lon_max)


def _iter_features(xml_path: str, local_name: str):
    """指定レイヤ要素を逐次 yield（メモリ節約のため処理後に解放）。"""
    tag = f"{{{FGD}}}{local_name}"
    ctx = etree.iterparse(xml_path, events=("end",), tag=tag)
    for _, el in ctx:
        yield el
        el.clear()
        # 先行兄弟を削除してメモリ常駐を防ぐ
        parent = el.getparent()
        if parent is not None:
            while el.getprevious() is not None:
                del parent[0]


def _building_rings(el) -> tuple[list | None, list[list]]:
    """BldA 要素 → (exterior_coords, [hole_coords,...])。
    PolygonPatch の exterior（外周）と interior（中庭の穴）を分離して返す。
    exterior が無ければ (None, [])。"""
    ext = None
    holes: list[list] = []
    for patch in el.iter(f"{{{GML}}}PolygonPatch"):
        ex = patch.find(f"{{{GML}}}exterior")
        if ex is not None:
            pl = ex.find(f".//{POSLIST}")
            if pl is not None and pl.text:
                c = _parse_poslist(pl.text)
                if len(c) >= 3 and ext is None:
                    ext = c
        for inr in patch.findall(f"{{{GML}}}interior"):
            pl = inr.find(f".//{POSLIST}")
            if pl is not None and pl.text:
                c = _parse_poslist(pl.text)
                if len(c) >= 3:
                    holes.append(c)
    return ext, holes


def load_buildings(xml_path: str, *, lat_min, lat_max, lon_min, lon_max) -> list[dict]:
    """BldA → [{"coords":[[lat,lon],...], "holes":[[[lat,lon],...],...],
               "tags":{"fgd_type","height_m"}}]
    coords=外周（後方互換）, holes=中庭の穴（per-building 高さ集約で空洞化に使う）。"""
    out = []
    for el in _iter_features(xml_path, "BldA"):
        ext, holes = _building_rings(el)
        if ext is None:
            continue
        if not _coords_intersect_bbox(ext, lat_min, lat_max, lon_min, lon_max):
            continue
        tp = el.findtext(f"{{{FGD}}}type") or ""
        h = BUILDING_HEIGHT_BY_TYPE.get(tp, DEFAULT_BUILDING_HEIGHT)
        out.append({"coords": ext, "holes": holes,
                    "tags": {"fgd_type": tp, "height_m": h}})
    return out


def load_roads(xml_path: str, *, lat_min, lat_max, lon_min, lon_max) -> list[dict]:
    """RdEdg → [{"coords":[...], "width_m":w, "tags":{"fgd_type"}}]"""
    out = []
    for el in _iter_features(xml_path, "RdEdg"):
        pl = el.find(f".//{POSLIST}")
        if pl is None or not pl.text:
            continue
        coords = _parse_poslist(pl.text)
        if len(coords) < 2:
            continue
        if not _coords_intersect_bbox(coords, lat_min, lat_max, lon_min, lon_max):
            continue
        tp = el.findtext(f"{{{FGD}}}type") or ""
        w = ROAD_WIDTH_BY_TYPE.get(tp, DEFAULT_ROAD_WIDTH)
        out.append({"coords": coords, "width_m": w, "tags": {"fgd_type": tp}})
    return out


def load_rail(xml_path: str, *, lat_min, lat_max, lon_min, lon_max) -> list[dict]:
    """RailCL（鉄道中心線）→ [{"coords":[[lat,lon],...], "tags":{"fgd_type"}}]。
    type 例: 普通鉄道 / 路面の鉄道 / 索道 等。道路と同じく polyline。"""
    out = []
    for el in _iter_features(xml_path, "RailCL"):
        pl = el.find(f".//{POSLIST}")
        if pl is None or not pl.text:
            continue
        coords = _parse_poslist(pl.text)
        if len(coords) < 2:
            continue
        if not _coords_intersect_bbox(coords, lat_min, lat_max, lon_min, lon_max):
            continue
        tp = el.findtext(f"{{{FGD}}}type") or ""
        out.append({"coords": coords, "tags": {"fgd_type": tp}})
    return out


def load_fgd_line_layer(xml_path: str, layer: str, *,
                        lat_min, lat_max, lon_min, lon_max) -> list[dict]:
    """任意の FG-GML **線**レイヤ（WStrL 水部構造物線 / Cstline 海岸線 / WL 水涯線 等）を
    → [{"coords":[[lat,lon],...], "tags":{"fgd_type"}}]。RailCL/RdEdg と同じ posList 経路。"""
    out = []
    for el in _iter_features(xml_path, layer):
        pl = el.find(f".//{POSLIST}")
        if pl is None or not pl.text:
            continue
        coords = _parse_poslist(pl.text)
        if len(coords) < 2:
            continue
        if not _coords_intersect_bbox(coords, lat_min, lat_max, lon_min, lon_max):
            continue
        tp = el.findtext(f"{{{FGD}}}type") or ""
        out.append({"coords": coords, "tags": {"fgd_type": tp}})
    return out


def load_water(xml_path: str, *, lat_min, lat_max, lon_min, lon_max) -> list[dict]:
    """WA / WStrA（水域面）→ [{"coords":[[lat,lon],...], "tags":{...}}]"""
    out = []
    for ln in ("WA", "WStrA"):
        try:
            for el in _iter_features(xml_path, ln):
                pl = el.find(f".//{POSLIST}")
                if pl is None or not pl.text:
                    continue
                coords = _parse_poslist(pl.text)
                if len(coords) < 3:
                    continue
                if not _coords_intersect_bbox(coords, lat_min, lat_max, lon_min, lon_max):
                    continue
                tp = el.findtext(f"{{{FGD}}}type") or ""
                out.append({"coords": coords, "tags": {"fgd_type": tp}})
        except Exception:
            pass
    return out


def load_fgd_buildings_roads(
    bld_xml: str | None,
    rdedg_xml: str | None,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    *,
    margin_deg: float = 0.0005,
    verbose: bool = True,
) -> dict:
    """
    FG-GML BldA / RdEdg を bbox でクリップして読み、
    fetch_osm_buildings_roads 互換 dict を返す。

    margin_deg : bbox を少し広げて境界の建物切れを防ぐ（~50m）
    bld_xml/rdedg_xml : 単一パス、リスト、または **カンマ区切り複数パス** を受け付ける。
      タイルが複数メッシュに跨る場合（例 御坊南端 002 = 503551+503561）に両方を union 読み。
    """
    la0, la1 = lat_min - margin_deg, lat_max + margin_deg
    lo0, lo1 = lon_min - margin_deg, lon_max + margin_deg

    def _as_list(x):
        if not x:
            return []
        if isinstance(x, (list, tuple)):
            return [str(p) for p in x if p]
        return [s.strip() for s in str(x).split(",") if s.strip()]

    buildings = []
    roads = []
    for bx in _as_list(bld_xml):
        if Path(bx).exists():
            buildings += load_buildings(bx, lat_min=la0, lat_max=la1,
                                        lon_min=lo0, lon_max=lo1)
    for rx in _as_list(rdedg_xml):
        if Path(rx).exists():
            roads += load_roads(rx, lat_min=la0, lat_max=la1,
                                lon_min=lo0, lon_max=lo1)

    if verbose:
        print(f"[fgd] buildings={len(buildings)}  roads={len(roads)}  "
              f"(bbox lat[{lat_min:.4f},{lat_max:.4f}] lon[{lon_min:.4f},{lon_max:.4f}])")

    return {
        "buildings": buildings,
        "roads": roads,
        "bbox": [lat_min, lat_max, lon_min, lon_max],
        "n_buildings": len(buildings),
        "n_roads": len(roads),
        "source": "fgd",
    }


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent.parent.parent / "kennkyuu20260114" / \
        "地形データ" / "FG-GML-503561-ALL-20251001"
    bld = base / "FG-GML-503561-BldA-20251001-0001.xml"
    rd = base / "FG-GML-503561-RdEdg-20251001-0001.xml"
    # 御坊駅周辺 1km スモークテスト
    BBOX = dict(lat_min=33.870, lat_max=33.880, lon_min=135.145, lon_max=135.158)
    d = load_fgd_buildings_roads(str(bld), str(rd), **BBOX)
    print(f"buildings={d['n_buildings']} roads={d['n_roads']}")
    if d["buildings"]:
        b = d["buildings"][0]
        print("sample building:", b["tags"], "verts=", len(b["coords"]),
              "first=", b["coords"][0])
    if d["roads"]:
        r = d["roads"][0]
        print("sample road:", r["tags"], "width_m=", r["width_m"], "verts=", len(r["coords"]))
