"""
ksj_evac.py
国土数値情報 P20（避難施設）GML を読み、指定 bbox 内の避難所を
[{"lat","lon","name","hazards"}] にして返す。

P20 構造:
  ksj:EvacuationFacilities
    ksj:position  → xlink:href="#<gml:Point id>"（位置参照）
    ksj:name      → 施設名
    ksj:tsunamiHazard / earthquakeHazard / windAndFloodDamage … → 対応災害(true/false)
  gml:Point[gml:id]/gml:pos = "lat lon"
"""
from __future__ import annotations
from lxml import etree

XLINK = "http://www.w3.org/1999/xlink"
_HAZARD_TAGS = {
    "tsunamiHazard": "津波",
    "earthquakeHazard": "地震",
    "windAndFloodDamage": "洪水",
    "volcanicHazard": "火山",
}


def load_evac_facilities(xml_path: str, *, lat_min, lat_max, lon_min, lon_max,
                         verbose: bool = True) -> list[dict]:
    root = etree.parse(xml_path).getroot()
    KSJ = root.nsmap.get("ksj")
    GML = root.nsmap.get("gml")

    # gml:Point id → (lat, lon)
    pts: dict[str, tuple[float, float]] = {}
    for pt in root.iter(f"{{{GML}}}Point"):
        pid = pt.get(f"{{{GML}}}id")
        pos = pt.find(f"{{{GML}}}pos")
        if pid and pos is not None and pos.text:
            la, lo = pos.text.split()[:2]
            pts[pid] = (float(la), float(lo))

    out: list[dict] = []
    for f in root.iter(f"{{{KSJ}}}EvacuationFacilities"):
        pr = f.find(f"{{{KSJ}}}position")
        href = pr.get(f"{{{XLINK}}}href") if pr is not None else None
        ll = pts.get(href.lstrip("#")) if href else None
        if ll is None:
            continue
        la, lo = ll
        if not (lat_min <= la <= lat_max and lon_min <= lo <= lon_max):
            continue
        nm_el = f.find(f"{{{KSJ}}}name")
        name = nm_el.text if nm_el is not None and nm_el.text else ""
        hazards = []
        for tag, label in _HAZARD_TAGS.items():
            el = f.find(f"{{{KSJ}}}{tag}")
            if el is not None and (el.text or "").strip().lower() in ("true", "1"):
                hazards.append(label)
        out.append({"lat": la, "lon": lo, "name": name, "hazards": hazards})

    if verbose:
        print(f"[evac] P20 避難所 {len(out)}件 "
              f"(bbox lat[{lat_min:.4f},{lat_max:.4f}] lon[{lon_min:.4f},{lon_max:.4f}])")
    return out
