"""
power_procedural.py
御坊（および日本の郊外）は OSM に個別の電柱 power=pole がほぼ無い
（実測 2026-08-12, Overpass, 御坊 bbox 33.855,135.125–33.915,135.205: pole=0本）。
配電柱は道路に沿って一定間隔・片側に立つので、FG-GML RdEdg（道路縁ポリライン）から
**手続き的に**電柱位置と架線を生成し、既存の terrain_render.add_power_blocks に
そのまま渡す（出力は power_osm.load_power と同じ {"lines":[...],"towers":[...]} 形式）。

配置根拠:
- 間隔 約30–35m: 全国 約3,600万本 ÷ 道路実延長 1,285,403.6km ≈ 28.1本/km（道路統計/MLIT）
- 片側のみ: 東京都「道路占用物件の配置基準」電柱・電話柱は同一路線上の片側
- 架線は路面上5m以上: 道路法施行令 第11条の2第1項第1号イ（本モジュールは点/線のみ生成、
  高さは add_power_blocks の voltage=0→約12m 基準に委ねる）

trap: 市街地は道路が密で、全 RdEdg 一律に生やすと過剰になる。tunnel/建物内の道路を除外し、
既存柱に近すぎる柱は min_gap_m で間引く（並走する道路の二重柱を防ぐ）。
"""
from __future__ import annotations

import math

# 生やさない道路種別（地下・建物内）
DEFAULT_EXCLUDE_TYPES = ("トンネル内の道路", "建物内の道路")


def _m_per_deg(lat: float) -> tuple[float, float]:
    """(m/緯度度, m/経度度) at lat。"""
    return 111320.0, 111320.0 * math.cos(math.radians(lat))


def poles_from_roads(roads, *, spacing_m: float = 33.0, offset_m: float = 3.0,
                     min_gap_m: float = 16.0, exclude_types=DEFAULT_EXCLUDE_TYPES,
                     verbose: bool = True) -> dict:
    """RdEdg 道路群 → {"lines":[...],"towers":[...]}（power_osm.load_power 互換）。

    roads: [{"coords":[[lat,lon],...], "width_m":w, "tags":{"fgd_type"}}]
    - 各道路をポリラインに沿って spacing_m 間隔で歩き、中心線から offset_m だけ片側へ
      ずらした位置に電柱（towers, kind="pole"）を置く。
    - 同一道路の連続柱を結ぶ架線（lines, voltage=0）を1本足す（径間ごとにカテナリ描画）。
    - 既存柱から min_gap_m 未満は間引く（並走路の二重柱防止）。空間ハッシュで O(n)。
    """
    towers: list = []
    lines: list = []
    cell = max(min_gap_m, 1.0)
    grid: dict = {}                     # (gx,gy) -> [(lat,lon),...]

    def _key(la, lo):
        mlat, mlon = _m_per_deg(la)
        return int(lo / (cell / mlon)), int(la / (cell / mlat))

    def _too_close(la, lo):
        gx, gy = _key(la, lo)
        mlat, mlon = _m_per_deg(la)
        for dgx in (-1, 0, 1):
            for dgy in (-1, 0, 1):
                for pla, plo in grid.get((gx + dgx, gy + dgy), ()):
                    dx = (lo - plo) * mlon
                    dy = (la - pla) * mlat
                    if dx * dx + dy * dy < min_gap_m * min_gap_m:
                        return True
        return False

    def _place(la, lo):
        gx, gy = _key(la, lo)
        grid.setdefault((gx, gy), []).append((la, lo))
        towers.append({"lat": la, "lon": lo, "kind": "pole"})

    n_roads = 0
    for r in roads:
        if r.get("tags", {}).get("fgd_type", "") in exclude_types:
            continue
        coords = r.get("coords") or []
        if len(coords) < 2:
            continue
        n_roads += 1
        road_poles: list = []
        cum = 0.0
        next_at = spacing_m * 0.5      # 交差点直上を避けて半間隔から
        for i in range(len(coords) - 1):
            la0, lo0 = coords[i]
            la1, lo1 = coords[i + 1]
            mlat, mlon = _m_per_deg((la0 + la1) * 0.5)
            dx = (lo1 - lo0) * mlon
            dy = (la1 - la0) * mlat
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                continue
            px, py = -dy / seg, dx / seg      # 片側への垂直単位ベクトル(m)
            while next_at <= cum + seg:
                f = (next_at - cum) / seg
                pla = la0 + (la1 - la0) * f
                plo = lo0 + (lo1 - lo0) * f
                mlat2, mlon2 = _m_per_deg(pla)
                ola = pla + (py * offset_m) / mlat2
                olo = plo + (px * offset_m) / mlon2
                if not _too_close(ola, olo):
                    _place(ola, olo)
                    road_poles.append([ola, olo])
                next_at += spacing_m
            cum += seg
        if len(road_poles) >= 2:
            lines.append({"coords": road_poles, "voltage": 0, "layer": 0,
                          "kind": "minor_line"})
    if verbose:
        print(f"  [power-poles] {len(towers)} 本を道路 {n_roads} 本沿いに手続き生成 "
              f"(間隔{spacing_m:.0f}m・片側offset{offset_m:.0f}m・{len(lines)}径間)")
    return {"lines": lines, "towers": towers}
