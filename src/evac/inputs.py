"""
evac/inputs.py — 避難経路解析の入力を 5 m 格子 (県 R8 津波 npz の格子そのまま) に揃える。

出力
  results/evac/inputs.npz      dem / t_arrive_sec / hmax / road(uint8) / building(uint8) / 施設 / 格子定義
  results/evac/roads.json      OSM highway の way 座標列 (グラフ構築用、roads.py が読む)
  results/evac/inputs_check.png 確認図 (陰影 DEM + hmax + 道路 + 建物 + 施設)

usage:  .venv/bin/python src/evac/inputs.py            (Overpass はキャッシュ → data_cache/osm)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from osm_cache import fetch_buildings_roads  # noqa: E402
sys.path.insert(0, str(REPO / "src/evac"))
from common import setup_japanese_font, hillshade  # noqa: E402

NPZ = REPO / "data_cache/wakayama_r8_tsunami/decoded/gobo_r8_tsunami_depth5m.npz"
EVAC_CSV = REPO / "data_cache/wakayama_r8_tsunami/evac/30205_2.csv"
OUT = REPO / "results/evac"

# 徒歩避難に使えない highway (自動車専用道 / 未供用) と、歩けるが敢えて除く種別
EXCLUDE_HIGHWAY = {"motorway", "motorway_link", "proposed", "construction", "abandoned",
                   "raceway", "bus_guideway", "platform", "corridor", "elevator"}
# 幅員は避難計算では無視 (指針) — 値はラスタ表示のみに使う
HIGHWAY_WIDTH_M = {"trunk": 12, "primary": 9, "secondary": 7, "tertiary": 6,
                   "residential": 5, "unclassified": 5, "service": 3, "footway": 2,
                   "path": 2, "steps": 2, "pedestrian": 3, "living_street": 4, "track": 3}


def grid_from_npz(g):
    return dict(lat_max=float(g["lat_max"]), lon_min=float(g["lon_min"]),
                res_lat=float(g["res_lat"]), res_lon=float(g["res_lon"]),
                H=int(g["dem"].shape[0]), W=int(g["dem"].shape[1]))


def latlon_to_rc(grid, lat, lon):
    """連続 (row, col)。row = (lat_max - lat)/res_lat, col = (lon - lon_min)/res_lon。"""
    return (grid["lat_max"] - np.asarray(lat)) / grid["res_lat"], (np.asarray(lon) - grid["lon_min"]) / grid["res_lon"]


def rc_to_latlon(grid, r, c):
    return grid["lat_max"] - np.asarray(r) * grid["res_lat"], grid["lon_min"] + np.asarray(c) * grid["res_lon"]


def cell_size_m(grid):
    """(row 方向 [m/cell], col 方向 [m/cell])"""
    lat_c = grid["lat_max"] - grid["res_lat"] * grid["H"] / 2
    return (grid["res_lat"] * 111320.0, grid["res_lon"] * 111320.0 * math.cos(math.radians(lat_c)))


def load_facilities(grid):
    rows = []
    with open(EVAC_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("津波") or "").strip() != "1":
                continue
            lat, lon = float(r["緯度"]), float(r["経度"])
            rr, cc = latlon_to_rc(grid, lat, lon)
            inside = 0 <= rr < grid["H"] - 1 and 0 <= cc < grid["W"] - 1
            rows.append(dict(id=r["共通ID"], name=r["施設・場所名"].strip(), addr=r["住所"],
                             lat=lat, lon=lon, row=float(rr), col=float(cc), inside=bool(inside),
                             note=(r.get("備考") or "").strip()))
    return rows


def rasterize_osm(grid, osm):
    """道路中心線 (all_touched で 1 セル幅) と建物面を 5 m 格子へ焼く。"""
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import LineString, Polygon

    tf = Affine(grid["res_lon"], 0, grid["lon_min"] - grid["res_lon"] / 2,
                0, -grid["res_lat"], grid["lat_max"] + grid["res_lat"] / 2)
    shp = (grid["H"], grid["W"])
    road_shapes, ways = [], []
    for w in osm["roads"]:
        ht = w["tags"].get("highway", "")
        if ht in EXCLUDE_HIGHWAY or len(w["coords"]) < 2:
            continue
        if w["tags"].get("access") in ("private", "no") and ht in ("service",):
            continue
        ls = LineString([(lo, la) for la, lo in w["coords"]])
        road_shapes.append((ls, 1))
        ways.append(dict(highway=ht, name=w["tags"].get("name", ""), coords=w["coords"],
                         bridge=w["tags"].get("bridge", ""), tunnel=w["tags"].get("tunnel", "")))
    road = features.rasterize(road_shapes, out_shape=shp, transform=tf, fill=0,
                              dtype="uint8", all_touched=True) if road_shapes else np.zeros(shp, np.uint8)
    b_shapes = []
    for b in osm["buildings"]:
        if len(b["coords"]) >= 4:
            b_shapes.append((Polygon([(lo, la) for la, lo in b["coords"]]), 1))
    building = features.rasterize(b_shapes, out_shape=shp, transform=tf, fill=0,
                                  dtype="uint8", all_touched=False) if b_shapes else np.zeros(shp, np.uint8)
    return road, building, ways


def check_figure(grid, dem, hmax, road, building, fac, path):
    import matplotlib
    matplotlib.use("Agg")
    setup_japanese_font()
    import matplotlib.pyplot as plt
    dy, dx = cell_size_m(grid)
    hs = hillshade(dem, dy, dx)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=110)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    sea = np.isnan(dem)
    ax.imshow(np.ma.masked_where(~sea, sea), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6, interpolation="nearest")
    hm = np.ma.masked_where(~(hmax >= 0.3), hmax)
    im = ax.imshow(hm, cmap="jet", vmin=0, vmax=10, alpha=0.45, interpolation="nearest")
    ax.imshow(np.ma.masked_where(building == 0, building), cmap="Greys", vmin=0, vmax=1.3, alpha=0.9, interpolation="nearest")
    ax.imshow(np.ma.masked_where(road == 0, road), cmap="autumn", vmin=0, vmax=1.5, alpha=0.9, interpolation="nearest")
    ins = [f for f in fac if f["inside"]]
    ax.scatter([f["col"] for f in ins], [f["row"] for f in ins], s=28, c="lime", edgecolors="k", zorder=5, label=f"指定緊急避難場所(津波) {len(ins)}")
    for f in ins:
        if "タワー" in f["name"] or f["name"] in ("御坊小学校", "御坊市役所"):
            ax.annotate(f["name"], (f["col"], f["row"]), fontsize=7, color="w", xytext=(3, 3), textcoords="offset points")
    plt.colorbar(im, ax=ax, fraction=0.03, label="R8 最大浸水深 hmax [m]")
    ax.set_title("evac inputs (5 m 格子): 陰影 DEM + R8 hmax + OSM 道路(橙)/建物(灰) + 避難場所")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    g = np.load(NPZ)
    grid = grid_from_npz(g)
    dem = g["dem"].astype(np.float32)
    T = g["t_arrive_sec"].astype(np.float32)
    hmax = g["hmax"].astype(np.float32)
    lat_min = grid["lat_max"] - grid["res_lat"] * (grid["H"] - 1)
    lon_max = grid["lon_min"] + grid["res_lon"] * (grid["W"] - 1)
    print(f"[inputs] grid {grid['H']}x{grid['W']}  lat[{lat_min:.5f},{grid['lat_max']:.5f}] lon[{grid['lon_min']:.5f},{lon_max:.5f}]"
          f"  cell {cell_size_m(grid)[0]:.2f}x{cell_size_m(grid)[1]:.2f} m")
    osm = fetch_buildings_roads(lat_min, grid["lat_max"], grid["lon_min"], lon_max,
                                highway_width_m=HIGHWAY_WIDTH_M, verbose=True)
    print(f"[inputs] OSM buildings={osm['n_buildings']} roads={osm['n_roads']}")
    road, building, ways = rasterize_osm(grid, osm)
    fac = load_facilities(grid)
    n_in = sum(f["inside"] for f in fac)
    print(f"[inputs] road cells={int(road.sum())} building cells={int(building.sum())}  facilities(津波=1)={len(fac)} inside grid={n_in}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "inputs.npz", dem=dem, t_arrive_sec=T, hmax=hmax, road=road, building=building,
                        lat_max=grid["lat_max"], lon_min=grid["lon_min"], res_lat=grid["res_lat"], res_lon=grid["res_lon"],
                        fac_id=np.array([f["id"] for f in fac]), fac_name=np.array([f["name"] for f in fac]),
                        fac_lat=np.array([f["lat"] for f in fac]), fac_lon=np.array([f["lon"] for f in fac]),
                        fac_row=np.array([f["row"] for f in fac]), fac_col=np.array([f["col"] for f in fac]),
                        fac_inside=np.array([f["inside"] for f in fac]), fac_note=np.array([f["note"] for f in fac]),
                        note="県R8津波 5m格子 (gobo_r8_tsunami_depth5m.npz) + OSM highway/building + GSI指定緊急避難場所(津波)")
    with open(OUT / "roads.json", "w", encoding="utf-8") as f:
        json.dump(dict(bbox=[lat_min, grid["lat_max"], grid["lon_min"], lon_max], ways=ways), f, ensure_ascii=False)
    check_figure(grid, dem, hmax, road, building, fac, OUT / "inputs_check.png")
    print(f"[inputs] wrote {OUT/'inputs.npz'}, roads.json ({len(ways)} ways), inputs_check.png")


if __name__ == "__main__":
    main()
