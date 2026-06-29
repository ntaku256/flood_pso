#!/usr/bin/env python3
"""gen_data_preview.py — 御坊 data_preview 生成器（永続版）

失われた scratchpad/gen_full_preview.py（full 高解像度ラスタ）と overview 合成図生成器を
統合・再構築したもの。今後 preview を再生成できるよう src/ に常駐させる。

二様式:
  - overview : 中心市街 bbox の軸付きマルチパネル合成図（構造物パネルは再生成、
               重いパネル(オルソ/土地利用/建物高/樹冠/洪水GT/建物FP)は既存 R_*.png を流用）。
               → results/data_preview/overview.png ＋ 各構造物パネル R_*.png
  - full     : 御坊全域・枠なし高解像度ラスタ（地形グレースケール背景＋フィーチャ色重畳）。
               → results/data_preview/gobo_full/gobo_full_R_*.png

新規統合レイヤ:
  - road+rail               : FGD RdEdg(灰) ＋ RailCL(赤)
  - bridge+tunnel+special   : OSM 橋 ＋ トンネル(破線) ＋ 電線(voltage色＋鉄塔点) ＋ 駐車場(塗り)

データは全てキャッシュ再利用（新規ネットワーク取得なし）。
使い方:
  python src/gen_data_preview.py --mode overview
  python src/gen_data_preview.py --mode full
  python src/gen_data_preview.py --mode both
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fgd_vector import load_roads, load_rail, load_water       # noqa: E402
from bridge_osm import load_bridges                            # noqa: E402
from power_osm import load_power                               # noqa: E402
from parking_osm import load_parking                           # noqa: E402

# ── パス ──────────────────────────────────────────────────────────────────
LID = ROOT / "data_cache/wakayama_lidar"
OSM = ROOT / "data_cache/osm"
INUND_NPZ = ROOT / "data_cache/inund_zeniki/inund_gt_w5.000.npz"
PREV_DIR = ROOT / "results/data_preview"
FULL_DIR = PREV_DIR / "gobo_full"

D561 = ROOT / "../kennkyuu20260114/地形データ/FG-GML-503561-ALL-20251001"
D551 = ROOT / "../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101"
FGD_RDEDG = [D561 / "FG-GML-503561-RdEdg-20251001-0001.xml",
             D551 / "FG-GML-503551-RdEdg-20260101-0001.xml"]
FGD_RAIL = [D561 / "FG-GML-503561-RailCL-20251001-0001.xml",
            D551 / "FG-GML-503551-RailCL-20260101-0001.xml"]
FGD_WA = [D561 / "FG-GML-503561-WA-20251001-0001.xml",
          D561 / "FG-GML-503561-WStrA-20251001-0001.xml",
          D551 / "FG-GML-503551-WA-20260101-0001.xml",
          D551 / "FG-GML-503551-WStrA-20260101-0001.xml"]
BRIDGES_JSON = OSM / "gobo_bridges_full_geom.json"
TUNNELS_JSON = OSM / "gobo_tunnels_geom.json"
POWER_JSON = OSM / "gobo_power_geom.json"
PARKING_JSON = OSM / "gobo_parking_geom.json"

# 中心市街 overview bbox（既存 R_road.png の範囲に合わせる。--bbox で上書き可）
OVERVIEW_BBOX = (33.873, 33.889, 135.155, 135.182)   # lat_min,lat_max,lon_min,lon_max

KOSEN = (33.8668, 135.1387)      # 和歌山高専（黄星）
FGD_SOUTH_LAT = 33.83333         # FGD 503561/503551 メッシュ境界（シアン線）
DEM_ROW0_NORTH = True            # npz dem の行 0 が北(lat_max)か。overlay 不一致なら反転。

# ── 色 ────────────────────────────────────────────────────────────────────
C_ROAD = "#555555"
C_RAIL = "#d62728"
C_BRIDGE = "#1f77b4"
C_TUNNEL = "#ff7f0e"
C_PARK = "#9467bd"
C_TOWER = "#8c564b"
C_WATER = "#3b78c2"
# 送電電圧 → 色（voltage[V] 段階）
def volt_color(v: int) -> str:
    if v >= 500000: return "#7b1fa2"
    if v >= 220000: return "#c2185b"
    if v >= 110000: return "#e64a19"
    if v >= 60000:  return "#f9a825"
    return "#9e9d24"


# ── データ読み込み ─────────────────────────────────────────────────────────
def _dem_npz_path() -> Path:
    cands = sorted(LID.glob("*grid0.666667m_pp.npz"), key=lambda p: p.name.count("+"))
    if not cands:
        raise FileNotFoundError("DEM mosaic npz not found in data_cache/wakayama_lidar")
    return cands[-1]   # '+' が最多 = 18図郭結合


def load_dem():
    p = _dem_npz_path()
    d = np.load(p, allow_pickle=True)
    dem = d["dem"].astype(np.float32)
    ext = (float(d["lat_min"]), float(d["lat_max"]),
           float(d["lon_min"]), float(d["lon_max"]))
    print(f"[dem] {p.name[:36]}… shape={dem.shape} "
          f"lat[{ext[0]:.4f},{ext[1]:.4f}] lon[{ext[2]:.4f},{ext[3]:.4f}]")
    return dem, ext


def load_inund():
    if not INUND_NPZ.exists():
        return None
    return np.load(INUND_NPZ, allow_pickle=True)["inund"].astype(np.float32)


def crop_grid(arr, dem_ext, bbox):
    """全域 grid 配列を bbox 矩形に切り出し、(crop, imshow_extent) を返す。"""
    lat_min, lat_max, lon_min, lon_max = dem_ext
    H, W = arr.shape
    rlat = (lat_max - lat_min) / H
    rlon = (lon_max - lon_min) / W
    blat0, blat1, blon0, blon1 = bbox
    if DEM_ROW0_NORTH:
        r0 = int((lat_max - blat1) / rlat); r1 = int((lat_max - blat0) / rlat)
    else:
        r0 = int((blat0 - lat_min) / rlat); r1 = int((blat1 - lat_min) / rlat)
    c0 = int((blon0 - lon_min) / rlon); c1 = int((blon1 - lon_min) / rlon)
    r0, r1 = max(0, r0), min(H, r1); c0, c1 = max(0, c0), min(W, c1)
    crop = arr[r0:r1, c0:c1]
    return crop, [blon0, blon1, blat0, blat1]


def _concat(loader, paths, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    out = []
    for p in paths:
        if Path(p).exists():
            out += loader(str(p), lat_min=lat_min, lat_max=lat_max,
                          lon_min=lon_min, lon_max=lon_max)
    return out


def load_features(bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    feats = {
        "roads": _concat(load_roads, FGD_RDEDG, bbox),
        "rails": _concat(load_rail, FGD_RAIL, bbox),
        "water": _concat(load_water, FGD_WA, bbox),
        "bridges": load_bridges(str(BRIDGES_JSON), lat_min=lat_min, lat_max=lat_max,
                                lon_min=lon_min, lon_max=lon_max),
        "tunnels": load_bridges(str(TUNNELS_JSON), lat_min=lat_min, lat_max=lat_max,
                                lon_min=lon_min, lon_max=lon_max),
        "power": load_power(str(POWER_JSON), lat_min=lat_min, lat_max=lat_max,
                            lon_min=lon_min, lon_max=lon_max),
        "parking": load_parking(str(PARKING_JSON), lat_min=lat_min, lat_max=lat_max,
                                lon_min=lon_min, lon_max=lon_max),
    }
    print(f"[feat] roads={len(feats['roads'])} rails={len(feats['rails'])} "
          f"water={len(feats['water'])} bridges={len(feats['bridges'])} "
          f"tunnels={len(feats['tunnels'])} power_lines={len(feats['power']['lines'])} "
          f"towers={len(feats['power']['towers'])} parking={len(feats['parking'])}")
    return feats


# ── 描画ヘルパ（coords=[lat,lon] → (x=lon, y=lat)） ──────────────────────────
def _segs(feats_coords):
    return [[(c[1], c[0]) for c in f["coords"]] for f in feats_coords if len(f["coords"]) >= 2]


def draw_polylines(ax, items, color, lw=0.8, ls="solid", alpha=1.0, zorder=2):
    segs = _segs(items)
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                         linestyles=ls, alpha=alpha, zorder=zorder))


def draw_polys(ax, items, facecolor, edgecolor="none", alpha=0.5, zorder=2):
    polys = [[(c[1], c[0]) for c in f["coords"]] for f in items if len(f["coords"]) >= 3]
    if polys:
        ax.add_collection(PolyCollection(polys, facecolors=facecolor, edgecolors=edgecolor,
                                         alpha=alpha, linewidths=0.4, zorder=zorder))


def draw_power(ax, power, lw=0.9, pt=4, zorder=3):
    for L in power["lines"]:
        seg = [(c[1], c[0]) for c in L["coords"]]
        if len(seg) >= 2:
            ax.add_collection(LineCollection([seg], colors=volt_color(L["voltage"]),
                                             linewidths=lw, zorder=zorder))
    if power["towers"]:
        xs = [t["lon"] for t in power["towers"]]; ys = [t["lat"] for t in power["towers"]]
        ax.scatter(xs, ys, s=pt, c=C_TOWER, marker="^", zorder=zorder + 1, linewidths=0)


# ── overview パネル群 ───────────────────────────────────────────────────────
def panel_terrain(ax, dem_crop, ext):
    ax.imshow(dem_crop, extent=ext, origin="upper" if DEM_ROW0_NORTH else "lower",
              cmap="terrain", aspect="auto")
    ax.set_title("1. Terrain (LiDAR grd 0.5m DEM)")


def panel_water(ax, feats, ext):
    draw_polys(ax, feats["water"], C_WATER, alpha=0.7)
    _finish(ax, ext, "6. Water (FG-GML WA/WStrA)")


def panel_road_rail(ax, feats, ext):
    draw_polylines(ax, feats["roads"], C_ROAD, lw=0.6, zorder=2)
    draw_polylines(ax, feats["rails"], C_RAIL, lw=1.6, zorder=3)
    _finish(ax, ext, "5. Road + Rail (FG-GML RdEdg/RailCL)")
    ax.legend(handles=[Line2D([0], [0], color=C_ROAD, lw=1.2, label=f"road ({len(feats['roads'])})"),
                       Line2D([0], [0], color=C_RAIL, lw=2.0, label=f"rail ({len(feats['rails'])})")],
              loc="upper left", fontsize=6, framealpha=0.8)


def panel_special(ax, feats, ext):
    draw_polys(ax, feats["parking"], C_PARK, edgecolor=C_PARK, alpha=0.45, zorder=2)
    draw_polylines(ax, feats["bridges"], C_BRIDGE, lw=1.8, zorder=4)
    draw_polylines(ax, feats["tunnels"], C_TUNNEL, lw=1.8, ls=(0, (3, 2)), zorder=4)
    draw_power(ax, feats["power"], lw=0.9, pt=6, zorder=5)
    _finish(ax, ext, "7. Bridge + Tunnel + Power + Parking")
    ax.legend(handles=[
        Line2D([0], [0], color=C_BRIDGE, lw=2, label=f"bridge ({len(feats['bridges'])})"),
        Line2D([0], [0], color=C_TUNNEL, lw=2, ls="--", label=f"tunnel ({len(feats['tunnels'])})"),
        Line2D([0], [0], color="#c2185b", lw=2, label=f"power ({len(feats['power']['lines'])})"),
        Line2D([0], [0], marker="^", color=C_TOWER, lw=0, label=f"tower ({len(feats['power']['towers'])})"),
        Patch(facecolor=C_PARK, alpha=0.45, label=f"parking ({len(feats['parking'])})"),
    ], loc="upper left", fontsize=5.5, framealpha=0.85)


def panel_flood(ax, inund_crop, ext):
    if inund_crop is None:
        ax.text(0.5, 0.5, "no inundation", ha="center", va="center"); ax.set_title("9. Flood")
        return
    m = np.ma.masked_less_equal(inund_crop, 0.02)
    ax.imshow(m, extent=ext, origin="upper" if DEM_ROW0_NORTH else "lower",
              cmap="Blues", vmin=0, vmax=5, aspect="auto")
    _finish(ax, ext, "9. Flood depth (DEM, water +5m)")


def _finish(ax, ext, title):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(title); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_aspect("auto")
    ax.tick_params(labelsize=6)


def panel_reuse(ax, png_name, fallback_title):
    p = PREV_DIR / png_name
    if p.exists():
        ax.imshow(mpimg.imread(p)); ax.axis("off")
    else:
        ax.text(0.5, 0.5, f"(missing {png_name})", ha="center", va="center", fontsize=7)
        ax.set_title(fallback_title); ax.axis("off")


# ── overview 合成図 ────────────────────────────────────────────────────────
def render_overview(bbox):
    dem, dem_ext = load_dem()
    inund = load_inund()
    feats = load_features(bbox)
    dem_c, ext = crop_grid(dem, dem_ext, bbox)
    inund_c = crop_grid(inund, dem_ext, bbox)[0] if inund is not None else None

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    fig.suptitle("Gobo data by ROLE — structural refresh "
                 "(road+rail / bridge+tunnel+power+parking added)", fontsize=14)
    A = axes.ravel()
    panel_terrain(A[0], dem_c, ext)
    panel_reuse(A[1], "R_bldh.png", "2. Building height")
    panel_reuse(A[2], "R_tree.png", "3. Tree canopy")
    panel_reuse(A[3], "R_bld.png", "4. Building footprint")
    panel_road_rail(A[4], feats, ext)
    panel_water(A[5], feats, ext)
    panel_special(A[6], feats, ext)
    panel_reuse(A[7], "R_surf.png", "8. Surface ortho")
    panel_flood(A[8], inund_c, ext)
    panel_reuse(A[9], "R_floodgt.png", "10. Flood GT (PSO)")
    panel_reuse(A[10], "R_landuse.png", "11. Land use")
    A[11].axis("off")

    PREV_DIR.mkdir(parents=True, exist_ok=True)
    out = PREV_DIR / "overview.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[overview] saved {out}")

    # 構造物パネルを個別 R_*.png にも保存
    for name, fn in (("R_road_rail", panel_road_rail), ("R_special", panel_special)):
        f, ax = plt.subplots(figsize=(6, 5))
        fn(ax, feats, ext)
        f.tight_layout(); f.savefig(PREV_DIR / f"{name}.png", dpi=130); plt.close(f)
        print(f"[overview] saved {PREV_DIR / (name + '.png')}")


# ── full 全域ラスタ ────────────────────────────────────────────────────────
def _full_base(dem, dem_ext, down=2):
    d = dem[::down, ::down]
    ext = [dem_ext[2], dem_ext[3], dem_ext[0], dem_ext[1]]   # lon0,lon1,lat0,lat1
    return d, ext


def _full_fig(dem_d, ext):
    H, W = dem_d.shape
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(dem_d, extent=ext, origin="upper" if DEM_ROW0_NORTH else "lower",
              cmap="gray", aspect="auto")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    return fig, ax


def _full_overlays(ax):
    # 巨大キャンバス(~4559×7924px)向けに太く大きく。
    ax.axhline(FGD_SOUTH_LAT, color="cyan", lw=4, alpha=0.7, zorder=8)
    ax.scatter([KOSEN[1]], [KOSEN[0]], s=2600, c="yellow", marker="*",
               edgecolors="k", linewidths=2, zorder=9)
    ax.annotate("KOSEN", (KOSEN[1], KOSEN[0]), color="yellow", fontsize=34,
                ha="left", va="bottom", zorder=9,
                xytext=(6, 6), textcoords="offset points")


_LEG_KW = dict(fontsize=46, framealpha=0.9, handlelength=2.6, borderpad=0.8,
               labelspacing=0.6)


def render_full(down=2):
    dem, dem_ext = load_dem()
    bbox_full = (dem_ext[0], dem_ext[1], dem_ext[2], dem_ext[3])
    feats = load_features(bbox_full)
    dem_d, ext = _full_base(dem, dem_ext, down=down)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    # road+rail（道路=黄, 鉄道=赤太線。暗い地形上でも視認できる太さ）
    fig, ax = _full_fig(dem_d, ext)
    draw_polylines(ax, feats["roads"], "#e6c84b", lw=0.5, alpha=0.9, zorder=3)
    draw_polylines(ax, feats["rails"], C_RAIL, lw=1.8, zorder=4)
    _full_overlays(ax)
    ax.legend(handles=[Line2D([0], [0], color="#e6c84b", lw=5, label=f"road ({len(feats['roads'])})"),
                       Line2D([0], [0], color=C_RAIL, lw=6, label=f"rail ({len(feats['rails'])})")],
              loc="upper right", **_LEG_KW).set_zorder(10)
    fig.savefig(FULL_DIR / "gobo_full_R_road_rail.png", dpi=100); plt.close(fig)
    print("[full] saved gobo_full_R_road_rail.png")

    # bridge+tunnel+power+parking（暗背景で映える明色・太線）
    fig, ax = _full_fig(dem_d, ext)
    draw_polys(ax, feats["parking"], "#d050ff", edgecolor="#d050ff", alpha=0.8, zorder=3)
    draw_polylines(ax, feats["bridges"], "#00e5ff", lw=2.4, zorder=5)
    draw_polylines(ax, feats["tunnels"], "#ff9100", lw=2.6, ls=(0, (3, 2)), zorder=5)
    draw_power(ax, feats["power"], lw=2.0, pt=14, zorder=4)
    _full_overlays(ax)
    ax.legend(handles=[
        Line2D([0], [0], color="#00e5ff", lw=6, label=f"bridge ({len(feats['bridges'])})"),
        Line2D([0], [0], color="#ff9100", lw=6, ls="--", label=f"tunnel ({len(feats['tunnels'])})"),
        Line2D([0], [0], color="#c2185b", lw=6, label=f"power line ({len(feats['power']['lines'])})"),
        Line2D([0], [0], marker="^", color=C_TOWER, lw=0, markersize=20, label=f"tower ({len(feats['power']['towers'])})"),
        Patch(facecolor="#d050ff", alpha=0.8, label=f"parking ({len(feats['parking'])})"),
    ], loc="upper right", **_LEG_KW).set_zorder(10)
    fig.savefig(FULL_DIR / "gobo_full_R_special.png", dpi=100); plt.close(fig)
    print("[full] saved gobo_full_R_special.png")


def main():
    ap = argparse.ArgumentParser(description="御坊 data_preview 生成器")
    ap.add_argument("--mode", choices=["overview", "full", "both"], default="both")
    ap.add_argument("--bbox", type=str, default=None,
                    help="overview bbox 'lat_min,lat_max,lon_min,lon_max'")
    ap.add_argument("--full-down", type=int, default=2, help="full ラスタ DEM 間引き率")
    args = ap.parse_args()
    bbox = OVERVIEW_BBOX
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
    if args.mode in ("overview", "both"):
        render_overview(bbox)
    if args.mode in ("full", "both"):
        render_full(down=args.full_down)


if __name__ == "__main__":
    main()
