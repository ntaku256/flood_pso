"""
evac/dijkstra_ref.py — 標準法: 道路網の時間依存 (最遅出発) Dijkstra による避難余裕時間と避難困難区域。

  目的地 (2 モード)
    fac  : 国土地理院 指定緊急避難場所 (津波=1) のうち格子内で道路へ 60 m 以内にスナップできたもの
    safe : fac + 浸水域外 (T=inf) の道路ノード  … 県指針の「避難目標地点 = 浸水想定区域外 or 避難ビル」に相当
  歩行速度 1.0 / 0.5 m/s、避難開始 5 分後。
  余裕 margin(v) = L(v) − 300 s。避難困難セル = 浸水域 (T<inf) かつ margin<0。

出力 results/evac/
  margin_{mode}_{v10|v05}.npz  raster margin [s] (NaN=道路 50 m 圏外/海), L, src
  difficult.png                4 パネル (fac/safe × v10/v05)
  dijkstra_stats.json / dijkstra_reach.csv  困難面積 [ha]、施設別到達可能圏
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, START_DELAY_S, V_WALK, load_inputs, setup_japanese_font, hillshade  # noqa: E402
from roads import RoadGraph, load_ways  # noqa: E402

LINK_M = 300.0   # 施設を最寄り道路ノードへ直線接続する上限 (高台など OSM に小道が無い施設のため)
RASTER_R_M = 50.0


def rasterize_margin(G, margin, grid, land):
    """道路ノードの margin を 5 m セルへ最近傍 (50 m 以内) で展開。海/圏外は NaN。"""
    rr, cc = np.mgrid[0:grid.H, 0:grid.W]
    x, y = grid.rc_to_xy(rr.ravel(), cc.ravel())
    d, i = G.tree.query(np.c_[x, y], distance_upper_bound=RASTER_R_M)
    out = np.full(grid.H * grid.W, np.nan)
    ok = np.isfinite(d)
    out[ok] = margin[i[ok]]
    out = out.reshape(grid.H, grid.W)
    out[~land] = np.nan
    return out


def main():
    inp = load_inputs()
    grid, dem, T, hmax, fac = inp["grid"], inp["dem"], inp["T"], inp["hmax"], inp["fac"]
    land = ~np.isnan(dem)
    ways = load_ways()
    t0 = time.time()
    fac_in = [f for f in fac if f["inside"]]
    G = RoadGraph(ways, grid, T, dem, extra_points=[(f["lat"], f["lon"]) for f in fac_in], max_link_m=LINK_M)
    print(f"[roads] nodes={G.N} edges={G.n_edges} components={G.n_comp} ({time.time()-t0:.1f}s)")

    fac_nodes, fac_used = [], []
    for f, i, d in zip(fac_in, G.extra_ids, G.extra_link_m):
        if i is None:
            print(f"  [skip] {f['name']} 道路まで {d:.0f} m > {LINK_M:.0f} m")
            continue
        fac_nodes.append(i)
        fac_used.append(dict(f, node=i, snap_m=d, T_node=float(G.T[i])))
    print(f"[roads] facilities snapped: {len(fac_nodes)}/{sum(f['inside'] for f in fac)}")
    dry_nodes = np.where(~np.isfinite(G.T) & ~G.sea)[0]
    inund_cells = land & np.isfinite(T)
    cell_ha = grid.dx_m * grid.dy_m / 1e4
    print(f"[roads] 浸水域 (T<inf) {inund_cells.sum()*cell_ha:.1f} ha  /  dry road nodes {len(dry_nodes)}")

    stats = dict(nodes=G.N, edges=G.n_edges, components=G.n_comp, facilities_snapped=len(fac_nodes),
                 inundated_area_ha=float(inund_cells.sum() * cell_ha), cell_ha=cell_ha, cases={})
    reach_rows = []
    rasters = {}
    tower_idx = [k for k, f in enumerate(fac_used) if "津波避難タワー" in f["name"]]
    print(f"[roads] towers: " + ", ".join(f"{fac_used[k]['name']}(T={fac_used[k]['T_node']/60:.0f}min)" for k in tower_idx))
    for mode in ("fac", "notower", "safe"):
        if mode == "notower":   # 内閣府 WG (2013, タワー建設前) との突合用: 3 タワーを目的地から外す
            targets = [n for k, n in enumerate(fac_nodes) if k not in tower_idx]
        else:
            targets = list(fac_nodes) + (list(map(int, dry_nodes)) if mode == "safe" else [])
        for vk, v in V_WALK.items():
            t1 = time.time()
            L, src = G.latest_departure(targets, v)
            margin = L - START_DELAY_S
            R = rasterize_margin(G, margin, grid, land)
            rasters[(mode, vk)] = R
            np.savez_compressed(OUT / f"margin_{mode}_{vk}.npz", margin=R.astype(np.float32), L=L.astype(np.float32), src=src,
                                node_xy=G.xy.astype(np.float32), node_latlon=G.latlon, targets=np.array(targets),
                                note=f"margin[s]=L-300; mode={mode} v_walk={v}; NaN=道路50m圏外/海; -inf=到達不能")
            covered = ~np.isnan(R)
            diff = inund_cells & covered & (R < 0)
            unreach = inund_cells & (R == -np.inf)
            n_in_nodes = int(np.isfinite(G.T).sum())
            n_bad_nodes = int((np.isfinite(G.T) & (margin < 0)).sum())
            case = dict(v_walk=v, mode=mode, difficult_ha=float(diff.sum() * cell_ha),
                        difficult_unreachable_ha=float(unreach.sum() * cell_ha),
                        inundated_road_covered_ha=float((inund_cells & covered).sum() * cell_ha),
                        inundated_nodes=n_in_nodes, inundated_nodes_margin_neg=n_bad_nodes,
                        elapsed_s=time.time() - t1)
            stats["cases"][f"{mode}_{vk}"] = case
            print(f"[{mode} {vk}] 困難 {case['difficult_ha']:.1f} ha (うち道路網で到達不能 {case['difficult_unreachable_ha']:.1f} ha) / "
                  f"浸水域の道路50m圏 {case['inundated_road_covered_ha']:.1f} ha ; 浸水ノード {n_bad_nodes}/{n_in_nodes} が margin<0")
            # 施設別到達可能圏 (margin>=0 で最終的にその施設に着くノード数・浸水ノード数)
            tmap = {t: k for k, t in enumerate(targets)}   # src は targets の index → fac_used の index へ
            for k, f in enumerate(fac_used):
                if f["node"] not in tmap:
                    continue
                sel = (src == tmap[f["node"]]) & (margin >= 0)
                reach_rows.append(dict(mode=mode, v_walk=v, facility=f["name"], lat=f["lat"], lon=f["lon"],
                                       T_facility_min=(G.T[f["node"]] / 60 if np.isfinite(G.T[f["node"]]) else ""),
                                       nodes_reach=int(sel.sum()), inundated_nodes_reach=int((sel & np.isfinite(G.T)).sum()),
                                       ))
    with open(OUT / "dijkstra_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    with open(OUT / "dijkstra_reach.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reach_rows[0].keys()))
        w.writeheader(); w.writerows(reach_rows)
    plot(grid, dem, hmax, inund_cells, rasters, fac_used, OUT / "difficult.png")
    plot_margin(grid, dem, rasters[("fac", "v10")], fac_used, OUT / "margin_fac_v10.png")


def plot(grid, dem, hmax, inund, rasters, fac_used, path):
    import matplotlib
    matplotlib.use("Agg"); setup_japanese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    hs = hillshade(dem, grid.dy_m, grid.dx_m)
    fig, axes = plt.subplots(2, 3, figsize=(21, 11), dpi=90)
    for ax, (mode, vk) in zip(axes.T.ravel(), [("fac", "v10"), ("fac", "v05"), ("notower", "v10"), ("notower", "v05"), ("safe", "v10"), ("safe", "v05")]):
        R = rasters[(mode, vk)]
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~np.isnan(dem), np.isnan(dem)), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~inund, inund), cmap="Blues", vmin=0, vmax=2.5, alpha=0.35, interpolation="nearest")
        cover = inund & ~np.isnan(R)
        ok = cover & (R >= 0)
        bad = cover & (R < 0)
        ax.imshow(np.ma.masked_where(~ok, ok), cmap="Greens", vmin=0, vmax=1.4, alpha=0.7, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~bad, bad), cmap="Reds", vmin=0, vmax=1.3, alpha=0.9, interpolation="nearest")
        fu = [f for f in fac_used if not (mode == "notower" and "タワー" in f["name"])]
        ax.scatter([f["col"] for f in fu], [f["row"] for f in fu], s=16, c="yellow", edgecolors="k", linewidths=0.5, zorder=5)
        for f in fu:
            if "タワー" in f["name"]:
                ax.annotate(f["name"].replace("地区津波避難タワー", ""), (f["col"], f["row"]), fontsize=8, color="w", xytext=(4, -8), textcoords="offset points")
        ha = (bad).sum() * grid.dx_m * grid.dy_m / 1e4
        ax.set_title(f"目的地={dict(fac='指定避難場所', notower='指定避難場所(3タワー除く)', safe='指定避難場所+浸水域外')[mode]}  歩行 {'1.0' if vk=='v10' else '0.5'} m/s   避難困難 {ha:.0f} ha", fontsize=11)
        ax.set_xlim(250, 1000); ax.set_ylim(720, 150)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(handles=[Patch(color="#2a9d2a", label="浸水域・余裕あり (margin≥0)"), Patch(color="#d62728", label="浸水域・避難困難 (margin<0)"),
                               Patch(color="#9ecae1", label="浸水域 (道路 50 m 圏外)"), Patch(color="yellow", label="指定緊急避難場所(津波)")],
                      loc="lower left", fontsize=9)
    fig.suptitle("県 R8 津波 (30 cm 到達時刻) × OSM 道路網 最遅出発 Dijkstra: 避難困難区域 (避難開始 5 分後)", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_margin(grid, dem, R, fac_used, path):
    import matplotlib
    matplotlib.use("Agg"); setup_japanese_font()
    import matplotlib.pyplot as plt
    hs = hillshade(dem, grid.dy_m, grid.dx_m)
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=100)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~np.isnan(dem), np.isnan(dem)), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6, interpolation="nearest")
    M = np.where(np.isfinite(R), R / 60.0, np.where(R == -np.inf, -40.0, np.where(R == np.inf, 60.0, np.nan)))
    im = ax.imshow(np.ma.masked_invalid(np.clip(M, -40, 60)), cmap="RdYlGn", vmin=-40, vmax=60, alpha=0.85, interpolation="nearest")
    ax.scatter([f["col"] for f in fac_used], [f["row"] for f in fac_used], s=16, c="cyan", edgecolors="k", linewidths=0.5, zorder=5)
    plt.colorbar(im, ax=ax, fraction=0.03, label="余裕時間 margin [min] (L − 5 min; 浸水しない道路は L=∞→60 で飽和)")
    ax.set_title("最遅出発余裕時間 margin_fac_v10 (目的地=指定避難場所, 1.0 m/s)")
    ax.set_xlim(250, 1000); ax.set_ylim(720, 150); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
