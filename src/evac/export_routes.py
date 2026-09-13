"""
evac/export_routes.py — 避難経路 (Dijkstra / PSO 最良 / CCPSO2 最良) と最遅出発余裕 margin を tizucra-walk の
overlay スキーマへ書き出す。

  routes.json  → <tizucra-walk>/overlay_out/gobo/routes.json
      {"map":"gobo","scenario":{...},"routes":[{id,label,color,width,start,goal,points:[{lat,lon,x,z,elev,t_min}],length_m,time_min,cost}],"source"}
      x,z = tizucra-walk/tools/profiles.py latlon_to_block(PROFILES["gobo"]) / elev = GSI 5 m DEM [m] / t_min = 5 分 + 距離/v_walk
  evac/margin_v10.png + .json  → 8 block/セル格子 (build_depth_grid.py と同じ 844×1961、block_to_latlon でセル中心)
      L 値 = clip(round(margin_min)+128, 1, 253)  (+inf→253, -inf/到達不能→1), 0 = 道路 50 m 圏外 (陸), 255 = 5 m 格子の範囲外/海
      margin_v05 も同様に出す。

usage: .venv/bin/python src/evac/export_routes.py [tizucra-walk のパス]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, START_DELAY_S, load_inputs  # noqa: E402
from roads import fill_nan_nearest  # noqa: E402

WALK = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[2] / "tizucra-walk"
sys.path.insert(0, str(WALK / "tools"))
from profiles import PROFILES, latlon_to_block, block_to_latlon  # noqa: E402

SOURCE = ("『最大クラスの巨大地震(南海トラフ巨大地震)の津波シミュレーション動画(R8)』(和歌山県) を復号した 30 cm 到達時刻、"
          "国土地理院 5 m DEM・指定緊急避難場所、OpenStreetMap (© OpenStreetMap contributors, ODbL) 道路網 を加工して作成")
COLORS = dict(dijkstra="#ffffff", pso="#9e9e9e", ccpso2="#ffd54f")
LABELS = dict(dijkstra="Dijkstra 参照 (道路網・時間依存)", pso="標準 PSO (最良 seed)", ccpso2="CCPSO2 (最良 seed)")


def export_routes(p, inp, summary, out_path):
    grid = inp["grid"]
    dem = fill_nan_nearest(inp["dem"].astype(np.float64))
    with open(OUT / "routes_best.json", encoding="utf-8") as f:
        rb = json.load(f)
    v = summary["v_walk"]
    routes = []
    for pid, ent in rb.items():
        for m in ("dijkstra", "pso", "ccpso2"):
            r = ent[m]
            lat = np.array(r["lat"]); lon = np.array(r["lon"])
            x, y = grid.latlon_to_xy(lat, lon)
            ds = np.hypot(np.diff(x), np.diff(y)); sc = np.r_[0.0, np.cumsum(ds)]
            # 2 m サンプルは多いので 6 m 毎に間引く (端点は保持)
            keep = np.r_[np.arange(0, len(lat) - 1, 3), len(lat) - 1]
            rr, cc = grid.latlon_to_rc(lat[keep], lon[keep])
            ri = np.clip(np.round(rr).astype(int), 0, grid.H - 1); ci = np.clip(np.round(cc).astype(int), 0, grid.W - 1)
            bx, bz = latlon_to_block(p, lat[keep], lon[keep])
            pts = [dict(lat=round(float(a), 7), lon=round(float(b), 7), x=round(float(xx), 1), z=round(float(zz), 1),
                        elev=round(float(e), 2), t_min=round(float(START_DELAY_S / 60 + s / v / 60), 2))
                   for a, b, xx, zz, e, s in zip(lat[keep], lon[keep], bx, bz, dem[ri, ci], sc[keep])]
            pr = ent["pair"]
            routes.append(dict(id=f"{pid}_{m}", label=f"{pr['label']} / {LABELS[m]}" + (f" seed {r['seed']}" if r.get("seed") is not None else ""),
                               method=m, pair=pid, color=COLORS[m], width=3,
                               start=dict(name=pr["start"]["name"], lat=pr["start"]["lat"], lon=pr["start"]["lon"]),
                               goal=dict(name=pr["goal"]["name"], lat=pr["goal"]["lat"], lon=pr["goal"]["lon"]),
                               points=pts, length_m=round(float(sc[-1]), 1), time_min=round(float(r["time_min"]), 2), cost=round(float(r["cost"]), 1)))
    doc = dict(map="gobo",
               scenario=dict(source="和歌山県 R8 (2026-03) 南海トラフ巨大地震 津波浸水想定 (30 cm 到達時刻)", v_walk=v,
                             start_delay_min=START_DELAY_S / 60, budget_evals=summary["budget"], seeds=summary["seeds"],
                             cost="経路長[m] + λ1·浸水通過長 + λ2·道路外 + λ3·建物内 + λ4·終点遅刻[s]", lam=summary["lam"],
                             pairs=summary["pairs"]),
               routes=routes, source=SOURCE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[export] {out_path}  routes={len(routes)}  ({out_path.stat().st_size/1024:.0f} KB)")


def export_margin(p, inp, vk, out_dir):
    grid = inp["grid"]
    cb = p["cell_blocks"]
    w = -(-(p["x_max"] + 1) // cb); h = -(-(p["z_max"] + 1) // cb)
    m = np.load(OUT / f"margin_fac_{vk}.npz")
    R = m["margin"].astype(np.float64)           # [s] NaN = 道路圏外/海, ±inf
    js, is_ = np.mgrid[0:h, 0:w]
    lat, lon = block_to_latlon(p, is_ * cb + cb / 2.0, js * cb + cb / 2.0)
    rr, cc = grid.latlon_to_rc(lat, lon)
    ri = np.round(rr).astype(int); ci = np.round(cc).astype(int)
    inside = (ri >= 0) & (ri < grid.H) & (ci >= 0) & (ci < grid.W)
    out = np.full((h, w), 255, np.uint8)
    rs = R[np.clip(ri, 0, grid.H - 1), np.clip(ci, 0, grid.W - 1)]
    land = ~np.isnan(inp["dem"])[np.clip(ri, 0, grid.H - 1), np.clip(ci, 0, grid.W - 1)]
    val = np.where(np.isnan(rs), 0, np.clip(np.round(np.clip(rs / 60.0, -200, 200)) + 128, 1, 253))
    val = np.where(rs == np.inf, 253, np.where(rs == -np.inf, 1, val))
    out[inside & land] = val[inside & land].astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "L").save(out_dir / f"margin_{vk}.png", optimize=True)
    vals, counts = np.unique(out, return_counts=True)
    n_road = int(((out >= 1) & (out <= 253)).sum()); n_neg = int(((out >= 1) & (out < 128)).sum())
    meta = dict(map="gobo", layer=f"evac_margin_{vk}", label=f"最遅出発余裕時間 (歩行 {float(vk[1:])/10:.1f} m/s, 目的地=指定緊急避難場所)",
                cellBlocks=cb, w=w, h=h, image=f"evac/margin_{vk}.png",
                encoding="L: 値 = clip(round(margin_min)+128, 1, 253) (margin_min = 最遅出発時刻 − 5 分; 253 = 浸水しない道路 (∞), 1 = 到達不能/-127 分以下), 0 = 道路 50 m 圏外の陸, 255 = 範囲外・海",
                definition="L(v)=min(T(v), max_u[min(L(u),T(u)) − d(v,u)/v_walk]) を指定緊急避難場所(津波) から逆向き Dijkstra、T=県 R8 30 cm 到達秒。道路ノード→8 block セルは 5 m 格子 (最近傍 50 m) 経由の最近傍。",
                v_walk=float(vk[1:]) / 10, start_delay_min=5, targets="国土地理院 指定緊急避難場所 (津波=1, 55 か所, 最寄り道路へ ≤300 m で接続)",
                stats=dict(road_cells=n_road, margin_negative_cells=n_neg, cells={int(a): int(b) for a, b in zip(vals, counts)}),
                source=SOURCE, license="県 HP 公開情報利用規約 (CC BY 4.0 互換) / GSI / ODbL")
    with open(out_dir / f"margin_{vk}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[export] {out_dir/f'margin_{vk}.png'}  road cells={n_road} negative={n_neg}")


def main():
    p = PROFILES["gobo"]
    inp = load_inputs()
    with open(OUT / "summary_pso_vs_ccpso2.json", encoding="utf-8") as f:
        summary = json.load(f)
    export_routes(p, inp, summary, WALK / "overlay_out/gobo/routes.json")
    for vk in ("v10", "v05"):
        export_margin(p, inp, vk, WALK / "overlay_out/gobo/evac")


if __name__ == "__main__":
    main()
