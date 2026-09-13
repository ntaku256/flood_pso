"""
evac/evac_ccpso2.py — 講演原稿 3 章の EVAC-PSO (中継点 + スプライン経路) を御坊実地形 (県 R8 津波) へ移植し、
標準 PSO (pyswarms GlobalBestPSO) / CCPSO2 (src/ccpso2.py) / Dijkstra 参照 を比較する。

問題設定
  粒子 x ∈ R^40 = K=20 中継点の (x,y) [m, 局所座標 (原点=始点, x 東 y 北)]。
  経路 = 始点 → 中継点 1..20 → 終点 を scipy.interpolate.splprep(k=3, s=0) で通し、約 DS=2 m 刻みでサンプル。
  通過時刻 t_pass(s) = 300 s (避難開始 5 分) + s / v_walk (1.0 m/s)。
  コスト [m] = 経路長 L
             + λ1 · Σ_{浸水: t_pass(s) ≥ T(s)} Δs                 (通過時に 30 cm 浸水しているサンプルの経路長)
             + λ2 · Σ_{道路外: EDT(s) > 3 m} (Δs + (EDT(s) − 3))  (道路外の経路長 + 道路からの超過距離。後者で道路へ寄せる勾配を与える)
             + λ3 · Σ_{建物内 or 海・河川 (DEM NaN) or 格子外} Δs
             + λ4 · max(0, t_pass(終点) − T(終点))  [s]
  λ1 = λ3 = 50 (浸水/建物 1 m につき 50 m 分の遠回りに相当), λ2 = 2, λ4 = 100 (遅刻 1 s = 100 m)。
  探索範囲 = 始点・終点の外接矩形を PAD=300 m 広げた矩形。
  予算 5000 評価 × 5 シード。PSO: 30 粒子 × 167 反復, c1=c2=1.5, w=0.7, bounds, bh="nearest", 速度クランプ ±20% 幅。
  CCPSO2: N=20, s=2 (K_g=20: 1 グループ = 1 中継点 (x,y)。ccpso2_sweep.py で s∈{1,2,4,8,40}/N/p_cauchy を比較し最良), p_cauchy=0.5, max_evals=5000
  (1 サイクル = N·K_g = 400 評価。サイクル途中で止まらない)。
  予算の公平化: 目的関数側で「最初の BUDGET 評価」までの最良値 (と粒子) を記録し、両手法ともその値を結果とする
  (PSO は 30×167=5010 評価、CCPSO2 は 20+13×400=5220 評価まで走るが、5001 評価目以降の改善は採用しない)。
  Dijkstra 参照 = 道路網の時間依存 (浸水セル通行不可) 最早到着経路を同じコスト関数で再評価 (折れ線 / 20 中継点スプライン化 の 2 通り)。
  ※「最適解」ではない (連続空間の最適値は未知)。

出力 results/evac/
  table_pso_vs_ccpso2.csv, summary_pso_vs_ccpso2.json, convergence.png, routes_overview.png, routes_pairs.png,
  routes_best.json (各ペア × 手法 の最良経路 lat/lon 列; export_routes.py が読む), evac_runs.npz (収束履歴)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import OUT, START_DELAY_S, load_inputs, setup_japanese_font, hillshade  # noqa: E402
from roads import RoadGraph, load_ways, fill_nan_nearest  # noqa: E402
from ccpso2 import CCPSO2  # noqa: E402

K = 20
D = 2 * K
DS = 2.0
PAD_M = 300.0
V_WALK = 1.0
LAMBDA = dict(flood=50.0, offroad=2.0, building=50.0, late=100.0)
OFFROAD_TOL_M = 3.0
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
BIG = 1e7
SUF = "" if BUDGET == 5000 else f"_b{BUDGET}"   # 既定予算以外は出力名に接尾辞 (補足実験)
CC_N, CC_S = 20, 2   # CCPSO2 粒子数 / グループ次元数 (ccpso2_sweep.py 参照)

PAIRS = [
    dict(id="naya", label="名屋 住宅地 → 名屋地区津波避難タワー",
         start=dict(name="名屋 住宅地 (道路上)", lat=33.878834, lon=135.155636),
         goal=dict(name="名屋地区津波避難タワー", lat=33.882881, lon=135.157136)),
    dict(id="sono", label="薗 住宅地 → 薗地区津波避難タワー",
         start=dict(name="薗 住宅地 (道路上)", lat=33.890765, lon=135.157293),
         goal=dict(name="薗地区津波避難タワー", lat=33.8871581, lon=135.161616)),
    dict(id="center", label="市街中心 → 御坊小学校",
         start=dict(name="御坊 市街中心 (道路上)", lat=33.889680, lon=135.159598),
         # 施設座標 (33.8919485, 135.154961 = 敷地中心) は道路から 57 m 離れ、直線接続部が校舎 (OSM building) を通って
         # Dijkstra 参照にも建物ペナルティが乗るため、敷地に最も近い道路ノード (校門前) を終点にする。
         goal=dict(name="御坊小学校 (校門前道路)", lat=33.892415, lon=135.154695)),
]
COLORS = dict(dijkstra="#ffffff", pso="#9e9e9e", ccpso2="#ffd54f")          # 陰影 DEM 上 (routes_overview / tizucra-walk)
PLOT_COLORS = dict(dijkstra="#7b1fa2", pso="#616161", ccpso2="#f9a825")     # 白背景 (routes_pairs; 道路が白なので濃色)


class Terrain:
    """5 m 格子ラスタ群 (局所メートル座標でのルックアップ)。"""

    def __init__(self, inp):
        self.grid = g = inp["grid"]
        dem = inp["dem"]
        self.land = ~np.isnan(dem)
        self.dem = fill_nan_nearest(dem.astype(np.float64))
        T = inp["T"].astype(np.float64)
        T[~self.land] = np.nan
        self.T = fill_nan_nearest(T)
        self.T[np.isnan(self.T)] = np.inf
        self.road = inp["road"].astype(bool)
        self.building = inp["building"].astype(bool)
        self.road_edt = distance_transform_edt(~self.road, sampling=(g.dy_m, g.dx_m))
        self.hmax = inp["hmax"]

    def lookup(self, xs, ys):
        r, c = self.grid.xy_to_rc(xs, ys)
        ri = np.round(r).astype(int); ci = np.round(c).astype(int)
        oob = (ri < 0) | (ri >= self.grid.H) | (ci < 0) | (ci >= self.grid.W)
        ri = np.clip(ri, 0, self.grid.H - 1); ci = np.clip(ci, 0, self.grid.W - 1)
        return ri, ci, oob


class RouteProblem:
    def __init__(self, terr: Terrain, start_xy, goal_xy, v_walk=V_WALK, lam=LAMBDA, budget=BUDGET):
        self.t = terr
        self.s = np.asarray(start_xy, float); self.g = np.asarray(goal_xy, float)
        self.v = v_walk; self.lam = lam; self.budget = budget
        lo = np.minimum(self.s, self.g) - PAD_M; hi = np.maximum(self.s, self.g) + PAD_M
        self.lb = np.tile(lo, K); self.ub = np.tile(hi, K)
        ri, ci, _ = terr.lookup(self.g[0:1], self.g[1:2])
        self.T_goal = float(terr.T[ri[0], ci[0]])
        self.n_eval = 0
        self.log: list = []
        self.best = np.inf
        self.best_x = None

    # ── 経路生成 ───────────────────────────────
    def spline_points(self, x):
        wp = np.asarray(x, float).reshape(K, 2)
        pts = np.vstack([self.s, wp, self.g])
        d = np.hypot(*np.diff(pts, axis=0).T)
        if (d < 1e-3).any():   # 重複点は splprep が拒否する → 僅かにずらす
            pts = pts + np.random.RandomState(0).normal(0, 1e-3, pts.shape) * (np.r_[0, d < 1e-3, 0] > 0)[:, None]
        tck, _ = splprep([pts[:, 0], pts[:, 1]], k=3, s=0)
        u = np.linspace(0, 1, 200)
        xs, ys = splev(u, tck)
        Lest = np.hypot(np.diff(xs), np.diff(ys)).sum()
        M = int(min(max(Lest / DS, 20), 20000))
        u = np.linspace(0, 1, M + 1)
        xs, ys = splev(u, tck)
        return np.asarray(xs), np.asarray(ys)

    @staticmethod
    def polyline_points(pts, ds=DS):
        pts = np.asarray(pts, float)
        xs, ys = [pts[0, 0]], [pts[0, 1]]
        for a, b in zip(pts[:-1], pts[1:]):
            L = np.hypot(*(b - a))
            n = max(1, int(np.ceil(L / ds)))
            for k in range(1, n + 1):
                p = a + (b - a) * k / n
                xs.append(p[0]); ys.append(p[1])
        return np.array(xs), np.array(ys)

    # ── コスト ────────────────────────────────
    def route_cost(self, xs, ys, detail=False):
        ds = np.hypot(np.diff(xs), np.diff(ys))
        s_cum = np.r_[0.0, np.cumsum(ds)]
        L = float(s_cum[-1])
        t_pass = START_DELAY_S + s_cum / self.v
        ri, ci, oob = self.t.lookup(xs, ys)
        seg_ds = np.r_[ds, 0.0]                      # サンプル i に続く区間長 (終点は 0)
        flooded = t_pass >= self.t.T[ri, ci]
        edt = self.t.road_edt[ri, ci]
        off = edt > OFFROAD_TOL_M
        bld = self.t.building[ri, ci] | oob | ~self.t.land[ri, ci]   # 建物 / 格子外 / 海・河川 (DEM NaN) は同じ λ3
        p_flood = self.lam["flood"] * float(seg_ds[flooded].sum())
        p_off = self.lam["offroad"] * float((seg_ds[off] + (edt[off] - OFFROAD_TOL_M)).sum())
        p_bld = self.lam["building"] * float(seg_ds[bld].sum())
        late = max(0.0, float(t_pass[-1]) - self.T_goal) if np.isfinite(self.T_goal) else 0.0
        p_late = self.lam["late"] * late
        cost = L + p_flood + p_off + p_bld + p_late
        if detail:
            return dict(cost=cost, length_m=L, time_min=float(t_pass[-1] / 60), p_flood=p_flood, p_offroad=p_off,
                        p_building=p_bld, p_late=p_late, n_flood=int(flooded.sum()), n_off=int(off.sum()), n_bld=int(bld.sum()))
        return cost

    def __call__(self, x):
        try:
            xs, ys = self.spline_points(x)
            c = self.route_cost(xs, ys)
        except Exception:
            c = BIG
        self.n_eval += 1
        if self.n_eval <= self.budget:      # 予算内の評価だけを結果に採用 (PSO / CCPSO2 の打ち切り単位の違いを吸収)
            if c < self.best:
                self.best = c; self.best_x = np.array(x, float).copy()
            self.log.append((self.n_eval, self.best))
        return c

    def batch(self, X):
        return np.array([self(x) for x in X])

    def reset_log(self):
        self.n_eval = 0; self.log = []; self.best = np.inf; self.best_x = None


# ─────────────────────────────────────────────────────────────
def run_pso(prob, seed, budget):
    from pyswarms.single import GlobalBestPSO
    n_p = 30
    iters = max(1, -(-budget // n_p))   # 予算以上回し、採用は目的関数側で budget 評価までに制限
    np.random.seed(seed)
    vmax = 0.2 * float((prob.ub - prob.lb).max())
    opt = GlobalBestPSO(n_particles=n_p, dimensions=D, options={"c1": 1.5, "c2": 1.5, "w": 0.7},
                        bounds=(prob.lb, prob.ub), bh_strategy="nearest", velocity_clamp=(-vmax, vmax), ftol=-np.inf)
    prob.reset_log()
    t0 = time.time()
    opt.optimize(prob.batch, iters=iters, verbose=False)
    return dict(cost=float(prob.best), x=np.asarray(prob.best_x), log=list(prob.log), evals=min(prob.n_eval, prob.budget),
                evals_run=prob.n_eval, elapsed=time.time() - t0)


def run_ccpso2(prob, seed, budget):
    prob.reset_log()
    t0 = time.time()
    cc = CCPSO2(prob, dim=D, n_particles=CC_N, group_size=CC_S, bounds=(prob.lb, prob.ub), p_cauchy=0.5, seed=seed)
    cc.run(max_evals=budget)
    return dict(cost=float(prob.best), x=np.asarray(prob.best_x), log=list(prob.log), evals=min(prob.n_eval, prob.budget),
                evals_run=prob.n_eval, elapsed=time.time() - t0)


def dijkstra_route(G, terr, pair, prob):
    s_id, sd = G.snap(pair["start"]["lat"], pair["start"]["lon"], 100)
    g_id = pair["_goal_node"]
    goal, t_arr, path = G.earliest_arrival(s_id, [g_id], START_DELAY_S, prob.v, respect_flood=True)
    flooded_ok = True
    if goal is None:
        flooded_ok = False
        goal, t_arr, path = G.earliest_arrival(s_id, [g_id], START_DELAY_S, prob.v, respect_flood=False)
    pts = np.vstack([prob.s, G.xy[path], prob.g]) if path else np.vstack([prob.s, prob.g])
    xs, ys = prob.polyline_points(pts)
    det = prob.route_cost(xs, ys, detail=True)
    # 20 中継点スプライン化 (経路長で等分した内点)
    ds = np.hypot(np.diff(xs), np.diff(ys)); sc = np.r_[0, np.cumsum(ds)]
    tgt = np.linspace(0, sc[-1], K + 2)[1:-1]
    wp = np.c_[np.interp(tgt, sc, xs), np.interp(tgt, sc, ys)]
    xs2, ys2 = prob.spline_points(wp.ravel())
    det2 = prob.route_cost(xs2, ys2, detail=True)
    return dict(poly=dict(xs=xs, ys=ys, **det, flood_respected=flooded_ok, snap_m=sd, graph_arrival_min=t_arr / 60),
                spline=dict(xs=xs2, ys=ys2, x=wp.ravel(), **det2))


# ─────────────────────────────────────────────────────────────
def main():
    inp = load_inputs()
    grid = inp["grid"]
    terr = Terrain(inp)
    G = RoadGraph(load_ways(), grid, inp["T"], inp["dem"], extra_points=[(p["goal"]["lat"], p["goal"]["lon"]) for p in PAIRS])
    for p, gid in zip(PAIRS, G.extra_ids):
        p["_goal_node"] = gid
    print(f"[evac] budget={BUDGET} seeds={SEEDS} K={K} D={D} λ={LAMBDA}")

    rows, summary, best_routes, histories = [], {}, {}, {}
    for pair in PAIRS:
        sx, sy = grid.latlon_to_xy(pair["start"]["lat"], pair["start"]["lon"])
        gx, gy = grid.latlon_to_xy(pair["goal"]["lat"], pair["goal"]["lon"])
        prob = RouteProblem(terr, (sx, sy), (gx, gy))
        straight = float(np.hypot(gx - sx, gy - sy))
        print(f"\n=== {pair['id']}: {pair['label']}  直線 {straight:.0f} m  T_goal={prob.T_goal/60:.0f} min  bounds {prob.ub[:2]-prob.lb[:2]} m")
        # 評価速度
        t0 = time.time(); [prob(prob.lb + (prob.ub - prob.lb) * np.random.RandomState(i).rand(D)) for i in range(200)]
        print(f"  eval speed {(time.time()-t0)/200*1e3:.2f} ms/eval")

        dj = dijkstra_route(G, terr, pair, prob)
        print(f"  Dijkstra poly cost={dj['poly']['cost']:.1f} L={dj['poly']['length_m']:.0f} m arrival {dj['poly']['time_min']:.1f} min "
              f"(flood {dj['poly']['n_flood']}, off {dj['poly']['n_off']}, bld {dj['poly']['n_bld']}) | spline20 cost={dj['spline']['cost']:.1f}")
        rows.append(dict(pair=pair["id"], method="dijkstra", seed="", **{k: v for k, v in dj["poly"].items() if k not in ("xs", "ys")}))
        rows.append(dict(pair=pair["id"], method="dijkstra_spline20", seed="", **{k: v for k, v in dj["spline"].items() if k not in ("xs", "ys", "x")}))

        res = {"pso": [], "ccpso2": []}
        for seed in SEEDS:
            for m, fn in (("pso", run_pso), ("ccpso2", run_ccpso2)):
                r = fn(prob, seed, BUDGET)
                xs, ys = prob.spline_points(r["x"])
                det = prob.route_cost(xs, ys, detail=True)
                r.update(det); r["xs"], r["ys"] = xs, ys
                res[m].append(r)
                histories[(pair["id"], m, seed)] = np.array(r["log"])
                rows.append(dict(pair=pair["id"], method=m, seed=seed, evals=r["evals"], evals_run=r["evals_run"], elapsed_s=round(r["elapsed"], 1),
                                 **{k: det[k] for k in det}))
                print(f"  seed {seed} {m:7s} cost={r['cost']:9.1f} L={r['length_m']:6.0f} flood={r['n_flood']:3d} off={r['n_off']:3d} bld={r['n_bld']:3d} late={r['p_late']:.0f} evals={r['evals']} {r['elapsed']:.0f}s")
        pc = np.array([r["cost"] for r in res["pso"]]); cc = np.array([r["cost"] for r in res["ccpso2"]])
        wins = int((cc < pc).sum())
        summary[pair["id"]] = dict(label=pair["label"], straight_m=straight, T_goal_min=prob.T_goal / 60,
                                   pso_median=float(np.median(pc)), pso_min=float(pc.min()), pso_max=float(pc.max()),
                                   ccpso2_median=float(np.median(cc)), ccpso2_min=float(cc.min()), ccpso2_max=float(cc.max()),
                                   ccpso2_wins=wins, pso_wins=len(SEEDS) - wins,
                                   dijkstra_cost=dj["poly"]["cost"], dijkstra_spline20_cost=dj["spline"]["cost"],
                                   dijkstra_length_m=dj["poly"]["length_m"], dijkstra_flood_respected=dj["poly"]["flood_respected"])
        print(f"  median PSO {np.median(pc):.1f} / CCPSO2 {np.median(cc):.1f} / Dijkstra {dj['poly']['cost']:.1f}  CCPSO2 wins {wins}/{len(SEEDS)}")
        bp, bc = res["pso"][int(np.argmin(pc))], res["ccpso2"][int(np.argmin(cc))]
        best_routes[pair["id"]] = dict(
            pair=pair, straight_m=straight,
            dijkstra=dict(xs=dj["poly"]["xs"], ys=dj["poly"]["ys"], cost=dj["poly"]["cost"], length_m=dj["poly"]["length_m"], time_min=dj["poly"]["time_min"]),
            pso=dict(xs=bp["xs"], ys=bp["ys"], x=bp["x"], cost=bp["cost"], length_m=bp["length_m"], time_min=bp["time_min"], seed=SEEDS[int(np.argmin(pc))]),
            ccpso2=dict(xs=bc["xs"], ys=bc["ys"], x=bc["x"], cost=bc["cost"], length_m=bc["length_m"], time_min=bc["time_min"], seed=SEEDS[int(np.argmin(cc))]),
            pso_all=[dict(xs=r["xs"], ys=r["ys"], cost=r["cost"]) for r in res["pso"]],
            ccpso2_all=[dict(xs=r["xs"], ys=r["ys"], cost=r["cost"]) for r in res["ccpso2"]],
            prob=prob)

    # ── 出力 ──
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["pair", "method", "seed", "cost", "length_m", "time_min", "p_flood", "p_offroad", "p_building", "p_late",
            "n_flood", "n_off", "n_bld", "evals", "evals_run", "elapsed_s", "flood_respected", "snap_m", "graph_arrival_min"]
    with open(OUT / f"table_pso_vs_ccpso2{SUF}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
        for pid, s in summary.items():
            w.writerow(dict(pair=pid, method="MEDIAN_pso", cost=round(s["pso_median"], 2)))
            w.writerow(dict(pair=pid, method="MEDIAN_ccpso2", cost=round(s["ccpso2_median"], 2)))
            w.writerow(dict(pair=pid, method=f"WINS_ccpso2_{s['ccpso2_wins']}_of_{len(SEEDS)}"))
    with open(OUT / f"summary_pso_vs_ccpso2{SUF}.json", "w", encoding="utf-8") as f:
        json.dump(dict(budget=BUDGET, seeds=SEEDS, K=K, D=D, lam=LAMBDA, v_walk=V_WALK, start_delay_s=START_DELAY_S,
                       ds_m=DS, pad_m=PAD_M, offroad_tol_m=OFFROAD_TOL_M, pairs=summary), f, ensure_ascii=False, indent=1)
    # 経路 (lat/lon) — export_routes.py 用
    rb = {}
    for pid, b in best_routes.items():
        ent = dict(pair={k: v for k, v in b["pair"].items() if not k.startswith("_")}, straight_m=b["straight_m"])
        for m in ("dijkstra", "pso", "ccpso2"):
            la, lo = grid.xy_to_latlon(b[m]["xs"], b[m]["ys"])
            ent[m] = dict(lat=la.tolist(), lon=lo.tolist(), cost=b[m]["cost"], length_m=b[m]["length_m"], time_min=b[m]["time_min"],
                          seed=b[m].get("seed"))
        rb[pid] = ent
    with open(OUT / f"routes_best{SUF}.json", "w", encoding="utf-8") as f:
        json.dump(rb, f, ensure_ascii=False)
    np.savez_compressed(OUT / f"evac_runs{SUF}.npz", **{f"{p}__{m}__{s}": h for (p, m, s), h in histories.items()})
    plot_convergence(histories, summary, OUT / f"convergence{SUF}.png")
    plot_routes_overview(inp, terr, best_routes, OUT / f"routes_overview{SUF}.png")
    plot_routes_pairs(terr, best_routes, OUT / f"routes_pairs{SUF}.png")
    print("\n[evac] wrote table_pso_vs_ccpso2.csv, summary_pso_vs_ccpso2.json, convergence.png, routes_overview.png, routes_pairs.png, routes_best.json")


# ─────────────────────────────────────────────────────────────
def _mpl():
    import matplotlib
    matplotlib.use("Agg"); setup_japanese_font()
    import matplotlib.pyplot as plt
    return plt


def plot_convergence(hist, summary, path):
    plt = _mpl()
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5.2 * len(PAIRS), 4.2), dpi=110)
    for ax, pair in zip(np.atleast_1d(axes), PAIRS):
        pid = pair["id"]
        for m, col, lab in (("pso", "#7f7f7f", "標準 PSO"), ("ccpso2", "#e6a817", "CCPSO2")):
            curves = []
            for s in SEEDS:
                h = hist[(pid, m, s)]
                ax.plot(h[:, 0], h[:, 1], color=col, alpha=0.35, lw=0.8)
                curves.append(np.interp(np.arange(1, BUDGET + 1), h[:, 0], h[:, 1]))
            ax.plot(np.arange(1, BUDGET + 1), np.median(curves, axis=0), color=col, lw=2.2, label=f"{lab} (中央値, 5 seed)")
        s = summary[pid]
        ax.axhline(s["dijkstra_cost"], color="k", ls="--", lw=1.2, label=f"Dijkstra 参照 (折れ線) {s['dijkstra_cost']:.0f}")
        ax.axhline(s["dijkstra_spline20_cost"], color="k", ls=":", lw=1.0, label=f"Dijkstra→20点スプライン {s['dijkstra_spline20_cost']:.0f}")
        ax.set_yscale("log"); ax.set_xlabel("評価回数"); ax.set_ylabel("経路コスト [m]")
        ax.set_title(pair["label"], fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"御坊 EVAC-PSO: 20 中継点 (40 次元) 経路コストの収束  (予算 {BUDGET} 評価 (両手法とも同数で打ち切り), v=1.0 m/s, 県 R8 津波)", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_routes_overview(inp, terr, best_routes, path):
    plt = _mpl()
    from matplotlib.lines import Line2D
    grid = terr.grid
    hs = hillshade(inp["dem"], grid.dy_m, grid.dx_m)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=110)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_where(terr.land, ~terr.land), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6, interpolation="nearest")
    im = ax.imshow(np.ma.masked_where(~(terr.hmax >= 0.3), terr.hmax), cmap="jet", vmin=0, vmax=10, alpha=0.4, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~terr.road, terr.road), cmap="Oranges", vmin=0, vmax=2.5, alpha=0.5, interpolation="nearest")
    fac = [f for f in inp["fac"] if f["inside"]]
    ax.scatter([f["col"] for f in fac], [f["row"] for f in fac], s=22, c="lime", edgecolors="k", zorder=5, label="指定緊急避難場所(津波)")
    rmin, rmax, cmin, cmax = 1e9, -1e9, 1e9, -1e9
    for pid, b in best_routes.items():
        for m, lw in (("dijkstra", 3.2), ("pso", 2.4), ("ccpso2", 2.4)):
            r, c = grid.xy_to_rc(b[m]["xs"], b[m]["ys"])
            ax.plot(c, r, color="k", lw=lw + 1.6, alpha=0.6, zorder=6)
            ax.plot(c, r, color=COLORS[m], lw=lw, zorder=7)
            rmin, rmax, cmin, cmax = min(rmin, r.min()), max(rmax, r.max()), min(cmin, c.min()), max(cmax, c.max())
        for key, mk, col in (("start", "s", "red"), ("goal", "*", "lime")):
            r, c = grid.latlon_to_rc(b["pair"][key]["lat"], b["pair"][key]["lon"])
            ax.plot(c, r, mk, color=col, ms=13 if mk == "*" else 9, mec="k", zorder=9)
            ax.annotate(b["pair"][key]["name"], (c, r), fontsize=8, color="w", xytext=(5, 5), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="k", alpha=0.5, ec="none"))
    ax.set_xlim(cmin - 40, cmax + 40); ax.set_ylim(rmax + 40, rmin - 40)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.03, label="R8 最大浸水深 hmax [m]")
    ax.legend(handles=[Line2D([], [], color=COLORS["dijkstra"], lw=3, label="Dijkstra 参照 (道路網・時間依存)"),
                       Line2D([], [], color=COLORS["pso"], lw=3, label="標準 PSO (最良 seed)"),
                       Line2D([], [], color=COLORS["ccpso2"], lw=3, label="CCPSO2 (最良 seed)"),
                       Line2D([], [], marker="s", color="red", ls="", mec="k", label="始点"),
                       Line2D([], [], marker="*", color="lime", ls="", mec="k", ms=12, label="終点 (避難場所)")], loc="lower left", fontsize=9,
              facecolor="#dddddd")
    ax.set_title("御坊 避難経路: 陰影 DEM + 県 R8 hmax + OSM 道路 + 3 手法の最良経路 (5 m 格子)", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_routes_pairs(terr, best_routes, path):
    """原稿 図2(b) と同構図: 障害物 (建物=黒, 道路外=灰), 中継点=赤点, 始点=赤■, 終点=緑■。"""
    plt = _mpl()
    grid = terr.grid
    fig, axes = plt.subplots(1, len(best_routes), figsize=(6.2 * len(best_routes), 6.4), dpi=110)
    for ax, (pid, b) in zip(np.atleast_1d(axes), best_routes.items()):
        prob = b["prob"]
        lo, hi = prob.lb[:2], prob.ub[:2]
        r0, c0 = grid.xy_to_rc(lo[0], hi[1]); r1, c1 = grid.xy_to_rc(hi[0], lo[1])
        r0, r1, c0, c1 = int(max(0, r0)), int(min(grid.H, r1)), int(max(0, c0)), int(min(grid.W, c1))
        sub = np.s_[r0:r1, c0:c1]
        ext = [lo[0], hi[0], lo[1], hi[1]]   # x 東 [m], y 北 [m]
        bg = np.full(terr.road[sub].shape + (3,), 0.82)
        bg[terr.road_edt[sub] <= OFFROAD_TOL_M] = 1.0
        Tm = terr.T[sub]
        bg[(Tm < np.inf) & (terr.road_edt[sub] <= OFFROAD_TOL_M)] = [0.85, 0.93, 1.0]
        bg[terr.building[sub]] = 0.0
        bg[~terr.land[sub]] = [0.55, 0.7, 0.9]
        ax.imshow(bg, extent=ext, origin="upper", interpolation="nearest")
        cs = ax.contour(np.linspace(lo[0], hi[0], Tm.shape[1]), np.linspace(hi[1], lo[1], Tm.shape[0]), np.where(np.isfinite(Tm), Tm / 60, 200),
                        levels=[20, 25, 30, 35, 40], colors="tab:blue", linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fmt="%d分", fontsize=7)
        for m, lw, z in (("dijkstra", 2.6, 3), ("pso", 2.0, 4), ("ccpso2", 2.0, 5)):
            ax.plot(b[m]["xs"], b[m]["ys"], color="w", lw=lw + 1.2, alpha=0.6, zorder=z)
            ax.plot(b[m]["xs"], b[m]["ys"], color=PLOT_COLORS[m], lw=lw, zorder=z + 0.1,
                    label=f"{dict(dijkstra='Dijkstra 参照', pso='標準 PSO', ccpso2='CCPSO2')[m]}  cost {b[m]['cost']:.0f}")
        wp = np.asarray(b["ccpso2"]["x"]).reshape(K, 2)
        ax.plot(wp[:, 0], wp[:, 1], "o", color="red", ms=3.5, zorder=8, label="CCPSO2 中継点 (20)")
        ax.plot(prob.s[0], prob.s[1], "s", color="red", ms=10, mec="k", zorder=9)
        ax.plot(prob.g[0], prob.g[1], "s", color="green", ms=10, mec="k", zorder=9)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xlabel("x [m] (東)"); ax.set_ylabel("y [m] (北)")
        ax.set_title(b["pair"]["label"], fontsize=10)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("白=道路 (EDT≤3 m), 淡青=浸水する道路, 灰=道路外, 黒=建物 (OSM), 青面=海・河川 (DEM NaN), 青線=30 cm 到達時刻 [分]", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
