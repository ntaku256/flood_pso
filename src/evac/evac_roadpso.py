"""
evac/evac_roadpso.py — 道路網に制約した「経由点 (via-point) 定式化」で PSO / CCPSO2 / Dijkstra 参照 を比較する。

背景
  evac_ccpso2.py (連続スプライン定式化) では、御坊の密な市街地で中継点スプラインが建物・道路外を避けきれず、
  λ3 (建物) と λ2 (道路外) のペナルティが支配的になって PSO / CCPSO2 とも道路 Dijkstra 参照に 1 桁負けた。
  本スクリプトは探索空間そのものを道路網へ落とし、「連続最適化が道路制約下なら参照に届くか」を検証する。

定式化 (via-point)
  粒子 x ∈ R^{2M} = M 個の経由点 (x, y) [m, 局所座標 (x 東 / y 北; 原点 = 5 m 格子の (row0, col0))]。M ∈ {5, 20} → D = 10 / 40。
  復号:
    1. 各経由点を最寄り道路ノードへ射影 (cKDTree; 対象 = 始点の連結成分に属する窓内ノードのみ → 必ず到達可能)
    2. 同じノードへの射影が連続したら重複を潰す (始点・終点との重複も潰す)
    3. 始点 → v1 → … → vM' → 終点 を、道路グラフ上の **静的 (長さ最小) 最短経路** で順に接続
    4. 得られた折れ線を DS = 2 m 間隔に再サンプル
  コスト [m] (道路上を歩くので道路外項 λ2 / 建物項 λ3 は不要):
    cost = L                                                    経路長 [m]
         + λ1 · Σ_{t_pass(s) ≥ T(s)} Δs                         通過時刻に 30 cm 浸水しているサンプルの経路長
         + λ4 · max(0, t_pass(終点) − T(終点))                  終点への遅刻 [s]
    t_pass(s) = 300 s (避難開始 = 地震 5 分後) + s / v_walk, v_walk = 1.0 m/s。
    λ1 = 50, λ4 = 100 — evac_ccpso2.py の LAMBDA["flood"] / LAMBDA["late"] をそのまま使う (import して同値を保証)。
    T = 県 R8 津波の 30 cm 到達秒 (5 m 格子ラスタ; evac_ccpso2.Terrain と同じ最近傍ルックアップ)。
  探索範囲 = 始点・終点の外接矩形を PAD_M = 300 m 広げた矩形 (evac_ccpso2.py と同じ)。
  道路グラフの窓 = 同 bbox + PAD_GRAPH_M = 500 m (窓端で経路が不自然に切れないよう探索範囲より広く取る)。

参照解 (どちらも上と同一のコスト関数で評価)
  (a) 静的 Dijkstra   : 窓内グラフの長さ最小経路 (浸水を無視) — via-point 復号の下限 (三角不等式より L ≥ この長さ)
  (b) 時間依存 Dijkstra: roads.RoadGraph.earliest_arrival (出発時 t < T(v) かつ 到着時 t + d/v < T(u) のみ通行可)
                         = dijkstra_ref.py の最遅出発ロジックと同じ通行条件の前向き版。PSO/CCPSO2 が届くべき目標。
  さらに (b) を M 個の経由点へ「符号化」し直した解も評価し、この符号化で参照解が表現可能かを示す。

最適化 (evac_ccpso2.py と同じ予算・seed 処理)
  予算 5000 評価 × 5 seed。PSO: pyswarms GlobalBestPSO 30 粒子 × 167 反復, c1=c2=1.5, w=0.7, bounds 付き,
  bh_strategy="nearest", 速度クランプ ±20% 幅。CCPSO2: src/ccpso2.py, N=20, p_cauchy=0.5, max_evals=5000。
  予算の公平化 (evac_ccpso2.py と同じ): 目的関数側で「最初の BUDGET 評価」までの最良値 (と粒子) だけを結果に採用し、
  PSO が 5010 評価、CCPSO2 が打ち切り単位の都合で 5000 超まで走っても 5001 評価目以降の改善は採らない。
  CCPSO2 のグループサイズ s は {1, 2, 5} を 3 ペア × seed {0,1} で軽くスイープし、参照コストで正規化した
  中央値が最小の s を M ごとに採用する (結果は summary_roadpso.json の ccpso2_group_sweep / roadpso_sweep.csv)。

高速化
  ペアごとに窓内サブグラフ (≤ 2.6 k ノード) の全点間最短距離・先行ノードを scipy.sparse.csgraph.dijkstra で
  1 回だけ前計算 (≈ 1 s, ≤ 76 MB)。1 評価は「KD 木射影 → 先行行列からの経路復元 (区間ごとにメモ化) →
  ラスタ参照」だけになり、実測 1 評価 ≲ 1 ms。射影ノード列が同一の解は復号結果をキャッシュする
  (評価回数 n_eval はキャッシュヒットでも増やす = 予算の数え方は evac_ccpso2.py と同じ)。

出力 results/evac/roadpso/
  table_roadpso.csv        ペア × M × 手法 × seed のコスト内訳
  summary_roadpso.json     中央値・勝敗・参照値・グループサイズスイープ・1 評価時間
  roadpso_sweep.csv        CCPSO2 グループサイズスイープの生値
  convergence_roadpso.png  収束曲線 (M × ペア)
  routes_roadpso.png       陰影 DEM + 30 cm 到達時刻 + 避難場所 + 各手法の道路上経路
  routes_roadpso_best.json 各ペアの参照 / PSO / CCPSO2 最良経路 (lat/lon 列)
  さらに <tizucra-walk>/overlay_out/gobo/routes_roadpso.json (export_routes.py と同じ overlay スキーマ)

usage:  .venv/bin/python src/evac/evac_roadpso.py [budget] [seeds] [tizucra-walk のパス]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import OUT, START_DELAY_S, load_inputs, setup_japanese_font, hillshade  # noqa: E402
from roads import RoadGraph, load_ways, fill_nan_nearest  # noqa: E402
from ccpso2 import CCPSO2  # noqa: E402

# evac_ccpso2 は import 時に sys.argv を予算として読むため、退避してから読み込む (同ファイルは変更しない)
_ARGV, sys.argv = sys.argv, sys.argv[:1]
import evac_ccpso2 as E  # noqa: E402
sys.argv = _ARGV

M_LIST = [5, 20]                      # 経由点数 (D = 2M = 10 / 40)
DS = E.DS                             # 2.0 m 再サンプル間隔
PAD_M = E.PAD_M                       # 300 m (経由点の探索範囲)
PAD_GRAPH_M = 500.0                   # 道路グラフの窓 (探索範囲より広く)
V_WALK = E.V_WALK                     # 1.0 m/s
LAM = dict(flood=E.LAMBDA["flood"], late=E.LAMBDA["late"])   # λ1 = 50, λ4 = 100
PAIRS = E.PAIRS
COLORS = dict(dijkstra="#ffffff", pso="#9e9e9e", ccpso2="#ffd54f")
REF_STYLE = dict(static=dict(ls=":", lw=2.0), timedep=dict(ls="-", lw=2.8))
CC_N = 20                             # CCPSO2 粒子数
SWEEP_S = [1, 2, 5]                   # CCPSO2 グループサイズ候補
SWEEP_SEEDS = [0, 1]
SNAP_MAX_M = 100.0

BUDGET = int(_ARGV[1]) if len(_ARGV) > 1 else 5000
SEEDS = [int(s) for s in _ARGV[2].split(",")] if len(_ARGV) > 2 else [0, 1, 2, 3, 4]
WALK = Path(_ARGV[3]) if len(_ARGV) > 3 else HERE.parents[2] / "tizucra-walk"
OUTD = OUT / "roadpso"
BIG = 1e7


# ── コスト ──────────────────────────────────────────────────────────
def _drop_zero_steps(xs, ys):
    keep = np.r_[True, np.hypot(np.diff(xs), np.diff(ys)) > 1e-9]
    return xs[keep], ys[keep]


def road_cost(terr, xs, ys, T_goal, v=V_WALK, lam=LAM, detail=False, t0=START_DELAY_S):
    """道路上折れ線 (xs, ys) [m] を DS 間隔に再サンプルし cost = L + λ1·浸水通過長 + λ4·遅刻 を返す。
    v = 歩行速度 [m/s], t0 = 避難開始時刻 [s] (地震発生からの遅延; 既定 START_DELAY_S = 300 s)。"""
    xs, ys = _drop_zero_steps(np.asarray(xs, float), np.asarray(ys, float))
    if len(xs) < 2:
        return BIG if not detail else dict(cost=BIG, length_m=0.0, time_min=t0 / 60, p_flood=0.0,
                                           p_late=0.0, late_s=0.0, n_flood=0, flood_len_m=0.0, n_off=0, n_bld=0,
                                           xs=xs, ys=ys)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    L = float(s[-1])
    n = int(np.clip(np.ceil(L / DS) + 1, 2, 20000))
    su = np.linspace(0.0, L, n)
    xi, yi = np.interp(su, s, xs), np.interp(su, s, ys)
    t_pass = t0 + su / v
    ri, ci, oob = terr.lookup(xi, yi)
    seg = np.r_[np.diff(su), 0.0]
    flooded = t_pass >= terr.T[ri, ci]
    flood_len = float(seg[flooded].sum())
    late = max(0.0, float(t_pass[-1]) - T_goal) if np.isfinite(T_goal) else 0.0
    cost = L + lam["flood"] * flood_len + lam["late"] * late
    if not detail:
        return cost
    edt = terr.road_edt[ri, ci]
    return dict(cost=cost, length_m=L, time_min=float(t_pass[-1] / 60), p_flood=lam["flood"] * flood_len,
                p_late=lam["late"] * late, late_s=late, n_flood=int(flooded.sum()), flood_len_m=flood_len,
                n_off=int((edt > E.OFFROAD_TOL_M).sum()), n_bld=int((terr.building[ri, ci] | oob).sum()),
                xs=xi, ys=yi)


# ── ペアごとの道路サブグラフ (全点間最短経路を前計算) ───────────────────
class PairGraph:
    def __init__(self, G: RoadGraph, pair, pad=PAD_GRAPH_M):
        self.G = G
        s_id, self.snap_m = G.snap(pair["start"]["lat"], pair["start"]["lon"], SNAP_MAX_M)
        g_id = pair["_goal_node"]
        if s_id is None or g_id is None:
            raise RuntimeError(f"{pair['id']}: 始点/終点を道路へスナップできない")
        lo = np.minimum(G.xy[s_id], G.xy[g_id]) - pad
        hi = np.maximum(G.xy[s_id], G.xy[g_id]) + pad
        win = np.where((G.xy[:, 0] >= lo[0]) & (G.xy[:, 0] <= hi[0]) &
                       (G.xy[:, 1] >= lo[1]) & (G.xy[:, 1] <= hi[1]))[0]
        win = np.union1d(win, [s_id, g_id])
        A = G.A[win][:, win].tocsr()
        ncomp, comp = sparse.csgraph.connected_components(A, directed=False)
        loc0 = {int(g): k for k, g in enumerate(win)}
        keep = np.where(comp == comp[loc0[s_id]])[0]           # 始点の連結成分のみ (= 必ず到達可能)
        if loc0[g_id] not in set(keep.tolist()):
            raise RuntimeError(f"{pair['id']}: 終点が始点と非連結")
        self.gid = win[keep]
        self.A = A[keep][:, keep].tocsr()
        self.N = len(self.gid)
        self.xy = G.xy[self.gid]
        self.tree = cKDTree(self.xy)
        loc = {int(g): k for k, g in enumerate(self.gid)}
        self.s_loc, self.g_loc = loc[s_id], loc[g_id]
        self.s_id, self.g_id = int(s_id), int(g_id)
        t0 = time.time()
        self.dist, self.pred = sparse.csgraph.dijkstra(self.A, directed=False, indices=None, return_predecessors=True)
        self.apsp_s = time.time() - t0
        self.n_comp_window = int(ncomp)
        self._leg: dict = {}

    # 経由点 (x, y) → 最寄りノード (局所 index)
    def project(self, pts):
        return self.tree.query(np.atleast_2d(pts))[1].astype(np.int32)

    def leg(self, a, b):
        key = (int(a), int(b))
        p = self._leg.get(key)
        if p is None:
            pr = self.pred[a]
            out = [int(b)]
            v = int(b)
            while v != a:
                v = int(pr[v])
                if v < 0:
                    return None
                out.append(v)
            p = np.array(out[::-1], np.int32)
            self._leg[key] = p
        return p

    def node_seq(self, vias):
        seq = [self.s_loc]
        for v in np.asarray(vias, int):
            if int(v) != seq[-1]:
                seq.append(int(v))
        if self.g_loc != seq[-1]:
            seq.append(self.g_loc)
        return seq

    def path_nodes(self, vias):
        seq = self.node_seq(vias)
        parts = []
        for a, b in zip(seq[:-1], seq[1:]):
            p = self.leg(a, b)
            if p is None:
                return None
            parts.append(p if not parts else p[1:])
        return np.concatenate(parts) if parts else np.array([self.s_loc], np.int32)

    def path_xy(self, nodes, start_xy=None, goal_xy=None):
        return self.path_xy_from_xy(self.xy[nodes], start_xy, goal_xy)

    @staticmethod
    def path_xy_from_xy(xy, start_xy=None, goal_xy=None):
        xy = np.asarray(xy, float)
        xs, ys = list(xy[:, 0]), list(xy[:, 1])
        if start_xy is not None and np.hypot(xs[0] - start_xy[0], ys[0] - start_xy[1]) > 1e-6:
            xs.insert(0, start_xy[0]); ys.insert(0, start_xy[1])
        if goal_xy is not None and np.hypot(xs[-1] - goal_xy[0], ys[-1] - goal_xy[1]) > 1e-6:
            xs.append(goal_xy[0]); ys.append(goal_xy[1])
        return np.array(xs), np.array(ys)


# ── 経由点問題 (目的関数) ────────────────────────────────────────────
class ViaProblem:
    def __init__(self, pg: PairGraph, terr, pair, M, budget=None, v_walk=V_WALK, start_delay_s=START_DELAY_S):
        """v_walk [m/s] と start_delay_s [s] (避難開始 = 地震発生 + start_delay_s) は road_cost / references の
        通過時刻計算に使う。既定は従来値 (1.0 m/s / 300 s) で、CLI の結果は変わらない。"""
        self.pg, self.t, self.M, self.D = pg, terr, M, 2 * M
        self.budget = BUDGET if budget is None else int(budget)
        self.v, self.t0 = float(v_walk), float(start_delay_s)
        self.s_xy = np.array(pg.G.xy[pg.s_id], float)
        self.g_xy = np.array(pg.G.xy[pg.g_id], float)
        lo = np.minimum(self.s_xy, self.g_xy) - PAD_M
        hi = np.maximum(self.s_xy, self.g_xy) + PAD_M
        self.lb, self.ub = np.tile(lo, M), np.tile(hi, M)
        ri, ci, _ = terr.lookup(self.g_xy[0:1], self.g_xy[1:2])
        self.T_goal = float(terr.T[ri[0], ci[0]])
        self.cache: dict = {}
        self.reset_log()

    def reset_log(self):
        self.n_eval = 0
        self.log = []
        self.best = np.inf
        self.best_x = None

    def decode(self, x, detail=False):
        v = self.pg.project(np.asarray(x, float).reshape(self.M, 2))
        nodes = self.pg.path_nodes(v)
        if nodes is None:
            return (BIG, None) if not detail else (dict(cost=BIG), None)
        xs, ys = self.pg.path_xy(nodes, self.s_xy, self.g_xy)
        return road_cost(self.t, xs, ys, self.T_goal, v=self.v, t0=self.t0, detail=detail), v

    def __call__(self, x):
        v = self.pg.project(np.asarray(x, float).reshape(self.M, 2))
        key = v.tobytes()
        c = self.cache.get(key)
        if c is None:
            nodes = self.pg.path_nodes(v)
            if nodes is None:
                c = BIG
            else:
                xs, ys = self.pg.path_xy(nodes, self.s_xy, self.g_xy)
                c = road_cost(self.t, xs, ys, self.T_goal, v=self.v, t0=self.t0)
            self.cache[key] = c
        self.n_eval += 1
        if self.n_eval <= self.budget:   # 予算内の評価だけを結果に採用 (PSO / CCPSO2 の打ち切り単位の違いを吸収)
            if c < self.best:
                self.best = c
                self.best_x = np.array(x, float).copy()
            self.log.append((self.n_eval, self.best))
        return c

    def batch(self, X):
        return np.array([self(x) for x in X])

    def encode_path(self, xy):
        """道路上折れ線 (n,2) → M 個の経由点 (経路長で等分した内点)。参照解がこの符号化で表現可能かの確認用。"""
        xy = np.asarray(xy, float)
        d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
        tgt = np.linspace(0, d[-1], self.M + 2)[1:-1]
        return np.c_[np.interp(tgt, d, xy[:, 0]), np.interp(tgt, d, xy[:, 1])].ravel()


# ── 最適化器 ────────────────────────────────────────────────────────
def run_pso(prob, seed, budget):
    from pyswarms.single import GlobalBestPSO
    n_p = 30
    np.random.seed(seed)
    vmax = 0.2 * float((prob.ub - prob.lb).max())
    opt = GlobalBestPSO(n_particles=n_p, dimensions=prob.D, options={"c1": 1.5, "c2": 1.5, "w": 0.7},
                        bounds=(prob.lb, prob.ub), bh_strategy="nearest", velocity_clamp=(-vmax, vmax), ftol=-np.inf)
    prob.reset_log()
    t0 = time.time()
    opt.optimize(prob.batch, iters=max(1, -(-budget // n_p)), verbose=False)   # 採用は目的関数側で budget 評価まで
    return dict(cost=float(prob.best), x=np.asarray(prob.best_x), log=list(prob.log),
                evals=min(prob.n_eval, prob.budget), evals_run=prob.n_eval, elapsed=time.time() - t0)


def run_ccpso2(prob, seed, budget, s):
    prob.reset_log()
    t0 = time.time()
    CCPSO2(prob, dim=prob.D, n_particles=CC_N, group_size=s, bounds=(prob.lb, prob.ub), p_cauchy=0.5, seed=seed).run(max_evals=budget)
    return dict(cost=float(prob.best), x=np.asarray(prob.best_x), log=list(prob.log),
                evals=min(prob.n_eval, prob.budget), evals_run=prob.n_eval, elapsed=time.time() - t0, s=s)


# ── 参照解 ──────────────────────────────────────────────────────────
def references(pg: PairGraph, prob: ViaProblem, terr):
    """(a) 静的 (長さ最小) Dijkstra と (b) 時間依存 Dijkstra を、via-point と同一のコスト関数で評価する。"""
    def wrap(xy, **extra):
        xs, ys = pg.path_xy_from_xy(xy, prob.s_xy, prob.g_xy)
        return dict(road_cost(terr, xs, ys, prob.T_goal, v=prob.v, t0=prob.t0, detail=True), node_xy=np.c_[xs, ys], **extra)

    out = {"static": wrap(pg.xy[pg.leg(pg.s_loc, pg.g_loc)], flood_respected=False,
                          graph_arrival_min=float(prob.t0 / 60 + pg.dist[pg.s_loc, pg.g_loc] / prob.v / 60))}
    G = pg.G
    goal, t_arr, gpath = G.earliest_arrival(pg.s_id, [pg.g_id], prob.t0, prob.v, respect_flood=True)
    ok = goal is not None
    if not ok:   # 浸水を守ると到達不能 → 浸水無視の最短時間経路へフォールバック (コストには λ1 が乗る)
        goal, t_arr, gpath = G.earliest_arrival(pg.s_id, [pg.g_id], prob.t0, prob.v, respect_flood=False)
    out["timedep"] = wrap(G.xy[gpath] if gpath else pg.xy[pg.leg(pg.s_loc, pg.g_loc)],
                          flood_respected=ok, graph_arrival_min=float(t_arr / 60))
    return out


# ── 本体 ────────────────────────────────────────────────────────────
def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    inp = load_inputs()
    grid = inp["grid"]
    terr = E.Terrain(inp)
    G = RoadGraph(load_ways(), grid, inp["T"], inp["dem"],
                  extra_points=[(p["goal"]["lat"], p["goal"]["lon"]) for p in PAIRS])
    for p, gid in zip(PAIRS, G.extra_ids):
        p["_goal_node"] = gid
    print(f"[roadpso] graph nodes={G.N} edges={G.n_edges} comps={G.n_comp} | budget={BUDGET} seeds={SEEDS} "
          f"M={M_LIST} λ1={LAM['flood']} λ4={LAM['late']} v={V_WALK} m/s")

    pgs, probs, refs = {}, {}, {}
    for pair in PAIRS:
        pg = PairGraph(G, pair)
        pgs[pair["id"]] = pg
        probs[pair["id"]] = {M: ViaProblem(pg, terr, pair, M, BUDGET) for M in M_LIST}
        refs[pair["id"]] = references(pg, probs[pair["id"]][M_LIST[0]], terr)
        r = refs[pair["id"]]
        print(f"  [{pair['id']}] sub N={pg.N} snap={pg.snap_m:.1f} m APSP {pg.apsp_s:.2f}s | "
              f"static cost={r['static']['cost']:.1f} (L={r['static']['length_m']:.0f} flood={r['static']['flood_len_m']:.0f} m) | "
              f"time-dep cost={r['timedep']['cost']:.1f} (L={r['timedep']['length_m']:.0f} arr={r['timedep']['time_min']:.1f} min, "
              f"flood_respected={r['timedep']['flood_respected']})")

    # 1 評価の実測時間
    ms = {}
    for pair in PAIRS:
        for M in M_LIST:
            pr = probs[pair["id"]][M]
            rs = np.random.RandomState(7)
            X = pr.lb + (pr.ub - pr.lb) * rs.rand(400, pr.D)
            pr.cache.clear(); pr.reset_log()
            t0 = time.time(); [pr(x) for x in X]; dt = (time.time() - t0) / len(X) * 1e3
            ms[f"{pair['id']}_M{M}"] = round(dt, 3)
            pr.cache.clear(); pr.reset_log()
    print(f"  [speed] ms/eval (cold cache, 400 random x): " + ", ".join(f"{k}={v:.2f}" for k, v in ms.items()))

    # ── CCPSO2 グループサイズ スイープ ──
    sweep_rows, chosen_s = [], {}
    for M in M_LIST:
        norm = {s: [] for s in SWEEP_S}
        for pair in PAIRS:
            ref = min(refs[pair["id"]]["static"]["cost"], refs[pair["id"]]["timedep"]["cost"])
            for s in SWEEP_S:
                for seed in SWEEP_SEEDS:
                    r = run_ccpso2(probs[pair["id"]][M], seed, BUDGET, s)
                    norm[s].append(r["cost"] / ref)
                    sweep_rows.append(dict(M=M, pair=pair["id"], s=s, seed=seed, cost=round(r["cost"], 2),
                                           ratio_to_ref=round(r["cost"] / ref, 4), evals=r["evals"],
                                           elapsed_s=round(r["elapsed"], 1)))
        med = {s: float(np.median(norm[s])) for s in SWEEP_S}
        chosen_s[M] = int(min(med, key=med.get))
        print(f"  [sweep M={M}] 参照比 中央値 " + " ".join(f"s={s}:{med[s]:.4f}" for s in SWEEP_S) +
              f"  → s={chosen_s[M]} を採用")

    # ── 本実験 ──
    rows, summary, best, hist = [], {}, {}, {}
    for pair in PAIRS:
        pid = pair["id"]
        pg, ref = pgs[pid], refs[pid]
        straight = float(np.hypot(*(probs[pid][M_LIST[0]].g_xy - probs[pid][M_LIST[0]].s_xy)))
        ent = dict(label=pair["label"], straight_m=straight, T_goal_min=probs[pid][M_LIST[0]].T_goal / 60,
                   sub_nodes=pg.N, snap_m=pg.snap_m,
                   ref=dict(static={k: v for k, v in ref["static"].items() if k not in ("xs", "ys", "node_xy")},
                            timedep={k: v for k, v in ref["timedep"].items() if k not in ("xs", "ys", "node_xy")}))
        for m, d in (("dijkstra_static", ref["static"]), ("dijkstra_timedep", ref["timedep"])):
            rows.append(dict(pair=pid, M="", method=m, seed="", **{k: v for k, v in d.items() if k not in ("xs", "ys", "node_xy")}))
        best[pid] = dict(pair=pair, straight_m=straight,
                         dijkstra_static=dict(xs=ref["static"]["xs"], ys=ref["static"]["ys"], **{k: ref["static"][k] for k in ("cost", "length_m", "time_min")}),
                         dijkstra_timedep=dict(xs=ref["timedep"]["xs"], ys=ref["timedep"]["ys"], **{k: ref["timedep"][k] for k in ("cost", "length_m", "time_min")}))
        print(f"\n=== {pid}: {pair['label']}  直線 {straight:.0f} m  T_goal={probs[pid][M_LIST[0]].T_goal/60:.0f} min")
        for M in M_LIST:
            prob = probs[pid][M]
            # 参照解 (時間依存) をこの符号化で表現した場合のコスト
            enc = prob.encode_path(ref["timedep"]["node_xy"])
            d_enc, _ = prob.decode(enc, detail=True)
            prob.reset_log(); prob.cache.clear()
            res = {"pso": [], "ccpso2": []}
            for seed in SEEDS:
                for meth in ("pso", "ccpso2"):
                    r = run_pso(prob, seed, BUDGET) if meth == "pso" else run_ccpso2(prob, seed, BUDGET, chosen_s[M])
                    det, v = prob.decode(r["x"], detail=True)
                    r.update({k: det[k] for k in det if k not in ("xs", "ys")})
                    r["xs"], r["ys"], r["vias"] = det["xs"], det["ys"], v
                    res[meth].append(r)
                    hist[(pid, M, meth, seed)] = np.array(r["log"])
                    rows.append(dict(pair=pid, M=M, method=meth, seed=seed, evals=r["evals"],
                                     evals_run=r["evals_run"], elapsed_s=round(r["elapsed"], 1), s_group=r.get("s", ""),
                                     **{k: det[k] for k in det if k not in ("xs", "ys")}))
                    print(f"  M={M:2d} seed {seed} {meth:6s} cost={r['cost']:8.1f} L={r['length_m']:6.0f} "
                          f"flood={r['flood_len_m']:5.0f} m late={r['late_s']:5.0f} s  {r['elapsed']:.0f}s")
            pc = np.array([r["cost"] for r in res["pso"]])
            cc = np.array([r["cost"] for r in res["ccpso2"]])
            rs, rt = ref["static"]["cost"], ref["timedep"]["cost"]
            rbest = min(rs, rt)
            ent[f"M{M}"] = dict(
                D=2 * M, ccpso2_group_size=chosen_s[M],
                pso_median=float(np.median(pc)), pso_min=float(pc.min()), pso_max=float(pc.max()),
                ccpso2_median=float(np.median(cc)), ccpso2_min=float(cc.min()), ccpso2_max=float(cc.max()),
                ccpso2_wins=int((cc < pc).sum()), pso_wins=int((pc < cc).sum()), ties=int((pc == cc).sum()),
                pso_reach_ref=int((pc <= rbest * 1.001).sum()), ccpso2_reach_ref=int((cc <= rbest * 1.001).sum()),
                pso_median_over_ref=float(np.median(pc) / rbest), ccpso2_median_over_ref=float(np.median(cc) / rbest),
                ref_timedep_encoded_cost=float(d_enc["cost"]), ref_best_cost=float(rbest))
            print(f"  → M={M}: 中央値 PSO {np.median(pc):.1f} / CCPSO2 {np.median(cc):.1f} | 参照 静的 {rs:.1f} / 時間依存 {rt:.1f} "
                  f"| 時間依存を M={M} で符号化 {d_enc['cost']:.1f} | CCPSO2 勝ち {int((cc<pc).sum())}/{len(SEEDS)} "
                  f"| 参照到達 PSO {int((pc<=rbest*1.001).sum())}/{len(SEEDS)} CCPSO2 {int((cc<=rbest*1.001).sum())}/{len(SEEDS)}")
            bp, bc = res["pso"][int(np.argmin(pc))], res["ccpso2"][int(np.argmin(cc))]
            best[pid][f"M{M}"] = dict(
                pso=dict(xs=bp["xs"], ys=bp["ys"], cost=bp["cost"], length_m=bp["length_m"], time_min=bp["time_min"],
                         seed=SEEDS[int(np.argmin(pc))], vias=pg.xy[bp["vias"]]),
                ccpso2=dict(xs=bc["xs"], ys=bc["ys"], cost=bc["cost"], length_m=bc["length_m"], time_min=bc["time_min"],
                            seed=SEEDS[int(np.argmin(cc))], vias=pg.xy[bc["vias"]]))
        summary[pid] = ent

    # ── 出力 ──
    keys = ["pair", "M", "method", "seed", "s_group", "cost", "length_m", "time_min", "p_flood", "p_late", "late_s",
            "flood_len_m", "n_flood", "n_off", "n_bld", "evals", "evals_run", "elapsed_s", "flood_respected",
            "graph_arrival_min"]
    with open(OUTD / "table_roadpso.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
        for pid, s in summary.items():
            for M in M_LIST:
                w.writerow(dict(pair=pid, M=M, method="MEDIAN_pso", cost=round(s[f"M{M}"]["pso_median"], 2)))
                w.writerow(dict(pair=pid, M=M, method="MEDIAN_ccpso2", cost=round(s[f"M{M}"]["ccpso2_median"], 2)))
                w.writerow(dict(pair=pid, M=M, method=f"WINS_ccpso2_{s[f'M{M}']['ccpso2_wins']}_of_{len(SEEDS)}"))
    with open(OUTD / "roadpso_sweep.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys())); w.writeheader(); w.writerows(sweep_rows)
    with open(OUTD / "summary_roadpso.json", "w", encoding="utf-8") as f:
        json.dump(dict(formulation="via-point on road graph (静的最短経路で接続)", budget=BUDGET, seeds=SEEDS,
                       M_list=M_LIST, lam=LAM, v_walk=V_WALK, start_delay_s=START_DELAY_S, ds_m=DS, pad_m=PAD_M,
                       pad_graph_m=PAD_GRAPH_M, ccpso2_n_particles=CC_N,
                       ccpso2_group_sweep=dict(candidates=SWEEP_S, seeds=SWEEP_SEEDS,
                                               chosen={f"M{M}": chosen_s[M] for M in M_LIST}),
                       ms_per_eval=ms, pairs=summary), f, ensure_ascii=False, indent=1)

    rb = {}
    for pid, b in best.items():
        e = dict(pair={k: v for k, v in b["pair"].items() if not k.startswith("_")}, straight_m=b["straight_m"])
        for m in ("dijkstra_static", "dijkstra_timedep"):
            la, lo = grid.xy_to_latlon(b[m]["xs"], b[m]["ys"])
            e[m] = dict(lat=la.tolist(), lon=lo.tolist(), **{k: b[m][k] for k in ("cost", "length_m", "time_min")})
        for M in M_LIST:
            for m in ("pso", "ccpso2"):
                r = b[f"M{M}"][m]
                la, lo = grid.xy_to_latlon(r["xs"], r["ys"])
                e[f"{m}_M{M}"] = dict(lat=la.tolist(), lon=lo.tolist(), seed=r["seed"],
                                      vias=[[float(a), float(c)] for a, c in r["vias"]],
                                      **{k: r[k] for k in ("cost", "length_m", "time_min")})
        rb[pid] = e
    with open(OUTD / "routes_roadpso_best.json", "w", encoding="utf-8") as f:
        json.dump(rb, f, ensure_ascii=False)
    np.savez_compressed(OUTD / "roadpso_runs.npz", **{f"{p}__M{M}__{m}__{s}": h for (p, M, m, s), h in hist.items()})

    plot_convergence(hist, summary, OUTD / "convergence_roadpso.png")
    plot_routes(inp, terr, best, OUTD / "routes_roadpso.png")
    export_overlay(grid, inp, summary, rb, WALK / "overlay_out/gobo/routes_roadpso.json")
    print(f"\n[roadpso] wrote {OUTD}/ table_roadpso.csv, summary_roadpso.json, roadpso_sweep.csv, "
          f"convergence_roadpso.png, routes_roadpso.png, routes_roadpso_best.json")


# ── 図 ──────────────────────────────────────────────────────────────
def _mpl():
    import matplotlib
    matplotlib.use("Agg"); setup_japanese_font()
    import matplotlib.pyplot as plt
    return plt


def plot_convergence(hist, summary, path):
    plt = _mpl()
    fig, axes = plt.subplots(len(M_LIST), len(PAIRS), figsize=(5.4 * len(PAIRS), 4.1 * len(M_LIST)), dpi=110, squeeze=False)
    for i, M in enumerate(M_LIST):
        for j, pair in enumerate(PAIRS):
            ax, pid = axes[i][j], pair["id"]
            s = summary[pid][f"M{M}"]
            for meth, col, lab in (("pso", "#7f7f7f", "標準 PSO"), ("ccpso2", "#e6a817", "CCPSO2")):
                curves = []
                for sd in SEEDS:
                    h = hist[(pid, M, meth, sd)]
                    ax.plot(h[:, 0], h[:, 1], color=col, alpha=0.3, lw=0.8)
                    curves.append(np.interp(np.arange(1, BUDGET + 1), h[:, 0], h[:, 1]))
                ax.plot(np.arange(1, BUDGET + 1), np.median(curves, axis=0), color=col, lw=2.2,
                        label=f"{lab} (中央値, {len(SEEDS)} seed)" + (f" s={s['ccpso2_group_size']}" if meth == "ccpso2" else ""))
            r = summary[pid]["ref"]
            ax.axhline(r["static"]["cost"], color="k", ls=":", lw=1.4, label=f"静的 Dijkstra {r['static']['cost']:.0f}")
            ax.axhline(r["timedep"]["cost"], color="tab:red", ls="--", lw=1.4, label=f"時間依存 Dijkstra {r['timedep']['cost']:.0f}")
            ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            ax.set_xlabel("評価回数"); ax.set_ylabel("経路コスト [m]")
            ax.set_title(f"{pair['label']}  (M={M} 経由点, D={2*M})", fontsize=10)
            ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"御坊 道路制約 via-point 定式化: コスト収束 (予算 {BUDGET} 評価, v={V_WALK} m/s, 県 R8 津波, "
                 f"cost = L + {LAM['flood']:.0f}·浸水通過長 + {LAM['late']:.0f}·遅刻)", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_routes(inp, terr, best, path):
    plt = _mpl()
    from matplotlib.lines import Line2D
    grid = terr.grid
    hs = hillshade(inp["dem"], grid.dy_m, grid.dx_m)
    fac = [f for f in inp["fac"] if f["inside"]]
    fac_xy = np.array([grid.rc_to_xy(f["row"], f["col"]) for f in fac]) if fac else np.zeros((0, 2))
    fig, axes = plt.subplots(len(M_LIST), len(PAIRS), figsize=(6.0 * len(PAIRS), 5.9 * len(M_LIST)), dpi=110, squeeze=False)
    for i, M in enumerate(M_LIST):
        for j, pair in enumerate(PAIRS):
            ax, pid = axes[i][j], pair["id"]
            b = best[pid]
            allx = np.concatenate([b[k]["xs"] for k in ("dijkstra_static", "dijkstra_timedep")] +
                                  [b[f"M{M}"][m]["xs"] for m in ("pso", "ccpso2")])
            ally = np.concatenate([b[k]["ys"] for k in ("dijkstra_static", "dijkstra_timedep")] +
                                  [b[f"M{M}"][m]["ys"] for m in ("pso", "ccpso2")])
            lo = np.array([allx.min(), ally.min()]) - 120.0
            hi = np.array([allx.max(), ally.max()]) + 120.0
            r0, c0 = grid.xy_to_rc(lo[0], hi[1]); r1, c1 = grid.xy_to_rc(hi[0], lo[1])
            r0, r1 = int(max(0, np.floor(r0))), int(min(grid.H, np.ceil(r1)))
            c0, c1 = int(max(0, np.floor(c0))), int(min(grid.W, np.ceil(c1)))
            sub = np.s_[r0:r1, c0:c1]
            ext = [c0 * grid.dx_m, c1 * grid.dx_m, -r1 * grid.dy_m, -r0 * grid.dy_m]
            ax.imshow(hs[sub], cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper", interpolation="nearest")
            ax.imshow(np.ma.masked_where(terr.land[sub], ~terr.land[sub]), cmap="Blues", vmin=0, vmax=1.5,
                      alpha=0.6, extent=ext, origin="upper", interpolation="nearest")
            ax.imshow(np.ma.masked_where(~terr.road[sub], terr.road[sub]), cmap="Oranges", vmin=0, vmax=2.6,
                      alpha=0.55, extent=ext, origin="upper", interpolation="nearest")
            ax.imshow(np.ma.masked_where(~terr.building[sub], terr.building[sub]), cmap="Greys", vmin=0, vmax=1.4,
                      alpha=0.55, extent=ext, origin="upper", interpolation="nearest")
            Tm = terr.T[sub] / 60.0
            wet = np.isfinite(Tm)
            ax.imshow(np.ma.masked_where(~wet, wet), cmap="Blues", vmin=0, vmax=2.6, alpha=0.30,
                      extent=ext, origin="upper", interpolation="nearest")
            gx = np.linspace(ext[0], ext[1], Tm.shape[1]); gy = np.linspace(ext[3], ext[2], Tm.shape[0])
            cl = ax.contour(gx, gy, np.where(wet, Tm, 999), levels=[20, 25, 30, 35, 40, 50],
                            colors="tab:blue", linewidths=0.8, alpha=0.85)
            ax.clabel(cl, fmt="%d分", fontsize=7)
            sel = (fac_xy[:, 0] > lo[0]) & (fac_xy[:, 0] < hi[0]) & (fac_xy[:, 1] > lo[1]) & (fac_xy[:, 1] < hi[1])
            ax.plot(fac_xy[sel, 0], fac_xy[sel, 1], "*", color="lime", ms=11, mec="k", zorder=8)
            same = abs(b["dijkstra_static"]["cost"] - b["dijkstra_timedep"]["cost"]) < 1e-6
            for key, style in REF_STYLE.items():
                k = f"dijkstra_{key}"
                nm = "静的" if key == "static" else "時間依存"
                if same and key == "static":
                    continue
                ax.plot(b[k]["xs"], b[k]["ys"], color="k", lw=style["lw"] + 2.0, alpha=0.75, zorder=4)
                ax.plot(b[k]["xs"], b[k]["ys"], color=COLORS["dijkstra"], zorder=5, **style,
                        label=(f"Dijkstra 参照 (静的=時間依存)  {b[k]['cost']:.0f}" if same else f"{nm} Dijkstra  {b[k]['cost']:.0f}"))
            for m, lw in (("pso", 2.2), ("ccpso2", 2.2)):
                r = b[f"M{M}"][m]
                ax.plot(r["xs"], r["ys"], color="k", lw=lw + 1.4, alpha=0.55, zorder=6)
                ax.plot(r["xs"], r["ys"], color=COLORS[m], lw=lw, zorder=7,
                        label=f"{'標準 PSO' if m=='pso' else 'CCPSO2'} 最良 (seed {r['seed']})  {r['cost']:.0f}")
                ax.plot(r["vias"][:, 0], r["vias"][:, 1], "o", ms=4.0, mec="k", mew=0.5,
                        color=COLORS[m], zorder=9)
            ax.plot(b["dijkstra_static"]["xs"][0], b["dijkstra_static"]["ys"][0], "s", color="red", ms=9, mec="k", zorder=10)
            ax.plot(b["dijkstra_static"]["xs"][-1], b["dijkstra_static"]["ys"][-1], "*", color="lime", ms=17, mec="k", zorder=10)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{pair['label']}  (M={M} 経由点)", fontsize=10)
            ax.legend(fontsize=7.2, loc="lower left", facecolor="#e8e8e8", framealpha=0.85)
    fig.suptitle("御坊 道路制約 via-point 経路: 陰影 DEM + 浸水域 (青) と 30 cm 到達時刻 [分] の等値線 + OSM 道路(橙)/建物(灰) "
                 "+ 指定緊急避難場所(★)  ／ ● = 経由点の射影先ノード", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ── tizucra-walk overlay ────────────────────────────────────────────
def route_points(grid, dem, lat, lon, v=V_WALK, t0=START_DELAY_S, to_block=None):
    """経路の lat/lon 列 (DS 間隔) → overlay の points 列と経路長 [m]。
    3 点に 1 点へ間引き (始点・終点は保持)、elev = dem (fill_nan_nearest 済み) の最近傍セル、
    t_min = t0/60 + s/v/60。to_block = (lat, lon) → (x, z) (tizucra-walk tools/profiles.latlon_to_block;
    None なら x/z を省く)。route_server.py と共用。"""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    x, y = grid.latlon_to_xy(lat, lon)
    sc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    keep = np.r_[np.arange(0, len(lat) - 1, 3), len(lat) - 1]
    rr, cc = grid.latlon_to_rc(lat[keep], lon[keep])
    ri = np.clip(np.round(rr).astype(int), 0, grid.H - 1)
    ci = np.clip(np.round(cc).astype(int), 0, grid.W - 1)
    bx, bz = to_block(lat[keep], lon[keep]) if to_block is not None else (None, None)
    pts = []
    for k, (a, b, el, s) in enumerate(zip(lat[keep], lon[keep], dem[ri, ci], sc[keep])):
        d = dict(lat=round(float(a), 7), lon=round(float(b), 7))
        if bx is not None:
            d["x"] = round(float(bx[k]), 1); d["z"] = round(float(bz[k]), 1)
        d["elev"] = round(float(el), 2)
        d["t_min"] = round(float(t0 / 60 + s / v / 60), 2)
        pts.append(d)
    return pts, float(sc[-1])


def export_overlay(grid, inp, summary, rb, out_path):
    sys.path.insert(0, str(WALK / "tools"))
    from profiles import PROFILES, latlon_to_block  # noqa: E402
    p = PROFILES["gobo"]
    dem = fill_nan_nearest(inp["dem"].astype(np.float64))
    labels = dict(dijkstra="Dijkstra 参照 (道路網)", pso="標準 PSO (経由点)", ccpso2="CCPSO2 (経由点)")
    routes = []
    for pid, e in rb.items():
        pr = e["pair"]
        pick = {}
        rs, rt = e["dijkstra_static"], e["dijkstra_timedep"]
        which = "timedep" if rt["cost"] <= rs["cost"] else "static"
        pick["dijkstra"] = dict(rt if which == "timedep" else rs,
                                note=("時間依存 (浸水セル通行不可)" if which == "timedep" else "静的 (長さ最小)"), M=None)
        for m in ("pso", "ccpso2"):
            cands = [(e[f"{m}_M{M}"]["cost"], M) for M in M_LIST]
            bM = min(cands)[1]
            pick[m] = dict(e[f"{m}_M{bM}"], note=f"M={bM} 経由点 (D={2*bM})", M=bM)
        for m, r in pick.items():
            pts, L = route_points(grid, dem, r["lat"], r["lon"], to_block=lambda la, lo: latlon_to_block(p, la, lo))
            lab = f"{pr['label']} / {labels[m]} {r['note']}" + (f" seed {r['seed']}" if r.get("seed") is not None else "")
            routes.append(dict(id=f"{pid}_{m}_road", label=lab, method=m, pair=pid, formulation="road_via_point",
                               M=r["M"], color=COLORS[m], width=3,
                               start=dict(name=pr["start"]["name"], lat=pr["start"]["lat"], lon=pr["start"]["lon"]),
                               goal=dict(name=pr["goal"]["name"], lat=pr["goal"]["lat"], lon=pr["goal"]["lon"]),
                               points=pts, length_m=round(L, 1), time_min=round(float(r["time_min"]), 2),
                               cost=round(float(r["cost"]), 1)))
    doc = dict(map="gobo",
               scenario=dict(source="和歌山県 R8 (2026-03) 南海トラフ巨大地震 津波浸水想定 (30 cm 到達時刻)",
                             v_walk=V_WALK, start_delay_min=START_DELAY_S / 60, budget_evals=BUDGET, seeds=SEEDS,
                             formulation="道路網に制約した経由点 (via-point) 定式化: M 個の経由点を最寄り道路ノードへ射影し、"
                                         "始点→v1→…→vM→終点 を静的最短経路で接続",
                             M_list=M_LIST, cost="経路長[m] + λ1·浸水通過長[m] + λ4·終点遅刻[s]", lam=LAM,
                             pairs=summary),
               routes=routes, source=E_SOURCE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[export] {out_path}  routes={len(routes)}  ({out_path.stat().st_size/1024:.0f} KB)")


E_SOURCE = ("『最大クラスの巨大地震(南海トラフ巨大地震)の津波シミュレーション動画(R8)』(和歌山県) を復号した 30 cm 到達時刻、"
            "国土地理院 5 m DEM・指定緊急避難場所、OpenStreetMap (© OpenStreetMap contributors, ODbL) 道路網 を加工して作成")


if __name__ == "__main__":
    main()
