"""
evac/route_server.py — 避難経路ラボ用ローカル HTTP API (tizucra-walk の dev_server.py が /api/* をここへ中継する)。

evac_roadpso.py の部品 (PairGraph / ViaProblem / references / run_pso / run_ccpso2 / route_points) をそのまま使い、
任意の始点 (lat/lon) → 終点 (指定緊急避難場所 or lat/lon) について Dijkstra 2 種 + PSO + CCPSO2 の道路上経路を返す。
標準ライブラリの http.server だけで動く (新規依存なし)。計算は threading.Lock で 1 件ずつ (取れなければ 409)。

起動
  cd flood_pso
  .venv/bin/python src/evac/route_server.py [--port 8766] [--host 127.0.0.1] [--walk ../tizucra-walk]
    --walk : tizucra-walk のパス (tools/profiles.py の latlon_to_block でブロック座標 x/z を付ける。無ければ x/z 省略)
  起動時に 1 回だけ 5 m 格子 (results/evac/inputs.npz) と OSM 道路グラフ (results/evac/roads.json) を読み、
  指定緊急避難場所 (津波=1・格子内) を仮想ノードとして道路へ接続する (実測 ≈ 1 s)。ログは stdout に 1 リクエスト 1 行。
  前提: src/evac/inputs.py を一度回して results/evac/inputs.npz と roads.json があること。

契約 (tizucra-walk 側と共通)
  GET  /api/info   → {"ok", "map":"gobo", "scenario":{source, v_walk_options, start_delay_min, budget_default, M_options,
                      M_default, ccpso2_group_size, cost, lam}, "bounds":{lat_min,lat_max,lon_min,lon_max},
                      "facilities":[{id,name,lat,lon,tower,reachable,addr}], "graph":{nodes,edges,comps}, "startup_s"}
  POST /api/route  (application/json)
      {"start": {"lat", "lon", "name"?},
       "goal":  {"facility_id"} | {"lat", "lon", "name"?},
       "v_walk": 1.0, "start_delay_min": 5, "budget": 5000 (200..50000), "M": 5 (1..40), "seeds": [0] (≤ 5 個),
       "methods": ["dijkstra_timedep", "dijkstra_static", "pso", "ccpso2"],
       "ccpso2_group_size": null (→ M=5: 5 / M=20: 2 / それ以外 min(5, M))}
    → 200 {"ok", "map", "scenario"(+ v_walk, start_delay_min, budget, M, seeds, ccpso2_group_size),
           "pair":{start:{name,lat,lon,snap_m}, goal:{name,lat,lon,facility_id?}, straight_m, T_goal_min, sub_nodes, apsp_s},
           "routes":[{id, label, method, seed, M, color, width, start, goal, points:[{lat,lon,x,z,elev,t_min}],
                      length_m, time_min, cost, flood_len_m, late_s, p_flood, flood_respected?, ratio_to_ref, evals,
                      elapsed_s, best}],
           "ref_cost", "timing":{total_s, graph_s, dijkstra_s, pso_s, ccpso2_s}, "source"}
    → 400 {"ok": false, "error": "..."} (始点/終点が道路から 100 m 以上 / 終点到達不能 / 施設 id 不明 / パラメータ不正 /
                                       始点と終点が離れすぎて道路窓が 8000 ノード超)
    → 409 {"ok": false, "busy": true, "error": "..."} (別ジョブ実行中)
  OPTIONS → 204。全応答に CORS ヘッダ (Allow-Origin: * / Allow-Headers: Content-Type / Allow-Methods: GET,POST,OPTIONS)。
  points は evac_roadpso.export_overlay と同じ (2 m サンプルを 3 点に 1 点へ間引き、始点・終点保持、elev = 5 m DEM、
  t_min = start_delay_min + s / v_walk / 60)。ref_cost = min(静的, 時間依存) のコスト (evac_roadpso の rbest と同じ)。
  best = 同じ手法の seed の中で最小コストの 1 本だけ true。

curl 例
  curl -s http://127.0.0.1:8766/api/info | python3 -m json.tool | head -40
  curl -s -X POST http://127.0.0.1:8766/api/route -H 'Content-Type: application/json' -d '{
    "start": {"lat": 33.878834, "lon": 135.155636, "name": "名屋 住宅地"},
    "goal": {"facility_id": "E3020500063201"}, "v_walk": 1.0, "budget": 5000, "M": 5, "seeds": [0]}'
  curl -s -X POST http://127.0.0.1:8766/api/route -H 'Content-Type: application/json' -d '{
    "start": {"lat": 33.878834, "lon": 135.155636}, "goal": {"lat": 33.882881, "lon": 135.157136},
    "methods": ["dijkstra_timedep", "pso"], "budget": 500}'
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# evac_roadpso / evac_ccpso2 は import 時に sys.argv を予算として読むため、退避してから読み込む
_ARGV, sys.argv = sys.argv, sys.argv[:1]
import evac_roadpso as R  # noqa: E402
sys.argv = _ARGV

from common import START_DELAY_S, load_inputs  # noqa: E402
from roads import RoadGraph, load_ways, fill_nan_nearest  # noqa: E402
import inputs as INP  # noqa: E402

MAP = "gobo"
SNAP_MAX_M = R.SNAP_MAX_M            # 100 m
V_WALK_OPTIONS = [1.0, 0.5]
M_OPTIONS = list(R.M_LIST)           # [5, 20]
M_DEFAULT = 5
BUDGET_DEFAULT, BUDGET_MIN, BUDGET_MAX = 5000, 200, 50000
M_MIN, M_MAX = 1, 40
MAX_SEEDS = 5
MAX_SUB_NODES = 8000                 # 道路窓 (bbox + 500 m) のノード数上限 (全点間最短距離 N² 行列 ≈ 0.75 GB / APSP ≈ 13 s)
GROUP_SIZE_DEFAULT = {5: 5, 20: 2}   # evac_roadpso のスイープで採用した s (summary_roadpso.json ccpso2_group_sweep.chosen)
METHODS = ["dijkstra_timedep", "dijkstra_static", "pso", "ccpso2"]
COLORS = dict(dijkstra_timedep="#ffffff", dijkstra_static="#80d8ff", pso="#9e9e9e", ccpso2="#ffd54f")
SCENARIO_SOURCE = "和歌山県 R8 (2026-03) 南海トラフ巨大地震 津波浸水想定 (30 cm 到達時刻)"


class BadRequest(Exception):
    pass


def default_group_size(M):
    return GROUP_SIZE_DEFAULT.get(M, min(5, M))


# ── 起動時に 1 回だけ組む状態 ───────────────────────────────────────
class Lab:
    def __init__(self, walk: Path | None):
        t0 = time.time()
        self.inp = inp = load_inputs()
        self.grid = grid = inp["grid"]
        self.terr = R.E.Terrain(inp)
        self.dem = fill_nan_nearest(inp["dem"].astype(np.float64))
        gd = dict(lat_max=grid.lat_max, lon_min=grid.lon_min, res_lat=grid.res_lat, res_lon=grid.res_lon, H=grid.H, W=grid.W)
        self.fac = [f for f in INP.load_facilities(gd) if f["inside"]]
        self.G = RoadGraph(load_ways(), grid, inp["T"], inp["dem"], extra_points=[(f["lat"], f["lon"]) for f in self.fac])
        self.fac_node = {f["id"]: nid for f, nid in zip(self.fac, self.G.extra_ids)}
        self.fac_link_m = {f["id"]: d for f, d in zip(self.fac, self.G.extra_link_m)}
        self.fac_by_id = {f["id"]: f for f in self.fac}
        self.to_block = None
        if walk is not None:
            try:
                sys.path.insert(0, str(walk / "tools"))
                from profiles import PROFILES, latlon_to_block  # noqa: E402
                p = PROFILES[MAP]
                self.to_block = lambda la, lo: latlon_to_block(p, la, lo)
            except Exception as e:  # noqa: BLE001
                print(f"[route_server] tizucra-walk の profiles を読めないため x/z を省略します: {walk} ({e})")
        self.lock = threading.Lock()
        self.startup_s = time.time() - t0
        print(f"[route_server] graph nodes={self.G.N} edges={self.G.n_edges} comps={self.G.n_comp} | "
              f"facilities={len(self.fac)} (reachable={sum(v is not None for v in self.fac_node.values())}) | "
              f"startup {self.startup_s:.1f}s")

    # ── /api/info ──
    def scenario(self):
        return dict(source=SCENARIO_SOURCE, v_walk_options=V_WALK_OPTIONS, start_delay_min=START_DELAY_S / 60,
                    budget_default=BUDGET_DEFAULT, M_options=M_OPTIONS, M_default=M_DEFAULT,
                    ccpso2_group_size={str(k): v for k, v in GROUP_SIZE_DEFAULT.items()},
                    cost="経路長[m] + λ1·浸水通過長[m] + λ4·終点遅刻[s]", lam=dict(flood=R.LAM["flood"], late=R.LAM["late"]))

    def info(self):
        g = self.grid
        return dict(ok=True, map=MAP, scenario=self.scenario(),
                    bounds=dict(lat_min=g.lat_min, lat_max=g.lat_max, lon_min=g.lon_min, lon_max=g.lon_max),
                    facilities=[dict(id=f["id"], name=f["name"], lat=f["lat"], lon=f["lon"], tower=("タワー" in f["name"]),
                                     reachable=self.fac_node[f["id"]] is not None, addr=f["addr"]) for f in self.fac],
                    graph=dict(nodes=int(self.G.N), edges=int(self.G.n_edges), comps=int(self.G.n_comp)),
                    startup_s=round(self.startup_s, 2))

    # ── /api/route ──
    def parse(self, req):
        if not isinstance(req, dict):
            raise BadRequest("リクエスト本文は JSON オブジェクトにしてください")

        def point(key):
            p = req.get(key)
            if not isinstance(p, dict):
                raise BadRequest(f"{key} は {{lat, lon}} のオブジェクトにしてください")
            try:
                lat, lon = float(p["lat"]), float(p["lon"])
            except (KeyError, TypeError, ValueError):
                raise BadRequest(f"{key} の lat / lon が数値ではありません")
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise BadRequest(f"{key} の lat / lon が範囲外です")
            name = p.get("name")
            return lat, lon, (str(name) if name else None)

        def number(key, default, lo=None, hi=None, integer=False):
            v = req.get(key, default)
            if v is None:
                v = default
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise BadRequest(f"{key} は数値にしてください")
            if integer:
                if float(v) != int(v):
                    raise BadRequest(f"{key} は整数にしてください")
                v = int(v)
            else:
                v = float(v)
                if not math.isfinite(v):
                    raise BadRequest(f"{key} は有限の数値にしてください")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                raise BadRequest(f"{key} は {lo}..{hi} の範囲にしてください")
            return v

        s_lat, s_lon, s_name = point("start")
        goal = req.get("goal")
        if not isinstance(goal, dict):
            raise BadRequest("goal は {facility_id} または {lat, lon} のオブジェクトにしてください")
        if "facility_id" in goal:
            fid = str(goal["facility_id"])
            f = self.fac_by_id.get(fid)
            if f is None:
                raise BadRequest(f"施設 id が不明です: {fid}")
            if self.fac_node[fid] is None:
                raise BadRequest(f"施設「{f['name']}」は道路へ接続できません (最寄り道路まで {self.fac_link_m[fid]:.0f} m)")
            g = dict(name=f["name"], lat=f["lat"], lon=f["lon"], facility_id=fid, node=self.fac_node[fid])
        else:
            g_lat, g_lon, g_name = point("goal")
            nid, d = self.G.snap(g_lat, g_lon, SNAP_MAX_M)
            if nid is None:
                raise BadRequest(f"終点が道路から {SNAP_MAX_M:.0f} m 以上離れています (最寄り {d:.0f} m)")
            g = dict(name=g_name or f"終点 ({g_lat:.5f}, {g_lon:.5f})", lat=g_lat, lon=g_lon, node=nid, snap_m=d)
        sid, d = self.G.snap(s_lat, s_lon, SNAP_MAX_M)
        if sid is None:
            raise BadRequest(f"始点が道路から {SNAP_MAX_M:.0f} m 以上離れています (最寄り {d:.0f} m)")
        if sid == g["node"]:
            raise BadRequest("始点と終点が同じ道路ノードに乗っています (離してください)")
        s = dict(name=s_name or f"始点 ({s_lat:.5f}, {s_lon:.5f})", lat=s_lat, lon=s_lon, node=sid, snap_m=d)

        v_walk = number("v_walk", R.V_WALK, lo=0.05, hi=10.0)
        delay_min = number("start_delay_min", START_DELAY_S / 60, lo=0.0, hi=600.0)
        budget = number("budget", BUDGET_DEFAULT, BUDGET_MIN, BUDGET_MAX, integer=True)
        M = number("M", M_DEFAULT, M_MIN, M_MAX, integer=True)
        seeds = req.get("seeds", [0])
        if seeds is None:
            seeds = [0]
        if not isinstance(seeds, list) or not seeds or len(seeds) > MAX_SEEDS or \
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
            raise BadRequest(f"seeds は 0 以上の整数のリスト (1..{MAX_SEEDS} 個) にしてください")
        methods = req.get("methods", METHODS)
        if methods is None:
            methods = METHODS
        if not isinstance(methods, list) or not methods or any(m not in METHODS for m in methods):
            raise BadRequest(f"methods は {METHODS} の部分集合 (1 つ以上) にしてください")
        methods = list(dict.fromkeys(methods))
        gs = req.get("ccpso2_group_size")
        if gs is None:
            gs = default_group_size(M)
        else:
            gs = number("ccpso2_group_size", gs, 1, 2 * M, integer=True)
        return dict(start=s, goal=g, v_walk=v_walk, start_delay_min=delay_min, budget=budget, M=M,
                    seeds=[int(x) for x in seeds], methods=methods, ccpso2_group_size=int(gs))

    def route(self, q):
        t_all = time.time()
        s, g = q["start"], q["goal"]
        v, t0 = q["v_walk"], q["start_delay_min"] * 60.0
        M, budget, gs = q["M"], q["budget"], q["ccpso2_group_size"]
        pair = dict(id="lab", label=f"{s['name']} → {g['name']}",
                    start=dict(name=s["name"], lat=s["lat"], lon=s["lon"]),
                    goal=dict(name=g["name"], lat=g["lat"], lon=g["lon"]), _goal_node=g["node"])
        t = time.time()
        xy = self.G.xy
        lo = np.minimum(xy[s["node"]], xy[g["node"]]) - R.PAD_GRAPH_M
        hi = np.maximum(xy[s["node"]], xy[g["node"]]) + R.PAD_GRAPH_M
        n_win = int(((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])).sum())
        if n_win > MAX_SUB_NODES:   # PairGraph は窓内の全点間最短距離を前計算する (N² メモリ) ので遠すぎるペアは断る
            raise BadRequest(f"始点と終点が離れすぎています (道路窓のノード数 {n_win} > {MAX_SUB_NODES})")
        try:
            pg = R.PairGraph(self.G, pair)
        except RuntimeError as e:
            if "非連結" in str(e):
                raise BadRequest("終点が始点と道路網で繋がっていません (別の連結成分)")
            raise BadRequest(str(e))
        prob = R.ViaProblem(pg, self.terr, pair, M, budget, v_walk=v, start_delay_s=t0)
        timing = dict(graph_s=time.time() - t, dijkstra_s=0.0, pso_s=0.0, ccpso2_s=0.0)

        t = time.time()
        refs = R.references(pg, prob, self.terr)
        timing["dijkstra_s"] = time.time() - t
        ref_cost = min(refs["static"]["cost"], refs["timedep"]["cost"])

        start = dict(name=s["name"], lat=s["lat"], lon=s["lon"])
        goal = dict(name=g["name"], lat=g["lat"], lon=g["lon"])

        def route_entry(rid, label, method, det, seed=None, Mv=None, evals=None, elapsed=0.0, **extra):
            lat, lon = self.grid.xy_to_latlon(det["xs"], det["ys"])
            pts, L = R.route_points(self.grid, self.dem, lat, lon, v=v, t0=t0, to_block=self.to_block)
            e = dict(id=rid, label=label, method=method, seed=seed, M=Mv, color=COLORS[method], width=3,
                     start=start, goal=goal, points=pts, length_m=round(L, 1), time_min=round(float(det["time_min"]), 2),
                     cost=round(float(det["cost"]), 1), flood_len_m=round(float(det["flood_len_m"]), 1),
                     late_s=round(float(det["late_s"]), 1), p_flood=round(float(det["p_flood"]), 1))
            e.update(extra)
            e.update(ratio_to_ref=round(float(det["cost"]) / ref_cost, 4) if ref_cost > 0 else None,
                     evals=evals, elapsed_s=round(float(elapsed), 2), best=False)
            return e

        routes = []
        for meth in q["methods"]:
            if meth == "dijkstra_timedep":
                d = refs["timedep"]
                routes.append(route_entry("lab_dijkstra_timedep", "Dijkstra 時間依存 (浸水セル通行不可)", meth, d,
                                          elapsed=timing["dijkstra_s"], flood_respected=bool(d["flood_respected"])))
            elif meth == "dijkstra_static":
                d = refs["static"]
                routes.append(route_entry("lab_dijkstra_static", "Dijkstra 静的 (長さ最小)", meth, d,
                                          elapsed=timing["dijkstra_s"], flood_respected=bool(d["flood_respected"])))
            else:
                for seed in q["seeds"]:
                    prob.cache.clear()
                    if meth == "pso":
                        r = R.run_pso(prob, seed, budget)
                        label = f"標準 PSO (経由点 M={M}) seed {seed}"
                    else:
                        r = R.run_ccpso2(prob, seed, budget, gs)
                        label = f"CCPSO2 (経由点 M={M}, s={gs}) seed {seed}"
                    timing[f"{meth}_s"] += r["elapsed"]
                    det, _ = prob.decode(r["x"], detail=True)
                    routes.append(route_entry(f"lab_{meth}_s{seed}", label, meth, det, seed=seed, Mv=M,
                                              evals=int(r["evals"]), elapsed=r["elapsed"]))
        for meth in q["methods"]:
            cand = [e for e in routes if e["method"] == meth]
            if cand:
                min(cand, key=lambda e: e["cost"])["best"] = True

        timing["total_s"] = time.time() - t_all
        timing = {k: round(float(x), 3) for k, x in timing.items()}
        straight = float(np.hypot(*(prob.g_xy - prob.s_xy)))
        scenario = dict(self.scenario(), v_walk=v, start_delay_min=q["start_delay_min"], budget=budget, M=M,
                        seeds=q["seeds"], ccpso2_group_size=gs, methods=q["methods"])
        pair_out = dict(start=dict(name=s["name"], lat=s["lat"], lon=s["lon"], snap_m=round(float(pg.snap_m), 1)),
                        goal=dict(name=g["name"], lat=g["lat"], lon=g["lon"]),
                        straight_m=round(straight, 1), T_goal_min=round(prob.T_goal / 60, 2), sub_nodes=int(pg.N),
                        apsp_s=round(float(pg.apsp_s), 3))
        if "facility_id" in g:
            pair_out["goal"]["facility_id"] = g["facility_id"]
        return dict(ok=True, map=MAP, scenario=scenario, pair=pair_out, routes=routes, ref_cost=round(float(ref_cost), 1),
                    timing=timing, source=R.E_SOURCE)


# ── JSON (非有限値・numpy 型を JSON にできる形へ) ──────────────────────
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


# ── HTTP ─────────────────────────────────────────────────────────────
def make_handler(lab: Lab):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "route_server/1.0"

        def log_message(self, fmt, *a):   # 既定のアクセスログは出さない (1 リクエスト 1 行は自前で出す)
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

        def _json(self, status, obj):
            body = json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _path(self):
            return self.path.split("?", 1)[0]

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self._path() == "/api/info":
                self._json(200, lab.info())
            else:
                self._json(404, dict(ok=False, error=f"不明なパス: {self._path()}"))

        def do_POST(self):
            if self._path() != "/api/route":
                self._json(404, dict(ok=False, error=f"不明なパス: {self._path()}"))
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n).decode("utf-8") if n else "{}")
            except (ValueError, UnicodeDecodeError) as e:
                self._json(400, dict(ok=False, error=f"JSON を読めません: {e}"))
                print(f"[route] 400 JSON parse error: {e}", flush=True)
                return
            if not lab.lock.acquire(blocking=False):
                self._json(409, dict(ok=False, busy=True, error="別の経路計算を実行中です。終わってからやり直してください"))
                print("[route] 409 busy", flush=True)
                return
            try:
                q = lab.parse(req)
                res = lab.route(q)
            except BadRequest as e:
                self._json(400, dict(ok=False, error=str(e)))
                print(f"[route] 400 {e}", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._json(500, dict(ok=False, error=f"計算中にエラー: {type(e).__name__}: {e}"))
                print(f"[route] 500 {type(e).__name__}: {e}", flush=True)
                return
            finally:
                lab.lock.release()
            self._json(200, res)
            tm = res["timing"]
            print(f"[route] 200 {q['start']['name']} → {q['goal']['name']} M={q['M']} budget={q['budget']} "
                  f"seeds={q['seeds']} v={q['v_walk']} delay={q['start_delay_min']}min gs={q['ccpso2_group_size']} "
                  f"routes={len(res['routes'])} ref={res['ref_cost']} | graph {tm['graph_s']:.2f}s "
                  f"dijkstra {tm['dijkstra_s']:.2f}s pso {tm['pso_s']:.2f}s ccpso2 {tm['ccpso2_s']:.2f}s "
                  f"total {tm['total_s']:.2f}s", flush=True)

    return H


def main():
    ap = argparse.ArgumentParser(description="避難経路ラボ用ローカル HTTP API (GET /api/info, POST /api/route)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--walk", type=Path, default=HERE.parents[2] / "tizucra-walk",
                    help="tizucra-walk のパス (tools/profiles.py でブロック座標 x/z を付ける)")
    a = ap.parse_args(_ARGV[1:])
    lab = Lab(a.walk if (a.walk / "tools" / "profiles.py").is_file() else None)
    if lab.to_block is None:
        print(f"[route_server] x/z 無し (tizucra-walk が見つからない: {a.walk})")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(lab))
    srv.daemon_threads = True
    print(f"[route_server] listening on http://{a.host}:{a.port}  (GET /api/info, POST /api/route)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
