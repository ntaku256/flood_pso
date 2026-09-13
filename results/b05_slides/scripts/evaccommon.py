"""避難経路まとめ図表の共通データロード (results/evac/* を読み、参照比を再計算する)。"""
import csv
import json

import numpy as np

from common import *  # noqa: F401,F403  (フォント・配色・OUT/FP)

EV = f"{FP}/results/evac"
PAIRS = ["naya", "sono", "center"]
PAIR_JA = {"naya": "名屋", "sono": "薗", "center": "市街中心"}
SEEDS = ["0", "1", "2", "3", "4"]
M_COLOR = {"PSO": C["orange"], "CCPSO2": C["blue"]}
PAIR_MARK = {"naya": "o", "sono": "s", "center": "^"}


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _agg(per_run, ref):
    """per_run[pair][method][seed] = cost → 参照比の中央値・ペア別中央値・勝敗。"""
    out = {}
    for m in ("PSO", "CCPSO2"):
        ratios, per_pair, costs = [], {}, {}
        for p in PAIRS:
            v = [per_run[p][m][s] / ref[p] for s in SEEDS]
            per_pair[p] = float(np.median(v))
            costs[p] = float(np.median([per_run[p][m][s] for s in SEEDS]))
            ratios += v
        out[m] = dict(ratios=[float(x) for x in ratios], ratio_median=float(np.median(ratios)),
                      ratio_per_pair=per_pair, cost_median_per_pair=costs,
                      ratio_min=float(min(ratios)), ratio_max=float(max(ratios)))
    w = l = t = 0
    for p in PAIRS:
        for s in SEEDS:
            a, b = per_run[p]["PSO"][s], per_run[p]["CCPSO2"][s]
            if b < a:
                w += 1
            elif b > a:
                l += 1
            else:
                t += 1
    out["ccpso2_wins"], out["ccpso2_losses"], out["ties"] = w, l, t
    out["ref"] = {p: float(ref[p]) for p in PAIRS}
    out["per_run"] = {p: {m: {s: float(per_run[p][m][s]) for s in SEEDS} for m in ("PSO", "CCPSO2")} for p in PAIRS}
    return out


def load_spline(budget):
    """連続スプライン定式化 (K=20 中継点, D=40)。参照 = 時間依存 Dijkstra の 20 点スプライン化。"""
    suf = "" if budget == 5000 else f"_b{budget}"
    rs = _rows(f"{EV}/table_pso_vs_ccpso2{suf}.csv")
    sm = json.load(open(f"{EV}/summary_pso_vs_ccpso2{suf}.json", encoding="utf-8"))
    ref = {r["pair"]: float(r["cost"]) for r in rs if r["method"] == "dijkstra_spline20"}
    ref_poly = {r["pair"]: float(r["cost"]) for r in rs if r["method"] == "dijkstra"}
    per_run = {p: {"PSO": {}, "CCPSO2": {}} for p in PAIRS}
    for r in rs:
        if r["seed"] == "" or r["method"] not in ("pso", "ccpso2"):
            continue
        per_run[r["pair"]]["PSO" if r["method"] == "pso" else "CCPSO2"][r["seed"]] = float(r["cost"])
    a = _agg(per_run, ref)
    a.update(budget=sm["budget"], D=sm["D"], K=sm["K"], lam=sm["lam"], seeds=sm["seeds"],
             ref_polyline={p: float(v) for p, v in ref_poly.items()},
             ref_kind="時間依存 Dijkstra 経路を 20 中継点スプラインに再符号化して同一コスト関数で評価")
    return a


def load_roadpso(M):
    """道路制約 via-point 定式化 (M 経由点, D=2M)。参照 = 時間依存 Dijkstra (道路グラフ上の厳密解)。"""
    rs = _rows(f"{EV}/roadpso/table_roadpso.csv")
    sm = json.load(open(f"{EV}/roadpso/summary_roadpso.json", encoding="utf-8"))
    ref = {r["pair"]: float(r["cost"]) for r in rs if r["method"] == "dijkstra_timedep"}
    ref_static = {r["pair"]: float(r["cost"]) for r in rs if r["method"] == "dijkstra_static"}
    per_run = {p: {"PSO": {}, "CCPSO2": {}} for p in PAIRS}
    for r in rs:
        if r["seed"] == "" or r["M"] != str(M) or r["method"] not in ("pso", "ccpso2"):
            continue
        per_run[r["pair"]]["PSO" if r["method"] == "pso" else "CCPSO2"][r["seed"]] = float(r["cost"])
    a = _agg(per_run, ref)
    a.update(budget=sm["budget"], D=2 * M, M=M, lam=sm["lam"], seeds=sm["seeds"],
             ccpso2_group_size=sm["pairs"]["naya"][f"M{M}"]["ccpso2_group_size"],
             ref_static={p: float(v) for p, v in ref_static.items()},
             ref_kind="道路グラフ上の時間依存 (最早到着) Dijkstra = この定式化の厳密解",
             reach_ref={p: dict(pso=sm["pairs"][p][f"M{M}"]["pso_reach_ref"],
                                ccpso2=sm["pairs"][p][f"M{M}"]["ccpso2_reach_ref"]) for p in PAIRS})
    return a


# 講演原稿 3 章 (b05.typ l.99, 表 1): 合成地形・20 中継点 = 40 次元。単一値 (seed 数・評価数の記載なし)
PAPER = dict(pso=1702.32, ccpso2=312.69, ref=309.63)
PAPER["pso_ratio"] = PAPER["pso"] / PAPER["ref"]
PAPER["ccpso2_ratio"] = PAPER["ccpso2"] / PAPER["ref"]


def load_dijkstra():
    return json.load(open(f"{EV}/dijkstra_stats.json", encoding="utf-8"))
