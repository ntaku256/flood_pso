"""S5 水面品質 (追加): 「1 m 階段率が減るのは定式化の効果か、探索器の差か」を測る。

s5_quality.py の C (CCPSO2 の解) と **同じ設定** で探索器だけを修正 PSO に替える。
  設定: HAND + depth∈[-4,20] (d0∈[0,16] + dd∈[-4,4])、K=16 (D=257)、実河道水源、λ=0、
        5,000 評価、seed 0/1/2
  D) 修正 PSO = pyswarms GlobalBestPSO、30 粒子、c1=c2=1.5、w=0.7、bh_strategy="nearest"、
     速度クランプ ±20% (vh_strategy="invert")。seeds21.py / hand_optimize20.py の PSO_clip と同じ。
     評価数は 30 × 166 = 4,980 (CCPSO2 は 20 × 17 × 14 = 4,760)。

あわせて決定的な参考値を 2 つ出す (探索なし。合計 10 秒程度):
  E) summary.md が「等方拡散」と書いている基準線 (面 IoU 0.7750) の水面品質。
     ⚠ 実体は downscale_baseline.py の 'iso' = **isolated-cell filter (孤立セル除去)** で、
        等方拡散ではない。Bryant et al. 2024 (CostGrow) 型の簡易版:
        25 m 粗セルの平均水面標高を最近傍で広げ → land_eff < WSE で切り → 粗図に連結する成分だけ残す。
  F) 同じ定式化で「探索しない」解。
     F1: dd=0 のまま d0 だけ 0.25 m 刻みで総当たり (= 一様な HAND 水深。1 次元)
     F2: 値域内の一様乱数 x (seed 0/1/2)。IoU は問わず、定式化が許す階段率の目安を見る。

再現性の確認として、s5_quality.json に保存済みの C の解 x を同じ格子で展開し直し、
指標が一致することを最初に確かめる (格子キャッシュを作り直しても同じ値になることの確認)。

出力: results/b05_detail/s5_quality_pso.json
実行: cd flood_pso && .venv/bin/python results/b05_detail/scripts/s5_quality_pso.py
      (3 シードを 3 プロセスで並列実行。前処理 約 1.5 分 + PSO 約 4〜5 分)
"""
import json
import os
import sys
import time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_detail import (CACHE, OUT, ST8, K_OPT, BUDGET, SEEDS,
                           load_grids, viz_metrics, HandObj, bounds_hand)

N_PART = 30
G = None  # fork で子プロセスに共有する格子


def depthfield_stair(o, x, mC, shape):
    """参考: HAND 水深場 tf 上の 1 m 階段 (WSE ではない。旧 e 表の定義)。"""
    H, W = shape
    _, tf = o.mask(x)
    tfl = np.full((H, W), np.nan)
    tfl[o.r0:o.r1, o.c0:o.c1] = tf
    Wb = np.where(mC, np.floor(tfl), np.nan)
    py = mC[:-1] & mC[1:]
    px = mC[:, :-1] & mC[:, 1:]
    st = int(np.count_nonzero((np.abs(np.diff(Wb, axis=0)) > 0.5) & py)
             + np.count_nonzero((np.abs(np.diff(Wb, axis=1)) > 0.5) & px))
    pr = int(np.count_nonzero(py) + np.count_nonzero(px))
    return st / pr if pr else 0.0


def metrics_of(o, x, name, verbose=True):
    g = G
    mC, wseC = o.expand(x, g)
    r = viz_metrics(name, mC, wseC, g["land"], g["valid"], g["gt"], verbose=verbose)
    r["stair_rate_on_depthfield"] = depthfield_stair(o, x, mC, g["land"].shape)
    r["iou_objective"] = float(o.iou(x))
    return r


def run_pso(sd):
    """修正 PSO 1 走 (seeds21.py の PSO_clip と同じ呼び出し)。"""
    os.chdir(CACHE)  # pyswarms が cwd に report.log を書くのでキャッシュ側へ逃がす
    from pyswarms.single import GlobalBestPSO
    t0 = time.time()
    g = G
    lb, ub, D = bounds_hand(K_OPT)
    o = HandObj(g, K_OPT, g["src_drain"], lam=0.0)

    def batch(X):
        return np.array([o(X[i]) for i in range(X.shape[0])])

    np.random.seed(sd)
    vc = 0.2 * (ub - lb)
    p = GlobalBestPSO(n_particles=N_PART, dimensions=D,
                      options={"c1": 1.5, "c2": 1.5, "w": 0.7}, bounds=(lb, ub),
                      bh_strategy="nearest", velocity_clamp=(-vc, vc),
                      vh_strategy="invert", ftol=-np.inf)
    best_cost, x = p.optimize(batch, iters=BUDGET // N_PART, verbose=False)
    n_evals = int(o.n_evals)
    r = metrics_of(o, x, f"D) 修正PSO seed={sd}", verbose=False)
    r.update(seed=sd, best_cost=float(best_cost), n_evals=n_evals,
             sec=float(time.time() - t0), x=np.asarray(x).tolist(),
             frac_on_bound=float(np.mean((np.asarray(x) <= lb + 1e-12) | (np.asarray(x) >= ub - 1e-12))))
    return r


def aggregate(per_seed, keys):
    agg = {}
    for k in keys:
        a = np.array([r[k] for r in per_seed], dtype=float)
        agg[k] = float(a.mean())
        agg[k + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return agg


KEYS = ("iou", "stair_rate", "ncomp", "nvals", "bad", "cells", "steps", "pairs",
        "stair_rate_on_depthfield")


def main():
    global G
    t0 = time.time()
    G = load_grids()
    g = G
    land, valid, gt = g["land"], g["valid"], g["gt"]
    H, W = land.shape
    lb, ub, D = bounds_hand(K_OPT)
    res = {"meta": dict(
        date=time.strftime("%Y-%m-%d"),
        setting="HAND + depth∈[-4,20] (d0∈[0,16], dd∈[-4,4]) / K=16 (D=257) / 実河道水源 / λ=0 / "
                "5000 評価 / seed 0-2 / 階段率は水面標高 WSE=tf+zd 上 (viz_metrics)",
        grid_shape=[int(H), int(W)], gt_cells=int(gt.sum()),
        src_drain_cells=int(g["src_drain"].sum()),
        pso="pyswarms GlobalBestPSO n=30 c1=c2=1.5 w=0.7 bh=nearest velocity_clamp=±0.2*(ub-lb) vh=invert "
            f"iters={BUDGET // N_PART} (= {N_PART * (BUDGET // N_PART)} 評価)",
        ccpso2="src/ccpso2.py n=20 group_size=16 (固定) p_cauchy=0.5 n_cycles=14 (= 4760 評価)")}
    print(f"[grids] {H}x{W}  GT {int(gt.sum()):,}  水源(実河道) {int(g['src_drain'].sum()):,}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── 0) 再現性: 保存済みの C の解を同じ格子で展開し直す ───────────────
    old = json.load(open(f"{OUT}/s5_quality.json"))
    chk = []
    for r0 in old["C_solution"]["per_seed"]:
        o = HandObj(g, K_OPT, g["src_drain"], lam=0.0)
        r = metrics_of(o, np.array(r0["x"]), f"C) CCPSO2 seed={r0['seed']} (保存解の再展開)")
        r["seed"] = r0["seed"]
        r["match_saved"] = bool(abs(r["iou"] - r0["iou"]) < 1e-9 and r["steps"] == r0["steps"]
                                and r["pairs"] == r0["pairs"] and r["ncomp"] == r0["ncomp"])
        chk.append(r)
    C = aggregate(chk, KEYS)
    C["name"] = "C) CCPSO2 の解 (s5_quality.json の保存解 x を再展開、seed 0-2 平均)"
    C["all_match_saved"] = bool(all(r["match_saved"] for r in chk))
    C["per_seed"] = chk
    res["C_ccpso2_reexpand"] = C
    print(f"   → 保存値と一致: {C['all_match_saved']}", flush=True)

    # ── D) 修正 PSO × 3 シード (並列) ────────────────────────────────
    t1 = time.time()
    print(f"[PSO] {len(SEEDS)} シードを並列実行 …", flush=True)
    with mp.get_context("fork").Pool(len(SEEDS)) as pool:
        per_seed = pool.map(run_pso, list(SEEDS))
    for r in per_seed:
        print(f"■ {r['name']}: IoU {r['iou']:.4f} / 1m階段 {r['steps']:,}/{r['pairs']:,} = "
              f"{100*r['stair_rate']:.2f}% (tf上 {100*r['stair_rate_on_depthfield']:.2f}%) / "
              f"連結成分 {r['ncomp']} / 水深値 {r['nvals']:,} / 違反 {r['bad']} / 浸水 {r['cells']:,} / "
              f"{r['n_evals']} 評価 {r['sec']:.0f}s / 境界張り付き {100*r['frac_on_bound']:.1f}%", flush=True)
    Dd = aggregate(per_seed, KEYS + ("n_evals", "sec", "frac_on_bound"))
    Dd["name"] = "D) 修正PSO の解 (実河道水源・λ=0, 4980評価, seed 0-2 平均)"
    Dd["per_seed"] = per_seed
    Dd["wall_sec_parallel"] = float(time.time() - t1)
    res["D_pso_clip"] = Dd

    # ── E) 「等方拡散」と書かれていた基準線 = Bryant 型簡易ダウンスケール (+孤立除去) ──
    from scipy.ndimage import distance_transform_edt, label as nd_label
    f = 5
    hh, ww = H // f, W // f
    blk = lambda a: a[:hh * f, :ww * f].reshape(hh, f, ww, f)
    wse_true = np.where(gt, land + np.maximum(g["gt_depth"], 0.0), 0.0)
    wcnt = blk(gt.astype(np.float64)).sum(axis=(1, 3))
    nval = blk(valid.astype(np.float64)).sum(axis=(1, 3))
    wse_c = np.divide(blk(wse_true).sum(axis=(1, 3)), np.maximum(wcnt, 1e-9))
    wet_c = (wcnt > 0) & (np.divide(wcnt, np.maximum(nval, 1)) >= 0.5)
    _, (iy, ix) = distance_transform_edt(~wet_c, return_indices=True)
    wf = np.full((H, W), np.nan)
    wf[:hh * f, :ww * f] = np.repeat(np.repeat(wse_c[iy, ix], f, axis=0), f, axis=1)
    wf = np.where(np.isnan(wf), np.nanmax(wse_c), wf)
    m0 = (g["land_eff"] < wf) & valid
    lab, _ = nd_label(m0, structure=ST8)
    seed = np.zeros((H, W), bool)
    seed[:hh * f, :ww * f] = np.repeat(np.repeat(wet_c, f, axis=0), f, axis=1)
    u = np.unique(lab[seed & m0]); u = u[u > 0]
    lut = np.zeros(lab.max() + 1, bool); lut[u] = True
    mE = lut[lab]
    E0 = viz_metrics("E0) 25m 粗図→5m (閾値のみ)", m0, wf, land, valid, gt)
    E = viz_metrics("E) 25m 粗図→5m (+孤立除去) ※summary.md の「等方拡散」", mE, wf, land, valid, gt)
    E["note"] = ("summary.md / outline_v2 が「等方拡散」と呼ぶ基準線 (面 IoU 0.7750) の実体。"
                 "downscale_baseline.py のキー 'iso' は isolated-cell filter (孤立セル除去) の略で、"
                 "等方拡散ではない。25 m 粗セルの平均水面標高を最近傍で広げ、land_eff<WSE で切り、"
                 "粗図に連結する成分だけを残す (Bryant et al. 2024 CostGrow 型の簡易版)。")
    res["E_downscale_isolated_filter_25m"] = E
    res["E0_downscale_threshold_only_25m"] = E0

    # ── F) 同じ定式化で「探索しない」解 ─────────────────────────────
    o = HandObj(g, K_OPT, g["src_drain"], lam=0.0)
    best = None
    for d0 in np.arange(0.0, 16.0 + 1e-9, 0.25):
        x = np.zeros(D); x[0] = d0
        v = o.iou(x)
        if best is None or v > best[0]:
            best = (v, d0)
    x = np.zeros(D); x[0] = best[1]
    F1 = metrics_of(o, x, f"F1) 探索なし: 一様な HAND 水深 d0={best[1]:.2f} m (dd=0、d0 のみ総当たり)")
    F1["d0"] = float(best[1])
    res["F1_uniform_hand_depth"] = F1
    rnd = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        x = rng.uniform(lb, ub)
        r = metrics_of(o, x, f"F2) 探索なし: 値域内の一様乱数 x seed={sd}")
        r["seed"] = sd
        rnd.append(r)
    F2 = aggregate(rnd, KEYS)
    F2["name"] = "F2) 探索なし: 値域内の一様乱数 x (seed 0-2 平均)"
    F2["per_seed"] = rnd
    res["F2_random_x"] = F2

    # ── 表 (A/B は s5_quality.json から転記) ─────────────────────────
    A, B = old["A_naive25m"], old["B_hand_oracle_K32"]
    rows = [("A 素朴拡大", A), ("B オラクル K=32", B), ("C CCPSO2", C), ("D 修正PSO", Dd),
            ("E 粗図DS+孤立除去", E), ("F1 一様HAND水深", F1), ("F2 乱数x", F2)]
    res["table"] = [dict(row=n, iou=float(r["iou"]), stair_rate=float(r["stair_rate"]),
                         ncomp=float(r["ncomp"]), nvals=float(r["nvals"]), bad=float(r["bad"]),
                         cells=float(r["cells"]),
                         stair_rate_sd=float(r.get("stair_rate_sd", 0.0)),
                         iou_sd=float(r.get("iou_sd", 0.0)),
                         ncomp_sd=float(r.get("ncomp_sd", 0.0))) for n, r in rows]
    res["meta"]["total_sec"] = float(time.time() - t0)

    print("\n| 行 | IoU | 1m階段率 | 連結成分 | 水深値の数 | 違反 | 浸水セル |", flush=True)
    print("|---|---|---|---|---|---|---|")
    for t in res["table"]:
        print(f"| {t['row']} | {t['iou']:.4f} | {100*t['stair_rate']:.2f}% | {t['ncomp']:.1f} | "
              f"{t['nvals']:,.0f} | {t['bad']:.0f} | {t['cells']:,.0f} |")
    json.dump(res, open(f"{OUT}/s5_quality_pso.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\nDONE {time.time()-t0:.0f}s  -> {OUT}/s5_quality_pso.json", flush=True)


if __name__ == "__main__":
    main()
