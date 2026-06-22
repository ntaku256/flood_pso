"""LSGO 標準ベンチで CCPSO2 vs 標準PSO を公平比較（最適化器そのものの強さを示す）。
自作の洪水ベンチと違い、コミュニティ標準の高次元テスト関数で評価する。
env: LSGO_DIMS, LSGO_BUDGET, LSGO_SEEDS, LSGO_FUNCS, LSGO_TAG。一時スクリプト。"""
import sys, os, json, time, collections
sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ccpso2 import CCPSO2
from pyswarms.single import GlobalBestPSO

# ── 標準テスト関数（最小化, global min = 0）。最適点を +0.3*ub にシフトし原点バイアスを排除 ──
FUNC_BOUNDS = {"sphere": 100.0, "rosenbrock": 30.0, "rastrigin": 5.12, "ackley": 32.0, "griewank": 600.0}
FUNC_KIND   = {"sphere": "separable/unimodal", "rosenbrock": "nonsep/unimodal",
               "rastrigin": "separable/multimodal", "ackley": "nonsep/multimodal", "griewank": "nonsep/multimodal"}

def make_func(name, D, ub):
    o = 0.3 * ub * np.ones(D)            # 最適点シフト
    if name == "sphere":
        return lambda x: float(np.sum((x - o) ** 2))
    if name == "rosenbrock":             # 最適 z=1 → x=o+1
        def f(x):
            z = x - o
            return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))
        return f
    if name == "rastrigin":
        def f(x):
            z = x - o
            return float(10 * z.size + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z)))
        return f
    if name == "ackley":
        def f(x):
            z = x - o
            return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2)))
                         - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + np.e)
        return f
    if name == "griewank":
        def f(x):
            z = x - o
            return float(np.sum(z ** 2) / 4000.0
                         - np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1)))) + 1.0)
        return f
    raise ValueError(name)

def group_size(D):
    if D <= 100: return 10
    if D <= 500: return 25
    return 50

DIMS    = [int(x) for x in os.environ.get("LSGO_DIMS", "100,500,1000").split(",")]
BUDGET  = int(os.environ.get("LSGO_BUDGET", "100000"))
PER_DIM = int(os.environ.get("LSGO_PER_DIM", "0"))   # >0 なら budget = PER_DIM*D（次元比例で公平に）
SEEDS   = [int(x) for x in os.environ.get("LSGO_SEEDS", "0,1,2").split(",")]
FUNCS   = os.environ.get("LSGO_FUNCS", "sphere,rosenbrock,rastrigin,ackley,griewank").split(",")
TAG     = os.environ.get("LSGO_TAG", "")
OUTJ    = f"results/benchmark/lsgo_benchmark{TAG}.json"
OUTP    = f"results/benchmark/lsgo_benchmark{TAG}.png"

class _LeanPSO(GlobalBestPSO):
    """pyswarms 標準 global-best PSO のまま、位置/速度履歴の保存だけ無効化（高次元×多反復の OOM 回避）。
    最適化ロジックは pyswarms そのものなので公平な強い baseline。"""
    def _populate_history(self, hist):
        self.cost_history.append(hist.best_cost)


def run_pso(f, D, ub, budget, seed):
    P = 50
    iters = max(1, budget // P)
    lb = np.full(D, -ub); hb = np.full(D, ub)
    def fb(X): return np.array([f(X[i]) for i in range(X.shape[0])])
    np.random.seed(seed)
    opt = _LeanPSO(n_particles=P, dimensions=D,
                   options={"c1": 1.49, "c2": 1.49, "w": 0.72},
                   bounds=(lb, hb), ftol=-np.inf)
    best, _ = opt.optimize(fb, iters=iters, verbose=False)
    return float(best), P * iters

def run_ccpso2(f, D, ub, budget, seed):
    # Li & Yao 2012 の適応的グループサイズ（候補集合 S から改善停滞時に選び直す）
    S = [s for s in (2, 5, 10, 25, 50, 100, 250) if s <= D] or [min(2, D)]
    N = 20
    lb = np.full(D, -ub); hb = np.full(D, ub)
    cc = CCPSO2(f, dim=D, n_particles=N, group_size=S[0], bounds=(lb, hb),
                p_cauchy=0.5, seed=seed, verbose=False, group_sizes=S)
    while cc.n_evals < budget:   # 評価数でバジェット厳密化（s が変わっても公平）
        cc.step()
    return float(cc.b_cost), cc.n_evals

print(f"[LSGO] dims={DIMS} budget={BUDGET} seeds={SEEDS} funcs={FUNCS}", flush=True)
rows = []
for fn in FUNCS:
    ub = FUNC_BOUNDS[fn]
    for D in DIMS:
        bud = PER_DIM * D if PER_DIM > 0 else BUDGET
        for sd in SEEDS:
            f = make_func(fn, D, ub)
            t0 = time.time()
            pso_best, pso_ev = run_pso(f, D, ub, bud, sd)
            cc_best, cc_ev = run_ccpso2(f, D, ub, bud, sd)
            rows.append({"func": fn, "D": D, "seed": sd, "budget": bud,
                         "pso": pso_best, "ccpso2": cc_best, "pso_ev": pso_ev, "cc_ev": cc_ev})
            print(f"  {fn:<10} D={D:<5} seed={sd}: PSO={pso_best:.4g}  CCPSO2={cc_best:.4g}  "
                  f"winner={'CCPSO2' if cc_best < pso_best else 'PSO'}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump({"dims": DIMS, "budget": BUDGET, "seeds": SEEDS, "funcs": FUNCS, "rows": rows},
                      open(OUTJ, "w"), indent=2)

# 集計（seed 中央値）＋表
agg = collections.defaultdict(dict)
for fn in FUNCS:
    for D in DIMS:
        sub = [r for r in rows if r["func"] == fn and r["D"] == D]
        agg[fn][D] = (float(np.median([r["pso"] for r in sub])), float(np.median([r["ccpso2"] for r in sub])))
print("\nfunc        D     | PSO(med)     CCPSO2(med)  | winner  ratio(PSO/CCP)", flush=True)
for fn in FUNCS:
    for D in DIMS:
        p, c = agg[fn][D]
        w = "CCPSO2" if c < p else "PSO"
        ratio = p / c if c > 0 else float("inf")
        print(f"{fn:<11} {D:<5} | {p:11.4g} {c:11.4g}  | {w:<7} {ratio:.2g}x", flush=True)

# プロット（関数ごとに D vs best、log軸）
ncol = min(3, len(FUNCS)); nrow = (len(FUNCS) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
for idx, fn in enumerate(FUNCS):
    ax = axes[idx // ncol][idx % ncol]
    ps = [agg[fn][D][0] for D in DIMS]; cs = [agg[fn][D][1] for D in DIMS]
    ax.plot(DIMS, ps, "o-", label="Standard PSO")
    ax.plot(DIMS, cs, "s-", label="CCPSO2")
    ax.set_yscale("log"); ax.set_xlabel("dimension D"); ax.set_ylabel("best cost (median, log)")
    ax.set_title(f"{fn} ({FUNC_KIND[fn]})"); ax.legend(); ax.grid(alpha=.3, which="both")
for j in range(len(FUNCS), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
fig.suptitle(f"LSGO standard benchmark: CCPSO2 vs Standard PSO (budget={BUDGET} FEs/run)", fontsize=13)
fig.tight_layout(); fig.savefig(OUTP, dpi=120)
print(f"\nsaved {OUTJ}\nsaved {OUTP}", flush=True)
