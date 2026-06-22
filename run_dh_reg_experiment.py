"""平滑化正則化 λ スイープ実験：loss + λ·roughness(Δh) が Δh RMSE と IoU に与える影響を、
Standard PSO と CCPSO2 で比較。評価(IoU/Δh RMSE)は素のまま（benchmark が再計算）。
env: EXP_K, EXP_BUDGET, EXP_SEEDS(comma), EXP_LAMBDAS(comma)。一時スクリプト。"""
import sys, os, json, time, collections
sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import benchmark as B

K       = int(os.environ.get("EXP_K", "16"))
BUDGET  = int(os.environ.get("EXP_BUDGET", "5000"))
SEEDS   = [int(s) for s in os.environ.get("EXP_SEEDS", "0").split(",")]
LAMBDAS = [float(x) for x in os.environ.get("EXP_LAMBDAS", "0,0.02,0.05,0.1,0.2").split(",")]
TAG     = os.environ.get("EXP_TAG", "")
OUTJ    = f"results/benchmark/dh_reg_experiment{TAG}.json"
OUTP    = f"results/benchmark/dh_reg_experiment{TAG}.png"

print("[setup] loading DEM ...", flush=True)
di = B.downsample(B.mosaic_tiles(str(B.DEM_DIR)), B.DS_FACTOR)
dem = di["dem"]
source = B.make_river_source(dem, lat_max=di["lat_max"], res_lat=di["res_lat"],
                             lon_min=di["lon_min"], res_lon=di["res_lon"],
                             river_bbox=B.HIDAKA_RIVER_BBOX, elev_max=B.RIVER_ELEV_MAX)
print(f"  DEM {dem.shape} src {int(source.sum())} | K={K} budget={BUDGET} seeds={SEEDS} lambdas={LAMBDAS}", flush=True)

rows = []
for lam in LAMBDAS:
    for sd in SEEDS:
        t0 = time.time()
        c = B.run_one_case(dem, source, K=K, budget=BUDGET, seed=sd, dh_tv=lam)
        rows.append({"lambda": lam, "seed": sd,
                     "pso_iou": c["pso"]["iou"], "pso_rmse": c["pso"]["dh_rmse"],
                     "cc_iou": c["ccpso2"]["iou"], "cc_rmse": c["ccpso2"]["dh_rmse"]})
        print(f"  [λ={lam:<5} seed={sd}] PSO iou={c['pso']['iou']:.4f} rmse={c['pso']['dh_rmse']:.3f} | "
              f"CCP iou={c['ccpso2']['iou']:.4f} rmse={c['ccpso2']['dh_rmse']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"K": K, "budget": BUDGET, "seeds": SEEDS, "lambdas": LAMBDAS, "rows": rows},
                  open(OUTJ, "w"), indent=2)

agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for k in ("pso_iou", "pso_rmse", "cc_iou", "cc_rmse"):
        agg[r["lambda"]][k].append(r[k])
lams = sorted(agg)
def m(l, k): return float(np.mean(agg[l][k]))
print("\nλ      | PSO_rmse CCP_rmse | PSO_iou  CCP_iou", flush=True)
for l in lams:
    print(f"{l:<6} | {m(l,'pso_rmse'):8.3f} {m(l,'cc_rmse'):8.3f} | {m(l,'pso_iou'):7.4f} {m(l,'cc_iou'):7.4f}", flush=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(lams, [m(l,"pso_rmse") for l in lams], "o-", label="Standard PSO")
a1.plot(lams, [m(l,"cc_rmse") for l in lams], "s-", label="CCPSO2")
a1.set_xlabel("lambda (smoothness reg)"); a1.set_ylabel("dh RMSE (lower = closer to truth)")
a1.set_title(f"Parameter recovery vs lambda (K={K}, D={1+K*K})"); a1.legend(); a1.grid(alpha=.3)
a2.plot(lams, [m(l,"pso_iou") for l in lams], "o-", label="Standard PSO")
a2.plot(lams, [m(l,"cc_iou") for l in lams], "s-", label="CCPSO2")
a2.set_xlabel("lambda (smoothness reg)"); a2.set_ylabel("IoU (higher = better fit)")
a2.set_title(f"Observation fit vs lambda (K={K})"); a2.legend(); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUTP, dpi=120)
print(f"\nsaved {OUTJ}\nsaved {OUTP}", flush=True)
