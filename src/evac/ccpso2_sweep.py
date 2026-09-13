"""evac/ccpso2_sweep.py — CCPSO2 の分解設定 (s, N, p_cauchy) 感度。ペア naya, 5000 評価, seed 0/1。
   結果 results/evac/ccpso2_group_sweep.csv。evac_ccpso2.py の CC_N/CC_S 決定根拠。"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evac_ccpso2 as E  # noqa: E402
from ccpso2 import CCPSO2  # noqa: E402

CONFIGS = [dict(N=20, s=8), dict(N=20, s=4), dict(N=20, s=2), dict(N=20, s=1), dict(N=20, s=40),
           dict(N=10, s=8), dict(N=30, s=8), dict(N=10, s=2), dict(N=5, s=2), dict(N=10, s=1),
           dict(N=20, gs="2,5,10"), dict(N=10, gs="1,2,4"),
           dict(N=20, s=8, pc=0.0), dict(N=20, s=8, pc=1.0), dict(N=20, s=2, pc=0.0), dict(N=10, s=2, pc=0.0)]


def main():
    inp = E.load_inputs(); grid = inp["grid"]; terr = E.Terrain(inp)
    pair = E.PAIRS[0]
    sx, sy = grid.latlon_to_xy(pair["start"]["lat"], pair["start"]["lon"])
    gx, gy = grid.latlon_to_xy(pair["goal"]["lat"], pair["goal"]["lon"])
    prob = E.RouteProblem(terr, (sx, sy), (gx, gy))
    rows = []
    for cfg in CONFIGS:
        cs = []
        for seed in (0, 1):
            prob.reset_log()
            kw = dict(dim=E.D, n_particles=cfg["N"], group_size=cfg.get("s", 8), bounds=(prob.lb, prob.ub),
                      p_cauchy=cfg.get("pc", 0.5), seed=seed)
            if "gs" in cfg:
                kw["group_sizes"] = [int(v) for v in cfg["gs"].split(",")]
            t0 = time.time()
            r = CCPSO2(prob, **kw).run(max_evals=E.BUDGET)
            cs.append(r["best_cost"])
            rows.append(dict(pair=pair["id"], N=cfg["N"], s=cfg.get("s", ""), group_sizes=cfg.get("gs", ""), p_cauchy=cfg.get("pc", 0.5),
                             seed=seed, cost=round(r["best_cost"], 1), evals=r["n_evals"], cycles=len(r["history"]) - 1, elapsed_s=round(time.time() - t0, 1)))
        print(cfg, [f"{c:.0f}" for c in cs], flush=True)
    with open(E.OUT / "ccpso2_group_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
