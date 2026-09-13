"""fig_holdout.png: 市松 holdout の train/test IoU (CCPSO2 vs 修正PSO, 3 シード)。"""
import json, numpy as np
from common import *

d = json.load(open(f"{RESCUE}/holdout.json"))
methods = ["CCPSO2", "修正PSO"]
cols = {"CCPSO2": C["blue"], "修正PSO": C["orange"]}
stat = {}
for m in methods:
    for split in ["train", "test", "full"]:
        v = np.asarray(d[m][split], float); assert v.shape == (3,)
        stat[(m, split)] = (float(v.mean()), float(v.std(ddof=1)), v)
summary = {f"{m}/{s}": dict(mean=stat[(m, s)][0], sd=stat[(m, s)][1], seeds=stat[(m, s)][2].tolist())
           for m in methods for s in ["train", "test", "full"]}
for m in methods:
    summary[f"{m}/gap(train-test)"] = stat[(m, "train")][0] - stat[(m, "test")][0]
    summary[f"{m}/paired test wins vs other"] = int(sum(
        stat[(m, "test")][2] > stat[(methods[1 - methods.index(m)], "test")][2]))
json.dump(summary, open(f"{OUT}/fig_holdout_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))

fig, ax = plt.subplots(figsize=(7.2, 4.6))
splits = ["train", "test"]; labels = {"train": "train (最適化に使った市松ブロック)", "test": "test (未使用の市松ブロック)"}
off = 0.18
for j, m in enumerate(methods):
    xs = np.arange(2) + (j - 0.5) * 2 * off
    for x, s in zip(xs, splits):
        mu, sd, v = stat[(m, s)]
        ax.hlines(mu, x - 0.11, x + 0.11, color=cols[m], lw=3, zorder=3, label=m if s == "train" else None)
        ax.scatter(np.full(3, x) + np.array([-0.04, 0, 0.04]), v, s=30, color=cols[m], alpha=0.55, zorder=2, edgecolor="white", linewidth=0.6)
        ax.text(x + 0.13, mu, f"{mu:.4f}", ha="left", va="center", fontsize=9.5, color=C["ink2"])
ax.set_xticks(range(2)); ax.set_xticklabels([labels[s] for s in splits])
ax.set_xlim(-0.55, 1.55); ax.set_ylim(0.60, 0.735); ax.set_ylabel("IoU (最大浸水域)"); ax.grid(axis="x", visible=False)
ax.set_title("市松 holdout: 片側の市松ブロックで最適化し、他方で評価 (3 シード, 太線=平均)", fontsize=11, loc="left")
ax.legend(frameon=False, loc="upper right")
g1 = summary["CCPSO2/gap(train-test)"]; g2 = summary["修正PSO/gap(train-test)"]
fig.text(0.125, -0.02, f"汎化ギャップ (train − test): CCPSO2 {g1:+.4f} / 修正PSO {g2:+.4f}\n"
        f"train は修正PSO が高く、test は CCPSO2 が高い → CCPSO2 の方が汎化 (test で CCPSO2 が上回ったのは 3 シード中 2)",
        fontsize=9.5, va="top", ha="left", color=C["ink2"])
fig.savefig(f"{OUT}/fig_holdout.png")
print("saved")
