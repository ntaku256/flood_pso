"""fig_bounds_param.png — 値域・パラメータ化を変えたときの CCPSO2 vs 修正PSO。

K=16 / 5000 評価 / 3 シード / 高速目的関数。
物理的に不当な「絶対標高の広い箱」だけ CCPSO2 の優位が消える。
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import OUT, C, plt

rows = json.load(open(f"{OUT}/summary.json"))["d_bounds"]
lab = [r["cond"].split("(")[0].strip() + f"\n幅 {r['width_m']:.0f} m"
       + ("" if r["physically_valid"] else "  ✕物理的に不当") for r in rows]
cc = [r["ccpso2"] for r in rows]
ccsd = [r["ccpso2_sd"] for r in rows]
ps = [r["pso_clip"] for r in rows]
pssd = [r["pso_clip_sd"] for r in rows]
orc = [r["oracle_K16"] for r in rows]

x = np.arange(len(rows))
w = 0.36
fig, ax = plt.subplots(figsize=(11.5, 4.8))
b1 = ax.bar(x - w / 2, cc, w, yerr=ccsd, capsize=4, color=C["blue"], label="CCPSO2")
b2 = ax.bar(x + w / 2, ps, w, yerr=pssd, capsize=4, color=C["orange"], label="修正PSO (clip+速度クランプ)")
for i, o in enumerate(orc):
    ax.hlines(o, i - 0.46, i + 0.46, color=C["muted"], lw=1.6, ls="--",
              zorder=5, label="ブロックオラクル K=16" if i == 0 else None)
for i, r in enumerate(rows):
    ax.text(i, max(cc[i], ps[i]) + max(ccsd[i], pssd[i]) + 0.013,
            f"{r['advantage']:+.4f}\n{r['win']}勝{r['lose']}敗", ha="center",
            fontsize=9.5, color=C["ink"] if r["advantage"] > 0.01 else C["red"])
for xi, v in zip(x - w / 2, cc):
    ax.text(xi, 0.5625, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, color="white")
for xi, v in zip(x + w / 2, ps):
    ax.text(xi, 0.5625, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, color="white")

ax.set_axisbelow(True)
ax.set_xticks(x)
ax.set_xticklabels(lab, fontsize=9.5)
ax.set_ylim(0.55, 0.90)
ax.set_ylabel("IoU (3 シード平均 ± sd)")
ax.set_title("値域とパラメータ化が探索器の優劣を決める — K=16 / 5000 評価 / 3 シード\n"
             "HAND では値域を広げるほど CCPSO2 の優位が増え、絶対標高の不当な箱では優位が消える",
             fontsize=12)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
fig.savefig(f"{OUT}/fig_bounds_param.png", dpi=170, bbox_inches="tight")
print("wrote fig_bounds_param.png")
