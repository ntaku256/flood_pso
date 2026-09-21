"""fig_calib21.png: HAND + depth∈[0,20] 21 シード比較 (修正PSO / CCPSO2 現行 / CCPSO2 忠実版)。"""
import json, numpy as np
from scipy import stats
from common import *

d = json.load(open(f"{RESCUE}/seeds21_all.json"))
names = {"PSO_clip": "修正PSO", "CC_current": "CCPSO2 (現行)", "CC_faithful": "CCPSO2 (忠実版)"}
arr = {names[k]: np.asarray(v, float) for k, v in d.items()}
for k, v in arr.items():
    assert v.shape == (21,), (k, v.shape)

pso, cc, ccf = arr["修正PSO"], arr["CCPSO2 (現行)"], arr["CCPSO2 (忠実版)"]
wins = int((cc > pso).sum()); losses = int((cc < pso).sum())
u = stats.mannwhitneyu(cc, pso, alternative="two-sided")
u1 = stats.mannwhitneyu(cc, pso, alternative="greater")
w = stats.wilcoxon(cc, pso, alternative="two-sided")
A12 = float(((cc[:, None] > pso[None, :]).sum() + 0.5 * (cc[:, None] == pso[None, :]).sum()) / (21 * 21))
summary = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), median=float(np.median(v)),
                   min=float(v.min()), max=float(v.max())) for k, v in arr.items()}
summary["CCPSO2現行 vs 修正PSO"] = dict(wins=wins, losses=losses, ties=21 - wins - losses,
    mannwhitney_two_sided_p=float(u.pvalue), mannwhitney_greater_p=float(u1.pvalue),
    wilcoxon_paired_p=float(w.pvalue), A12=A12,
    faithful_vs_current_wins=int((ccf > cc).sum()))
json.dump(summary, open(f"{OUT}/fig_calib21_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))

fig, ax = plt.subplots(figsize=(8.4, 4.6))
order = ["修正PSO", "CCPSO2 (現行)", "CCPSO2 (忠実版)"]
rng = np.random.default_rng(0)
bp = ax.boxplot([arr[k] for k in order], positions=range(3), widths=0.42, showfliers=False,
                patch_artist=True, medianprops=dict(color=C["ink"], lw=1.6),
                whiskerprops=dict(color=C["muted"]), capprops=dict(color=C["muted"]),
                boxprops=dict(facecolor="white", edgecolor=C["muted"]), zorder=2)
for i, k in enumerate(order):
    x = i + rng.uniform(-0.13, 0.13, 21)
    ax.scatter(x, arr[k], s=34, color=METHOD_COLOR[k], edgecolor="white", linewidth=0.8, zorder=3)
    m, s = summary[k]["mean"], summary[k]["sd"]
    ax.text(i, 0.7215, f"{m:.4f} ± {s:.4f}", ha="center", va="bottom", fontsize=10, color=C["ink2"])
ax.set_xticks(range(3)); ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel("IoU (最大浸水域, HAND 基準・depth∈[0,20] m)")
ax.set_ylim(0.650, 0.729)
ax.set_title("21 シード比較: 各手法 5000 評価・K=16 (上の数値は平均 ± 標準偏差)", fontsize=11.5, loc="left")
txt = (f"CCPSO2 (現行) vs 修正PSO: {wins} 勝 {losses} 敗 (同一シード対)\n"
       f"Wilcoxon 符号順位 (対応あり) p = {w.pvalue:.1e},  A12 = {A12:.3f}")
ax.text(0.99, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f5f4f1", ec="none"))
ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/fig_calib21.png")
print("saved")
