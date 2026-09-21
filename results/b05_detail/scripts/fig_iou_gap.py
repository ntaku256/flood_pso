"""fig_iou_gap.png — IoU ギャップの 5 分解 (積み上げ棒)。

実測 (CCPSO2 / 絶対標高[1,10] / bbox水源 / K=16 / 5000評価) から IoU=1 までの
0.3796 を、ブロック単位オラクルの階層で 5 段に分ける。
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import OUT, C, plt

S = json.load(open(f"{OUT}/summary.json"))["c_gap"]
base = S["base_iou"]
st = S["stages"]

COLORS = [C["blue"], C["aqua"], C["yellow"], C["violet"], C["muted"]]
SHORT = ["探索", "パラメータ化", "値域", "解像度", "到達不能"]

X0 = 0.55
fig, ax = plt.subplots(figsize=(10.5, 3.4))
ax.barh(0, base - X0, left=X0, color=C["ink"], height=0.60)
ax.text((X0 + base) / 2, 0, f"実測\n{base:.4f}", ha="center", va="center",
        color="white", fontsize=10, fontweight="bold")
left = base
for (s, col, sh) in zip(st, COLORS, SHORT):
    w = s["width"]
    ax.barh(0, w, left=left, color=col, height=0.60,
            edgecolor="white", linewidth=1.2)
    if w < 0.03:            # 細い帯はラベルが入らないので下に出す
        ax.plot([left + w / 2, left + w / 2], [-0.30, -0.05], color=col, lw=1.2)
        ax.text(left + w / 2, -0.33, f"{sh} {w:+.4f} ({100*s['share']:.1f}%)",
                ha="center", va="top", fontsize=9.5, color=C["ink"])
    else:
        ax.text(left + w / 2, 0, f"{sh}\n{w:+.4f}\n{100*s['share']:.1f}%",
                ha="center", va="center", fontsize=9.5,
                color="white" if sh in ("探索", "解像度", "到達不能") else C["ink"])
    left += w

for i, s in enumerate(st[:-1]):
    y = 0.44 if i % 2 == 0 else 0.62
    ax.plot([s["reach"], s["reach"]], [0.30, y], color=C["muted"], lw=0.8, ls=":")
    ax.text(s["reach"], y + 0.02, f"{s['reach']:.4f}", ha="center", va="bottom",
            fontsize=8.5, color=C["ink2"])

ax.set_xlim(X0, 1.005)
ax.set_ylim(-0.72, 0.95)
ax.set_yticks([])
ax.set_xlabel("IoU (公式 25m 想定図との面の一致)")
ax.set_title("IoU ギャップの 5 分解 — 到達不能が 41.2% で最大、探索が 24.4%\n"
             "(オラクル = 各ブロックで誤分類最小の閾値を選んだ上限。いずれも連結性制約なし)",
             fontsize=12)
ax.grid(axis="x", color="#e4e3df")
ax.set_axisbelow(True)
fig.savefig(f"{OUT}/fig_iou_gap.png", dpi=170, bbox_inches="tight")
print("wrote fig_iou_gap.png")
