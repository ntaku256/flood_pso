"""fig_evac_summary: 左 = 定式化 × 手法の参照比 (log), 右 = 最遅出発 Dijkstra の避難困難区域面積。"""
import json

import numpy as np

from evaccommon import *

S5, S20, R5, R20 = load_spline(5000), load_spline(20000), load_roadpso(5), load_roadpso(20)
DJ = load_dijkstra()

GROUPS = [
    ("原稿 3 章\n合成ベンチ\nD=40 (予算 記載なし)", None),
    ("御坊 スプライン\n自由平面\nD=40 (5k)", S5),
    ("御坊 スプライン\n自由平面\nD=40 (20k)", S20),
    ("御坊 道路制約\nvia-point\nD=10 (5k)", R5),
    ("御坊 道路制約\nvia-point\nD=40 (5k)", R20),
]

fig = plt.figure(figsize=(15.4, 5.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.19)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ---------------- 左: 参照比 ----------------
rng = np.random.default_rng(0)
OFF = {"PSO": -0.185, "CCPSO2": 0.185}
ax.axvspan(2.5, 3.5, color="#f4f8f4", zorder=0)
ax.axhspan(1.0, 1.2, color="#dcefe0", zorder=0)
ax.axhline(1.0, color=C["green"], lw=1.3, ls="--", zorder=1)
for gi, (name, a) in enumerate(GROUPS):
    for m in ("PSO", "CCPSO2"):
        x0 = gi + OFF[m]
        if a is None:
            vals = {"naya": [PAPER["pso_ratio"] if m == "PSO" else PAPER["ccpso2_ratio"]]}
            med = vals["naya"][0]
        else:
            vals = {p: [a["per_run"][p][m][s] / a["ref"][p] for s in SEEDS] for p in PAIRS}
            med = a[m]["ratio_median"]
        for p, v in vals.items():
            v = np.asarray(v)
            jx = x0 + rng.uniform(-0.085, 0.085, v.size)
            ax.scatter(jx, v, s=40, marker=PAIR_MARK[p], color=M_COLOR[m], alpha=0.72,
                       edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot([x0 - 0.16, x0 + 0.16], [med, med], color=C["ink"], lw=2.6, zorder=4,
                solid_capstyle="butt")
        lab = f"{med:.2f}" if med < 10 else (f"{med:.1f}" if med < 100 else f"{med:.0f}")
        ax.annotate(lab, (x0, med), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=9.5, color=C["ink"], fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82))
for gi in range(1, len(GROUPS)):
    ax.axvline(gi - 0.5, color="#e4e3df", lw=1.0, zorder=0)
ax.set_yscale("log")
ax.set_ylim(0.82, 2200)
ax.set_xlim(-0.55, len(GROUPS) - 0.45)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g[0] for g in GROUPS], fontsize=10)
ax.set_ylabel("参照解に対するコスト比 (対数軸)\n1.0 = Dijkstra 参照/最適解と同着")
ax.set_title("(a) 定式化 × 手法 の参照比 — 点 = 3 ペア × 5 seed の各実行, 太線 = 中央値", fontsize=11.5, loc="left")
ax.text(len(GROUPS) - 0.6, 1.055, "参照解の 1.0〜1.2 倍", fontsize=9.5, color=C["green"], va="bottom", ha="right")
ax.annotate("道路網に制約すると\n参照に届く", (3, 1.6), fontsize=10, color=C["green"], ha="center", va="bottom", fontweight="bold")
ax.grid(axis="x", visible=False)
h_m = [plt.Line2D([], [], marker="o", ls="", color=M_COLOR[m], label=m) for m in ("PSO", "CCPSO2")]
h_p = [plt.Line2D([], [], marker=PAIR_MARK[p], ls="", color=C["muted"], label=PAIR_JA[p]) for p in PAIRS]
h_md = [plt.Line2D([], [], color=C["ink"], lw=2.6, label="中央値")]
lg = ax.legend(handles=h_m + h_md + h_p, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.135),
               frameon=False, fontsize=10, handletextpad=0.5, columnspacing=1.4)
ax.add_artist(lg)

# ---------------- 右: 避難困難区域 ----------------
MODES = [("fac", "指定避難場所\n(津波, 55 施設)"), ("notower", "指定避難場所\n(3 タワーを除く)"), ("safe", "指定避難場所\n+ 浸水域外の道路")]
VS = [("v10", "歩行 1.0 m/s", C["aqua"]), ("v05", "歩行 0.5 m/s", C["violet"])]
w = 0.34
bars = {}
for k, (vk, vlab, col) in enumerate(VS):
    xs = np.arange(len(MODES)) + (k - 0.5) * w
    ys = [DJ["cases"][f"{m}_{vk}"]["difficult_ha"] for m, _ in MODES]
    bars[vk] = ys
    ax2.bar(xs, ys, width=w, color=col, edgecolor="white", linewidth=0.8, label=vlab, zorder=3)
    for x, y in zip(xs, ys):
        t = "0" if y == 0 else ("<0.01" if y < 0.05 else f"{y:.0f}")
        ax2.annotate(t, (x, y), xytext=(0, 4), textcoords="offset points", ha="center",
                     fontsize=10, color=C["ink"], fontweight="bold", zorder=4)
ax2.set_xticks(range(len(MODES)))
ax2.set_xticklabels([m[1] for m in MODES], fontsize=10)
ax2.set_ylabel("避難困難区域 [ha]")
ax2.set_ylim(0, 168)
ax2.set_xlabel("目的地の定義", fontsize=10)
ax2.set_title("(b) 最遅出発 Dijkstra の避難困難区域 (避難開始 = 地震 5 分後)", fontsize=11.5, loc="left")
ax2.legend(frameon=False, fontsize=10, loc="upper left")
ax2.grid(axis="x", visible=False)
inu = DJ["inundated_area_ha"]
cov = DJ["cases"]["fac_v10"]["inundated_road_covered_ha"]
ax2.text(0.0, -0.235, f"浸水域 (30 cm 以上) {inu:.0f} ha のうち道路 50 m 圏の {cov:.0f} ha が評価対象。"
                      f"\n避難困難 = 浸水域かつ (最遅出発余裕 − 避難開始 300 s) < 0。"
                      f"\n道路網 (OSM, {DJ['nodes']:,} ノード / {DJ['edges']:,} 辺) 上の時間依存 Dijkstra。",
         transform=ax2.transAxes, fontsize=9, color=C["ink2"], va="top", linespacing=1.5)

fig.suptitle("御坊市の避難経路実験: 連続最適化は道路網 Dijkstra に勝てるか / 歩行速度が避難困難区域を決める",
             fontsize=13.5, x=0.008, ha="left", y=1.005)
fig.savefig(f"{OUT}/fig_evac_summary.png")

stats = dict(
    ratio_median={k: {m: v[m]["ratio_median"] for m in ("PSO", "CCPSO2")}
                  for k, v in dict(spline_5000=S5, spline_20000=S20, roadpso_M5=R5, roadpso_M20=R20).items()},
    ratio_per_pair={k: {m: v[m]["ratio_per_pair"] for m in ("PSO", "CCPSO2")}
                    for k, v in dict(spline_5000=S5, spline_20000=S20, roadpso_M5=R5, roadpso_M20=R20).items()},
    ratio_minmax={k: {m: [v[m]["ratio_min"], v[m]["ratio_max"]] for m in ("PSO", "CCPSO2")}
                  for k, v in dict(spline_5000=S5, spline_20000=S20, roadpso_M5=R5, roadpso_M20=R20).items()},
    wins={k: dict(ccpso2_wins=v["ccpso2_wins"], ccpso2_losses=v["ccpso2_losses"], ties=v["ties"])
          for k, v in dict(spline_5000=S5, spline_20000=S20, roadpso_M5=R5, roadpso_M20=R20).items()},
    paper=PAPER,
    difficult_ha={f"{m}_{vk}": DJ["cases"][f"{m}_{vk}"]["difficult_ha"] for m, _ in MODES for vk, _, _ in VS},
    inundated_area_ha=inu, inundated_road_covered_ha=cov,
    roadpso_reach_ref=dict(M5=R5["reach_ref"], M20=R20["reach_ref"]))
json.dump(stats, open(f"{OUT}/fig_evac_summary_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(dict(ratio_median=stats["ratio_median"], wins=stats["wins"],
                      difficult_ha=stats["difficult_ha"]), ensure_ascii=False, indent=1))
print("saved fig_evac_summary.png / _stats.json")
