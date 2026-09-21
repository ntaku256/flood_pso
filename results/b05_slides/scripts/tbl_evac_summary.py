"""tbl_evac_summary: 避難経路実験 (原稿 3 章 合成 / 御坊 スプライン / 御坊 道路制約) の一枚まとめ表。"""
import csv
import json

from evaccommon import *

S5 = load_spline(5000)
S20 = load_spline(20000)
R5 = load_roadpso(5)
R20 = load_roadpso(20)

ROWS = [
    dict(key="paper", name="原稿 3 章 合成ベンチ\n(中継点スプライン, 人工障害物)", dim="D = 40\n(K=20)",
         budget="評価数・seed 数の\n記載なし (単一値)",
         pso=f'{PAPER["pso_ratio"]:.2f} 倍\n(コスト {PAPER["pso"]:.2f})',
         cc=f'{PAPER["ccpso2_ratio"]:.2f} 倍\n(コスト {PAPER["ccpso2"]:.2f})',
         ref=f'証明済み最適解\n{PAPER["ref"]:.2f}',
         win="CCPSO2 の勝ち\n(1 事例, 統計なし)"),
    dict(key="spline5k", name="御坊 実市街地\n連続スプライン\n(K=20 中継点, 自由平面)", dim="D = 40", budget="5,000 評価\n× 5 seed × 3 ペア", a=S5),
    dict(key="spline20k", name="同上 (予算 4 倍)\n連続スプライン", dim="D = 40", budget="20,000 評価\n× 5 seed × 3 ペア", a=S20),
    dict(key="road5", name="御坊 実市街地\n道路制約 via-point\n(M=5 経由点)", dim="D = 10", budget="5,000 評価\n× 5 seed × 3 ペア", a=R5),
    dict(key="road20", name="御坊 実市街地\n道路制約 via-point\n(M=20 経由点)", dim="D = 40", budget="5,000 評価\n× 5 seed × 3 ペア", a=R20),
]


def fmt_ratio(x):
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def cell_method(a, m):
    r = a[m]
    pp = " / ".join(fmt_ratio(r["ratio_per_pair"][p]) for p in PAIRS)
    return f'{fmt_ratio(r["ratio_median"])} 倍\n[{pp}]'


REF_LABEL = {"spline5k": "時間依存 Dijkstra\nの 20 点スプライン化",
             "spline20k": "時間依存 Dijkstra\nの 20 点スプライン化",
             "road5": "時間依存 Dijkstra\n(= 道路上の厳密解)",
             "road20": "時間依存 Dijkstra\n(= 道路上の厳密解)"}


def cell_ref(a, key):
    v = " / ".join(f'{a["ref"][p]:.0f}' for p in PAIRS)
    return f'{REF_LABEL[key]}\n[{v}] m'


for r in ROWS:
    a = r.get("a")
    if a is None:
        continue
    r["pso"] = cell_method(a, "PSO")
    r["cc"] = cell_method(a, "CCPSO2")
    r["ref"] = cell_ref(a, r["key"])
    tie = f' {a["ties"]} 分' if a["ties"] else ""
    r["win"] = f'CCPSO2 {a["ccpso2_wins"]} 勝 {a["ccpso2_losses"]} 敗{tie}\n(同一 seed 15 対戦)'

REF_POLY = " / ".join("%.0f" % S5["ref_polyline"][p] for p in PAIRS)

stats = dict(paper=PAPER, spline_5000=S5, spline_20000=S20, roadpso_M5=R5, roadpso_M20=R20,
             dijkstra=load_dijkstra()["cases"])
json.dump(stats, open(f"{OUT}/tbl_evac_summary_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# --- CSV (表と同じ内容 + ペア別の生値) ---
cols = ["定式化", "次元", "予算", "PSO 中央値 (参照比)", "CCPSO2 中央値 (参照比)", "参照 (Dijkstra/最適)", "CCPSO2 対 PSO"]
with open(f"{OUT}/tbl_evac_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in ROWS:
        w.writerow([r["name"].replace("\n", " "), r["dim"].replace("\n", " "), r["budget"].replace("\n", " "),
                    r["pso"].replace("\n", " "), r["cc"].replace("\n", " "),
                    r["ref"].replace("\n", " "), r["win"].replace("\n", " ")])
    w.writerow([])
    w.writerow(["--- 生値 (ペア別中央値コスト [m] と参照比) ---"])
    w.writerow(["定式化", "ペア", "参照コスト[m]", "PSO中央値[m]", "PSO参照比", "CCPSO2中央値[m]", "CCPSO2参照比"])
    for r in ROWS:
        a = r.get("a")
        if a is None:
            w.writerow(["原稿3章 合成40次元", "-", PAPER["ref"], PAPER["pso"], round(PAPER["pso_ratio"], 4),
                        PAPER["ccpso2"], round(PAPER["ccpso2_ratio"], 4)])
            continue
        for p in PAIRS:
            w.writerow([r["name"].replace("\n", " "), PAIR_JA[p], round(a["ref"][p], 2),
                        round(a["PSO"]["cost_median_per_pair"][p], 2), round(a["PSO"]["ratio_per_pair"][p], 4),
                        round(a["CCPSO2"]["cost_median_per_pair"][p], 2), round(a["CCPSO2"]["ratio_per_pair"][p], 4)])

# --- PNG ---
hdr = ["定式化 (探索空間)", "次元", "計算予算",
       "標準 PSO\n参照比 中央値", "CCPSO2\n参照比 中央値", "参照解 (Dijkstra / 最適)\nとそのコスト", "CCPSO2 対 PSO\n(同一 seed 対戦)"]
cells = [[r["name"], r["dim"], r["budget"], r["pso"], r["cc"], r["ref"], r["win"]] for r in ROWS]
cw = [0.215, 0.065, 0.135, 0.125, 0.125, 0.185, 0.15]

fig, ax = plt.subplots(figsize=(15.2, 5.9))
ax.set_axis_off()
tb = ax.table(cellText=cells, colLabels=hdr, loc="upper center", cellLoc="center",
              bbox=[0, 0.05, 1, 0.87], colWidths=cw)
tb.auto_set_font_size(False)
tb.set_fontsize(9.3)
GOOD, BAD = "#e8f4ea", "#fdeceb"
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor("#d9d8d3")
    c.set_height(0.155)
    if i == 0:
        c.set_facecolor("#efeeea")
        c.set_text_props(weight="bold")
        c.set_height(0.115)
        continue
    r = ROWS[i - 1]
    if j == 0:
        c._loc = "left"
        c.set_text_props(color=C["ink"], weight="bold" if r["key"].startswith("road5") else "normal")
    if j in (3, 4):
        a = r.get("a")
        rt = (PAPER["pso_ratio"] if j == 3 else PAPER["ccpso2_ratio"]) if a is None else \
             a["PSO" if j == 3 else "CCPSO2"]["ratio_median"]
        c.set_facecolor(GOOD if rt < 1.5 else (BAD if rt > 3 else "white"))
ax.set_title(
    "避難経路最適化のまとめ: 合成ベンチ (原稿 3 章) → 御坊 実市街地・連続スプライン → 御坊 実市街地・道路制約 via-point\n"
    "参照比 = (手法のコスト) / (参照解のコスト)。1.00 倍 = 参照解と同着。中央値は 3 ペア × 5 seed = 15 実行の中央値、[ ] 内はペア別中央値 (名屋 / 薗 / 市街中心)",
    fontsize=12, loc="left", pad=14)
note = (
    "コスト [m] = 経路長 + λ1·(通過時に 30 cm 以上浸水しているサンプルの経路長) + λ4·(避難先への遅刻秒) "
    "( + 連続スプライン版のみ λ2·道路外, λ3·建物/河川内)。λ1=λ3=50, λ2=2, λ4=100。歩行 1.0 m/s, 避難開始 = 地震 5 分後, サンプル間隔 2 m。\n"
    "3 ペア = 名屋 住宅地→名屋地区津波避難タワー (津波 27 分) / 薗 住宅地→薗地区津波避難タワー (32 分) / 市街中心→御坊小学校 (34 分)。seed = 0〜4。\n"
    "参照 (連続スプライン版) = 道路網の時間依存 Dijkstra 経路を 20 中継点スプラインに再符号化し同一コスト関数で評価した値 "
    f"(折れ線のままなら {REF_POLY} m)。連続空間の真の最適値は未知のため「到達すべき目安」。\n"
    "参照 (道路制約版) = 同じ道路グラフ上の時間依存 (最早到着) Dijkstra = この定式化の厳密解。静的 Dijkstra と一致した "
    "(= 歩行 1.0 m/s では浸水制約が binding にならない)。\n"
    f'CCPSO2 のグループ次元数 s: 連続スプライン版 s=2、道路制約版は M=5 で s={R5["ccpso2_group_size"]}, M=20 で s={R20["ccpso2_group_size"]} '
    "(3 ペア × seed 0,1 のスイープで選択)。PSO は pyswarms GlobalBestPSO 30 粒子, c1=c2=1.5, w=0.7。")
ax.text(0, 0.0, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.8, color=C["ink2"], linespacing=1.5)
fig.savefig(f"{OUT}/tbl_evac_summary.png")
print(json.dumps({k: dict(pso=v["PSO"]["ratio_median"], cc=v["CCPSO2"]["ratio_median"],
                          w=v["ccpso2_wins"], l=v["ccpso2_losses"], t=v["ties"])
                  for k, v in dict(spline5k=S5, spline20k=S20, road5=R5, road20=R20).items()},
                 ensure_ascii=False, indent=1))
print("saved tbl_evac_summary.png / .csv / _stats.json")
