"""tbl_r8_vs_manual: 御坊市避難マニュアル p7 表 (H25 想定) と R8 復号値の比較。"""
import re, csv, json, fitz, numpy as np
from r8common import *
doc = fitz.open(f"{R8}/pdf/goboudai1-hinann.pdf")
page = [p for p in doc if "各地点での想定津波到達時間" in p.get_text()][0]
lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
z = str.maketrans("０１２３４５６７８９．ｍ", "0123456789.m")
i0 = lines.index("浸水深") + 1
toks = lines[i0:]
# 4 列 (地域, 地点, 到達時間, 浸水深) 繰り返し。地点名が 2 行に折り返す場合を吸収する
rows = []; j = 0
while j < len(toks):
    area = toks[j]; j += 1
    name = toks[j]; j += 1
    while not re.fullmatch(r"[０-９0-9]+分", toks[j]):
        name += toks[j]; j += 1
    tmin = int(toks[j].translate(z).rstrip("分")); j += 1
    dep = float(toks[j].translate(z).rstrip("m")); j += 1
    rows.append(dict(area=area, name=name, manual_min=tmin, manual_depth=dep))
print(len(rows), "rows from PDF")
d, g = load_r8(); tarr, hmax, dem = d["t_arrive_sec"], d["hmax"], d["dem"]; H, W = dem.shape
evac = load_evac()
alias = {"薗津波避難タワー": "薗地区津波避難タワー", "オークワロマンシティ": "オークワ ロマンシティ御坊店",
         "藤田小学校（運動場）": "藤田小学校", "熊野会館（駐車場）": "熊野会館", "和歌山高専（運動場）": "国立和歌山高専"}
out = []
for r in rows:
    key = r["name"] if r["name"] in evac else alias.get(r["name"])
    rec = dict(r, csv_name=key or "", match=("同名" if r["name"] in evac else ("別名一致" if key else "座標なし")))
    if key:
        la, lo = evac[key]; rr, cc = ll2rc(g, la, lo); ri, ci = int(rr), int(cc)
        rec.update(lat=la, lon=lo)
        if 0 <= ri < H and 0 <= ci < W:
            v = tarr[ri, ci]; hm = hmax[ri, ci]
            # 近傍 (半径 5 セル ≈ 26-31 m) の最短到達・最大深も併記 (座標のずれ吸収)
            sl = (slice(max(ri-5,0), ri+6), slice(max(ci-5,0), ci+6))
            vn = tarr[sl]; vn = vn[np.isfinite(vn)]
            rec.update(r8_min=(round(float(v)/60, 1) if np.isfinite(v) else None),
                       r8_depth=round(float(hm), 2), dem_m=round(float(dem[ri, ci]), 2) if np.isfinite(dem[ri, ci]) else None,
                       r8_min_nb=(round(float(vn.min())/60, 1) if vn.size else None),
                       r8_depth_nb=round(float(np.nanmax(hmax[sl])), 2), in_grid=True)
        else:
            rec.update(in_grid=False)
    out.append(rec)
cols = ["area", "name", "manual_min", "manual_depth", "match", "csv_name", "lat", "lon", "in_grid", "dem_m", "r8_min", "r8_depth", "r8_min_nb", "r8_depth_nb"]
with open(f"{OUT}/tbl_r8_vs_manual.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in out: w.writerow({k: r.get(k, "") for k in cols})
cmp_ = [r for r in out if r.get("r8_min") is not None]
dt = np.array([r["r8_min"] - r["manual_min"] for r in cmp_]); dd = np.array([r["r8_depth"] - r["manual_depth"] for r in cmp_])
dtn = np.array([r["r8_min_nb"] - r["manual_min"] for r in cmp_ if r.get("r8_min_nb") is not None])
stats = dict(n_manual=len(rows), n_with_coord=sum(1 for r in out if r["match"] != "座標なし"),
             n_compared=len(cmp_), arrival_MAE_min=float(np.abs(dt).mean()), arrival_bias_min=float(dt.mean()),
             arrival_max_abs_min=float(np.abs(dt).max()), arrival_within_1min=int((np.abs(dt) <= 1).sum()),
             arrival_MAE_min_neighborhood=float(np.abs(dtn).mean()),
             depth_MAE_m=float(np.abs(dd).mean()), depth_bias_m=float(dd.mean()),
             compared_points=[r["name"] for r in cmp_])
json.dump(stats, open(f"{OUT}/tbl_r8_vs_manual_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
for r in out: print(r["name"], r["match"], r.get("r8_min"), r.get("r8_depth"), r.get("in_grid"))

# PNG 表
hdr = ["地域", "地点 (マニュアル)", "到達 [分]\nH25 マニュアル", "到達 [分]\nR8 セル", "到達 [分]\nR8 周辺30m", "差 [分]\n(周辺)", "浸水深 [m]\nマニュアル", "浸水深 [m]\nR8 セル", "浸水深 [m]\nR8 周辺30m", "座標の出所"]
cells = []
for r in out:
    if r["match"] == "座標なし":
        cells.append([r["area"], r["name"], r["manual_min"], "座標なし", "座標なし", "—", f'{r["manual_depth"]:.1f}', "座標なし", "座標なし", "座標なし"]); continue
    if not r.get("in_grid"):
        cells.append([r["area"], r["name"], r["manual_min"], "格子外", "格子外", "—", f'{r["manual_depth"]:.1f}', "格子外", "格子外", r["match"]]); continue
    cm = f'{r["r8_min"]:.0f}' if r.get("r8_min") is not None else "未到達"
    nb = f'{r["r8_min_nb"]:.0f}' if r.get("r8_min_nb") is not None else "未到達"
    dif = f'{r["r8_min_nb"]-r["manual_min"]:+.0f}' if r.get("r8_min_nb") is not None else "—"
    cells.append([r["area"], r["name"], r["manual_min"], cm, nb, dif, f'{r["manual_depth"]:.1f}', f'{r["r8_depth"]:.1f}', f'{r["r8_depth_nb"]:.1f}', r["match"]])
cmp_nb = [r for r in out if r.get("r8_min_nb") is not None]
dtn2 = np.array([r["r8_min_nb"] - r["manual_min"] for r in cmp_nb]); ddn = np.array([r["r8_depth_nb"] - r["manual_depth"] for r in cmp_nb])
stats.update(n_compared_neighborhood=len(cmp_nb), arrival_MAE_min_neighborhood=float(np.abs(dtn2).mean()),
             arrival_bias_min_neighborhood=float(dtn2.mean()), arrival_within_1min_neighborhood=int((np.abs(dtn2) <= 1).sum()),
             depth_MAE_m_neighborhood=float(np.abs(ddn).mean()), depth_bias_m_neighborhood=float(ddn.mean()),
             compared_points_neighborhood=[r["name"] for r in cmp_nb])
json.dump(stats, open(f"{OUT}/tbl_r8_vs_manual_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in stats.items() if "neighborhood" in k}, ensure_ascii=False, indent=1))
fig, ax = plt.subplots(figsize=(13, 0.36 * len(cells) + 1.6)); ax.set_axis_off()
tb = ax.table(cellText=cells, colLabels=hdr, loc="upper center", cellLoc="center", bbox=[0, 0, 1, 0.9],
              colWidths=[0.06, 0.27, 0.09, 0.08, 0.09, 0.07, 0.09, 0.08, 0.09, 0.08])
tb.auto_set_font_size(False); tb.set_fontsize(9.5)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor("#d9d8d3")
    if i == 0: c.set_facecolor("#efeeea"); c.set_text_props(weight="bold")
    elif cells[i-1][9] == "座標なし": c.set_text_props(color="#9a9994")
    if i > 0 and j == 1: c._loc = "left"
ax.set_title(f"御坊市 地域別津波避難マニュアル (御坊第一地区, R2.3, H25 想定) p7 の 30 cm 到達時間・浸水深 と 県 R8 復号値の比較\n"
             f"セル = 施設座標の 5 m セル (比較 {stats['n_compared']} 地点: 到達 MAE {stats['arrival_MAE_min']:.1f} 分, ±1 分以内 {stats['arrival_within_1min']}/{stats['n_compared']}, 浸水深 MAE {stats['depth_MAE_m']:.1f} m) / "
             f"周辺 30 m = 半径 5 セル (~30 m) の最早到達・最大深 (比較 {stats['n_compared_neighborhood']} 地点: 到達 MAE {stats['arrival_MAE_min_neighborhood']:.1f} 分, "
             f"±1 分以内 {stats['arrival_within_1min_neighborhood']}/{stats['n_compared_neighborhood']}, 浸水深 MAE {stats['depth_MAE_m_neighborhood']:.1f} m)\n"
             f"座標は 国土地理院 指定緊急避難場所 (30205_2.csv) の同名/別名施設。マニュアルの 21 地点中 {stats['n_with_coord']} 地点に座標あり (うち 2 地点は復号格子の南端 lat 33.870 より南で格子外)",
             fontsize=10, loc="left")
fig.savefig(f"{OUT}/tbl_r8_vs_manual.png")
print("saved")
