"""fig_r8_timeline.png: 県 R8 津波の時間展開 (6 時刻) + 最大浸水深 + 30cm 到達時刻 の 8 パネル。"""
import numpy as np
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm, LinearSegmentedColormap
from r8common import *
import matplotlib.patheffects as pe
d, g = load_r8()
depth, t_sec, hmax, tarr, dem = d["depth"], d["t_sec"], d["hmax"], d["t_arrive_sec"], d["dem"]
H, W = dem.shape
DX, DY = cell_dims(g, H); CELL_HA = DX * DY / 1e4
lat_min = g["lat_max"] - H * g["res_lat"]; lon_max = g["lon_min"] + W * g["res_lon"]
ext = [g["lon_min"], lon_max, lat_min, g["lat_max"]]
asp = 1 / np.cos(np.radians((g["lat_max"] + lat_min) / 2))

ls = LightSource(azdeg=315, altdeg=45)
dem_f = np.where(np.isnan(dem), 0.0, dem)
hs = ls.hillshade(dem_f.astype(float), vert_exag=1.0, dx=DX, dy=DY)
base = np.dstack([hs * 0.55 + 0.45] * 3)          # 灰の陰影 (0.45〜1.0)
sea = np.isnan(dem)
base[sea] = np.array([0.80, 0.87, 0.93])            # 海 = 淡青

# 浸水深: 単色 (青) 逐次ランプ、境界は A40 区分に合わせる
bounds = [0.3, 1, 2, 3, 5, 10]
ramp = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
cm_d = ListedColormap(ramp); nm_d = BoundaryNorm(bounds, cm_d.N)

evac = load_evac()
fac = {"新町地区津波避難タワー": "▲", "薗地区津波避難タワー": "▲", "名屋地区津波避難タワー": "▲",
       "御坊市役所": "●", "御坊小学校": "●", "日高高等学校": "●"}
short = {"新町地区津波避難タワー": "新町タワー", "薗地区津波避難タワー": "薗タワー", "名屋地区津波避難タワー": "名屋タワー",
         "御坊市役所": "市役所", "御坊小学校": "御坊小", "日高高等学校": "日高高校"}

times = [25, 30, 35, 40, 60, 90]
fig = plt.figure(figsize=(16, 8.3))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.045], hspace=0.16, wspace=0.05, left=0.01, right=0.99, top=0.92, bottom=0.07)
axs = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
cax_d = fig.add_subplot(gs[2, 0:3]); cax_t = fig.add_subplot(gs[2, 3])
def panel(ax, title):
    ax.imshow(base, extent=ext, aspect=asp, interpolation="nearest", zorder=0)
    ax.set_title(title, fontsize=11, loc="left"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
im_d = None
for ax, tm in zip(axs, times):
    i = int(np.argmin(np.abs(t_sec - tm * 60))); assert abs(t_sec[i] - tm * 60) < 1
    D = depth[i].astype(np.float32); D = np.ma.masked_where(D < 0.3, D)
    area_ha = float((depth[i] >= 0.3).sum()) * CELL_HA
    panel(ax, f"地震発生 {tm} 分後  (浸水 ≥0.3 m: {area_ha:,.0f} ha)")
    im_d = ax.imshow(D, cmap=cm_d, norm=nm_d, extent=ext, aspect=asp, interpolation="nearest", zorder=2)
# 最大浸水深
ax = axs[6]; Hm = np.ma.masked_where(hmax < 0.3, hmax)
area_ha = float((hmax >= 0.3).sum()) * CELL_HA
panel(ax, f"最大浸水深 (全 120 分, ≥0.3 m: {area_ha:,.0f} ha)")
ax.imshow(Hm, cmap=cm_d, norm=nm_d, extent=ext, aspect=asp, interpolation="nearest", zorder=2)
cb = fig.colorbar(im_d, cax=cax_d, orientation="horizontal")
cb.set_label("浸水深 [m] (η − DEM, 凡例上限 10 m で飽和)", fontsize=10); cb.ax.tick_params(labelsize=9)
cax_d.set_position([0.22, 0.045, 0.45, 0.02])
# 到達時刻
ax = axs[7]
tmin = tarr / 60.0; tmin = np.ma.masked_where(~np.isfinite(tarr), tmin)
b2 = [20, 25, 30, 35, 40, 50, 60, 90, 120]
cm_t = LinearSegmentedColormap.from_list("warm", ["#5a0a0a", "#c0392b", "#eb6834", "#f6a56d", "#fbd3b3", "#fdeee0"], N=len(b2) - 1)
nm_t = BoundaryNorm(b2, cm_t.N)
panel(ax, "30 cm 浸水の到達時刻 [分] と避難施設")
im_t = ax.imshow(tmin, cmap=cm_t, norm=nm_t, extent=ext, aspect=asp, interpolation="nearest", zorder=2)
order = ["名屋地区津波避難タワー", "新町地区津波避難タワー", "薗地区津波避難タワー", "御坊市役所", "御坊小学校", "日高高等学校"]
legend_lines = []
for k, name in enumerate(order, 1):
    la, lo = evac[name]; mk = fac[name]
    ax.scatter(lo, la, marker="^" if mk == "▲" else "o", s=110, facecolor="white", edgecolor=C["ink"], linewidth=1.4, zorder=5)
    ax.annotate(str(k), (lo, la), xytext=(6, 5), textcoords="offset points", ha="left", va="bottom", fontsize=9.5, color=C["ink"],
                fontweight="bold", zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    r, c = ll2rc(g, la, lo); ri, ci = int(r), int(c); v = tarr[ri, ci]
    sl = (slice(max(ri - 5, 0), ri + 6), slice(max(ci - 5, 0), ci + 6)); vn = tarr[sl]; vn = vn[np.isfinite(vn)]
    cell = f"{v/60:.0f} 分" if np.isfinite(v) else "未到達"
    nb = f"{vn.min()/60:.0f}" if vn.size else "—"
    legend_lines.append(f"{k} {mk} {short[name]}: {cell} (周辺 {nb})")
ax.text(0.02, 0.98, "\n".join(legend_lines), transform=ax.transAxes, fontsize=8.6, va="top", ha="left", linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9), zorder=6)
ax.text(0.98, 0.02, "セル値 (施設座標の 5 m セル) / 周辺 = 半径 5 セル (~30 m) の最早到達", transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=6)
cb2 = fig.colorbar(im_t, cax=cax_t, orientation="horizontal")
cb2.set_label("到達時刻 [分]", fontsize=10); cb2.ax.tick_params(labelsize=9)
cax_t.set_position([0.76, 0.045, 0.22, 0.02])
fig.suptitle(f"和歌山県 R8 (2026-03) 南海トラフ巨大地震 津波浸水想定 — 御坊市 (県公表動画から復号, 5 m 格子, 陰影 = GSI DEM5A, セル {DX:.1f}×{DY:.1f} m)", fontsize=12.5, x=0.01, ha="left", y=0.975)
fig.savefig(f"{OUT}/fig_r8_timeline.png", dpi=170, bbox_inches="tight")
print("saved")
