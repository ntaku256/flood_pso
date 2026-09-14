"""fig_surface_naive_vs_converted.png — 想定図の素朴拡大 vs 変換後の水面。

左  : 公式 25m 想定図のランク代表深を 5m 格子に載せた水面標高 (= 素朴拡大)
中  : 変換後の水面標高 (実測解: HAND + depth∈[-4,20], 実河道水源, λ=0, CCPSO2)
右  : 1m 丸めで段差になる箇所 (素朴拡大 = オレンジ / 変換後 = 青)
いずれも陰影起伏 DEM の上に重ねる。
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import (OUT, WIN, C, plt, crop, hillshade, km_axes,
                       load_grids, load_surfaces, stair_edges)
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

g = load_grids(verbose=False)
S = load_surfaces()
land, valid = g["land"], g["valid"]

hs = hillshade(np.where(valid, land, np.nan))
hs = np.nan_to_num(crop(hs), nan=0.9)
LANDC = crop(np.where(valid, land, np.nan))

mA, wseA = crop(S["mA"]), crop(S["wseA"])
mC, wseC = crop(S["mC"]), crop(S["wseC"])
stA = crop(stair_edges(S["mA"], S["wseA"]))
stC = crop(stair_edges(S["mC"], S["wseC"]))

sq = json.load(open(f"{OUT}/s5_quality.json"))
rA, rC = sq["A_naive25m"], sq["C_solution"]

lo = float(np.nanpercentile(np.where(mA, wseA, np.nan), 1))
hi = float(np.nanpercentile(np.where(mA, wseA, np.nan), 99))

fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6), gridspec_kw=dict(wspace=0.30))
cmap = plt.get_cmap("viridis").copy()

for ax, (m, w, ttl) in zip(axes[:2], [
        (mA, wseA, f"(左) 想定図の素朴拡大\n面IoU {rA['iou']:.3f} / 1m階段 {100*rA['stair_rate']:.1f}%"),
        (mC, wseC, f"(中) 変換後の水面 (実測解, 3シード平均)\n面IoU {rC['iou']:.3f} / 1m階段 {100*rC['stair_rate']:.2f}%")]):
    ax.imshow(hs, cmap="gray", vmin=0.15, vmax=1.0, interpolation="nearest")
    im = ax.imshow(np.where(m, w, np.nan), cmap=cmap, vmin=lo, vmax=hi,
                   alpha=0.85, interpolation="nearest")
    ax.set_title(ttl, fontsize=11.5)
    km_axes(ax)
cb = fig.colorbar(im, ax=axes[:2], fraction=0.020, pad=0.012)
cb.set_label("水面標高 [m]")

ax = axes[2]
ax.imshow(hs, cmap="gray", vmin=0.15, vmax=1.0, interpolation="nearest")
ov = np.zeros(stA.shape + (4,))
ov[stA] = [*[int(C["orange"][i:i + 2], 16) / 255 for i in (1, 3, 5)], 1.0]
ax.imshow(ov, interpolation="nearest")
ov2 = np.zeros(stC.shape + (4,))
ov2[stC] = [*[int(C["blue"][i:i + 2], 16) / 255 for i in (1, 3, 5)], 1.0]
ax.imshow(ov2, interpolation="nearest")
ax.set_title("(右) 1m 丸めで段差になる箇所\n"
             f"素朴拡大 {100*rA['stair_rate']:.1f}%  /  変換後 {100*rC['stair_rate']:.2f}%"
             f"  ({rA['stair_rate']/max(rC['stair_rate'],1e-9):.1f} 倍)", fontsize=11.5)
km_axes(ax)
ax.legend(handles=[Line2D([], [], color=C["orange"], lw=6, label="素朴拡大の段差"),
                   Line2D([], [], color=C["blue"], lw=6, label="変換後の段差")],
          loc="lower right", framealpha=0.9, fontsize=9.5)

fig.suptitle("浸水想定図 (25m・5段ランク) → 地形に整合した連続水面への変換 — 御坊市街〜日高川",
             fontsize=13, y=1.03)
fig.savefig(f"{OUT}/fig_surface_naive_vs_converted.png", dpi=170, bbox_inches="tight")
print("wrote fig_surface_naive_vs_converted.png")

# ── 断面 1 本 (川を横切る) ────────────────────────────────────
r0, r1, c0, c1 = WIN
# 両方が十分長く浸水している列のうち、素朴拡大の 1m 段差が最も多い列を選ぶ
wid = (mA & mC).sum(axis=0)
score = np.where(wid >= 0.55 * wid.max(), stA.sum(axis=0), -1)
col = int(np.argmax(score))
wet = np.where(mA[:, col] | mC[:, col])[0]
lo_i, hi_i = max(0, wet.min() - 20), min(mA.shape[0], wet.max() + 20)
sl = slice(lo_i, hi_i)
ter = LANDC[sl, col]
wA = np.where(mA[sl, col], wseA[sl, col], np.nan)
wC = np.where(mC[sl, col], wseC[sl, col], np.nan)
x = np.arange(lo_i, hi_i) * 5.0 / 1000.0

fig2, ax = plt.subplots(figsize=(11.5, 4.4))
base = np.nanmin(ter) - 3
ax.fill_between(x, ter, base, color="#d9d5cb", zorder=1, label="地形 (DEM 5m)")
ax.plot(x, ter, color=C["ink2"], lw=1.2, zorder=3)
ax.step(x, np.floor(wA), where="mid", color=C["orange"], lw=1.0, alpha=0.55, zorder=4)
ax.step(x, np.floor(wC), where="mid", color=C["blue"], lw=1.0, alpha=0.55, zorder=6)
ax.plot(x, wA, color=C["orange"], lw=2.2, zorder=5,
        label="素朴拡大の水面 (= 地形 + ランク代表深)")
ax.plot(x, wC, color=C["blue"], lw=2.2, zorder=7, label="変換後の水面 (連続)")
ax.plot([], [], color=C["muted"], lw=1.0, label="↑ それぞれを 1m 丸めたもの (Minecraft のブロック高)")
ax.set_xlabel("断面に沿った距離 [km] (北→南)")
ax.set_ylabel("標高 [m]")
fin = np.concatenate([wA[np.isfinite(wA)], wC[np.isfinite(wC)]])
ax.set_ylim(base, min(np.nanmax(ter), fin.max() + 4.0) + 0.5)
ax.set_title(f"浸水域を横切る断面 (列 {c0+col}) — 素朴拡大の「水面」は地形の凹凸をそのままなぞるので\n"
             "1m 丸めると全面が階段になる。変換後は水面として滑らかに繋がる", fontsize=12)
ax.legend(loc="upper left", framealpha=0.95, fontsize=9.5)
fig2.savefig(f"{OUT}/fig_stairs_profile.png", dpi=170, bbox_inches="tight")
print(f"wrote fig_stairs_profile.png (col={c0+col})")
