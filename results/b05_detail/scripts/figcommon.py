"""b05_detail の図の共通部品 (フォント・配色は results/b05_slides/scripts/common.py を流用)。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_detail import CACHE, OUT, load_grids   # noqa: E402
from common import plt, C                          # noqa: E402  (b05_slides/scripts/common.py)

# 図の窓 (御坊市街 〜 日高川。5m 格子で 600×900 セル = 約 3.0 km × 4.5 km)
WIN = (60, 660, 80, 980)


def hillshade(z, azdeg=315.0, altdeg=45.0, ve=2.0, res=5.0):
    """陰影起伏 (0-1)。"""
    gy, gx = np.gradient(z * ve, res, res)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(360.0 - azdeg + 90.0)
    alt = np.deg2rad(altdeg)
    s = (np.sin(alt) * np.sin(slope)
         + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip((s + 1) / 2, 0, 1)


def load_surfaces():
    z = np.load(f"{CACHE}/surfaces.npz")
    return {k: z[k] for k in z.files}


def crop(a, win=WIN):
    r0, r1, c0, c1 = win
    return a[r0:r1, c0:c1]


def stair_edges(mask, wse):
    """1m 丸めで隣接セル間に段差が出るセル (どちらか片方を True にする)。"""
    Wb = np.where(mask, np.floor(wse), np.nan)
    out = np.zeros_like(mask)
    py = mask[:-1] & mask[1:]
    px = mask[:, :-1] & mask[:, 1:]
    by = (np.abs(np.diff(Wb, axis=0)) > 0.5) & py
    bx = (np.abs(np.diff(Wb, axis=1)) > 0.5) & px
    out[:-1, :] |= by
    out[1:, :] |= by
    out[:, :-1] |= bx
    out[:, 1:] |= bx
    return out


def km_axes(ax, win=WIN, res_m=5.0):
    r0, r1, c0, c1 = win
    h, w = r1 - r0, c1 - c0
    ax.set_xticks(np.arange(0, w + 1, 1000 / res_m))
    ax.set_xticklabels([f"{i}" for i in range(len(np.arange(0, w + 1, 1000 / res_m)))])
    ax.set_yticks(np.arange(0, h + 1, 1000 / res_m))
    ax.set_yticklabels([f"{i}" for i in range(len(np.arange(0, h + 1, 1000 / res_m)))])
    ax.set_xlabel("km")
    ax.grid(False)
