"""
visualize.py
結果の可視化ユーティリティ。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def plot_dem(dem: np.ndarray, title="DEM [m]", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    valid = dem[~np.isnan(dem)]
    vmin, vmax = valid.min(), min(valid.max(), 50)
    im = ax.imshow(dem, cmap="terrain", origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Elevation [m]")
    ax.set_title(title)
    ax.set_xlabel("Column (West → East)")
    ax.set_ylabel("Row (North → South)")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, ax


def plot_flood_overlay(dem: np.ndarray, inundation: np.ndarray,
                       source_mask: np.ndarray = None,
                       ref_mask: np.ndarray = None,
                       title="Flood inundation", save_path=None):
    """
    DEMを背景に、浸水域・水源・参照マスクを重ね描きする。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    valid = dem[~np.isnan(dem)]
    vmin, vmax = valid.min(), min(valid.max(), 30)

    for ax, show_ref in zip(axes, [False, True]):
        ax.imshow(dem, cmap="terrain", origin="upper", vmin=vmin, vmax=vmax, alpha=0.8)

        # 浸水域（青系カラーマップ）
        flood_masked = np.where(inundation > 0.05, inundation, np.nan)
        flood_cmap = plt.cm.Blues
        flood_cmap.set_bad(alpha=0)
        ax.imshow(flood_masked, cmap=flood_cmap, origin="upper",
                  alpha=0.65, vmin=0, vmax=inundation[~np.isnan(inundation)].max() if np.any(~np.isnan(inundation)) else 1)

        # 水源
        if source_mask is not None:
            source_overlay = np.where(source_mask, 1.0, np.nan)
            cmap_src = mcolors.LinearSegmentedColormap.from_list("src", ["cyan", "cyan"])
            cmap_src.set_bad(alpha=0)
            ax.imshow(source_overlay, cmap=cmap_src, origin="upper", alpha=0.9)

        # 参照マスク（輪郭のみ）
        if ref_mask is not None and show_ref:
            from scipy.ndimage import binary_erosion
            edge = ref_mask & ~binary_erosion(ref_mask)
            edge_overlay = np.where(edge, 1.0, np.nan)
            cmap_edge = mcolors.LinearSegmentedColormap.from_list("edge", ["red", "red"])
            cmap_edge.set_bad(alpha=0)
            ax.imshow(edge_overlay, cmap=cmap_edge, origin="upper", alpha=1.0)

        legend = [Patch(color="blue", alpha=0.6, label="Simulated flood")]
        if source_mask is not None:
            legend.append(Patch(color="cyan", label="Water source"))
        if ref_mask is not None and show_ref:
            legend.append(Patch(color="red", label="Reference boundary"))
        ax.legend(handles=legend, loc="upper right", fontsize=8)
        ax.set_title(title if not show_ref else title + " + Reference")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, axes


def plot_pso_history(history, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best cost (1 - IoU)")
    ax.set_title("PSO convergence")
    ax.grid(True, alpha=0.4)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, ax


def plot_inundation_depth_histogram(inundation: np.ndarray, save_path=None):
    flooded = inundation[inundation > 0.05]
    if flooded.size == 0:
        print("No flooded cells to plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flooded, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Inundation depth [m]")
    ax.set_ylabel("Cell count")
    ax.set_title("Distribution of inundation depth")
    ax.grid(True, alpha=0.4)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig, ax
