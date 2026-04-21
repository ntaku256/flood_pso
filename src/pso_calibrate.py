"""
pso_calibrate.py
PSO を使って洪水シミュレーションのパラメータを校正する。

最適化変数 (2次元):
  x[0]: water_level  [m]    : 洪水水面標高。範囲 [0, 15]
  x[1]: sigma        [cells]: DEM 平滑化 sigma。範囲 [0, 3]

目的関数: 1 - IoU (シミュレーション浸水域 vs 参照浸水域)
"""

import numpy as np
import pyswarms as ps
from pyswarms.single import GlobalBestPSO

from flood_sim import simulate_flood, iou_loss


def build_objective(dem, source_mask, ref_mask):
    """pyswarms 用目的関数を生成する。"""
    def objective(X):
        losses = []
        for row in X:
            water_level = float(row[0])
            sigma       = float(np.clip(row[1], 0.0, 5.0))
            inundation  = simulate_flood(dem, source_mask,
                                         water_level=water_level,
                                         sigma=sigma)
            losses.append(iou_loss(inundation, ref_mask))
        return np.array(losses)
    return objective


def run_pso(dem, source_mask, ref_mask,
            n_particles=20, n_iter=60, verbose=True):
    """
    PSO を実行し、最適パラメータと最終浸水マップを返す。

    Returns
    -------
    best_params     : dict  {'water_level', 'sigma'}
    best_cost       : float
    best_inundation : np.ndarray
    history         : list[float]
    """
    objective = build_objective(dem, source_mask, ref_mask)

    bounds = (
        np.array([0.0, 0.0]),   # 下限: water_level, sigma
        np.array([15.0, 3.0]),  # 上限
    )
    options = {"c1": 1.5, "c2": 1.5, "w": 0.7}

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds,
        ftol=1e-5,
        ftol_iter=20,
    )

    best_cost, best_pos = optimizer.optimize(
        objective, iters=n_iter, verbose=verbose
    )

    best_params = {
        "water_level": float(best_pos[0]),
        "sigma":       float(np.clip(best_pos[1], 0.0, 5.0)),
    }

    best_inundation = simulate_flood(
        dem, source_mask,
        water_level=best_params["water_level"],
        sigma=best_params["sigma"],
    )

    return best_params, best_cost, best_inundation, optimizer.cost_history


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    H, W = 200, 300
    y, x = np.meshgrid(np.linspace(0, 10, H), np.linspace(0, 15, W), indexing="ij")
    dem_test = y + 0.5 * x + np.random.rand(H, W) * 0.3
    dem_test[80:120, 50:80] = -1.0

    source = np.zeros((H, W), dtype=bool)
    source[80:120, 50:80] = True

    from flood_sim import make_reference_mask
    ref_mask = make_reference_mask(dem_test, 3.5)

    print("Running PSO (synthetic DEM)...")
    params, cost, inundation, history = run_pso(
        dem_test, source, ref_mask,
        n_particles=15, n_iter=40, verbose=True
    )
    print(f"\nBest: {params}  IoU={1-cost:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dem_test, cmap="terrain")
    axes[0].set_title("DEM")
    axes[1].imshow(ref_mask, cmap="Blues")
    axes[1].set_title("Reference")
    axes[2].imshow(inundation > 0.05, cmap="Blues")
    axes[2].set_title(f"PSO result (IoU={1-cost:.3f})")
    plt.tight_layout()
    plt.savefig("test_pso.png", dpi=100)
    print("Saved test_pso.png")
