"""
pso_calibrate_hd.py
高次元 PSO 校正：ブロック単位水位補正 Δh(K×K) + global water_level。

最適化変数 (D = 1 + K*K):
  x[0]            : water_level_global ∈ [w_min, w_max]
  x[1 : 1+K*K]    : dh_map.flatten()   ∈ [dh_min, dh_max]

目的関数: 1 - IoU(simulated, ground_truth_mask)
"""

from __future__ import annotations
import time
import numpy as np
from pyswarms.single import GlobalBestPSO

from flood_sim import simulate_flood_hd, iou_loss


# ─────────────────────────────────────────────────────────────
# 合成 ground truth 生成
# ─────────────────────────────────────────────────────────────

def make_synthetic_ground_truth(dem: np.ndarray, source_mask: np.ndarray,
                                K: int,
                                water_level_true: float,
                                dh_amp: float = 1.5,
                                seed: int = 42,
                                sigma: float = 0.5) -> dict:
    """
    固定シードで真の Δh マップを生成し、それで作った浸水マスクを返す。

    Returns
    -------
    {
      'dh_true'      : (K,K) 真の補正マップ
      'water_true'   : float
      'gt_mask'      : (H,W) bool 真の浸水マスク（参照）
      'gt_inundation': (H,W) float 浸水深
    }
    """
    rng = np.random.RandomState(seed)
    dh_true = rng.uniform(-dh_amp, +dh_amp, size=(K, K)).astype(np.float64)
    # 隣接ブロックで急変しすぎないよう軽く平滑化（リアリスティックさのため）
    from scipy.ndimage import gaussian_filter as gf
    dh_true = gf(dh_true, sigma=0.8)

    inundation = simulate_flood_hd(dem, source_mask,
                                   water_level_global=water_level_true,
                                   dh_map=dh_true,
                                   sigma=sigma)
    gt_mask = inundation > 0.05
    return {
        "dh_true":       dh_true,
        "water_true":    float(water_level_true),
        "gt_mask":       gt_mask,
        "gt_inundation": inundation,
        "K":             K,
        "sigma":         sigma,
    }


# ─────────────────────────────────────────────────────────────
# 目的関数
# ─────────────────────────────────────────────────────────────

def make_objective_hd(dem, source_mask, gt_mask, K, sigma):
    """pyswarms 用の高次元目的関数。"""
    def objective(X):
        n_particles = X.shape[0]
        losses = np.empty(n_particles, dtype=np.float64)
        for i in range(n_particles):
            x = X[i]
            water = float(x[0])
            dh    = x[1:1 + K * K].reshape(K, K)
            sim   = simulate_flood_hd(dem, source_mask,
                                      water_level_global=water,
                                      dh_map=dh,
                                      sigma=sigma)
            losses[i] = iou_loss(sim, gt_mask)
        return losses
    return objective


# ─────────────────────────────────────────────────────────────
# 標準 PSO 実行
# ─────────────────────────────────────────────────────────────

def run_pso_hd(dem, source_mask, gt_mask, K,
               sigma=0.5,
               w_bounds=(3.0, 8.0),
               dh_bounds=(-2.0, 2.0),
               n_particles=30, n_iter=100,
               pso_options=None,
               verbose=True):
    """
    高次元 (1+K²) で標準 GlobalBestPSO を実行。

    Returns
    -------
    {
      'best_water', 'best_dh',
      'best_cost', 'best_iou',
      'history'   : list[float] best_cost の履歴
      'elapsed_s' : float
    }
    """
    D = 1 + K * K
    objective = make_objective_hd(dem, source_mask, gt_mask, K, sigma)

    lb = np.empty(D); ub = np.empty(D)
    lb[0]   = w_bounds[0]; ub[0]   = w_bounds[1]
    lb[1:]  = dh_bounds[0]; ub[1:] = dh_bounds[1]
    bounds = (lb, ub)

    options = pso_options or {"c1": 1.5, "c2": 1.5, "w": 0.7}
    optimizer = GlobalBestPSO(n_particles=n_particles,
                              dimensions=D,
                              options=options,
                              bounds=bounds,
                              ftol=1e-6,
                              ftol_iter=30)

    t0 = time.time()
    best_cost, best_pos = optimizer.optimize(objective, iters=n_iter, verbose=verbose)
    elapsed = time.time() - t0

    return {
        "best_water": float(best_pos[0]),
        "best_dh":    best_pos[1:1 + K * K].reshape(K, K).copy(),
        "best_cost":  float(best_cost),
        "best_iou":   float(1.0 - best_cost),
        "history":    list(optimizer.cost_history),
        "elapsed_s":  elapsed,
        "D":          D,
        "K":          K,
        "n_particles": n_particles,
        "n_iter":     n_iter,
    }
