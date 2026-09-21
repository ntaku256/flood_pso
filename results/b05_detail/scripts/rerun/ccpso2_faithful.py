"""Li & Yao (2012) Algorithm 2 に忠実な CCPSO2。

現行 `flood_pso/src/ccpso2.py` の 4 つの逸脱を全部直したもの:
  1. ✅ ring lbest (近傍サイズ3: i-1, i, i+1) — 現行は全体 gbest
  2. ✅ **pbest の協調適合度を毎サイクル再評価** — 現行は単一の単調スカラーで再評価なし (最大の劣化要因)
  3. ✅ ランダムグルーピングを毎サイクル — 現行は停滞時のみ
  4. ✅ 評価回数 2·K_g·N /サイクル — 現行は K_g·N

論文 Algorithm 2 の擬似コード:
  1: K 個のサブスウォーム (各 N 粒子 × s 次元) を作り初期化
  2: repeat
  3:   f(b) が改善しなければ S から s を選び直し、K = ceil(D/s) で再グルーピング
  4:   for j = 1 to K:                          # 各サブスウォーム
  5:     for i = 1 to N:
  6:       if f(b(j, P_ij)) < f(b(j, y_ij)): y_ij = P_ij       # pbest 更新 (2 評価)
  7:       if f(b(j, y_ij)) < f(b(j, ŷ_j)):   ŷ_j = y_ij       # サブスウォーム best
  8:     for i = 1 to N: ŷ'_ij = リング近傍 {i-1,i,i+1} の best
  9:     for i = 1 to N, for d in block j:
 10:       確率 p で   P_ijd = y_ijd  + Cauchy(1)   * |y_ijd − ŷ'_ijd|
 11:       それ以外は  P_ijd = ŷ'_ijd + Normal(0,1) * |y_ijd − ŷ'_ijd|
 12: until 終了条件
  b は各サイクルで ŷ_j により更新される。

境界処理は原著の「1回だけの鏡面反射」ではなく **clip** を既定にした。
理由: 物理量に上下限がある逆問題では鏡面反射だと解が範囲外に飛ぶ
(実測で Δh_RMSE が 130、最悪 389 まで発散した)。`bound="mirror"` で原著挙動も選べる。

スケール係数は論文式(4) どおり `|y − ŷ'|`。著者の Java 実装は `/2.0` しており論文と食い違うので、
`scale_half=True` で著者実装に合わせられる。
"""
from __future__ import annotations
import time
import numpy as np


class CCPSO2Faithful:
    def __init__(self, objective_full, dim: int, n_particles: int = 30,
                 group_sizes=None, group_size: int | None = None,
                 bounds: tuple | None = None, p_cauchy: float = 0.5,
                 seed: int | None = None, bound: str = "clip",
                 scale_half: bool = False, verbose: bool = False):
        self.f = objective_full
        self.D = dim
        self.N = n_particles
        self.lb = np.asarray(bounds[0], dtype=np.float64)
        self.ub = np.asarray(bounds[1], dtype=np.float64)
        self.p = p_cauchy
        self.bound = bound
        self.scale_half = scale_half
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

        if group_sizes:
            self.S = sorted({int(s) for s in group_sizes if 1 <= s <= dim})
        else:
            self.S = [int(group_size if group_size else max(1, dim // 10))]
        self.s = int(self.rng.choice(self.S))

        # 位置 P (N×D) と personal best y (N×D)
        self.P = self.lb + (self.ub - self.lb) * self.rng.uniform(size=(self.N, self.D))
        self.y = self.P.copy()

        self.n_evals = 0
        # context vector b は初期化時に1回だけ全評価してベストを採る
        costs = np.empty(self.N)
        for i in range(self.N):
            costs[i] = self._eval(self.P[i])
        i0 = int(np.argmin(costs))
        self.b = self.P[i0].copy()
        self.b_cost = float(costs[i0])
        self.history = [self.b_cost]
        self._regroup()

    # ─────────────────────────────────────────────
    def _eval(self, x):
        self.n_evals += 1
        return float(self.f(x))

    def _eval_block(self, dims, z):
        """context vector b の block dims を z に差し替えた完全解を評価する。"""
        x = self.b.copy()
        x[dims] = z
        return self._eval(x)

    def _regroup(self):
        self.K_g = (self.D + self.s - 1) // self.s
        perm = self.rng.permutation(self.D)
        self.groups = [perm[g * self.s: min((g + 1) * self.s, self.D)]
                       for g in range(self.K_g)]
        self.groups = [g for g in self.groups if len(g) > 0]

    def _apply_bound(self, z, lo, hi):
        if self.bound == "mirror":            # 原著: 1回だけの鏡面反射
            z = np.where(z < lo, 2 * lo - z, z)
            z = np.where(z > hi, 2 * hi - z, z)
            return z
        return np.clip(z, lo, hi)             # 既定

    # ─────────────────────────────────────────────
    def step(self):
        prev = self.b_cost
        for dims in self.groups:
            lo, hi = self.lb[dims], self.ub[dims]

            # ── line 5-7: pbest を「現 context 上で」毎回再評価する (これが原著の核) ──
            fy = np.empty(self.N)
            for i in range(self.N):
                fP = self._eval_block(dims, self.P[i, dims])          # 評価 1
                fyi = self._eval_block(dims, self.y[i, dims])         # 評価 2
                if fP < fyi:
                    self.y[i, dims] = self.P[i, dims]
                    fy[i] = fP
                else:
                    fy[i] = fyi

            # サブスウォーム best ŷ_j と context vector の更新
            ib = int(np.argmin(fy))
            if fy[ib] < self.b_cost:
                self.b_cost = float(fy[ib])
                self.b[dims] = self.y[ib, dims]

            # ── line 8: ring lbest (近傍サイズ 3) ──
            im1 = np.roll(np.arange(self.N), 1)
            ip1 = np.roll(np.arange(self.N), -1)
            trio = np.stack([fy[im1], fy, fy[ip1]])                   # (3, N)
            which = np.argmin(trio, axis=0)                           # 0→i-1, 1→i, 2→i+1
            nb = np.where(which == 0, im1, np.where(which == 1, np.arange(self.N), ip1))

            # ── line 9-11: 位置更新 (速度なし) ──
            yb = self.y[:, dims]                                      # (N, s)
            lb_ = self.y[nb][:, dims]                                 # lbest の該当ブロック
            scale = np.abs(yb - lb_)
            if self.scale_half:
                scale = scale / 2.0
            u = self.rng.uniform(size=(self.N, len(dims)))
            cauchy = np.tan(np.pi * (self.rng.uniform(size=(self.N, len(dims))) - 0.5))
            gauss = self.rng.normal(size=(self.N, len(dims)))
            newpos = np.where(u < self.p, yb + scale * cauchy, lb_ + scale * gauss)
            self.P[:, dims] = self._apply_bound(newpos, lo, hi)

        self.history.append(self.b_cost)
        # ── line 3: 改善が無ければ s を選び直して再グルーピング。改善があっても毎サイクル再グルーピング ──
        if self.b_cost >= prev - 1e-12 and len(self.S) > 1:
            self.s = int(self.rng.choice(self.S))
        self._regroup()
        return self.b_cost

    def run(self, n_cycles: int = 10 ** 9, max_evals: int | None = None) -> dict:
        t0 = time.time()
        it = 0
        while it < n_cycles:
            self.step()
            it += 1
            if max_evals is not None and self.n_evals >= max_evals:
                break
            if self.verbose and it % 5 == 0:
                print(f"  [faithful] cycle {it} b_cost={self.b_cost:.5f} evals={self.n_evals}")
        return {"best_x": self.b.copy(), "best_cost": self.b_cost,
                "history": list(self.history), "elapsed_s": time.time() - t0,
                "n_evals": self.n_evals, "D": self.D, "s": self.s,
                "K_g": self.K_g, "N": self.N, "cycles": it}
