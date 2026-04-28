"""
ccpso2.py
Cooperatively Coevolving Particle Swarm Optimization 2 (CCPSO2)
Li, X., Yao, X. (2012) "Cooperatively Coevolving Particle Swarms for Large Scale Optimization"

特徴:
  - D 次元の最適化問題を K_g = D / s のサブスウォームに分割
  - 各サブスウォームは s 次元の部分問題を担当
  - Context Vector b を共有し、部分解の評価時に b の該当ブロックを差し替えて完全解を構築
  - 速度概念なし。Cauchy / Gaussian サンプリングで位置を更新
  - 改善が止まったらランダムにグループを再構成（Random Grouping）

依存: numpy のみ。
"""

from __future__ import annotations
import time
import numpy as np


class CCPSO2:
    def __init__(self,
                 objective_full,
                 dim: int,
                 n_particles: int,
                 group_size: int,
                 bounds: tuple,
                 p_cauchy: float = 0.5,
                 seed: int | None = None,
                 verbose: bool = False):
        """
        Parameters
        ----------
        objective_full : callable, x∈R^D → float (small is better)
        dim            : D
        n_particles    : 各サブスウォームに対し共通の N（全粒子は D 次元の自身の位置を持つ）
        group_size     : s — サブスウォームの担当次元数。dim は s で割り切れる必要あり。
        bounds         : (lb, ub), 各 R^D
        p_cauchy       : 位置更新で Cauchy 分布を選ぶ確率
        seed           : 乱数シード
        """
        # 割り切れない場合、最終グループだけ小さくする（CCPSO2 標準の柔軟運用）
        self.objective_full = objective_full
        self.D  = dim
        self.s  = group_size
        self.K_g = (dim + group_size - 1) // group_size  # ceil(dim/s)
        self.N   = n_particles
        self.lb = np.asarray(bounds[0], dtype=np.float64)
        self.ub = np.asarray(bounds[1], dtype=np.float64)
        self.p_cauchy = p_cauchy
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

        # 初期化
        self.pos        = self.lb + (self.ub - self.lb) * self.rng.uniform(size=(self.N, self.D))
        self.pbest      = self.pos.copy()
        self.pbest_cost = np.full(self.N, np.inf)

        # 全粒子を一度評価し、ベストを context vector b にする
        for i in range(self.N):
            c = self._eval(self.pos[i])
            self.pbest_cost[i] = c
        i_best = int(np.argmin(self.pbest_cost))
        self.b      = self.pos[i_best].copy()
        self.b_cost = float(self.pbest_cost[i_best])

        # 初回ランダムグルーピング
        self._regroup()

        # 履歴
        self.history: list[float] = [self.b_cost]
        self.n_evals: int = self.N

    # ─────────────────────────────────────────────────────────
    def _eval(self, x: np.ndarray) -> float:
        return float(self.objective_full(x))

    def _regroup(self):
        perm = self.rng.permutation(self.D)
        self.groups = []
        for g in range(self.K_g):
            blk = perm[g * self.s:min((g + 1) * self.s, self.D)]
            if len(blk) > 0:
                self.groups.append(blk)

    def _sample_new(self, pbest_g: np.ndarray, gbest_g: np.ndarray) -> np.ndarray:
        s = pbest_g.shape[0]
        scale = np.abs(pbest_g - gbest_g) + 1e-12
        if self.rng.uniform() < self.p_cauchy:
            # Cauchy(0,1) via inverse CDF
            u = self.rng.uniform(size=s)
            new = pbest_g + scale * np.tan(np.pi * (u - 0.5))
        else:
            new = gbest_g + scale * self.rng.normal(size=s)
        return new

    # ─────────────────────────────────────────────────────────
    def step(self) -> float:
        """1 サイクル（全サブスウォームを1巡）"""
        prev_b = self.b_cost

        for dims in self.groups:
            # 1) 各粒子を「現 context vector b 上で当該ブロックを差し替えた解」として評価
            for i in range(self.N):
                x_full = self.b.copy()
                x_full[dims] = self.pos[i, dims]
                c = self._eval(x_full)
                self.n_evals += 1
                # pbest 更新（粒子内ベスト）
                if c < self.pbest_cost[i]:
                    self.pbest_cost[i] = c
                    self.pbest[i, dims] = self.pos[i, dims]
                # context vector 更新
                if c < self.b_cost:
                    self.b_cost = c
                    self.b[dims] = self.pos[i, dims]

            # 2) サブスウォーム gbest = pbest_cost が最小の粒子の当該ブロック
            i_g = int(np.argmin(self.pbest_cost))
            gbest_g = self.pbest[i_g, dims]

            # 3) 全粒子の当該ブロックを Cauchy/Gaussian サンプリングで更新
            lb_g = self.lb[dims]
            ub_g = self.ub[dims]
            for i in range(self.N):
                pbest_g = self.pbest[i, dims]
                new = self._sample_new(pbest_g, gbest_g)
                self.pos[i, dims] = np.clip(new, lb_g, ub_g)

        self.history.append(self.b_cost)

        # 改善なしなら再グルーピング
        if self.b_cost >= prev_b - 1e-12:
            self._regroup()

        return self.b_cost

    # ─────────────────────────────────────────────────────────
    def run(self, n_cycles: int) -> dict:
        t0 = time.time()
        for it in range(n_cycles):
            self.step()
            if self.verbose and (it % max(1, n_cycles // 20) == 0):
                print(f"  [CCPSO2] cycle {it:3d}/{n_cycles}  b_cost={self.b_cost:.5f}  evals={self.n_evals}")
        elapsed = time.time() - t0
        return {
            "best_x":    self.b.copy(),
            "best_cost": self.b_cost,
            "history":   list(self.history),
            "elapsed_s": elapsed,
            "n_evals":   self.n_evals,
            "D":         self.D,
            "s":         self.s,
            "K_g":       self.K_g,
            "N":         self.N,
        }


# ─────────────────────────────────────────────────────────────
# 動作確認：Rosenbrock 関数（高次元の標準ベンチマーク）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def rosenbrock(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))

    D = 20
    bounds = (np.full(D, -5.0), np.full(D, 5.0))
    cc = CCPSO2(rosenbrock, dim=D, n_particles=20, group_size=5,
                bounds=bounds, p_cauchy=0.5, seed=0, verbose=True)
    res = cc.run(n_cycles=80)
    print(f"\nFinal best cost: {res['best_cost']:.6f}  evals={res['n_evals']}  elapsed={res['elapsed_s']:.2f}s")
    print(f"x_best (first 5): {res['best_x'][:5]}")
