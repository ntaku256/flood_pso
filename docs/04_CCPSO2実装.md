# 04 CCPSO2 実装

> 2026-04-28
> ファイル: `src/ccpso2.py`

## アルゴリズム要旨

参考: Li, X., Yao, X. (2012) "Cooperatively Coevolving Particle Swarms for Large Scale Optimization"

```
入力: f: R^D → R, D, N (粒子数), s (グループサイズ), bounds
1. 全 N 粒子を D 次元で一様初期化
2. 全粒子を評価し、最良の粒子を context vector b として採用
3. ランダム置換で D 個の次元を K_g = D/s 個のサブグループに分割
4. for cycle = 1..T:
     prev = b.cost
     for each group g (担当次元集合 dims):
       (a) for each particle i:
             x_full = b と同じだが dims のブロックだけ pos[i, dims] に置換
             c = f(x_full)
             if c < pbest_cost[i]: pbest[i, dims] = pos[i, dims], pbest_cost[i] = c
             if c < b.cost:        b[dims] = pos[i, dims], b.cost = c
       (b) gbest_g = (pbest_cost が最小の粒子の dims ブロック)
       (c) for each particle i:
             scale = |pbest[i, dims] - gbest_g| + ε
             if rand < p_cauchy:
                 pos[i, dims] = pbest[i, dims] + scale * Cauchy(0,1)
             else:
                 pos[i, dims] = gbest_g + scale * Normal(0,1)
             clip to bounds
     if b.cost did not improve: regroup()
```

## 実装上のポイント

| 項目 | 設計判断 |
|---|---|
| 各粒子の次元 | D 次元（部分位置ではなく全位置を保持） |
| 部分解の評価 | `b` のコピーに `dims` ブロックだけ差し替えて完全解を構築し `f` に渡す |
| pbest の文脈 | 「過去のいずれかの context vector で評価したベスト」。context が変わると stale になりうるが、単調非増加を保つので問題なし |
| Cauchy 分布 | 標準 Cauchy(0,1) は逆累積分布関数 `tan(π(U−0.5))` で生成 |
| スケール | `|pbest_g − gbest_g| + ε` — 両者が近いほど局所探索、離れていれば大域探索になる自己適応性 |
| Random Grouping | サイクル中に b が改善しなかった場合のみ再グルーピング（CCPSO2 標準） |
| 速度概念 | なし — 速度発散の問題を回避 |
| 境界処理 | `np.clip` で各次元の bounds 内に強制 |

## 自己テスト結果（Rosenbrock D=20）

```
$ .venv/bin/python src/ccpso2.py
...
Final best cost: 274.71  evals=6420  elapsed=0.08s
```

D=20 Rosenbrock は局所解が多い悪性関数。初期コスト数千 → 274 まで降下する挙動を確認。アルゴリズム自体の動作 OK。

## flood_pso 問題への接続

D=65（K=8 ブロック分割 + global water）の場合、推奨設定：
- `s = 5`（K_g=13）または `s = 13`（K_g=5）
- `N = 20–30`
- `T = 80–150` サイクル

D=257（K=16）でこそ CCPSO2 の真価が現れる想定。次フェーズでベンチマーク。
