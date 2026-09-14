# results/b05_detail — 洪水パート「詳細版 進捗報告」用の再計算一式

洪水パートの位置づけは **「浸水想定図 (25 m・5 段ランク) を地形に整合した連続水面へ変換する
高次元逆問題」** であり、洪水シミュレーション/推定ではない (降雨・時間を入力に取らない)。
したがって IoU は「想定図との整合の目安」にとどめ、主指標は **水面品質**
(1 m 階段率 / 連結成分数 / 水面<地形の違反 / 取り得る水深値の数) に置く。

数値は `summary.json` (機械可読) と `summary.md` (発表用の表) に入っている。

## 生成物

| ファイル | 内容 |
|---|---|
| `summary.json` | a〜f の全数値 (機械可読) |
| `summary.md` | a〜f の表 + 期待値 (memory) との差異 |
| `s5_quality.json` | f (水面品質 A/B/C) の生結果。C はシード別も保持 |
| `fig_surface_naive_vs_converted.png` | 素朴拡大 / 変換後 / 1m 段差の発生箇所 (陰影 DEM 上) |
| `fig_stairs_profile.png` | 川を横切る断面の地形・素朴拡大水面・変換後水面 |
| `fig_iou_gap.png` | IoU ギャップ 5 分解の積み上げ棒 |
| `fig_bounds_param.png` | 値域・パラメータ化ごとの CCPSO2 vs 修正PSO |
| `raw/*.json` | 2026-07-25 の実験の生データ (退避ディレクトリからコピー) |
| `scripts/*.py` | 再計算・集計・作図のスクリプト |
| `scripts/rerun/*.py` | a〜e の生データを作った元スクリプト (パスのみ修正して移植) |

既存図 `results/b05_slides/fig_calib21.png` (21 シード) と `fig_holdout.png` (市松 holdout) は
再生成不要。a / b の数値はそれらと同じ生データから来ている。

## 各数値の出所

| 表 | 内容 | 出所 | 再計算の有無 |
|---|---|---|---|
| a | 21 シード統計・Wilcoxon・A12 | `raw/seeds21_all.json` (`seeds21.py`) | 統計量のみ再計算 |
| b | 市松 holdout train/test/full | `raw/holdout.json` (`holdout.py`) | 統計量のみ再計算 |
| c | IoU ギャップ 5 分解・達成率 | `raw/hand_oracle.json` (`hand_oracle.py`) + `raw/box_verify.json` + `raw/iou_ceiling.json` | 算術のみ再計算 |
| d | 値域・パラメータ化の比較 | `raw/box_verify.json` (絶対標高 [1,10] / [-17,28]) + `raw/hand_optimize.json` (HAND [-2,10]) + `raw/hand_optimize20.json` (HAND [-4,20]) | 既存結果を使用 (再実行不要) |
| e | 水源マスク × λ 7 設定 | `raw/src_reg.json` + `raw/src_reg_final.json` (`src_reg.py` / `src_reg_final.py`) | 既存結果を使用 |
| f | 水面品質 A/B/C | **本ディレクトリで全て再計算** (`scripts/s5_quality.py`) | A/B は決定的、C は CCPSO2 5000 評価 × seed 0-2 |

元スクリプトは読み取り専用の退避ディレクトリ `/home/ntaku/research-rescue-20260728/` にある。
`src/` には一切触っていない。移植は 2 種類:

- `scripts/*.py` — f (水面品質) と集計・作図のために**書き直した**もの。
  格子の前処理と S5 指標の定義を `common_detail.py` に一本化した。
- `scripts/rerun/*.py` — a〜e の生データを作った元スクリプトを**ほぼそのまま**置いたもの。
  変更は 2 箇所だけで、(1) scratchpad を指す `sys.path.insert` → 同ディレクトリ、
  (2) 出力先 `/tmp/.../scratchpad/*.json` → `results/b05_detail/raw/*.json`。
  今回は既存の生データを使ったので**これらは実行していない** (構文と import のみ確認済み)。
  `seeds21.py` はシードを 3 分割する設計 (`python seeds21.py a|b|c` を並列に回して
  `seeds21_{a,b,c}.json` を作り、手で `seeds21_all.json` に束ねる。3 並列で約 4.6 時間)。

## 再現コマンド

```bash
cd /home/ntaku/laravel-project/flood_pso
.venv/bin/python results/b05_detail/scripts/s5_quality.py     # f + 図用の水面を作る (約 13 分)
.venv/bin/python results/b05_detail/scripts/stats_tables.py   # a〜e を集計 → summary.json (1 秒)
.venv/bin/python results/b05_detail/scripts/make_summary_md.py  # summary.md (1 秒)
.venv/bin/python results/b05_detail/scripts/fig_surface.py    # 図 1・図 2 (約 40 秒)
.venv/bin/python results/b05_detail/scripts/fig_iou_gap.py    # 図 3 (1 秒)
.venv/bin/python results/b05_detail/scripts/fig_bounds_param.py  # 図 4 (1 秒)
```

初回は DEM モザイク + 想定図タイル + HAND の前処理に約 1.5 分かかり、
結果は `$B05D_CACHE` (既定は scratchpad) の `grids.npz` にキャッシュされる。
想定図タイルは `data_cache/disaportal/` にキャッシュ済みなのでネットワークは不要。

所要時間の実測: 前処理 92 秒 / `s5_quality.py` 全体 約 13 分 (CCPSO2 1 走 約 3.9 分 × 3 シード) /
集計と図は合計 1 分未満。**合計 15 分程度**。5 m での再最適化はしていない。

## 実験設定 (f の C = 現行の最善設定)

- 浸水条件: `HAND(x,y) < depth_field(x,y)`、`HAND = land − land[最近傍排水路]`
  (排水路 = `land − minimum_filter(land, 201) < 0.5`、5,892 セル = 0.18%)
- `depth_field = d0 + bilinear_upsample(dd_map)`、`d0∈[0,16]`, `dd∈[-4,4]` → `depth ∈ [-4,20]`
- ブロック分割 K=16 → 決定変数 D = 1 + 16×16 = 257 次元
- 水源マスク: **実河道** (谷底 ∩ 河川 bbox ∩ 標高≤5 m、2,830 セル = 0.085%)
- 正則化 λ = 0
- 探索: CCPSO2 (粒子 20・グループ 16・p_cauchy 0.5)、5000 評価、seed 0/1/2
- 水面標高 `WSE = depth_field + land[最近傍排水路]`

## 既知の訂正・留保

1. 🔴 **e 表の「1 m 階段」は HAND 水深場 `tf` 上で測ったもの**で、水面標高 `WSE = tf + zd` 上の値ではない。
   `zd` (最近傍排水路の標高) は排水路の Voronoi 境界で不連続なので、その段差が数えられていなかった。
   **f 表が正しい値** (定義は `s5_vizquality.py` に一致)。
   影響: 実測解の 1 m 階段率は **1.71% → 2.36%**、素朴拡大に対する改善は
   **12.1 倍 → 8.1 倍**。`tf` 上で測り直すと 1.71% を再現するので原因は確定している
   (`scripts/s5_quality.py` が両方を出す)。
2. 🔴 **「隣接段差 非ゼロ率」は指標として誤り**なので出していない。
   実装は「隣接浸水ペアで |ΔWSE| > 1e-9 の割合」= 区分定数か連続かを測っているだけで、
   連続水面なら構成上ほぼ 100% になり、それが望ましい挙動。
3. ⚠ c 表のオラクルは**全て連結性制約なし**、実測は**連結性あり**。
   「探索律速」の一部は探索失敗ではなく連結性制約のコスト (連結性ありオラクルは未測定)。
4. ⚠ d 表の HAND[-4,20] の「3 シード完全分離」は 21 シードでは崩れる
   (修正PSO の max 0.6990 > CCPSO2 の min 0.6851)。主張は 21勝0敗・p=9.5e-07・A12=0.966 に置く。
5. ⚠ GT セル数は再計算で 406,410 (`iou_ceiling.json` は 407,144)。
   `gt & valid` の取り方の差で、IoU への影響は 4 桁目以下。
