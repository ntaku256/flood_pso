# 06 NBT 管理（実装系研究軸）

> 2026-04-28
> 関連スクリプト: `src/nbt_export.py`, `src/make_nbt.py`, `src/make_nbt_hd.py`

## 動機

研究の軸足を「PSO アルゴリズム研究」から **「PSO 改善手法の結果を Minecraft NBT 形式でアーカイブし、誰でも閲覧・再現可能にする実装系研究」** にも広げる。

既存の研究記録（`kennkyuu20260114/`）はアルゴリズム重視だが、本人の専門性（Minecraft 系 Web エンジニアリング、NBT 共有アプリ等の開発経験）を活かして以下を狙う：

1. **再現性**: NBT に最適化メタデータ（手法・パラメータ・loss・dh_map・seed・git rev）を埋め込み、ファイル単体で実験条件が完結
2. **公開性**: NBT を開けるツール（既存の Web ビューワやマイクラ本体）があれば、第三者が浸水域を 3D で歩いて確認できる
3. **拡張性**: 同じスキーマで「研究室メンバーごとの結果」「他都市」「別アルゴリズム（IA など来年度本筋）」を追加できる

## アーキテクチャ

```
benchmark.py の出力 (case_K{K}_seed{S}.json)
        │  最適 water, dh_map, loss, IoU, seed, …
        ▼
make_nbt_hd.py
        │  フル解像度 5m DEM + simulate_flood_hd で再シミュ
        │  meta dict を組み立てて nbt_export に渡す
        ▼
nbt_export.py: write_nbt_structure
        │  Minecraft Structure NBT (gzip)
        │   ├ DataVersion / size / palette / blocks / entities
        │   └ flood_pso_meta  ← NEW
        ▼
results/nbt/hd/gobo_hd_K{K}_seed{S}_{preset}_{method}.nbt
```

## flood_pso_meta スキーマ（schema_version=1）

```yaml
flood_pso_meta:
  schema_version: 1
  generator:        flood_pso/nbt_export.py
  timestamp_utc:    2026-04-28T...Z
  git_revision:     <short hash>

  experiment:       flood_pso_HD_benchmark
  method:           pso | ccpso2 | gt | baseline_2d_pso
  method_long:      "CCPSO2 (s=16, custom impl)"
  loss_kind:        depth | iou
  K:                16
  D:                257
  seed:             0
  budget:           5000

  water_level_global_m: 5.133
  sigma:                0.5
  dh_amp_m:             1.5
  dh_bounds_m:          [-2.0, 2.0]
  dh_map:               [256個のFloat]   # K*K
  dh_map_shape:         [16, 16]
  loss:                 0.115
  iou:                  0.960
  dh_rmse:              1.450
  n_evals:              5120
  elapsed_s:            15.8

  river_bbox:           [33.855, 33.905, 135.145, 135.215]
  river_elev_max_m:     5.0
  dem_source:           "FG-GML-503561-DEM5A-20250620 (国土地理院 5m DEM)"
  study_area:           "Gobo city / Hidaka river, Wakayama, Japan"

  preset:               md_5m
  center_lat:           33.875
  center_lon:           135.168
  width_m:              5000
  depth_m:              5000
  h_res_m_per_block:    5
  v_res_m_per_block:    1
  v_exag:               2
  structure_size_xyz:   [968, 490, 802]
  n_block_entries:      3,085,855

  ref_doc:              flood_pso/docs/05_ベンチマーク結果.md
```

`build_meta_compound()`（`src/nbt_export.py`）が辞書を NBT Compound に変換：
- `np.ndarray` → `List[Float]` + `_shape` を別キーで保存
- `bool/int/float/str` → `Byte/Int/Double/String` に正規化
- ネスト dict/list にも対応

NBT 上で読むには標準の `nbtlib.load(path)["flood_pso_meta"]` でアクセス可能。

## プリセット一覧

| name | 範囲 [m] | 解像度 [m/block] | 推奨メモリ | 想定 NBT サイズ |
|---|---|---|---|---|
| xs_overview | 2,000×2,000 | 10 | 4 GB+ | ~0.6 MB |
| sm_5m | 2,500×2,500 | 5 | 8 GB+ | ~3 MB |
| md_5m | 5,000×5,000 | 5 | 16 GB+ | ~10 MB |
| lg_10m | 10,000×10,000 | 10 | 16 GB+ | ~13 MB |
| **xl_5m** | **10,000×10,000** | **5** | **16 GB+** | **~40 MB** |
| **huge_5m** | **15,000×15,000** | **5** | **32 GB+** | **~90 MB** |

末尾2つは新規追加（16-32 GB メモリ環境用）。Web ビューワでは厳しいが、デスクトップビューワやマイクラ本体なら閲覧可能。

## 実行例

```bash
# 1) 2変数ベースラインの NBT（複数プリセット）
.venv/bin/python src/make_nbt.py --preset md_5m
.venv/bin/python src/make_nbt.py --preset xl_5m
.venv/bin/python src/make_nbt.py --preset huge_5m

# 2) 高次元 PSO vs CCPSO2 vs GT の比較 NBT
.venv/bin/python src/benchmark.py                       # 先に benchmark を実行
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset md_5m
.venv/bin/python src/make_nbt_hd.py --K 8  --seed 0 --preset md_5m
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset huge_5m   # 大型版
```

## 出力ファイル

```
results/nbt/
├── gobo_xs_overview.nbt     # 2変数ベースライン（小）
├── gobo_sm_5m.nbt
├── gobo_md_5m.nbt
└── hd/
    ├── gobo_hd_K16_seed0_md_5m_pso.nbt      # 標準PSO  (IoU=0.949)
    ├── gobo_hd_K16_seed0_md_5m_ccpso2.nbt   # CCPSO2  (IoU=0.960)  ★ 高次元勝者
    └── gobo_hd_K16_seed0_md_5m_gt.nbt       # 合成 ground truth   (IoU=1.000)
```

## ブロックパレット

`nbt_export.PALETTE` 参照。要点：
- `water` → `minecraft:blue_stained_glass`（Web ビューワでアニメーションテクスチャがチラつく問題への回避策）
- `blue_ice` → `minecraft:cyan_stained_glass`（同上、未使用）
- 地表は標高で sand / gravel / grass を使い分け
- 地盤は地表3ブロック下まで stone

`Properties` も付与（`grass_block` に `snowy: false`）して Structure 互換性を保証。

## 既知の制約

1. **Block entry が冗長**: 各ブロックを `{pos, state}` 個別 Compound として記述しているため、フラット ByteArray にした schematic 形式より NBT サイズが大きい。マイクラ Structure 公式仕様に合わせるため現状はこの形を維持。
2. **大規模プリセットでの Python 側メモリ**: huge_5m (3000×3000) は変換中に Python メモリ 8-12 GB を消費する見込み。要計測。
3. **flood_pso_meta は非標準キー**: マイクラ本体は無視するだけなので互換性影響なし。Web ビューワは独自パース必要。

## 研究的位置づけ

中間発表の構成（`kennkyuu20260114/doc/05_中間発表概要_再構成版.md`）に対する追加章：

> **6. 実装系の貢献：NBT を介した研究成果のアーカイブ**
>
> 提案する PSO 改善手法（CCPSO2）を御坊市・日高川の高次元洪水校正問題に適用した結果を、Minecraft Structure NBT 形式（schema_version=1）でアーカイブ可能にした。`flood_pso_meta` コンパウンドに最適化条件を埋め込むことで、ファイル単体で実験条件が完結し、第三者の再現・閲覧を可能にする。NBT は拡張容易で、来年度の IA 適用結果や他河川への展開も同スキーマで蓄積できる。

## 次のステップ候補

1. **schemati（Sponge Schematic）** との相互変換を追加 — 既存 NBT エコシステムとの互換性向上
2. **タイムシリーズ NBT**: 浸水進行（時間ステップ別）を複数 NBT に分割保存
3. **メタデータからの逆引きツール**: `nbt_inspect.py` を作って `--query method=ccpso2 --query K=16` で集計
4. **実ハザードマップとのオーバーレイ**: `kennkyuu20260114/地形データ/hidaka_hanran.pdf` を画像化して NBT 内に追加レイヤとして書き込む
