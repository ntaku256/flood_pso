# flood_pso

御坊市・日高川を題材とした **PSO 改善手法（CCPSO2）の高次元最適化** 実証実験。

`kennkyuu20260114/` の研究軸と接続：
- メイン仮説：**CCPSO2 は高次元最適化問題で標準 PSO を上回る**
- 応用：洪水浸水シミュレーションのパラメータ校正を高次元化（K×K ブロックの水位補正マップ）

## ディレクトリ

```
flood_pso/
├── src/
│   ├── dem_parser.py          # 国土地理院 DEM5A GML → numpy モザイク
│   ├── flood_sim.py           # バスタブ + 連結成分浸水モデル（HD版含む）
│   ├── pso_calibrate.py       # 既存 2変数 PSO 校正
│   ├── pso_calibrate_hd.py    # 高次元 PSO 校正 + 合成 GT 生成
│   ├── ccpso2.py              # CCPSO2 自作実装（汎用）
│   ├── benchmark.py           # 標準PSO vs CCPSO2 多シードベンチマーク
│   ├── main.py                # 既存 2変数版エントリ
│   ├── main_hd.py             # 高次元 標準PSO 単独エントリ
│   ├── nbt_export.py          # Minecraft Structure NBT 出力（flood_pso_meta 付き）
│   ├── make_nbt.py            # 2D ベースラインから NBT 生成
│   ├── make_nbt_hd.py         # benchmark 結果（PSO/CCPSO2/GT）から NBT 生成
│   └── visualize.py / view_3d.py
├── docs/                      # 全実装の時系列記録
│   ├── 00_開発記録.md
│   ├── 01_環境構築.md
│   ├── 02_ベースライン再現.md
│   ├── 03_高次元化設計.md
│   ├── 04_CCPSO2実装.md
│   ├── 05_ベンチマーク結果.md
│   └── 06_NBT管理.md
├── results/
│   ├── *.png                  # 既存 2変数版の結果
│   ├── hd/                    # 高次元 標準PSO 単独実行の結果
│   ├── benchmark/             # 比較ベンチマーク結果
│   └── nbt/                   # 出力 NBT 群（hd/ にHD版）
└── requirements.txt
```

## セットアップ

```bash
cd /home/moriken/web-app/flood_pso
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# DEM XML は git LFS 管理
export PATH="$HOME/.local/bin:$PATH"   # git-lfs を local 導入した場合
cd ../kennkyuu20260114
git lfs install --local
git lfs pull --include "地形データ/FG-GML-503561-DEM5A-20250620/*.xml"
```

## 実行

```bash
cd /home/moriken/web-app/flood_pso

# 1. 既存 2変数版（IoU校正）
.venv/bin/python src/main.py

# 2. 高次元 標準PSO 単独
.venv/bin/python src/main_hd.py

# 3. 標準PSO vs CCPSO2 ベンチマーク（メイン）
.venv/bin/python src/benchmark.py
# 損失を切り替えるには環境変数 FLOOD_PSO_LOSS=iou または depth (default depth)

# 4. NBT 出力（実装系研究軸）
.venv/bin/python src/make_nbt.py --preset md_5m            # 2D ベースライン
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset md_5m   # PSO/CCPSO2/GT 比較
# プリセット: xs_overview / sm_5m / md_5m / lg_10m / xl_5m / huge_5m
# NBT 内 flood_pso_meta コンパウンドに手法・loss・dh_map・seed 等を全て埋め込み

# 5. 歩行可能な高精細ワールド（Layer C / 防災教育）
#    1m/block・真スケール(v_exag=1)・FG-GML 建物/道路を載せた「歩ける御坊」
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt --use-fgd
# --use-fgd : 国土地理院 FG-GML の BldA(建物)/RdEdg(道路) をローカルから配置（API不要）
#             ※初回は kennkyuu20260114 側で `git lfs pull` で BldA/RdEdg を実体化
# 既定の地形は GSI 5m を 1m に bilinear 補間（滑らか）。

# 5b. 真の 1m 地形（和歌山県 LiDAR グラウンド点群）で歩ける御坊
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt \
    --use-fgd --wakayama-grd data_cache/wakayama_lidar/06RC802_grd.txt --tag-suffix lidar
# --wakayama-grd : 系VI(JGD2011)点群を 1m 緯度経度グリッド DEM 化（src/wakayama_pcd.py、要 pyproj/pandas）
#                  GSI DEM1A は御坊未整備のため和歌山県オープンデータを利用。
#                  size/center 未指定ならタイル全域(2km×1.5km等)を出力。--wakayama-org で DSM(建物高さ)。
# --scale 1.3    : 1ブロックを細かく（1block≈0.77m）して全体を1.3倍に拡大（ブロック数~1.69倍、重い）。
#                  LiDAR もその解像度で再グリッド。h/v 両方を細かくする。

# 5c. 地表色を GSI 空中写真から（写真駆動の地表）
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt \
    --use-fgd --wakayama-grd data_cache/wakayama_lidar/06RC802_grd.txt --surface-ortho --tag-suffix lidar
# --surface-ortho : seamlessphoto を取得し、各地表セルを viewer 既知のバニラブロックへ色マッチ
#                   （草/砂/砂利/石/水/岩盤）。src/ortho_surface.py。NBT は Minecraft 互換のまま。

# 6. Litematica (.litematic) — make_nbt_hd は既定で .nbt と一緒に自動出力（redtact / Litematica mod 用）
#    抑止したいときだけ --no-litematic。既存 .nbt の個別変換は:
.venv/bin/python src/nbt_to_litematic.py results/nbt/hd/<file>.nbt [out.litematic]
# contiguous bit packing・YZX index・palette を redtact のローダ(@taku128/java-schematic)規約に厳密一致。
# _parse_fast 流用で高速（数秒）。redtact は .litematic/.litematica/.schem/.nbt いずれも読める。
```

## 歩行ワールド生成オプション一覧（`make_nbt_hd.py`）

御坊を真1m・写真地表・全リアル化で出す基本形（⭐＝2026-06 追加のリアル化オプション）:

```bash
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt \
  --use-fgd --wakayama-grd data_cache/wakayama_lidar/06RC802_grd.txt \
  --surface-ortho --trees --tree-mode sparse --evac --scale 1.5 \
  --center-lat 33.882 --center-lon 135.164 --width 1000 --depth 1000 --tag-suffix demo
```

**データソース**
| オプション | 既定 | 説明 |
|---|---|---|
| `--use-fgd` | off | FG-GML 建物(BldA)/道路(RdEdg) をローカル配置（API不要） |
| `--fgd-bld` / `--fgd-rdedg` | 503561 | 建物/道路 GML。**カンマ区切りで複数メッシュ**（境界タイルは `503551,503561`） |
| `--fgd-wa` ⭐ | 503561(ON) | 水域 WA/WStrA を水面に（河川・池）。空文字で無効 |
| `--wakayama-grd <grd[,grd…]>` ⭐ | – | 和歌山LiDAR 1m DEM。**カンマ区切りで複数図郭を mosaic**（タイル境界跨ぎ） |
| `--wakayama-org <org>` | 自動 | DSM(建物実高さ)の明示パス（無指定なら `_grd→_org` 自動） |
| `--use-esa` ⭐ | off | ESA WorldCover 土地利用を地表に重ね（cropland→田畑 等。rasterio 要） |
| `--use-osm` | off | OSM 建物/道路を Overpass から取得 |

**地表・見た目**
| `--surface-ortho` | off | GSI 空中写真色で地表ブロックを決定 |
| `--ortho-zoom` / `--ortho-saturation` | 18 / 1.4 | 写真解像度(18≈0.6m/px) / 彩度ブースト |

**樹木（LiDAR class3）⭐**
| `--trees` | off | 植生点→樹冠高で樹木配置（高さ別樹種＝低木birch/中木oak/高木spruce、緑フィルタ、建物/道路/水域/海は除外） |
| `--tree-mode canopy\|sparse` | canopy | canopy=密な森 / sparse=間引き個別木(球/円錐・千鳥配置) |
| `--no-veg-filter` | off | 建物DSMから植生class3を除外しない |

**建物・洪水**
| `--no-building-heights` / `--building-height <m>` | off / 6.0 | LiDAR実高さを使わず一律高さ / その値 |
| `--no-flood-barrier` ⭐ | off | 建物を浸水バリアにしない（既定は水が建物を避ける） |

**橋・避難 ⭐**
| `--bridges-json <json>` | gobo_bridges | OSM橋を立体化（桁+坂+橋脚）。空文字で無効 |
| `--evac` / `--evac-xml <gml>` ⭐ | off / P20-12_30 | 国土数値情報P20 指定緊急避難場所を緑光柱マーカーで配置 |

**範囲・解像度・出力**
| `--center-lat` / `--center-lon` | 御坊中心 | 出力エリア中心 |
| `--width <m>` / `--depth <m>` | preset | 東西 / 南北幅 |
| `--scale <r>` | 1.0 | 1.5 で 1block≈0.667m（細かく拡大、LiDARも再グリッド） |
| `--tiles RxC` | – | 重なりなくグリッド分割（大スケール時の OOM 回避） |
| `--tag-suffix <s>` / `--no-litematic` | "" / off | 出力名サフィックス / .litematic 抑止 |

> 注: 御坊中心の広範囲(建物1700+/水域多数)は描画が重く OOM 気味 → `--width/--depth` で範囲を絞るか `--tiles` で分割する。
> 建物の屋根に来た砂利/砂は andesite/sandstone に、海底の砂/砂利は最下に deepslate 土台層を地形なりに敷いて重力落下を防止（自動）。

## 主要結果（多シード平均）

| K | D=1+K² | Standard PSO loss (depth MAE) | CCPSO2 loss |
|---|---|---|---|
| 4 | 17 | **0.0007** | 0.0155 |
| 8 | 65 | **0.0135** | 0.0967 |
| 16 | 257 | 0.1629 | **0.0987** |

D=257 で CCPSO2 が逆転（loss −39%、IoU +1.9pt）、安定性（分散）でも勝る（CCPSO2 std 0.0133 < PSO std 0.0422）。CCPSO2 は適応的グループサイズ（候補集合 s∈{2,5,10,25,50,100,250} から改善停滞時に選び直す、Li & Yao 2012 準拠）を採用。詳細は `docs/05_ベンチマーク結果.md`。

> 注: 正則化なしの素のベンチでは CCPSO2 の Δh 復元（Δh_RMSE）は PSO より悪い（過適合）が、平滑化正則化 λ≈0.02 で PSO 同等（≈0.73）になり IoU 優位は全 seed で保たれる。詳細は `docs/LayerB_Δh逆問題とPP-PSO論文_整理.md` §6。

## NBT による研究成果アーカイブ

各最適化結果を Minecraft Structure NBT として保存。`flood_pso_meta` コンパウンドに **手法・パラメータ・loss・dh_map・seed・git_revision** を埋め込み、ファイル単体で実験条件が再現可能。詳細は `docs/06_NBT管理.md`。

```python
import nbtlib
f = nbtlib.load("results/nbt/hd/gobo_hd_K16_seed0_md_5m_ccpso2.nbt")
m = f["flood_pso_meta"]
print(m["method"], m["iou"], m["dh_map_shape"])
# → ccpso2  0.96  [16, 16]
```

## データ出典・ライセンス

本リポジトリおよび生成物（NBT/litematic/可視化）には第三者データを含む。公開・配布時は以下を明記すること。
データの役割・規約の詳細は **`docs/08_データとPSO位置づけ.md`**。

- **国土地理院**: 基盤地図情報（建物 BldA・道路 RdEdg・水域・5m DEM）、地理院タイル（空中写真 seamlessphoto・洪水浸水想定区域）を加工して利用。各利用規約に従い**出典明示が必須**（「出典：国土地理院」）。商用・大規模公開時は測量成果の承認要否を要確認。
- **和歌山県**: 3次元点群オープンデータ（航空レーザ測量, `06RC802`）を加工して利用。正確なライセンスは配布元で要確認、**出典明示**。原データは大容量のため `data_cache/`（.gitignore）で未収録。
- **OpenStreetMap**: 橋データ（`data_cache/osm/gobo_bridges_geom.json`）は OSM 由来。**© OpenStreetMap contributors**、**ODbL 1.0**。当 JSON を含む再配布は ODbL（継承）に従う。

> 出力は座標＋バニラブロック ID のみで、Minecraft のテクスチャ等アセットは含まない。
