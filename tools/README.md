# tools/ — 橋・トンネル・地形を直すための検証ツール

`src/nbt_preview.py` は **俯瞰(トップダウン)投影しか出せない**ため、橋のデッキ高・
トンネルの貫通・崖の穴といった **高さ方向の不具合が一切判定できなかった**。
ここにあるのはその穴を埋めるための道具。

| ツール | 何をするか | 入力 |
|---|---|---|
| `nbt_section.py` | 指定した線に沿った **鉛直断面** を PNG 化 + 空洞/水没/段差の統計 | `.nbt` / Anvil world dir |
| `bridge_profile.py` | 橋チェーンのデッキ高プロファイル図 + 埋没/水没/段差の PASS-FAIL | `BRIDGE_DUMP` の npz |
| `make_crop_dem.py` | 結合 mosaic DEM から小 bbox を切り出してローダのキャッシュに置く | `data_cache/wakayama_lidar/*.npz` |

すべて標準ライブラリ + 既存依存 (numpy / PIL / matplotlib / nbt) だけで動く。
`.venv/bin/python` で実行すること。

---

## 0. まず断面の読み方

`nbt_section.py` が出す画像の色の意味:

| 色 | 意味 |
|---|---|
| ブロック色 (`src/block_palette.py`) | 実際に置かれているブロック。地表・道路・建物・橋デッキが材質で見分けられる |
| **青** | 水 (blue/cyan stained glass, `minecraft:water`, ice 系) |
| **白に近い灰** | 空 (sky) = その柱の最上位固体より上の air |
| **黄** | **空洞 (cavity)** = 上下を固体に挟まれた高さ `--cavity-max`(既定16) 以下の air。<br>トンネル内空・建物内部・桁下がここに出る。**検査対象はこの黄色** |
| **ベージュ** | 上下は閉じているが背が高い air。山の内部シェル(地表の下 `deep_ground=8` しか地盤を書かないため中空)や高い高架の桁下 |
| **青灰** | VOID = 下が抜けている air（最下スラブより下・モデル化されていない地下） |
| **赤の斜線ハッチ** | **NO DATA** = その station の chunk / 列そのものが取れなかった（タイル外・未生成 chunk）。<br>**空(白)ではない**。ここは「貫通していない」のではなく「**見えていない**」 |
| **マゼンタ** | `--mark <ブロック名>` で強調したブロック（橋デッキなら `andesite`） |
| **濃いマゼンタ (220,40,220) 単色** | `block_palette.py` に無い未知ブロック。標準出力に `[warn] block_palette に無い…` が出る |

判定の目安:

* 橋が **埋没** → デッキのマゼンタが地形色に覆われる / `BURIED?(接している固体 >2 blocks)` が >0
* 橋が **水没** → デッキの上に青 / `SUBMERGED` が >0
* 橋に **段差** → デッキの折れ線が階段状に飛ぶ / `marked top y ... maxstep` が大きい
* トンネルが **貫通していない** → 黄色の帯が途中で切れる（**赤ハッチで切れているのは欠損**）
* トンネルが **平坦地で箱になっている** → 黄色の帯が地表より上に浮く
* 崖に **穴** → 崖の斜面に黄色が食い込む

> **`cover above` / `BURIED?` は「デッキに接している固体の連続段数」**。デッキの直上から
> 連続する固体だけを数え、空気に当たった時点で 0 に戻る。`--trees` の樹冠や OSM 送電線が
> デッキの数ブロック上を通っていても **埋没にはならない**（旧版はここを「その柱の最上位固体
> までの距離」で測っていたため、触れていない樹冠だけで `BURIED?` が立った）。
> 接していない上空の固体は `(参考) 最上位固体まで max=N block` として別行に出る。

---

## 1. 橋を直すときの手順

### 1-1. 速いループ用の小さい世界を作る

```bash
# (a) Makefile の crop ターゲット（南部4図郭から中心 400m 角を1タイル。Makefile の実測値で
#     初回 5分35秒 = 洪水 sim 込み、2回目以降 26s = sim キャッシュ(data_cache/inund_south4)再利用）
#     ※ 13分34秒 は `make world-test`(南部4図郭の中央2km角 2x2タイル)の実測値。crop ではない。
make crop CLAT=33.8355 CLON=135.1866 W=500

# (b) さらに速くしたいときは DEM 自体を小さく切る（洪水 sim も小さくなる）
.venv/bin/python tools/make_crop_dem.py secbridge 33.8320 33.8390 135.1820 135.1910
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km \
  --methods gt --scale 1.5 --no-building-heights \
  --wakayama-grd data_cache/wakayama_lidar/secbridge_grd.txt \
  --center-lat 33.8355 --center-lon 135.1866 --width 640 --depth 560 \
  --use-fgd \
  --fgd-bld  "../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101/FG-GML-503551-BldA-20260101-0001.xml" \
  --fgd-rdedg "../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101/FG-GML-503551-RdEdg-20260101-0001.xml" \
  --fgd-wa   "../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101/FG-GML-503551-WA-20260101-0001.xml" \
  --bridges-json data_cache/osm/gobo_bridges_south4_geom.json \
  --tunnels-json '' --power-json '' --parking-json '' \
  --no-litematic --tag-suffix sectest
```

> 南部4図郭には `gobo_bridges_full_geom.json` / `gobo_tunnels_geom.json` /
> `gobo_power_geom.json` / `gobo_parking_geom.json` が **無い**。
> 橋は `make` が図郭別 JSON を結合して作る `gobo_bridges_south4_geom.json` を使い、
> 他は空文字で無効化する（パスを渡してファイルが無いと `make_nbt_hd.py` は停止する）。

### 1-2. どの橋があるか調べる

```bash
.venv/bin/python tools/nbt_section.py --list-ways \
  --bbox 33.8320,33.8390,135.1820,135.1910
# way    385194099  L=  174m n=  6  c=(33.83550,135.18661)  hw=tertiary layer=1 bridge=yes
```

### 1-3. 橋 1 本の断面を出す

```bash
.venv/bin/python tools/nbt_section.py results/nbt/hd/..._sectest.nbt \
  --way 385194099 --margin 40 --mark andesite -o /tmp/bridge.png
```

`--margin 40` で way の両端を 40m 延長し、取り付け部（アプローチが地形に刺さって
いないか）まで見る。`--mark andesite` で橋デッキ・橋脚・欄干がマゼンタになる。

標準出力の見どころ（実測: `results/nbt/hd/..._zeniki_r10c4.nbt` の way 385194099）:

```
  y range      : 0 .. 57   stations=383 (with blocks: 274, no data: 109)
                                          ↑ 109 station はタイル外＝PNG では赤ハッチ
  top solid y  : min=37 max=57 maxstep=1 block  @10->11/383
                 (隣接有効 station 間のみ。欠損で 109 対をスキップ)   ← 地表の段差
  MARKED (--mark) : 604 cells on 259/383 stations
    marked top y  : 35 .. 56  maxstep=2 block @53->54/383            ← デッキの段差
    cover above   : min=0 median=0 max=2 block   ← デッキに**接している**固体の連続段数
    BURIED?(接している固体 >2 blocks): 0/259 stations                 ← 埋没
    SUBMERGED (water above marked top) : 0/259 stations              ← 水没
```

`maxstep` は **隣接する有効 station 同士**でのみ取る（欠損を跨いだ幻の段差を作らない）。
`@i->i+1/N` の N は全 station 数。

### 1-4. ワールドを書き出さずに数値だけ見る（最速）

`src/terrain_render.py::add_bridge_blocks` は環境変数を見ている:

```bash
BRIDGE_DUMP=/tmp/bd.npz make crop                       # make は環境をそのまま recipe に渡す
BRIDGE_DUMP=/tmp/bd.npz .venv/bin/python src/make_nbt_hd.py ... --bridges-json ...
BRIDGE_DEBUG=1          .venv/bin/python src/make_nbt_hd.py ...   # 1行ずつ標準出力へ
```

env-gate なので **未設定なら本番のオーバヘッドは 0**。

`BRIDGE_DUMP` は **ブロックを置く直前の解析値**（デッキ Y / 直下地形 Y / 橋脚底 Y /
水面 Y）を全 station 分そのまま保存するので、Anvil / NBT を書き出す前に判定できる。

```bash
.venv/bin/python tools/bridge_profile.py /tmp/bd.npz -o /tmp/bd.png
#   ok   main L=   318b n= 640 deckY[41,69] maxstep=1b buried=0 flush=76 submerged=0/0 low_pier=0
#   TOTAL: ... → defective chains 0/1
```

修正前後の比較と自動判定:

```bash
BRIDGE_DUMP=/tmp/bd_before.npz make crop      # 修正前
#   ...コード修正...
BRIDGE_DUMP=/tmp/bd_after.npz  make crop      # 修正後
.venv/bin/python tools/bridge_profile.py /tmp/bd_after.npz --compare /tmp/bd_before.npz
#   COMPARE vs bd_before.npz: chains 6→6  buried 676→0  submerged 3→0  maxstep 1→1

.venv/bin/python tools/bridge_profile.py /tmp/bd.npz --no-plot --fail-on-defect   # 不具合なら exit 1
```

判定項目: `buried`(dy<terr) / `submerged`(dy<wsurf) / `flush`(dy==terr) /
`maxstep`(隣接 station の段差) / `low_pier`(橋脚の底がデッキより上)。

### 1-5. まとめ（橋の推奨順序）

1. `make crop` （または `tools/make_crop_dem.py` + `make_nbt_hd.py`）で小さい世界を作る
2. `BRIDGE_DUMP` 付きで再生成 → `tools/bridge_profile.py` で **数値** を見る（速い）
3. 数値が通ったら `tools/nbt_section.py --way ... --mark andesite` で **実ブロック** を見る
   （BRIDGE_DUMP は解析値なので、実際に置かれたブロックの確認は断面が必要）
4. 直したら 2 に戻る。最後に `--compare` で before/after を残す

---

## 2. トンネルを直すときの手順

トンネルには `BRIDGE_DUMP` に相当する計装が無いので、**断面が唯一の判定手段**。

```bash
# どのトンネルがあるか
.venv/bin/python tools/nbt_section.py --list-ways \
  --geom-json data_cache/osm/gobo_tunnels_geom.json --bbox <lat0,lat1,lon0,lon1>

# 坑口から坑口まで way に沿って縦断（--margin で坑口の外まで延長）
.venv/bin/python tools/nbt_section.py results/anvil/gobo_crop \
  --center 33.8337,135.1789 --size 400x400 --scale 1.5 \
  --way <TUNNEL_WAY_ID> --geom-json data_cache/osm/gobo_tunnels_geom.json \
  --margin 30 -o /tmp/tunnel.png

# 内空の形を見る（拡大。--ymin/--ymax は分類の後に切るので空洞判定は壊れない）
.venv/bin/python tools/nbt_section.py <world> --from-block 680,493 --to-block 780,493 \
  --ymin 32 --ymax 62 --px 8 -o /tmp/tunnel_zoom.png
```

見るポイント:

* **黄色の帯が坑口から坑口まで途切れず続いているか** = 貫通しているか
  * ただし **赤の斜線ハッチ**で切れているのは欠損（タイル外/未生成 chunk）で、貫通の失敗ではない。
    標準出力の `no data: N` と `[warn] N/M stations had no chunk/column data` を必ず確認する
* 帯の上に必ず固体（天井）があるか = 密閉されているか（空が見えたら天井が抜けている）
* 平坦地で帯が地表より上に浮いていないか = 地表に箱が生えていないか
* 内空高さは 5〜9 block（`--cavity-max 16` の既定はこれを黄色にする値）

> **貫通確認は必ず `--thick 1`（既定）で**。`--thick>1` は法線方向の柱を **OR 合成**
> （どれか 1 本でも固体なら固体）するので、幅が thick 未満の内空は埋まって黄色が消える。
> 実測（合成シーン: 中央行だけに幅1・高さ5 の内空 20 station）:
> `--thick 1` → `CAVITY cells: 100 (stations with cavity: 20/60)` /
> `--thick 3` → `CAVITY cells: 0 (0/60)`。
> thick>1 を指定したときは thick=1 相当の空洞数を `[warn]` で出すので、差が出たら thick を下げる。

> 山の内部が一面ベージュになるのは **バグではなく仕様**（地表の下 `deep_ground=8`
> ブロックしか地盤を書かないので山は中空）。トンネルだけを見たいときは
> `--cavity-max` を 12 前後に下げると黄色がトンネルだけになる。

---

## 3. 崖の穴・その他の地形チェック

```bash
# 崖を横切る短い線を厚み 3 で切る（線が 1 ブロックずれても構造を捉える）
.venv/bin/python tools/nbt_section.py <world> \
  --from 33.83506,135.18690 --to 33.83506,135.18760 --thick 3 --px 6 -o /tmp/cliff.png
```

崖の斜面に黄色（空洞）が食い込んでいたらそこが穴。標準出力の
`top solid y ... maxstep=NN block @i->i+1/N` が大きい station が崖の位置。

`--thick 3` は線が 1 ブロックずれても崖の壁を捉えるための設定で、**穴(空洞)を探す用途では
過小評価になる**（柱を OR 合成するため細い穴が埋まる）。穴の有無を数えるときは `--thick 1`
と併用して `[warn] --thick 3 は … thick=1 なら CAVITY N cells` の差を見ること。

---

## 4. ジオリファレンス（緯度経度 ↔ ブロック座標）

| 入力 | 既定 | 明示指定 |
|---|---|---|
| `.nbt` | `flood_pso_meta` の `center_lat/center_lon/width_m/depth_m/h_res_m_per_block` から自動（実測 誤差 <0.5m） | 不要 |
| Anvil dir | メタが無いので **必須** | `--center LAT,LON --size WxD --scale 1.5`<br>または `--bbox lat_min,lat_max,lon_min,lon_max --size WxD` |

* `--center/--size/--scale` は **生成時に渡したのと同じ値** を書く。
* `--bbox` は生成ログの `[patch_bbox] lat[...] lon[...]` 行をそのまま使うと厳密。
* 緯度経度が分からない/要らないときは `--from-block X,Z --to-block X,Z`。
* `--tiles` で複数タイルを1ワールドに merge した Anvil はタイルごとに原点がずれるので、
  `--origin X,Z`（そのタイルの world ブロック原点）を併せて指定する。
* Anvil の y は **ワールド絶対 y**（`--world-base-y -50` なら -50 始まり）。
  `.nbt` の y は構造ローカル（0 始まり）。同じ場所でも軸のオフセットが違う。

`.nbt` は `flood_pso_meta` がファイル末尾にあるため初回だけ全体を流し読みする
（実測 59.9MB / 21M ブロックで 1.0s）。結果は `data_cache/section_cache/` に
キャッシュされ、2 回目以降は不要。

---

## 5. 注意

* 出力 PNG の既定先は `results/inspect/section_<stem>.png`。使い捨ての検証は `-o /tmp/...`
  へ出して results/ を太らせないこと。
* `make_crop_dem.py` が作る npz は `data_cache/wakayama_lidar/` に残る。
  使い終わったら `tools/make_crop_dem.py --clean <name>`。
* `--margin` で線を延長するとタイル外に出ることがある。その場合
  `[warn] N/M stations had no chunk/column data` が出て、その station は
  **PNG では赤の斜線ハッチ**、統計（`top solid y` の段差など）からは除外される。
  タイル分割された世界の橋/トンネルを丸ごと見たいときは、隣のタイルも `--origin` を
  合わせて切るか、merge 済みの Anvil world を使うこと。
* `--ymin/--ymax` は **表示だけ**を切る。空洞判定は常に全高で行うので、
  切った端で天井/床が消えて誤判定することはない（`--ymin > --ymax` はエラー、
  断面の y 範囲と重ならない指定は `[warn]` を出して全高を描く）。
* `--thick` は **奇数のみ**。偶数を渡すと `[warn]` を出して +1 した値で動く。
* m/block は `.nbt` なら `flood_pso_meta`、無ければ `--h-res` → `--scale` の順。
  どれも無いときは `--scale 1.5` を仮定して `[warn]` を出す（生成時の scale と違うと
  同じ 79 ブロックが 79m にも 53m にもなる）。Anvil は必ず `--scale` か `--h-res` を書くこと。
* Anvil の chunk キャッシュは LRU で 192 枚上限（region file は 8 個）。長い線を切っても
  メモリは頭打ちになる。実測（6000 ブロックの対角線）: キャッシュ 528 枚 24.3MB → 192 枚 5.8MB。

---

## 6. 実測（このツールを作ったときの検証）

南部4図郭 (`06RC904/913/002/011`) の LiDAR を 640×560m / 0.667m per block で生成し、
OSM way 385194099（日高川支流を渡る 174m の橋）で確認した:

* `nbt_section.py --way 385194099 --mark andesite`
  → デッキ Y=41→68 が谷（地形 Y=20）の上を連続、`buried=0` `maxstep=2block`、
    桁下が空洞として抜けている
* 同じ世界を Anvil でも出して両方の reader で切ったところ、
  cavity 3804 cells / big gap 8666 cells / `top solid maxstep=22 @48` が **完全に一致**
  （Anvil 側は y が -50 オフセット）
* `bridge_profile.py` を過去の既知バグ dump `results/inspect/bridgedump_hwbug.npz` に
  かけると `buried=676 low_pier=673 → defective 4/6` を検出し exit 1、
  修正後 dump `bridgedump_hwfix.npz` では `0/6` になることを確認
* 合成トンネル way を通して生成した世界では、断面の黄色い帯が坑口から坑口まで
  連続していること（＝貫通）と内空 8〜9 block を確認

---

## 7. 指標の修正（2 回目の検証で見つかった偽陽性・見落とし）

独立検証で出た 5 件を直した。合成シーン (`tools/` 外・使い捨て) と実タイル
`results/nbt/hd/..._zeniki_r10c4.nbt` (way 385194099, `--margin 40`) での修正前→修正後:

| 症状 | 修正前 | 修正後 |
|---|---|---|
| 触れていない樹冠/架線で埋没判定 | `cover max=6` `BURIED? 15/30` | `cover max=3` `BURIED? 5/30`（本物の埋没 5 station のみ）|
| `--thick 3` が細い空洞を消す | `CAVITY 0 (0/60)` と表示して終わり | 同じ 0 でも `[warn] thick=1 なら CAVITY 100 cells / 20/60` |
| 欠損と空が同色 | 右 28% が純白 (250,250,252) | 赤の斜線ハッチ + 凡例 `NO DATA x109 (not sky!)` |
| 欠損を跨ぐ幻の段差 | `maxstep=26 block @19/40`（N が詰めた後の数）| `maxstep=0 block`（`欠損で 31 対をスキップ`, N=70）|
| `minecraft:water` が UNKNOWN 色 | マゼンタ (220,40,220) = `--mark` とほぼ同色 | 青 (40,100,200) |
| 引数ミスが traceback / 無言 | `--bbox` 3 要素で `IndexError`、`--ymin>--ymax` は無視 | いずれも 1 行のエラーメッセージ |
| chunk キャッシュ無制限 | 528 枚 24.3MB（線の長さに比例） | LRU 192 枚 5.8MB で頭打ち |

実タイルでの回帰確認: `CAVITY 706 cells` / `big gap 3388` / `MARKED 604 cells on 259/383` /
`BURIED? 0` / `SUBMERGED 0` は修正前後で完全一致（変わったのは欠損の描画と段差の数え方だけ）。
