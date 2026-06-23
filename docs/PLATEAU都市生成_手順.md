# PLATEAU 都市3Dワールド生成 手順（和歌山セッション まとめ → 大阪市へ）

> 2026-06-23 作成。別PC・別セッションで大阪市を生成するための引き継ぎ。
> このリポジトリ(flood_pso)の `src/` 変更込みでコミット済み。新PCでは **git pull → データ入手 → §4のコマンド** でOK。

---

## 0. このセッションでやったこと（要約）

和歌山市（和歌山城周辺）で、LiDAR真1m地形＋樹木＋空中写真を土台に **建物だけ2方式**の歩行ワールド(litematic)を生成した。

| 方式 | 建物ソース | フラグ |
|---|---|---|
| **いつも通り** | OSM footprint ＋ LiDAR DSM 高さ | `--osm-bld` |
| **PLATEAU 高精度** | CityGML 正確footprint＋実測高さ(measuredHeight)、用途別壁材、LOD2屋根 | `--plateau-bld` (+`--plateau-lod2`) |

追加実装：①用途(usage)・構造で壁材を9種に分化、②屋根色は空中写真の実色（既存 `color_building_roofs`）、③PLATEAU LOD2 の屋根形状を高さに反映（和歌山城天守＝309面/33mの連立天守を再現）。

**FG-GML が無い地域でも建物を載せられる汎用化**が肝（和歌山も大阪も FG-GML 同梱なし）。

---

## 1. 新PCのセットアップ

```bash
git clone <flood_pso のリポジトリ> flood_pso   # or git pull で最新化
cd flood_pso
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # 無ければ: numpy scipy matplotlib nbtlib pyproj Pillow pyswarms requests
```
- PLATEAU パーサは標準ライブラリ `xml.etree` のみ（追加依存なし）。
- matplotlib の日本語図が要るなら日本語フォント（Yu Gothic / Noto Sans CJK 等）。図に不要なら気にしない。
- **メモリ目安**：1タイルあたり 0.5–0.8GB の地形 + ブロックリスト。OOM 回避は §4.3。

---

## 2. 大阪市のデータ入手

### 2.1 PLATEAU CityGML（建物）
- **G空間情報センター**で「大阪市」PLATEAU（市コード **27100**）の **CityGML** を入手。
- 使うのは `udx/bldg/*.gml`（建物。`51〜` のメッシュコード名）。dem/tran/fld も同梱されるが今回は bldg だけでOK。
- パス例：`.../27100_osaka-shi_..._citygml_.../udx/bldg`

### 2.2 LiDAR DSM/DEM（地形・建物高さ・樹木）
- 大阪府/大阪市 または 国土地理院のオープンデータで **grd（地物除去後DEM）** と **org（DSM・class列付き）** のテキストCSV を入手。
  - 形式：`id,easting,northing,Z[,class]`（和歌山の `06QC502_grd.txt` / `_org.txt` と同形式）。
  - **座標系：平面直角座標系 第VI系 = EPSG:6674**（京都/大阪/奈良/和歌山 共通。和歌山と同じなので `wakayama_pcd.py` がそのまま使える）。
- **grd と org が別ディレクトリ**になりがち。両方のフルパスを控える（§4 で `--wakayama-grd` と `--wakayama-org` に渡す。引数名は wakayama だが汎用）。

> ⚠ 大阪は広大。**容量・メモリの都合で「必要な数km四方」だけ**入手・生成すること。全域は非現実的。

---

## 3. 追加されたコマンド（make_nbt_hd.py）

| フラグ | 意味 |
|---|---|
| `--plateau-bld <udx/bldg>` | PLATEAU 建物（高精度。用途別壁材＋実測高さ） |
| `--plateau-lod2` | LOD2 の屋根形状を高さに反映（城など。`--plateau-bld` と併用、やや重い） |
| `--osm-bld` | OSM 建物 footprint ＋ LiDAR 高さ（いつも通り版） |
| `--fgd-wa ""` | **必須**（FG-GML 水域を無効化。既定は Gobo の水域パスなので空にする） |

`--use-fgd` は使わない（大阪に FG-GML 同梱が無いため）。

---

## 4. 生成手順（コマンドテンプレート）

### 4.1 LiDAR の範囲を調べて中心を決める
```bash
# easting/northing の min/max
awk -F, 'NR==1{e1=e2=$2;n1=n2=$3}{if($2<e1)e1=$2;if($2>e2)e2=$2;if($3<n1)n1=$3;if($3>n2)n2=$3}END{print e1,e2,n1,n2}' <grd.txt>
# → pyproj で緯度経度に（EPSG:6674 → 4326, always_xy で (E,N)→(lon,lat)）
python3 -c "from pyproj import Transformer;t=Transformer.from_crs('EPSG:6674','EPSG:4326',always_xy=True);print(t.transform(<E>,<N>))"
```
→ LiDAR が覆う緯度経度範囲が出る。その中の市街中心や **ランドマーク（大阪城＝34.6873,135.5259）** を中心に選ぶ。

### 4.2 単一ワールド生成（小エリア。まずはこれで動作確認）
```bash
WAKA=<データ親ディレクトリ>          # 変数名は流用、実体は大阪
GRD=<.../grd.txt>; ORG=<.../org.txt>
BLDG=<.../udx/bldg>
# PLATEAU 高精度版（600m四方の例）
python3 src/make_nbt_hd.py --K 4 --seed 0 --preset gobo_walk_1km --methods gt \
  --center-lat <LAT> --center-lon <LON> --width 600 --depth 600 \
  --wakayama-grd "$GRD" --wakayama-org "$ORG" --surface-ortho --trees --tree-mode sparse \
  --plateau-bld "$BLDG" --fgd-wa "" --scale 1.5 --tag-suffix osaka_plateau
# いつも通り版（建物だけ差し替え。OSM=要ネット/Overpass）
#   上の --plateau-bld "$BLDG" を  --osm-bld  に置換、--tag-suffix osaka_osm
```
- `--surface-ortho`（空中写真の地表色＋屋根色）, `--trees`（class3 樹木）, `--scale 1.5`（1ブロック0.667m）は和歌山と同じ。
- 出力：`results/nbt/hd/gobo_hd_..._<tag>.litematic`(+.nbt)。**プレフィックスは "gobo_hd" 固定**なので `--tag-suffix` で区別し、後で `mv` でリネーム。

### 4.3 大きいエリアは分割（OOM回避）
- 目安：**1タイル 〜15M ブロック以下**（RAM12GB だと 19M 超で落ちた。新PCのRAM次第で緩められる）。
- scale1.5 だと **約500–760m/タイル** が安全。`--tiles COLSxROWS` で分割：
```bash
  ... --width 1000 --depth 1500 --tiles 2x2 ...   # 4タイル(各~500x760m)
```
- タイル名は `_r{行}c{列}`（r0=北/r1=南, c0=西/c1=東）。配置：`c→c は +タイル幅(X東)`, `r→r は +タイル深(Z南)` で密着。

### 4.4 ランドマーク（大阪城）を LOD2 屋根で
```bash
  ... --plateau-bld "$BLDG" --plateau-lod2 ...
```
- `--plateau-lod2` を足すだけ。LOD2 を持つ建物（城・主要建築）の屋根形状が高さに反映される。
- z幅≥20m の LOD2 は自動で「ランドマーク」＝白壁(white_concrete)扱い（城の漆喰）。
- **大阪城がどのタイルに入るか**は §4.1 で確認（34.6873, 135.5259）。そのタイルだけ `--plateau-lod2` で個別生成すれば軽い（城天守付近の単一タイルを center/width 指定で出す）。
- 確認：LOD2を持つメッシュかは `grep -c lod2 <mesh>.gml`。和歌山は 24/209 メッシュのみ LOD2 だった（大阪も一部のみの可能性）。

---

## 5. ハマりどころ（重要）

1. **`--fgd-wa ""` を必ず付ける**（無いと Gobo の水域GMLを探して失敗）。
2. **grd/org が別ディレクトリ** → `--wakayama-org` を明示（自動の `_grd.txt`→`_org.txt` 置換は同ディレクトリ前提）。
3. **座標系は EPSG:6674（系VI）**。関西は共通。違う県だと系が変わるので注意（中部=VII, 関東=IX 等）。
4. **OOM**：分割必須。落ちたら `Killed`。タイルを小さく。
5. **OSM版はネット必須**（Overpass）。地方都市は被覆が疎なことがある。PLATEAUはローカルで完結。
6. **PLATEAU の用途コード**は `udx/.../codelists/Building_usage.xml` で和訳確認可（411=共同住宅 等）。壁材マップは `src/plateau.py` の `USAGE_CAT` ＋ `terrain_render.BUILDING_WALL_BY_TYPE`。色を足したい時はここ。
7. **このデータセットのLOD上限はLOD2**（窓・壁の作り込み=LOD3/4は無い）。城も「屋根形状＋白壁」レベル。

---

## 6. メッシュコード→緯度経度（範囲特定に便利）
`src/plateau.py` の `mesh_bbox(code)` が標準地域メッシュ8桁→(lat,lon)範囲を返す。逆に、ある緯度経度を含むメッシュを探すには bbox を回して `mesh_bbox` で判定（`load_plateau_buildings` が内部でやっている）。

---

## 7. このセッションの成果物（和歌山。参考。大阪では作り直し）
- `results/nbt/hd/wakayama_城周辺600_{plateau,osm}.litematic`（600m比較）
- `results/nbt/hd/wakayama_東半分_{plateau,osm}_r{0,1}c{0,1}.litematic`（東半分2x2、計8）。`r1c1` は**城LOD2版**。
- 図：`results/nbt_preview/waka_bld_compare.png`（PLATEAU701 vs OSM496棟）、`waka_wall_materials.png`（用途別壁材）、`waka_castle_roof.png`（城LOD2屋根）。
- ※ nbt/litematic は容量大なので git には入れていない（コマンドで再生成可）。

## 8. 触ったコード（git に入っている）
- `src/plateau.py`（新規）：PLATEAU 建物ローダー、用途→壁材、LOD2 屋根抽出。
- `src/make_nbt_hd.py`：`--plateau-bld` / `--osm-bld` / `--plateau-lod2`、建物リスト読み込み。
- `src/nbt_export.py`：`building_list` 引数（建物ソースの汎用化）。
- `src/terrain_render.py`：用途別 壁/屋根パレット、`HIP_ROOF_TYPES`、`_rasterize_lod2_roof`、height_m=None 対応。
