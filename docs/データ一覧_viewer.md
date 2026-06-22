# 多源データ一覧 ＋ viewer（御坊歩行ワールド生成 2026-06）

今回の生成（御坊全域 10 LiDAR図郭 × scale1.5・4分割 + 洪水校正GT）に使った全データ（**11種 A-K**。2026-06に I 水域 / J 避難所 / K ESA土地利用 を追加）を種類ごとにグループ化。データ撮影・図化の参照用。

## 1. 代表データ一覧（**役割ごと**に1つ。overview図 `results/data_preview/overview.png` の11役割に対応）

| # | 役割 | 内容 | 代表ファイル | 形式 | サイズ |
|---|---|---|---|---|---|
| 1 | 地形 | 地物除去後の真の1m地形(DEM)。起伏・標高 | `06RC802_grd.txt` | CSV 4列(系VI EPSG:6674/JGD2011) | 169MB×10 |
| 2 | 建物高さ | LiDAR DSM−DEM（class列で植生除外）＝建物の実高さ | `06RC802_org.txt`（class≠3を使用） | CSV 5列 (…,Z,class) | 967MB×10 |
| 3 | 樹木 | **同じ org の class3(植生)** の樹冠高 → 高さ別樹種を配置 | `06RC802_org.txt`（class3を使用） | CSV 5列 | （2と同ファイル） |
| 4 | 建物 | FG-GML footprint。type別壁・寄棟屋根・複数メッシュ union | `FG-GML-503561-BldA-20251001-0001.xml` | GML(XML) | 29.0MB |
| 5 | 道路 | FG-GML 道路縁。幹線=舗装/細道=砂利 | `FG-GML-503561-RdEdg-20251001-0001.xml` | GML(XML) | 11.5MB |
| 6 | 水域 | FG-GML WA/WStrA → 河川・池を水面に（`--fgd-wa`） | `FG-GML-503561-WA-20251001-0001.xml` | GML(XML) 面 | 1.9MB |
| 7 | 橋 | OSM bridge → 桁+坂+橋脚を立体化、橋下は洪水水位まで充水 | `gobo_bridges_geom.json` | GeoJSON(Overpass) | – |
| 8 | 地表色 | GSI 空中写真 → ~80バニラブロックへ色マッチ | `104805.jpg` | JPGタイル(seamlessphoto z18≈0.6m/px) | – |
| 9 | 洪水フォワード地形 | GSI 5m DEM。氾濫計算のフォワード地形 | `FG-GML-5035-61-02-DEM5A-20250620.xml` | GML(XML) 標高 | 0.7MB |
| 10 | 浸水GT | 洪水校正ベンチの浸水域(GT)。ワールドに重ねる | `case_K16_seed0.json` | JSON | 0.2MB |
| 11 | 土地利用 | ESA WorldCover。田畑/森/市街の補助（`--use-esa`, 既定OFF） | `ESA_WorldCover_10m_2021_v200_N33E135_Map.tif` | GeoTIFF 10m | 36.7MB |
| ＋ | 避難所(防災レイヤ) | 国土数値情報P20 指定緊急避難場所 → 緑の光柱マーカー(`--evac`)。**多源データ overview図には含めず、防災レイヤとして配置** | `P20-12_30.xml` | GML(XML) point | 0.3MB(御坊32件) |

> 役割で整理。**同じ LiDAR org から「建物高さ(DSM−DEM, 植生除外)」と「樹木(class3 樹冠高)」を別役割**として使う。
> 避難所は避難の目的地マーカー（防災レイヤ）で、多源データの overview 図からは除外（B方針）。

## 2. 種類ごとの全ファイル

### A. LiDAR 地形 (grd)
- **形式**: テキストCSV 4列: id, easting, northing, Z（平面直角座標系VI=EPSG:6674/JGD2011）
- **役割**: 地物除去後の真の1m地形(DEM)。歩行ワールドの起伏・標高
- **取得元**: 和歌山県 3次元点群オープンデータ(航空レーザ測量)

| ファイル | タイル/メッシュ | サイズ | 先頭サンプル |
|---|---|---|---|
| `06RC703_grd.txt` | 06RC703 | 169.4MB | `-1,-78929.02,-233794.24,5.75` |
| `06RC704_grd.txt` | 06RC704 | 160.9MB | `-1,-77758.21,-233659.96,2.15` |
| `06RC801_grd.txt` | 06RC801 | 150.2MB | `-1,-78122.96,-234752.97,0.03` |
| `06RC802_grd.txt` | 06RC802 | 133.3MB | `-1,-76317.15,-234792.77,11.90` |
| `06RC804_grd.txt` | 06RC804 | 161.0MB | `-1,-77804.43,-236094.57,-0.18` |
| `06RC902_grd.txt` | 06RC902 | 137.1MB | `-1,-76124.91,-237922.60,37.34` |
| `06RC904_grd.txt` | 06RC904 | 140.3MB | `-1,-76041.78,-239158.26,15.67` |
| `06RC913_grd.txt` | 06RC913 | 99.8MB | `-1,-75984.68,-239937.80,21.93` |
| `06SC002_grd.txt` | 06SC002 | 69.4MB | `-1,-76052.00,-240057.33,17.67` |
| `06SC011_grd.txt` | 06SC011 | 120.9MB | `-1,-75459.22,-240690.59,0.59` |

### B. LiDAR DSM (org)
- **形式**: テキストCSV 5列: id, easting, northing, Z, class
- **役割**: 建物・樹木込みの表層(DSM)。DSM−DEM=物体高(建物高さ)、class列で植生分離
- **取得元**: 和歌山県 3次元点群オープンデータ

| ファイル | タイル/メッシュ | サイズ | 先頭サンプル |
|---|---|---|---|
| `06RC703_org.txt` | 06RC703 | 967.5MB | `1,-79999.99,-232509.43,0.76,1` |
| `06RC704_org.txt` | 06RC704 | 896.0MB | `1,-78000.00,-232528.02,4.25,1` |
| `06RC801_org.txt` | 06RC801 | 784.0MB | `1,-79992.84,-234000.02,0.36,1` |
| `06RC802_org.txt` | 06RC802 | 997.5MB | `1,-77977.22,-234000.01,3.16,3` |
| `06RC804_org.txt` | 06RC804 | 945.2MB | `1,-77999.94,-235788.61,8.12,1` |
| `06RC902_org.txt` | 06RC902 | 1074.3MB | `1,-77999.96,-237056.66,-0.13,1` |
| `06RC904_org.txt` | 06RC904 | 758.6MB | `1,-77295.61,-238500.03,0.10,1` |
| `06RC913_org.txt` | 06RC913 | 1280.4MB | `1,-75999.94,-238777.08,33.26,1` |
| `06SC002_org.txt` | 06SC002 | 404.1MB | `1,-77999.98,-240047.07,0.29,1` |
| `06SC011_org.txt` | 06SC011 | 1247.9MB | `1,-76000.00,-240070.89,21.57,1` |

### C. FG-GML 建物 (BldA)
- **形式**: GML(XML)。建物外周ポリゴン + type(普通/堅ろう/無壁舎)
- **役割**: 建物フットプリント。high/type で壁材、LiDARで高さ。複数メッシュ union
- **取得元**: 国土地理院 基盤地図情報(基本項目)

| ファイル | タイル/メッシュ | サイズ | 先頭サンプル |
|---|---|---|---|
| `FG-GML-503561-BldA-20251001-0001.xml` | FG-GML-503561-BldA-20251001-0001.xml | 29.0MB | `—` |
| `FG-GML-503551-BldA-20260101-0001.xml` | FG-GML-503551-BldA-20260101-0001.xml | 6.6MB | `—` |

### D. FG-GML 道路 (RdEdg)
- **形式**: GML(XML)。道路縁ポリライン + type(真幅道路/徒歩道…)
- **役割**: 道路網。幅バッファで安山岩の路面に
- **取得元**: 国土地理院 基盤地図情報

| ファイル | タイル/メッシュ | サイズ | 先頭サンプル |
|---|---|---|---|
| `FG-GML-503561-RdEdg-20251001-0001.xml` | FG-GML-503561-RdEdg-20251001-0001.xml | 11.5MB | `—` |
| `FG-GML-503551-RdEdg-20260101-0001.xml` | FG-GML-503551-RdEdg-20260101-0001.xml | 3.5MB | `—` |

### E. GSI 5m DEM (DEM5A)
- **形式**: GML(XML) 標高グリッド。94 メッシュ分割
- **役割**: 洪水校正(PSO)のフォワード地形。25m に粗化して氾濫計算
- **取得元**: 国土地理院 基盤地図情報(数値標高モデル DEM5A)
- **枚数**: 94 メッシュ xml。代表: `FG-GML-5035-61-02-DEM5A-20250620.xml`

### F. GSI 空中写真 (ortho)
- **形式**: JPGタイル(地理院タイル seamlessphoto, zoom18≈0.6m/px)。1978 枚キャッシュ
- **役割**: 地表色・屋根色・橋デッキ上面。~80バニラブロックへ色マッチ
- **取得元**: 国土地理院 地理院タイル(seamlessphoto)
- **枚数**: 1978 タイル（zoom18, x/y 階層）。代表: `data_cache/gsi_ortho/18/229474/104805.jpg`
- 合計 ≈ 24MB

### G. OSM 橋 (bridge geom)
- **形式**: GeoJSON(Overpass out geom)。way bridge=yes + highway + layer + width
- **役割**: 橋の位置・幅・層。Tellus流に桁+坂+橋脚を立体化
- **取得元**: OpenStreetMap (Overpass API)  © OpenStreetMap contributors, ODbL

| ファイル | タイル/メッシュ | サイズ | 先頭サンプル |
|---|---|---|---|
| `gobo_bridges_002_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_011_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_703_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_704_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_801_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_804_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_902_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_904_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_913_geom.json` | gobo | 0.0MB | — |
| `gobo_bridges_geom.json` | gobo | 0.0MB | — |

### H. ベンチ case (GT浸水)
- **形式**: JSON。PSO/CCPSO2/GT の最適化結果(water, dh_map, IoU, loss…)
- **役割**: 歩行ワールドに重ねる浸水域(GT)。洪水校正ベンチの生データ
- **取得元**: src/benchmark.py 出力(本研究で生成)

| ファイル | タイル/メッシュ | サイズ | 先頭サンプル |
|---|---|---|---|
| `case_K16_seed0.json` | case | 0.2MB | — |

## 3. viewer（データ別の見方・撮影方法）

| 種類 | 見方 / 可視化方法 |
|---|---|
| A. LiDAR 地形 (grd) | src/wakayama_pcd.py でグリッド化→matplotlib imshow（標高カラー） |
| B. LiDAR DSM (org) | 同上。class==3 を色分け、DSM−DEM を物体高として可視化 |
| C. FG-GML 建物 (BldA) | src/fgd_vector.py で読み→matplotlib でポリゴン塗り |
| D. FG-GML 道路 (RdEdg) | fgd_vector で読み→matplotlib で線描画 |
| E. GSI 5m DEM (DEM5A) | src/dem_parser.py(mosaic_tiles)→matplotlib imshow |
| F. GSI 空中写真 (ortho) | JPG を直接表示 / タイル結合してモザイク |
| G. OSM 橋 (bridge geom) | geojson.io にドロップ / matplotlib で polyline |
| H. ベンチ case (GT浸水) | json 読み→dh_map/浸水mask を matplotlib |

> LiDAR/DEM/FG-GML/浸水は matplotlib で画像化、空中写真は JPG 直接、OSMは geojson.io が手軽。各データの代表プレビューPNGも生成可能（要望があれば作成）。