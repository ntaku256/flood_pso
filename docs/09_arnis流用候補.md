# arnis → flood_pso 流用候補カタログ（御坊地形生成）

> arnis（Rust製・OpenStreetMap+標高→Minecraftワールド生成, v2.9）の地形生成コードを7軸で精読し、
> flood_pso（御坊市 DEM5A/FGD/オルソ/点群 → 歩けるMinecraft NBT生成）に流用できる箇所を抽出したもの。
> マルチエージェント調査（7軸×抽出→統合）の結果を、各項目の根拠・arnis参照(file:line)・移植時の落とし込み先付きで保存。

**全体結論**: 最も価値が高いのは `arnis/src/elevation/postprocess.rs` のDEM後処理アルゴリズム群。
docs/07が自認する「TerrainAnomalyRepair未移植」「外れ値/NaN処理の欠落」をそのまま埋める純アルゴリズムで、
Rust固有要素（所有権共有・rayon・f32最適化・symlink処理）は捨て、scipy/numpy/skimage/rasterio/pyprojの既存関数か数十行の純numpyに低〜中コストで落とせる。
校正本体（PSO/CCPSO2/inundation）には不干渉で、前処理・レンダ・書き出し層のみを改善できる。

凡例: 🔴 must=最優先 / 🟡 nice=効果あり余裕があれば / ⚪ skip=参考程度。コストは low/medium/high。

## おすすめ流用ポイント Top8

| # | 施策 | 優先 | コスト | arnis参照 | flood_pso適用先 |
|---|------|------|--------|-----------|-----------------|
| 1 | 5x5中央値+MAD反復で地形異常修復（尾根保存しつつ継ぎ目スパイク/点群誤反射を除去） | must | med | `postprocess.rs:22` | `dem_parser.py`/`wakayama_pcd.py`出口 |
| 2 | 反復3x3平均でNaN穴埋め（方向性アーティファクト無し） | must | low | `postprocess.rs:1061` | `dem_parser.mosaic_tiles` |
| 3 | IQR×3+5%ガードの外れ値クリップ | must | low | `postprocess.rs:1117` | 全DEM経路の前段 |
| 4 | `{pos,state}`列挙→密3D numpy配列（8-12GB問題の根治） | must | med | `world_editor/common.rs:84-364` | `nbt_export.py` |
| 5 | タイル原点を固定ブロックステップへfloor整列+halo描画 | must | med | `tile.rs:56-260` | `make_nbt_hd.py:331-358` |
| 6 | floodfillを `nd.label+isin`→`binary_propagation`化（PSO内側ループ高速化） | must | low | `floodfill.rs:115-197` | `flood_sim.py` |
| 7 | 色マッチを RGB-L2 → Oklab知覚距離 | must | low | `colors.rs:114` | `ortho_surface.classify_rgb_to_palette` |
| 8 | 二値水マスクを σ=3 gaussian+0.5アイソラインで軟化 | must | low | `land_cover.rs:104` | `terrain_render.py` |

---

## 実装メモ（施策①実装時の実測補正 2026-06-26, branch `feat/arnis-dem-postprocess`）

軸1の must 3点を `src/dem_postprocess.py` に移植し、`dem_parser.mosaic_tiles` /
`wakayama_pcd.load_wakayama_dem` の出口に配線（環境変数 `FLOOD_PSO_DEM_POSTPROCESS=0`
で無効化可、wakayama は `_pp` 別キャッシュ）。実 DEM（FG-GML-503561, 1491×2241）で
検証した結果、arnis 既定そのままでは御坊地形に**不適合な点が2つ**判明し補正した:

- **IQR 外れ値除去（Top8 #3）は既定 OFF にした。** 御坊 DEM は平野(中央値51m)から
  山地(最大488m)まで標高が連続分布し、山頂は全体の0.86%しかない。arnis の
  「分布外側<5%＝破損」前提が崩れ、IQR 上側バウンド(Q3+3IQR≈348m)が**実在の山頂
  約2.3万セルを誤って NaN 化**してしまう（max 488→348m に decapitation）。DEM5A は
  NoData を parse 段で NaN 化済みなので、突出値はローカルな LiDAR スパイク/継ぎ目段差
  のみ。これは**グローバル IQR でなくローカル MAD 修復（#1）で安全に除去**できる。
  IQR は明確な大域破損が分かっている時だけ `outliers=True` で opt-in。
- **NaN 補間（Top8 #2）は `fill_skip_border=True` 必須。** 御坊は沿岸で、NaN 662k の
  うち**65万セルが単一の巨大成分＝海・データ外**（縁連結）、内部の真の欠損は約1万セル
  だけ。全 NaN を埋めると**海を陸の標高で塗りつぶす**（fill が467反復に膨張）。縁連結
  成分を除外することで、内部欠損(約1万)だけ8反復で補間し、海(65万)は NaN 保持→下流の
  海陸分離を維持。MAD 修復(#1)は NaN セルに書き込まないので山頂・海とも安全。

結果（既定 = MAD 修復 + 内部 NaN 補間, IQR off）: max 488.3m 保存・ローカルスパイク
195セル修復・内部欠損のみ補間で海 65万セルは NaN 保持。`src/dem_postprocess.py` の
`_selftest()` で合成データの両経路を検証済み。

---

## 軸別 詳細カタログ

### 軸1: 標高/DEMパイプライン（取得・複数ソース選択・欠損補間・平滑化・キャッシュ）

**flood_pso 現状**: DEM取得は3経路に分散し統合層がない。(1) dem_parser.py: 国土地理院DEM5A GMLをパースしモザイク合成。NoData(<=-9000)→NaN、エッジ欠損はNaNパディング、モザイクは「先勝ち上書き」(mosaic_tiles L119-123)で重なり平均なし。downsampleはnanmeanブロック平均のみ。外れ値除去・継ぎ目修復・NaN補間は一切なし(NaNが残る)。(2) wakayama_pcd.py: 県1m点群をbincountでセル平均グリッド化、内部NaN欠損はscipy distance_transform_edtの最近傍コピーで穴埋め(_fill_nan_nearest L56)、.npzキャッシュ、LiDAR class列で植生除外。(3) tellus_data.py: Mapzen terrarium PNGタイル取得+モザイク、reproject_to_grid(map_coordinates bilinear)、_http_get_with_retry(指数バックオフ,404即raise)、ディスクタイルキャッシュ(期限切れ/破損検知なし)。GSIオルソ・ESA WorldCoverも同様。docs/07で既にTellus由来のcliff-aware smoothing等を移植済だがTerrainAnomalyRepairは「未移植」と明記。docs/精度向上候補で「Mapzen+GSIハイブリッド(高解像+連続性)」が将来課題。複数ソースのcoverage/解像度ベース自動選択・フォールバックの仕組みは無い。スケーリングはv_exag手動指定でビルド高フィット自動圧縮なし。

**arnis 読込ファイル**: `src/elevation/mod.rs`, `src/elevation/selector.rs`, `src/elevation/provider.rs`, `src/elevation/postprocess.rs`, `src/elevation/cache.rs`, `src/elevation_data.rs`, `src/elevation/providers/aws_terrain.rs`

#### 1. [🔴 must / medium] 5x5中央値+MAD(中央絶対偏差)による反復的な地形異常修復。中心値が|center-median|>6m かつ >3×MAD のときだけ近傍中央値で置換し、尾根/谷の実地形は保存。10パス反復で多画素アーティファクト塊を外周から侵食。

- **なぜ御坊に効くか**: flood_pso最大の欠落。dem_parserはDEM5Aタイル継ぎ目やLiDAR誤分類スパイク(docs/07の針葉樹ピン化原因の高周波成分)を一切除去せずNBT化している。docs/07で『TerrainAnomalyRepair未移植』と自認している穴をそのまま埋める。MAD基準なので日高川の谷壁や堤防の実エッジを潰さない。
- **arnis参照**: `postprocess.rs:22 repair_terrain_anomalies`
- **適合方法（移植）**: numpy化容易。scipy.ndimage.generic_filterは遅いので、5x5近傍をnp.lib.stride_tricks.sliding_window_viewで取り出し、軸方向にnp.median、MAD=median(|x-med|)をベクトル計算。閾値マスクで置換し2-3パス反復。NaNはマスクで除外。閾値6m/3xはDEM5A(5m格子)に合わせ要調整(継ぎ目は数m段差)。

#### 2. [🔴 must / low] 反復3x3近傍平均によるNaN穴埋め。各反復前にスナップショットを取り走査順バイアスを排除、収束まで(変化なしまで)ループ。

- **なぜ御坊に効くか**: dem_parser.mosaic_tilesはNaNを残したまま返し下流が海と誤認する(docs/07で言及)。wakayama_pcdの_fill_nan_nearestは単一最近傍値のコピーで、広い欠損で平坦ブロック/方向性アーティファクトを生む。反復平均は滑らかに拡散補間し、エッジタイルのNaNパディング・点群取りこぼし両方に効く。
- **arnis参照**: `postprocess.rs:1061 fill_nan_values`
- **適合方法（移植）**: low。scipy.ndimage.uniform_filterでNaN→0版とマスク版を畳み込み、count>0セルだけsum/countで更新、NaNが消えるまでwhileループ。dem_parser/wakayama_pcd両方の出口に共通関数として挿入。

#### 3. [🔴 must / medium] 複数標高プロバイダの解像度順選択+カバレッジ判定+空データ自動フォールバック。ElevationProviderトレイト(coverage_bboxes/native_resolution_m/fetch_raw)で抽象化し、解像度の細かい順に最初にbboxが重なるものを選ぶ。返ってきた格子のNaN率>0.5なら粗いグローバルソースへ自動退避。

- **なぜ御坊に効くか**: docs/精度向上候補が掲げる『Mapzen+GSIハイブリッド: 御坊中心はGSI/県1m、外周・海域はMapzenで接続』を実現する設計そのもの。現状3経路が手動分岐で、生成中心がLiDARタイル外だとDEM空→ZeroDivisionError(docs/07 8.5で既知バグ)になる。NaN率フォールバックがこれを構造的に防ぐ。
- **arnis参照**: `selector.rs:26 select_provider / provider.rs:14 ElevationProvider / mod.rs:133-177 + compute_nan_ratio mod.rs:289`
- **適合方法（移植）**: medium。Pythonでは抽象基底class/Protocolで Provider(coverage_bbox, res_m, load(bbox,shape)->dem) を定義し、[Wakayama1m, GSI5m, Mapzen]を解像度順リスト化。各経路の既存関数(load_wakayama_dem/mosaic_tiles/fetch_mapzen_dem)を薄くラップ。np.isnan(dem).mean()>0.5で次段へ。座標系はWGS84 bboxに統一(wakayamaはpyproj変換済)。

#### 4. [🔴 must / low] IQR×3グローバル外れ値クリップ+5%カウントガード。Q1-3IQR/Q3+3IQRを超える値をNaN化するが、超過セルが全体の5%超なら『実地形(谷底/山)』とみなしそのバウンドをスキップ。修復後fill_nan_valuesで穴埋め。

- **なぜ御坊に効くか**: 点群(wakayama_pcd)のマルチパス誤反射や水面のスペックル、DEMの破損値を低コストで除去。カウントガードにより日高川の連続した低標高谷底や海面下の値を誤って消さない。flood_pso全DEM経路に外れ値除去は皆無。
- **arnis参照**: `postprocess.rs:1117 filter_elevation_outliers`
- **適合方法（移植）**: low。np.nanpercentile(dem,[25,75])でIQR算出、below/above count比較、マスクでnp.nan代入後に上記fill_nan。repair_terrain_anomaliesの前段に置く(arnisと同順: outlier→anomaly→fill)。

#### 5. [🟡 nice / medium] 水面のヒストグラムモード平準化+流水の局所中央値勾配保存。連結成分のIQR>5mなら『流水(勾配あり河川)』と判定し半径12セルの局所中央値で各セルの水面を作り勾配維持、IQR小さければ『止水』とみなしヒストグラム最頻ビン(MODE_BIN=1m)で単一水面に平坦化。上下尾を持つDSM汚染にロバスト。

- **なぜ御坊に効くか**: 洪水研究の本丸=日高川。日高川は勾配を持つ流水なので、単純な一律平坦化では下流端の壁標高に全河川がクランプされ『峡谷を横断する平帯』アーティファクトが出る。局所中央値法はその回避策。flood_psoはFG-GML WA/WStrA水域面とESA water(精度向上候補で『手元・未使用』)を持つので、ESAより正確な水マスクで適用できる。
- **arnis参照**: `postprocess.rs:249 level_water_surfaces / 472 local_water_median / 544 histogram_mode`
- **適合方法（移植）**: medium。水マスクはFG-GML WA/WStrAポリゴンをラスタ化 or ESA cover==80。scipy.ndimage.labelで連結成分、各成分のnp.percentileでIQR、流水はscipy.ndimage.median_filterを水マスク限定で(半径12)。表示用DEMに適用。ただし校正用inundationは別レイヤ生成なので校正値には不干渉(レンダ層のみ)。

#### 6. [🟡 nice / medium] 羽毛化マスクによるGaussian平滑ブレンド。二値の対象クラスマスク自体を同じカーネルでぼかし0..1重みにし out=(1-m)*orig+m*blur で境界の継ぎ目をなくす。separable(水平→垂直)Gaussianはエッジで有効サンプルだけ重み再正規化し縁の暗化を防ぐ。σ<1.5セルなら平滑無効化。

- **なぜ御坊に効くか**: flood_psoは既にcliff_aware_smooth(docs/07)を持つが二値ON/OFFで市街地境界に継ぎ目が出やすい。羽毛化マスクのlerpは市街地(ESA built=50)だけ平滑し自然地形のエッジを残す『部分平滑』をシームレスに行える。edge再正規化はNaN/タイル境界を含むDEMで縁アーティファクトを防ぐ実装テク。
- **arnis参照**: `postprocess.rs:833 smooth_built_up_gaussian / 937 gaussian_blur_grid_reported`
- **適合方法（移植）**: nice。scipy.ndimage.gaussian_filterはNaN非対応なので、arnis同様にNaN→0の値とマスクを別々にぼかしsum/wsumで割る自作関数が必要(=NaN-aware gaussian)。これは現状cliff_aware_smoothのNaN処理改善にも流用可。

#### 7. [🟡 nice / low] 実標高レンジをビルド可能Y幅へ自動圧縮スケーリング。ideal=range*scaleが利用可能Y幅を超えたら compression=avail/range で圧縮し、圧縮比(1block=Xm)をログ。アフィン係数(min_height_m, blocks_per_meter)を返し逆変換(雪線等の閾値→Y)を可能に。

- **なぜ御坊に効くか**: flood_psoはv_exag手動(docs/07で3→1.5に試行錯誤)。scale1.5タイル分割でビルド高319を超える山地が出ると現状クランプで頂上潰れ。自動圧縮なら超過時のみ縮め、収まる時はフル誇張を保つ。blocks_per_meter保持は浸水深→Y変換の一貫性にも使える。
- **arnis参照**: `postprocess.rs:1202 scale_to_minecraft`
- **適合方法（移植）**: nice。純粋なnumpyスカラ計算でほぼそのまま移植可。MAX_Y/TERRAIN_BUFFERを御坊の地盤レベル設定に合わせる。タイル間で同一min/max/圧縮比を共有しないと境界段差が出るので、全タイル共通の標高レンジを先に計算して渡す設計が必要(連結窓整合の延長)。

#### 8. [🟡 nice / low] タイルキャッシュの破損/切詰検知と年齢ベース自動掃除。キャッシュtileはサイズ<1000バイトor画像デコード失敗で再DL。7日超の古いタイルを再帰的に削除(symlinkは辿らない安全実装)。

- **なぜ御坊に効くか**: tellus_dataのMapzen/GSIオルソタイルキャッシュは破損検知も期限切れも無く、途中で切れたPNGが永続化すると黒タイルが地形に残る。研究で長期運用するとdata_cacheが肥大化する。
- **arnis参照**: `aws_terrain.rs:304 fetch_or_load_tile / cache.rs:166 cleanup_old_cached_files`
- **適合方法（移植）**: low。fetch_mapzen_tile/fetch_gsi_ortho_tileに os.path.getsize<1000 と PIL.Image.open例外時の再DL分岐を追加。掃除はpathlibでst_mtime比較しunlink。symlink安全実装まではPython研究用途では不要。

#### 9. [⚪ skip / medium] 海岸陸地の水面方向プルダウン(マルチソースBFS)。確定水面セルを種にBFSで距離と水位を伝播、max_distance内の陸セルを距離重みで水位へ線形に引き下げ。ただしorig-水位>15mは実崖として除外。

- **なぜ御坊に効くか**: 紀伊水道側の海岸線で、DSM由来の建物高バイアスが水際に『立ち上がる傾斜』を作る問題への対策。docs/07で海岸表現改善を課題にしているので、海岸の階段状崖の緩和に使える。
- **arnis参照**: `postprocess.rs:729 pull_coastal_land_toward_water`
- **適合方法（移植）**: skip寄り。御坊の主対象は日高川氾濫で海岸は周縁。実装はscipy multi-source距離変換+水位伝播でやや手間。優先度は流水処理より低い。

**この軸で流用しない/不要なもの**:

- f64→f32ダウンキャストによるメモリ半減(mod.rs:29,265): arnisは16384²≈2GB格子が前提だが、flood_psoは単一都市タイル(数千²)で点群/GMLも小さくメモリ律速でない。f32化は得が薄い。
- シンボリックリンク安全なキャッシュ削除/symlink_metadata厳密処理(cache.rs:63-156): OSS配布のセキュリティ要件で、研究用ローカルスクリプトには過剰。Pythonのshutil/pathlibで十分。
- rayon並列(postprocess全域のpar_iter): numpyベクトル化で同等以上の速度が出るため明示並列は不要。移植時は並列構造を捨ててベクトル化するのが正解。
- AWS terrarium直接bilinearサンプリング+タイル境界クロスオーバ(aws_terrain.rs:115-203): flood_psoは既にfetch_mapzen_demで全mosaic構築→reproject_to_grid(map_coordinates)を行っており、より単純で等価。巨大bboxのメモリ最適化(mosaicを作らず直接サンプル)が要る場面でのみ参考。
- OSM/land_coverプロバイダ取得インフラ(selectorのIgnFrance/Usgs3dep等): 海外WMS/WCS前提で御坊には無関係。flood_psoはGSI/県点群/Mapzenの独自ソースを持つ。
- GUI進捗emit(emit_gui_progress_update, gaussian_blur_grid_reportedのreportコールバック): CLIスクリプトのflood_psoには不要。

> 軸まとめ: arnisのpostprocess.rsはflood_psoのDEMパイプラインの最大の弱点(docs/07が自認する『TerrainAnomalyRepair未移植』と外れ値/NaN処理の欠落)を埋める純アルゴリズム集で、移植価値が最も高い。must級は4つ: (1)5x5中央値+MAD反復修復(postprocess.rs:22)でDEM5A継ぎ目スパイクと点群誤反射を尾根保存しつつ除去、(2)反復3x3平均NaN補間(postprocess.rs:1061)でdem_parserが残すNaNと最近傍コピーの方向性アーティファクトを解消、(3)IQR×3+5%ガードの外れ値クリップ(postprocess.rs:1117)、(4)解像度順provider選択+NaN率0.5フォールバック(selector.rs:26/mod.rs:133)でdocs/精度向上候補の『GSI+Mapzenハイブリッド』とZeroDivisionErrorガードを同時実現。いずれもscipy/numpyで低〜中コスト移植可能で、arnisのパイプライン順(outlier→anomaly→NaN-fill→landcover平滑→scale)をそのまま御坊DEM生成の前処理段に挿入するのが推奨。洪水研究の本丸である日高川には流水の局所中央値水面処理(postprocess.rs:249)がnice級で刺さる(FG-GML水域面マスクを併用)。スケール自動圧縮(postprocess.rs:1202)はscale1.5タイル分割のビルド高超過対策に有用。Rustのrayonとf32化・symlink安全キャッシュは文脈差で不要。

---

### 軸2: 地形ボクセル化/グラウンド生成

**flood_pso 現状**: 中核は terrain_render.py:dem_to_blocks_enhanced (724-1355)。(1) cliff_aware_smooth でDEM全解像度にgaussian、slope>=0.4の崖は元値保持、(2) nanmean で h_res_block へダウンサンプル、(3) make_sea_mask(NaN or elev<=sea_level)で海陸分離、(4) 海は単一の scipy distance_transform_edt(陸シード)で海岸距離→depth_per_m=max/200の線形+グローバルmax_depth、海底に sand/gravel、(5) classify_surface_block_grid で標高/slope(中央差分gradient)/convexity(ラプラシアン)/海岸距離のハード閾値で sand/gravel/stone/grass、(6) 陸柱は固定 deep_ground=8 の stone、(7) 浸水水柱・建物(空洞化・窓・寄棟屋根、LOD2は_rasterize_lod2_roofで平面lstsqのheightfield)・樹木・橋・凡例層。各ブロックをPythonループで nbtlib.Compound として append。make_nbt_hd.py がタイル分割して書き出す。

**arnis 読込ファイル**: `src/ground.rs`, `src/ground_generation.rs`, `src/water_depth.rs`, `src/models_3d/voxelize.rs`, `src/models_3d/pipeline.rs`, `src/bedrock_block_map.rs`

#### 1. [🔴 must / low] 近傍参照の可変深さアンダーフィル: 各列のstone柱の深さを「8近傍の最低地表Yまで+1」をclamp(2,64)で決め、崖面はその差分だけ充填、平地は2ブロックで済ます

- **なぜ御坊に効くか**: flood_psoは固定 deep_ground=8。doc07は『deep_ground=8は地下stoneを増やすだけで見た目に寄与しない』と認めつつ+77%ブロック増を招いている(07章実測)。一方で8より高い崖では依然voidが出る。近傍最低Y基準にすれば平地のブロックを激減させ(=NBT軽量化・タイル分割で効く)、かつ崖面の底抜けを完全に塞げる。性能と品質の両弱点を同時解消
- **arnis参照**: `ground_generation.rs:716-758 (depth計算) と 1076-1103 (全列を確実にsealする無条件パス)`
- **適合方法（移植）**: scipy.ndimage.minimum_filter(y_surf_land, size=3) で8近傍最低Yグリッドを作り、各列で fill 範囲を max(_lift, neighbor_min)..y_top-1 に。numpyでベクトル化可能。海底柱(945-948)にも同手法を適用

#### 2. [🔴 must / low] 二値マスクを先にgaussianで平滑化→bilinearサンプル→0.5等値線でハード閾値、で海岸/河川/土地被覆境界を曲線contour化する(矩形grid edge除去)

- **なぜ御坊に効くか**: flood_psoは water_mask/cover_ds/sea_mask を最近傍の二値でそのまま地表に焼くため、ESA10mやFGD水域境界が四角いギザギザになる。doc07の弱点『海岸線が雑』に直結。DEMにはcliff_aware_smoothを掛けているのにマスク側は平滑化していないのが穴。マスクをgaussian+閾値0.5にすると御坊海岸線・日高川の縁が滑らかな曲線になる
- **arnis参照**: `ground.rs:295-335 water_blend と ground.rs:514 refresh_water_blend_grid、ground_generation.rs:383-384 is_esa_water判定`
- **適合方法（移植）**: scipy.ndimage.gaussian_filter(mask.astype(float), sigma) を sea/water/cover_water に適用し >0.5 で再二値化してから surf_block へ焼く。cover_ds の最近傍ダウンサンプル(805)もこの blur+threshold に置換できる

#### 3. [🔴 must / medium] 決定論的value noise(4隅coord_hash+smoothstep bilinear)で表層素材を有機的blob化し、細かいper-block hashでpeek-through。scree/bare/混合を単色面でなく石・砂利・粗土の自然な斑に

- **なぜ御坊に効くか**: flood_psoの classify_surface_block_grid はハード閾値で領域が均一ブロックになり、doc07が繰り返し挙げる『緑/灰の柱が密集する針葉樹林のような見た目』『地表が不自然』の主因。value noiseで石/砂利/粗土を~5-6ブロック解像度の有機的斑に散らすと、山地のstone一色(07章で70%)や海岸gravelベルトの単調さが緩和。ESAなし運用でもDEM由来量だけで効く
- **arnis参照**: `ground_generation.rs:1227-1251 value_noise_01、利用箇所479-557(LC_SHRUBLAND/LC_BARE/slopeティアの混合)`
- **適合方法（移植）**: numpyでvalue_noise実装(整数格子4隅をハッシュ→smoothstep補間、座標配列で一括)。classify_surface_block_grid の stone/gravel 判定に noise<th 条件を足し andesite/cobblestone/coarse_dirt を混ぜる。block_palette に必要ブロック追加

#### 4. [🟡 nice / medium] 水面Yを急斜面でのみ局所最小へスナップ、ただしスナップ半径超の落差(本物の崖/滝)は跨がない

- **なぜ御坊に効くか**: flood_psoはDEM5AとFGD水域ポリゴンを別ソースで重ねるため、川縁でDEM標高と水域分類が数mズレると水面が階段状(terrace)になる。arnisのsnapはDEM-water微小ズレを補正しつつ滝/堰の段差は保つ。日高川の護岸・河川縁の不自然な段差解消に有効
- **arnis参照**: `ground.rs:381-417 water_level (SNAP_RADIUS=3, center-min>radiusなら自セル維持)`
- **適合方法（移植）**: y_surf_land に対し water_mask セルで近傍radius内の最小Yを取り、落差>radiusなら自セル維持。numpyのminimum_filter+条件マスクでベクトル化

#### 5. [🟡 nice / medium] 水域の連結成分ごとに最大幅(距離変換のmax)を出し、本体サイズで最大水深をティア化(小池=2,大海=6)。岸からの平坦shoal帯を設けてからスロープ開始

- **なぜ御坊に効くか**: flood_psoは海・河川・池をまとめてグローバル distance_transform_edt+単一 depth_per_m/max_depth で処理(836-843)。小さな池や日高川の細い水路まで紀伊水道と同じ深さプロファイルになり不自然。連結成分ごとに本体幅で最大水深を変えれば、海は深く・河川や溜池は浅く、かつ岸際にshoal(浅瀬)帯ができて海岸の『のっぺり』(doc07弱点)が改善
- **arnis参照**: `water_depth.rs:113-211 compute_big_water_field(per-component BFS), 262-279 polygon_local_max(幅ティア)/depth_from_dt(SHOAL_DT_UNITS=9の平坦帯)`
- **適合方法（移植）**: scipy.ndimage.label で水マスクを連結成分化、各成分の distance_transform_edt 最大値でmax_depthをティア選択、shoal幅ぶんは深さ0に固定してから線形ramp。海(sea_mask)と内陸(water_mask)で別パラメータ可

#### 6. [🟡 nice / low] 対向近傍チェックによる単セル水ギャップ補完(N&S or E&W or 斜め対が水ならその穴も水に)

- **なぜ御坊に効くか**: flood_psoの水マスクはラスタ化やダウンサンプルで1セルの穴が空き、川の連続性が切れることがある。対向近傍補完は安価に河川/海岸の連結性を保つ
- **arnis参照**: `ground_generation.rs:358-375 osm_gap判定`
- **適合方法（移植）**: numpyで水マスクの shift 4方向/斜めを取り (N&S)|(E&W)|(対角対) を元マスクにORするだけ。binary closingでも近似可

#### 7. [🟡 nice / high] 三角形メッシュのDDAボクセル化(WorldTransformでintrinsic+worldのscale/yaw/translationを合成、頂点色/テクスチャ平均色→最近傍パレット)

- **なぜ御坊に効くか**: flood_psoのLOD2は _rasterize_lod2_roof(426-452)で屋根上面を平面lstsqのheightfieldにするだけで、庇の張り出し・オーバーハング・城の複雑形状や橋桁の真の3D形状を表現できない。真のメッシュボクセライザがあれば御坊城などランドマークLOD2や3Dモデルを忠実にscale1.5でボクセル化できる。WorldTransformのscale合成はタイル分割座標系への配置に綺麗に対応
- **arnis参照**: `voxelize.rs:214-332 voxelize_glb / 335-359 voxelize_uniform_triangles / 38-123 WorldTransform`
- **適合方法（移植）**: DDA三角形ボクセル化をPython/numpyで実装(または trimesh.voxelized/自前3D Bresenham)。ただし歩行地形の主目的にはheightfieldで足りるため、ランドマーク建物限定の任意機能とする。Rustのdda-voxelizeクレート相当の自前移植が必要で重い

#### 8. [⚪ skip / low] slopeをSTEP距離(4ブロック)離れた4基本方位の(max-min)で算出する広域ステンシル

- **なぜ御坊に効くか**: flood_psoのcompute_slopeは隣接1セル中央差分でブロック解像度ではノイジー。STEP離れたmax-minは局所起伏をより安定に捉え表層分類のちらつきを減らす。ただし効果は限定的
- **arnis参照**: `ground.rs:345-361 slope (STEP=4, max-min, saturating_sub)`
- **適合方法（移植）**: numpyで roll を±STEP にして4方位max-min。既存gradient法と差し替えるかブレンド

**この軸で流用しない/不要なもの**:

- snow_line_meters/snow_threshold_for (ground.rs:48-72): 緯度別雪線で高所をsnow_layer化。御坊は温暖湿潤Cfaでdoc07が明記する通り雪不要。標高も低く対象外
- ChunkGroundCache (ground_generation.rs:50-107): チャンク内Yを256配列にキャッシュし~20回の近傍lookupを高速化するのはRustの逐次get_ground_level前提の最適化。flood_psoは既にnumpyでグリッド全体をベクトル処理しており、minimum_filter等で近傍を一括取得するため不要
- fillground/bulk_fill_chunk_sections_below (ground_generation.rs:225-256,1176-1185): region/section単位のRLE一括充填はarnisのWorldEditor(リージョンファイル)構造前提。flood_psoはNBT structure形式で各ブロックをlist要素として持つため同型の最適化は効かない(litematic palette側で対応すべき別レイヤ)
- bedrock_block_map.rs (BedrockBlock/states): Bedrock版のstring identifier+state変換。flood_psoはJava NBT/litematic専用でBedrock出力をしないため不要
- models_3d/pipeline.rs (3DMR/Wikidata/stadium/plane の外部3Dモデル置換オーケストレーション): OSM tag→外部glTFモデル取得の文脈。flood_psoはFGD/PLATEAU/点群が一次ソースで外部3DモデルDB連携の予定がなく対象外
- steep_overrideのブラックリスト式素材置換(ground_generation.rs:653-684): 道路標示/建物を保護しつつ崖面素材を上書き。flood_psoは表層分類→道路/水を後段で上書きする順序制御で既に等価に対処済み

> 軸まとめ: flood_psoの表層生成(terrain_render.py)は既にTellus由来でslope/convexity/海岸距離/cliff-aware smoothを実装済みで、arnisと設計思想は重なる。差分として実利が大きいのは4点: (1)固定deep_ground=8を近傍最低Y基準の可変アンダーフィル(ground_generation.rs:716-758)に変えるとNBTを軽量化しつつ崖の底抜けを完全に塞げる=doc07が認めるブロック+77%の無駄と底抜けを同時解消、(2)二値マスクをgaussian+0.5閾値で焼く(ground.rs:295-335 water_blend)と海岸線/河川縁の四角いギザギザが曲線化=doc07の最重要弱点『海岸線が雑』に直撃、(3)value_noise_01(ground_generation.rs:1227-1251)で表層素材を有機的斑にすると『針葉樹林のような単調な見た目』が緩和、(4)水域の連結成分別ティア水深(water_depth.rs:113-279)で海と河川/池の深さを分離。いずれもscipy.ndimage(minimum_filter/gaussian_filter/label/distance_transform_edt)で低〜中コストにPython移植でき、PSO/CCPSO2の校正本体(inundation)には不干渉でレンダ層のみ改善する。メッシュDDAボクセル化(voxelize.rs)はランドマーク建物向けの高コスト任意機能、snow/Bedrock/region最適化は御坊文脈/NBT形式に不適合。

---

### 軸3: 座標変換/投影/タイル整合/スケール

**flood_pso 現状**: flood_psoの緯度経度→ブロック座標は「平面近似 (equirectangular) + DEMグリッド直接クロップ」方式。(1) どこでも `lat_per_m=1/111320`, `lon_per_m=1/(111320*cos(lat_center))` のマジック定数で変換 (nbt_export.py:395-396, make_nbt_hd.py:339-340, view_3d.py:41-42)。lon scaleは出力中心緯度1点のcosのみ使用。(2) ブロック格子は実質DEM格子のリサンプルで、export_to_nbt が中心(lat,lon)を row_c/col_c に round して half_rows/half_cols 分を整数indexクロップ (nbt_export.py:542-566)。真の地図投影は持たず、tellus_world経路のみ anvil_loader.py:36-59 に Tellus Mercator の forward を移植 (inverse抽象なし、定数 111319.49/6378137 が散在)。(3) DEMモザイクは先頭タイルの代表解像度を全体に適用し round で配置、重なりは先勝ち (dem_parser.py:101-123) で多タイルでドリフトしうる。(4) --tiles はwidth/depthをCOLS×ROWS等分し各タイル中心を _lon/_lat_per_m で算出 (make_nbt_hd.py:331-358)。col0=西/row0=北。タイルは公称無重なりで、配置表は nx=752固定+752ステップのマジック値 (docs/御坊全域v2_配置表.md)。境界の隙間は過去 ×0.985マージンでバグ化し「実データの重なりで密着」させる運用 (make_nbt_hd.py:298-303) = 継ぎ目整合が弱点。(5) 系VI(EPSG:6674)→WGS84 は pyproj で正確 (wakayama_pcd.py:122-131)。グリッド間再投影は scipy bilinear (nbt_export.py:455-462)。

**arnis 読込ファイル**: `src/coordinate_system/transformation.rs`, `src/coordinate_system/mod.rs`, `src/projection/web_mercator.rs`, `src/map_transformation/transform_map.rs`, `src/map_transformation/operator.rs`, `src/tile.rs`

#### 1. [🔴 must / medium] タイル境界を固定ブロックステップ(region境界)へ floor/ceil 整列させる create_tiles 方式。ワールドbboxを 512の倍数へ aligned_min=(min>>9)<<9 / aligned_max=((max+512)>>9)<<9 で丸め、tile_size刻みで隙間なく敷き詰める。

- **なぜ御坊に効くか**: flood_psoの --tiles は width/depthをCOLS等分し各タイル中心をfloat演算で出すため、丸め誤差で隣接タイル列が1ブロックずれ/重複しうる(過去 ×0.985 で隙間バグ→配置表の 752/+752 マジックで手当て)。タイル原点を固定ブロックステップへ整列すれば、隣接タイルが必ず同一ブロック列を共有し継ぎ目が原理的に一致する。
- **arnis参照**: `tile.rs create_tiles (lines 56-93), 特に 61-64 のビットシフト整列`
- **適合方法（移植）**: 御坊はlitematic/NBT structureでありAnvil regionではないので 512 ではなく tile幅(例 nx=752 や h_res由来の固定step)を整列単位にする。lat/lon中心ベースの現行ループを捨て、まずグローバルbboxをブロック座標 (col=int((lon-lon_min)/res_lon), row=int((lat_max-lat)/res_lat)) に落とし、tile_x0 = floor(col0/step)*step として整数ブロックでタイルを切る。各タイルの (lat,lon,width,depth) はそのブロック範囲から逆算。make_nbt_hd.py:331-358 を置換。

#### 2. [🔴 must / medium] タイルにハロー(halo)を持たせ、継ぎ目を跨ぐ要素を各タイルで完全描画する expanded bounds + AABB割当。TILE_EDITOR_HALO=64 を最大要素半幅以上に設定し、要素は厳密境界でなく halo拡張境界に重なる全タイルへ割り当てる。

- **なぜ御坊に効くか**: flood_psoはFG-GML建物/道路/樹木をタイル毎に描く。建物footprintやorthoパッチが継ぎ目を跨ぐと、現行の整数indexクロップ(nbt_export.py:551-566)で切断され、隣接litematicで建物が割れる/道路が途切れる。haloクロップで両タイルに完全に焼き込めば、litematic重ね合わせ時に整合する。
- **arnis参照**: `tile.rs TileBounds::expanded (lines 28-35), assign_elements_to_tiles (lines 187-260), HALO定義 41-52`
- **適合方法（移植）**: export_to_nbt のクロップ r0:r1/c0:c1 を、建物最大半幅(御坊なら~30m→res_lon換算セル)分だけ拡張した r0-halo:r1+halo で「ベクタ/orthoのみ」描画し、地形DEMは厳密境界で書く。あるいは building_list/FGDを AABB で halo拡張タイルbboxへ事前割当(arnis assign_elements_to_tiles相当)してから描画。重複ブロックはlitematic配置側で上書き一致するので無害。

#### 3. [🟡 nice / low] ブロック寸法を Haversine 実距離から決める geo_distance。lat方向は緯度差Haversine、lon方向は2点の平均緯度でcos補正したHaversineを使う(中心1点でなく区間平均)。

- **なぜ御坊に効くか**: flood_psoは全域(南北~15km, 緯度33.85→33.99)で lon_per_m を出力中心緯度1点のcosで固定している。御坊全域v2のように北端〜南端でcos(lat)が変わると、端のタイルで東西スケールが系統的にずれ、タイル間でブロック密度が不整合になる。区間平均緯度のHaversineなら各タイルで正しい東西mが出る。
- **arnis参照**: `transformation.rs geo_distance (lines 175-182), lon_distance (186-194), lat_distance (198-205)`
- **適合方法（移植）**: マジック 111320 を捨て、tileごとに lon_distance((lat_min+lat_max)/2, lon_min, lon_max) で width_m、lat_distance で depth_m を算出。pyprojのgeod.line_length でも等価実装可。make_nbt_hd.py:300-302 と nbt_export.py:395-396 のflat近似を置換。地球半径は研究で使う 6378137(WGS84長半径, anvil_loaderと同じ)に統一。

#### 4. [🟡 nice / medium] 投影クラス(Projection trait)に forward/inverse を対で持たせ、origin を z_offset で 0 に正規化、round-trip テストで保証する設計。

- **なぜ御坊に効くか**: flood_psoは anvil_loader.py に Mercator forward だけ移植し inverse は export_to_nbt 内に arctan/sinh で手書き散在(nbt_export.py:445-448)。定数 111319.49/6378137 が複数ファイルに重複。forward/inverse を1クラスに集約しround-tripテストを置けば、tellus_world/mapzen/gsi の座標往復ミスを防げる。
- **arnis参照**: `web_mercator.rs WebMercatorProjection::new の z_offset事前計算 (lines 26-39), forward/inverse (42-73), round-trip test (96-118)`
- **適合方法（移植）**: anvil_loader.py の lat_to_blockZ/lon_to_blockX/逆変換を Projection クラス(forward(lat,lon)->(x,z), inverse(x,z)->(lat,lon), origin_lat/lon, scale)へ統合。origin を御坊中心にして z_offset で 0 化すると、litematic配置のワールド座標(配置表のX/Zオフセット)を一貫管理できる。pytest で forward→inverse 誤差<1e-8 を assert。

#### 5. [🟡 nice / low] bboxを4隅投影→軸並行エンベロープで確定する with_projection。NW/NE/SW/SEを投影し x_min..x_max,z_min..z_max を floor/ceil で囲う。

- **なぜ御坊に効くか**: flood_psoは格子が北up軸並行前提で、投影歪み・将来の回転(系VIの真北とWGS84の差や、図郭斜め配置)に対し四隅が必ず収まる保証がない。4隅エンベロープなら投影非線形でもクロップ範囲が全領域を確実に内包し、端欠けを防ぐ。
- **arnis参照**: `transformation.rs with_projection (lines 89-104)`
- **適合方法（移植）**: DEMが既にlat/lon軸並行格子なので必須ではない。Mercator/回転を導入する場合のみ、タイルbbox確定を「4隅を Projection.forward して min/max を floor/ceil」に置換。御坊小領域では効果小なので回転導入時のみ。

#### 6. [🟡 nice / medium] AABBが到達しうる region セル範囲だけを走査する region_range ビットシフト + region→tile の HashMap で O(1) 要素割当。

- **なぜ御坊に効くか**: flood_psoは多タイル出力でFG-GML(BldA 29MB等)をタイル毎に再パース/全走査しがちで、全域v2の40タイルでは無駄が大きい。要素を一度パースしAABBで該当タイルへ空間割当すれば、タイル毎の描画コストを要素数に比例させられる。
- **arnis参照**: `tile.rs region_range (lines 142-151), tile_grid HashMap (198-202), node O(1)割当 (206-214)`
- **適合方法（移植）**: fgd_vector で建物/道路を一度読み、各ポリゴンの (row,col) AABB を計算→ region_range 相当(タイルstepでシフト)で該当タイルindexへ push。タイル描画時はそのリストだけ描く。FG-GMLはOSMタグでないので is_linear_element の代わりに BldA/RdEdg の種別で halo を分岐。

#### 7. [⚪ skip / low] メートル距離を floor してからスケール倍し整数ブロック寸法を得る (scale=blocks/m を全軸一様適用)。

- **なぜ御坊に効くか**: flood_psoの --scale は h_res/v_res を割るだけで、タイルブロック数が int() 切り捨てでタイル毎に1ブロック揺れうる。先にメートルをfloorして決定論的なブロック数にすればタイル間でブロック数が揃い継ぎ目が合う。
- **arnis参照**: `transformation.rs llbbox_to_xzbbox (lines 54-56) の scale_factor.floor()*scale`
- **適合方法（移植）**: タイル整列(must項目1)と併せ、各タイルのブロック数 = floor(tile_width_m/h_res) を固定値にして全タイル同一にする。make_nbt_hd.py:341 の _tw=width_m/n_cols を整数ブロック step に置換。

**この軸で流用しない/不要なもの**:

- 主DEM経路へのWeb Mercator全面導入: 御坊対象域は南北~15km・東西~10kmで、equirectangular近似の歪みは<0.1%。さらにflood_psoのDEMは既にGSI/pyprojでlat/lon等間隔格子化済みなので、Mercatorを噛ませると逆に元DEM格子とミスアラインする。arnisがMercator必須なのは任意の全球都市を slippy-map タイルへ整合させるため。御坊では geo_distance/4隅エンベロープ等の精度補正だけ流用し、投影そのものの置換は不要。
- map_transformation の rotate/translate オペレータ pipeline (transform_map.rs, operator.rs): 御坊は実地形を真北upでそのまま出すのでマップ回転/平行移動は不要。OSM都市を任意配置するarnis特有の要件。ただし「ワールド原点へのtranslate正規化」概念だけは配置表の手動X/Zオフセット管理に転用余地あり(低優先)。
- is_linear_element 等のOSMタグ駆動の要素分類 (tile.rs:166-174): flood_psoのベクタはFG-GML(BldA/RdEdg/WA)であり highway/railway 等OSMタグを持たない。ハロー分岐の『線要素は半幅、面要素はeditor halo』という概念は有用だが、タグ判定ロジック自体はFG-GML種別へ読み替えが必要。
- 512=Minecraft region への整列定数: flood_psoの出力は litematic/NBT structure でAnvil region単位ではないため 512 そのものは不要。整列の『固定ブロックステップへ floor/ceil で丸める』原理のみ流用し、ステップは御坊のタイル幅(nx=752等)にする。

> 軸まとめ: 最も価値が高いのはタイル整合の2点(must)。flood_psoの継ぎ目問題(配置表の752マジック、過去の×0.985隙間バグ)は、arnis tile.rs の (a) 固定ブロックステップへの floor/ceil 整列(create_tiles:56-93) と (b) ハロー拡張による継ぎ目跨ぎ要素の完全描画(expanded/assign_elements_to_tiles:28-260) で原理的に解消できる。現行の lat/lon中心float等分(make_nbt_hd.py:331-358)を整数ブロック座標ベースのタイル切りに置換し、建物/orthoをhaloクロップで両タイルに焼き込むのが具体策。精度面では geo_distance のHaversine+区間平均緯度(transformation.rs:175-205)でマジック111320と中心1点cosを置換すると全域南北のlonスケール系統誤差を除ける(nice)。投影そのもの(Web Mercator)とmap回転は御坊小領域・北up・既存lat/lon格子の前提では不要かつ逆効果で、forward/inverseを1クラスに集約しround-trip保証する設計上の整理(web_mercator.rs:26-73)のみ refactor 価値あり。RustからPythonへの移植は数十行規模で低〜中コスト、numpy/pyproj/scipyで等価実装可能。

---

### 軸4: 水/洪水表現・floodfill

**flood_pso 現状**: flood_sim.py がコア。バスタブモデル+連結成分ラベリング: (1) DEMをwater_level閾値で2値化(candidate), (2) scipy.ndimage.label で全グリッドの連結成分を計算, (3) source_mask に触れるラベルだけ np.isin で抽出, (4) 浸水深 = water_level - dem。sigma による Gaussian blur で地表粗度/DEM誤差をモデル化。HD版 simulate_flood_hd は K×K の dh_map をバイリニア補間して空間変動水位 water_field を作り、sigma_map 用に5段階 gaussian_filter スタックを毎回構築。水源 make_river_source は緯度経度bbox ∩ (dem<=elev_max) の矩形マスク(粗い)。hazard_gt.py は GSI ハザードタイルをファイルキャッシュ(404は空マーカ)で取得し凡例色→水深の最近傍マッチ、mercatorモザイク→DEM格子へ map_coordinates 再サンプル。評価は iou_loss / depth_loss、正則化に dh_roughness。PSO/CCPSO2 が water_level・sigma・dh_map を多数回評価するため simulate_flood が内側ループでホットになる。

**arnis 読込ファイル**: `src/floodfill.rs`, `src/floodfill_cache.rs`, `src/water_depth.rs`, `src/element_processing/water_areas.rs`, `src/land_cover_osm_water_override.rs`

#### 1. [🔴 must / low] 全グリッド nd_label+np.isin を「水源シードからの制約付きflood(モルフォロジー再構成)」に置換し、浸水域だけを訪問する。arnisのシード起点BFS(visited bitmapで膨張、領域外セルは一切触らない)と数学的に等価。

- **なぜ御坊に効くか**: simulate_flood/simulate_flood_hd は毎回 H×W 全体に nd_label(全成分ラベル付け)→ さらに np.isin で全画素を再走査する。PSO は population×iterations で数千〜万回評価するためここが支配的。arnis流に水源から連結セルのみ展開すれば、河道に繋がらない無関係な盆地(全成分の大半)を計算せずに済み、ラベル配列と isin の確保も不要になる。
- **arnis参照**: `floodfill.rs optimized_flood_fill_area (L115-197): queueにseedを入れ4近傍をvisited.insertで膨張、polygon.contains を満たすセルだけ拡張。FloodBitmap (L16-60)。`
- **適合方法（移植）**: Rust BFS をそのまま移植する必要はない。scipy.ndimage.binary_propagation(source_valid, mask=candidate) または skimage.morphology.flood が『シードから mask 内へ連結拡張』そのもので、nd_label+set+isin を1コールに圧縮できる。connectivity は structure 引数(4/8連結)で再現。HD版も candidate を water_field で作った後同様に置換。semantics(水源連結成分のみ)は不変。

#### 2. [🔴 must / low] 連結/浸水計算は御坊フルドメインで1回だけ行い、タイル分割は『書き出し段でstrictなタイル境界に絞って縦方向のみ書く』に限定する。グローバル計算とタイル描画を分離してシームを構造的に消す。

- **なぜ御坊に効くか**: flood_pso は scale1.5 でタイル分割してNBT化し、メモリにも『タイル境界は連結窓で整合』とある弱点。もし浸水をタイルごとに flood すると、境界をまたぐ連結成分がタイルAでは水源連結と判定されBでは孤立扱い…というシーム不整合が起きる。arnisの『グローバルに連結判定→タイルは単なる切り出し』方式なら境界整合は自明に保証され、連結窓のオーバーラップ処理が不要になる。
- **arnis参照**: `water_depth.rs: BigWaterField を水域サブ矩形全体で一度だけ構築(compute_big_water_field L113-211, 成分BFS L160-202)し、carve_lc_water_region (L586-623) は iter_min/max で渡されたタイル境界に交差させて『縦方向のみ』書く。コメント『writes are vertical-only so output is identical regardless of tiling』。`
- **適合方法（移植）**: simulate_flood は御坊DEM全域で動かしグローバル inundation 配列を得る。NBT化段で inundation[z0:z1, x0:x1] をタイル境界で純粋にスライスするだけにする(タイル単位 flood を廃止)。書き込みが各(x,z)で独立(縦カラムのみ)なら順序・分割に依存しない。

#### 3. [🔴 must / medium] make_river_source の矩形bboxを、FGD の河川/水域ポリゴンの『偶奇ルール scanline ラスタライズ(穴=内環をsubtract)』に置き換え、正確な水源/水域マスクを作る。

- **なぜ御坊に効くか**: 現状 make_river_source (flood_sim L29-50) は緯度経度の矩形 ∩ dem<=elev_max で河道近似しており、蛇行を取りこぼし・河道外の低地を誤って水源化する。これが IoU/depth_loss の系統誤差源。FGD には実際の河川・水域ポリゴンがあり、これを正しくラスタライズすれば水源・水域マスクが大幅に精緻化する。
- **arnis参照**: `water_areas.rs: compute_scanline_spans (L274-319) 偶奇ルールでスパン算出、union_spans (L322-349) 複数外環の和、subtract_spans (L355-387) 内環(島/穴)の差。land_cover_osm_water_override.rs fill_polygon_scanline (L224-340) と edge_crossings_at_z (L342-359) はグリッド解像度へスケールしつつ同手法でラスタライズ。`
- **適合方法（移植）**: RustのscanlineをPythonへ手移植せず rasterio.features.rasterize か skimage.draw.polygon2mask を使う。穴(内環)は別マスクを作り XOR/減算で処理(subtract_spans相当)。DEM格子(lat/lon)へは flood_pso 既存の res_lat/res_lon から行列インデックスへ写像。複数外環は OR で union。

#### 4. [🟡 nice / medium] 水源標高をmulti-source BFSで全セルへ伝播し、空間変動の『参照水面高』を作って water_field / dh_map の物理的初期値(下流方向に下がる河川水面プロファイル)にする。

- **なぜ御坊に効くか**: simulate_flood_hd の water_field は一様 water_level_global に PSO探索の dh_map を足すだけで、河川が下流へ水面低下する事実を事前情報として持たない。dh逆問題は ill-posed(dh_roughnessで正則化中)。最近傍水源標高を伝播した水面を baseline にすれば、PSO は小さな残差 dh だけ探索すればよく、可同定性と物理妥当性が上がる。
- **arnis参照**: `land_cover_osm_water_override.rs compute_nearest_water_y (L618-661): 全LC_WATERセルをqueueに入れ、各陸セルへ最近傍水セルの地形YをBFSで伝播。`
- **適合方法（移植）**: scipy.ndimage.distance_transform_edt(return_indices=True) で各セルの最近傍水源インデックスを取り、その源のDEM/水面値を割り当てる(BFS手書き不要)。または collections.deque で4近傍BFS。得た場を water_field 基準にし dh_map は残差として最適化。

#### 5. [🟡 nice / low] 『最近傍水面高 + 許容差』を超えるセルは連結でも浸水から除外する標高ガードを足し、sigmaブラー由来の1画素リークによる過大浸水を抑える。

- **なぜ御坊に効くか**: flood_pso は sigma で DEM を平滑化して『水が障壁を越えやすく』するが、これは緩斜面で1画素の低点を通じて尾根上まで連結リークし過大浸水を生む副作用がある。連結後に dem > water_field + tol のセルを落とすガードでリークを抑制でき、IoUの過検出を減らせる。
- **arnis参照**: `land_cover_osm_water_override.rs passes_water_guard (L663-686) と ELEVATION_TOLERANCE_BLOCKS(=1.5)。cell_y <= nearest_water_y + tol を満たさないセルは水化しない。`
- **適合方法（移植）**: flood_mask 確定後に numpy で flood_mask &= (dem <= water_field + tol) を掛けるだけ。tol を新パラメータ化して PSO に含めても良い。R4の伝播水面と組み合わせると効果的。

#### 6. [🟡 nice / medium] バスタブ前提の平坦水深に対し、岸からの距離変換ベースの水深勾配(浅瀬→深部のなだらかな掘り込み)を、DEMが水面平坦/欠損で水底情報が無い河道に適用してLayer C可視化の自然さを上げる。

- **なぜ御坊に効くか**: flood_pso の浸水深 = water_level - dem は、河道のように水底DEMが水面に張り付いて平坦な領域では一様スラブになり、歩けるNBTで不自然。岸距離に応じた掘り込みで自然な河床断面が出る。ただし御坊は実DEMがあり氾濫原の水深自体は物理的に正しいので、適用はあくまでNBT見栄え用で損失関数には使わない。
- **arnis参照**: `water_depth.rs chamfer_3_4_dt (L213-260) 2スイープ距離変換、depth_from_dt (L277-293) 浅瀬オフセットSHOAL_DT_UNITS後に勾配を成分最大幅でtier-clamp、成分ごと最大DT算出 (compute_big_water_field L160-202, polygon_local_max L263-274)。`
- **適合方法（移植）**: chamfer3-4を移植する必要はなく scipy.ndimage.distance_transform_edt(厳密EDT)で水域マスクの岸距離を取り、shoal offset + tier slope で深さに写像。成分ごと最大幅で深さ上限を変える tier 思想は np.label の成分ごとに distance.max() を取って再現可能。可視化段のみ。

#### 7. [🟡 nice / low] PSO内側ループ不変量(固定sigma_levelsのDEM平滑化スタック)を粒子ループの外で一度だけ前計算してキャッシュする。arnisがflood fillを逐次処理前に並列前計算する設計の発想。

- **なぜ御坊に効くか**: simulate_flood_hd は sigma_map 経路で毎回5回の gaussian_filter(land) を再計算する(L157-161)が、これは DEM と固定 sigma_levels のみに依存し water_level/dh には依存しない。PSO の population×iterations 全評価で同じスタックを再構築しており完全な冗長計算。
- **arnis参照**: `floodfill_cache.rs precompute (L281-322): 必要な要素のfloodfillをrayonで一括前計算し、本処理ではキャッシュ参照(Arc refcount)に。empty sentinel (L26-29)。`
- **適合方法（移植）**: stack=[gaussian_filter(land,s) for s in sigma_levels] とNaN穴埋め land を PSO 開始前に1回計算して関数へ渡す/クロージャに束ねる。Pythonの並列はGIL回避に multiprocessing か、そもそも前計算1回なので並列不要。simulate_flood の単一sigmaも同様にキャッシュ可。

**この軸で流用しない/不要なもの**:

- OSM way/relation のリング閉合・断片連結(water_areas.rs verify_closed_rings L177-197, merge_way_segments, land_cover_osm_water_override の merge/stitch)は、flood_pso の入力が DEM ラスタ + FGD GML(ポリゴンとして直接ラスタライズ可能)でありOSMトポロジ断片の再構成が不要なため移植不要。FGDがポリゴンで来る限りR3のラスタライズだけで足りる。
- Arc<Vec> 参照カウント共有・rayon スレッドプール構成(floodfill_cache.rs FloodFillResult L20, configure_rayon_thread_pool L480-509)は Rust 所有権モデル前提。Python は numpy 配列の view/コピー方針と multiprocessing で対応すべきで直接対応物が無い(GILのため共有メモリ最適化の意味も薄い)。
- 水中ダンス/海草/ケルプ/河床ブロックのテクスチャリング(water_depth.rs place_underwater_dunes L462-496, place_underwater_vegetation L500-566, carve_water_column のブロック選択 L364-410)はMinecraft海洋の見栄え専用で、御坊の淡水河川氾濫(歩ける地形)の対象外。value_noiseドメインワープも同様。
- 1bit/座標の visited bitmap(floodfill.rs FloodBitmap, floodfill_cache.rs CoordinateBitmap, count_in_range のpopcount L174-251)は、Python では numpy bool 配列(1byte/セル)で既に十分軽量。御坊25m格子は小さくHashSet回避のためのbit-packは投資対効果が低い。
- MAX_FLOOD_FILL_AREA の面積上限による棄却・面積でアルゴリズム切替(floodfill.rs L10, L96-111)は、海隣接の巨大OSMポリゴン暴走を防ぐためのもの。flood_pso は御坊の固定小領域で動くため不要。
- GSIタイルの404空マーカ・ファイルキャッシュは flood_pso hazard_gt.py L52-77 が既に同等実装済み(空バイト書き込みで再取得回避)。arnis側に追加で学ぶ点は無い。

> 軸まとめ: arnisは『OSMポリゴンを多角形内塗りで水化し、岸距離DTで水底を掘る』成熟実装で、flood_pso(DEM閾値バスタブ+連結成分)とは水の定義が逆向き(arnis=ポリゴン所与・深さを合成 / flood_pso=DEM所与・連結で水域を導出)。そのため depth carving や海洋装飾はそのまま使えないが、(A)連結計算の効率化と(B)タイル境界整合と(C)水源マスク精緻化の3点で直接効く知見がある。最優先3件: [must-1] nd_label+isin の全グリッド処理を scipy.ndimage.binary_propagation による水源シードflood(arnis floodfill.rs L115-197相当)へ置換し PSO内側ループを高速化、[must-2] 連結/浸水はフルドメインで1回計算しタイルは縦方向書き出しの単純スライスに限定(water_depth.rs L586-623の tile-invariant 思想)してシームを構造的に排除、[must-3] make_river_source の矩形bboxを FGD河川/水域ポリゴンの scanline+内環subtract ラスタライズ(water_areas.rs L274-387, land_cover_osm_water_override.rs L224-340)へ置換して水源精度=損失精度を底上げ。次点で最近傍水源標高のmulti-source BFS伝播(L618-661)を water_field/dh の物理プライアに、標高+許容差ガード(L663-686)でsigmaブラーの過大浸水リーク抑制。移植コストはいずれもscipy/skimage/rasterioの既存関数で吸収でき、Rust BFSの手書き移植は不要。」

---

### 軸5: NBT/ワールド書き出し

**flood_pso 現状**: flood_psoのNBT/書き出しは3経路。(1) src/nbt_export.py: DEM+浸水をMinecraft Structure NBT(.nbt gzip)化。dem_to_blocks()/dem_to_blocks_enhanced()が各ブロックを個別の nbtlib.Compound{pos:[x,y,z], state:int} として Python list に積み、write_nbt_structure()で palette(block_palette.pyの単一真実源~80ブロック)+blocks listをgzip保存。docs/06_NBT管理.mdの既知制約#1「block entryが冗長」#2「huge_5mで変換中Python 8-12GB消費」が示す通り、密表現でなくスパースなCompound列挙のため巨大・高メモリ。(2) src/nbt_to_litematic.py: 出力済.nbtをnbt_preview._parse_fastでnumpy抽出→dense YZX配列→pack_litematica_bits()でlong跨ぎcontiguous bit-pack→.litematic。ここは唯一の密ビットパック実装でチャンク処理(8M毎)でメモリ抑制済。(3) src/anvil_loader.py: TellusWorldがmod生成済Anvil(.mca)を読むのみ(RegionFile/get_chunk、_decode_section)。書き出しは未対応。全経路Python単スレッド。御坊の歩けるNBTは現状scale1.5でタイル分割しlitematic化、地形自体はTellus mod依存(project_flood_pso_walkable_gen)。native .mca書き出し・並列化・heightmap/biome生成・section単位パレットは未実装。

**arnis 読込ファイル**: `src/world_editor/mod.rs`, `src/world_editor/java.rs`, `src/world_editor/common.rs`, `src/block_definitions.rs`, `/home/ntaku/laravel-project/flood_pso/src/nbt_export.py`, `/home/ntaku/laravel-project/flood_pso/src/make_nbt.py`, `/home/ntaku/laravel-project/flood_pso/src/nbt_to_litematic.py`, `/home/ntaku/laravel-project/flood_pso/src/anvil_loader.py`, `/home/ntaku/laravel-project/flood_pso/docs/06_NBT管理.md`

#### 1. [🔴 must / medium] ブロックを「整数idの密3D配列(numpy)」として保持し、{pos,state} Compoundの列挙を完全に廃止する。arnisはBlockStorage::Uniform(block)/Full(Vec<u8>)/FullWide(Vec<Block>)の3段で、全同一(空気/石)セクションは1ブロック値だけ持ち、混在時のみ4096要素配列。書き出し直前のto_section()でだけpaletteとLongArrayを生成する。

- **なぜ御坊に効くか**: flood_psoの最大の弱点=docs既知制約#1#2。dem_to_blocks()が数百万のnbtlib.Compoundを生成するためhuge_5mで8-12GB消費・ファイル肥大。御坊全域(15km四方/5m)を一度に出すには密表現が必須。
- **arnis参照**: `common.rs:84-174 (BlockStorage enum / get/set/try_compact), common.rs:280-364 (to_section dense pack)`
- **適合方法（移植）**: nbt_export.dem_to_blocks/_enhanced を、Compound listでなく dense = np.full((sy,sz,sx), air_idx, dtype=int32) に直接書く形へ改修(nbt_to_litematic.py が既にこのdense YZX配列とpack_litematica_bits()を持つので、その入力を「.nbt経由」でなく生成段から直接渡せば.nbt中間生成自体を省ける)。Uniform判定はnp.all(slice==v)で代替。移植コスト低〜中。

#### 2. [🔴 must / high] native Anvil(.mca region/chunk)書き出し器を持つ。arnisはxz→chunk(>>4)→region(>>5)の三層へblockをルーティングし、region毎にr.X.Z.mcaへchunk NBTを書く。chunk NBTは1.18+の素のroot形式(Level wrapper無し): DataVersion/xPos/yPos/zPos/Status='minecraft:full'/sections/Heightmaps/structures/PostProcessing/block_entities。section内はYZX indexのpalette+data(LongArray, bits=max(4,ceil(log2(len))))。

- **なぜ御坊に効くか**: 御坊の歩けるワールドが現状Tellus mod依存(anvil_loaderは読むだけ)。native .mca出力できればTellus不要、Structure/litematicビューア上限(~1000^2推奨)も撤廃でき、御坊全域をそのまま開けるワールドにできる。最終目標(歩けるNBT)に直結。
- **arnis参照**: `java.rs:744-841 (create_chunk_nbt), java.rs:844-876 (build_section_value), java.rs:201-289 (create_region_file/write_region_to_disk), common.rs:267-269 (index YZX)`
- **適合方法（移植）**: flood_psoは既にnbt.region.RegionFileをimport済(anvil_loader.py:25)。同ライブラリはwrite_chunkも持つので、Pythonでchunk root dict→nbtlib/NBTFile→RegionFile.write_chunkで実装可能。bits=max(4,...)のsection packは pack_litematica_bits を「longを跨がない」版(64//bits毎リセット)に変えるだけ。section index式とindices_per_longは anvil_loader._decode_section の逆。中規模実装。

#### 3. [🟡 nice / medium] region単位の並列書き出し。arnisはself.world.regions.par_iter()(rayon)で各.mcaを独立に直列化・書込し、AtomicBool should_stop + Mutex first_errorで最初のI/Oエラーで全体中断。

- **なぜ御坊に効くか**: flood_psoは全Python単スレッド。御坊大型プリセット(xl_5m/huge_5m)の変換・書出が遅い。region/タイルは独立なので並列化の効果が高い。
- **arnis参照**: `java.rs:78-177 (save_java, par_iter 131-168)`
- **適合方法（移植）**: multiprocessing.Pool または concurrent.futures.ProcessPoolExecutor で region(タイル)毎にworkerを割当。dense配列はreadonly共有(shared_memory)し各workerが自region分だけpack+書込。GILを避けるためthreadでなくprocess。numpyの重い箇所(pack)はprocess並列が効く。中規模。

#### 4. [🟡 nice / medium] stream-to-disk eviction: 完成したregionを背景スレッドへ送り、compact+直列化+書込をI/Oオーバーラップさせ、bounded channel(sync_channel)でbackpressureしRAM上の保留region数を上限化。

- **なぜ御坊に効くか**: docs既知制約#2(変換中メモリ)への王道解。全blockを保持してから書くのでなく、生成済タイル/regionを逐次ディスクへ流せばピークRAMがregion1個分+αに下がり、御坊全域を低RAM環境でも出せる。
- **arnis参照**: `mod.rs:1435-1507 (FlushWorker spawn/send/finish), mod.rs:336-358 (flush_region_via), java.rs:1456 (ctx.write)`
- **適合方法（移植）**: Pythonでは queue.Queue(maxsize=N) + writer threadで実装(ディスクI/Oはthreadでもオーバーラップ可、GIL外で待つため)。タイルループで1タイルdense完成毎にqueueへput、writerがpack+RegionFile.write_chunk。errorはthread側でevent.set()して主ループ中断。低〜中。

#### 5. [🟡 nice / low] Heightmap生成(MOTION_BLOCKING等)とpack_heightmap_values。各16×16カラムの最上non-airのY-min+1を求め、bits=ceil(log2(total_height+1))(>=9)・longを跨がない方式でLongArray化。4種(MOTION_BLOCKING/_NO_LEAVES/OCEAN_FLOOR/WORLD_SURFACE)を同一データで埋める(サーバは読込時再計算)。

- **なぜ御坊に効くか**: native Anvil出力する場合に必須(無いと一部ビューア/サーバでchunkが正しくロードされない・スポーン高が壊れる)。flood_psoのdem由来surface YがそのままMOTION_BLOCKING値になるので計算も自明。
- **arnis参照**: `java.rs:884-999 (compute_heightmaps), java.rs:1003-1028 (pack_heightmap_values)`
- **適合方法（移植）**: dense配列があれば各(x,z)列のargmax(非air)をnumpyで一括算出→pack。flood_psoはsurface Yをdem_to_blocks段で既知なので、その2D surface_y gridを+min_block_y補正して流用。低。

#### 6. [🟡 nice / medium] タイル境界のauthoritative-bounds merge(halo write-if-air)。各タイルはhalo付きで生成し、merge時にauth領域内はnon-airで常に上書き、auth外(halo)はdestがairの時のみ書く。これで木の樹冠や建物がタイル境界を跨いでも片側で消えない。

- **なぜ御坊に効くか**: flood_psoのタイル分割は『連結窓で整合』というアドホック対応(project_flood_pso_walkable_gen)。建物/橋/樹冠がタイル境界で切れる問題に対し、arnisのauth+halo方式は実証済みでシームを綺麗に縫合できる。
- **arnis参照**: `common.rs:770-952 (merge), common.rs:1029-1064 (merge_section_auth_overwrite_nonair), common.rs:987-1022 (merge_section_write_if_air)`
- **適合方法（移植）**: 各タイルにoverlap(halo)幅を持たせdense生成→隣接タイルへ書込む際 auth矩形内は上書き・外はnp.where(dest==air, src, dest)。Pythonのnumpyスライス代入で素直に実装可。中。

#### 7. [🟡 nice / low] section単位パレット+set-if-absent(先書き優先)。arnisは整数Block idを唯一の真実源にし(block_definitions.rsのid→name表)、name/Properties文字列化は直列化時のto_section()のみで行う。set_with_props_if_absentでairの時だけ書く=重ね順制御。

- **なぜ御坊に効くか**: flood_psoは全palette80ブロックのグローバルindexを毎セル文字列/Compoundで持つ。section毎パレットなら使用色だけでbitsが小さくなりファイル縮小。先書き優先は地盤→地表→建物→水の重ね合わせ制御に有効(現状dem_to_blocksは後勝ちでlist append)。
- **arnis参照**: `block_definitions.rs:91-115 (Block id), common.rs:619-669 (set_with_props_if_absent), common.rs:304-364 (id直引きpalette構築)`
- **適合方法（移植）**: flood_psoは既にblock_palette.pyでint index単一真実源を持つので相性良。dense int配列→section毎にnp.uniqueでローカルpalette→再index→pack。先書き優先はdense[mask & (dense==air)] = id で表現。低〜中。

#### 8. [🟡 nice / low] underground一括充填の高速化: bulk_fill_chunk_sections_below(空Uniform(AIR)セクションをUniform(block)へ丸ごと差替え)とfill_column(region/chunkを1回だけ解決して縦列を埋める)。

- **なぜ御坊に効くか**: flood_psoのdem_to_blocksは地盤柱を for dy in range(y_top-3,y_top) のPythonループでCompound生成。御坊全域で地盤を深く(deep_ground)埋めると爆発的。Uniform section差替えなら定数コスト。
- **arnis参照**: `common.rs:720-750 (bulk_fill_chunk_sections_below), common.rs:673-716 (fill_column)`
- **適合方法（移植）**: dense表現に移れば不要に近いが、native Anvilのsection直列化時に『全石section』をUniform扱いしdataを省く最適化として効く。surface Yより十分下のsectionはpaletteを[stone]だけ・data無しで出力。低。

#### 9. [⚪ skip / low] block entity(看板/旗)書込スキーマ。chunkのblock_entities listへid='minecraft:sign'/front_text.messages 等のCompoundを追加し、対応位置にSIGNブロックを置く。banner はpatterns list。

- **なぜ御坊に効くか**: flood_psoはevac_facilities(避難所)とbridgesを既に立体化(nbt_export evac_xml/bridges_json)。避難所に看板で施設名を表示すれば防災可視化として有用。Structure NBTでもStructure仕様のblock_entities枠に同形式で入れられる。
- **arnis参照**: `mod.rs:708-767 (set_sign), mod.rs:552-581 (set_banner_block_entity_absolute), mod.rs:523-547 (insert_block_entity)`
- **適合方法（移植）**: Structure NBT経路ではpalette要素にminecraft:oak_sign+Propertiesを足し、blocksのその要素にnbt(block entity data)を付与(Structure仕様はblock entry内にnbtキー可)。native Anvil経路ならarnis同様chunk.block_entities listへ。低。

#### 10. [⚪ skip / high] per-chunk lighting bake(SkyLight/BlockLight nibble配列)とheightベースのflood-fill伝播。

- **なぜ御坊に効くか**: off-diskのLODレンダラ/Webビューアで自前ライティングする場合に必要。
- **arnis参照**: `java.rs:612-709 (compute_lighting/propagate_light), java.rs:599-609 (pack_light_nibble)`
- **適合方法（移植）**: Minecraft本体は読込時に再ライティングするので、御坊を本体やlitematicaで見る用途では不要。Webビューアを高度化する時のみ検討。

**この軸で流用しない/不要なもの**:

- Bedrock(.mcworld)書き出し・bannerのBedrock整数color/短縮pattern変換(mod.rs:592-706)— flood_psoはJava NBT/litematic専用で対象外
- Luanti/Minetest出力(mod.rs new_luanti, luanti::save_luanti_world)— 配信先が違う
- lighting bake(compute_lighting)— Minecraft本体/litematicaは読込時再ライティングするため通常不要(Webビューア自前描画時のみ)
- road_surface_overrides / get_ground_level のroad-aware地形(mod.rs:138, 393-455)— OSM道路を平坦化する文脈固有で、DEM5A地形のflood_psoでは河川堤防表現として転用余地はあるが本軸(書出)ではない
- build_deterministic_uuid(mod.rs:1401-1416)— entity配置用UUID。flood_psoは静的地形/浸水でmob entity不要
- is_disk_full_error のエラーチェーン解析(mod.rs:54-84)— Rust固有のError downcast。Pythonでは OSError.errno==28 を見るだけで足り、移植不要
- region.template同梱バイナリ(java.rs:211)— nbt.region.RegionFileが空headerを自前生成するためPythonでは不要

> 軸まとめ: 本軸での最大の収穫は2点。(A) メモリ/サイズの根治: flood_psoのStructure NBTは{pos,state} Compoundをセル毎に列挙するためhuge_5mで8-12GB・巨大ファイル(docs既知制約#1#2)。arnisのBlockStorage(Uniform/Full)=密3D表現+書出直前packが正解で、flood_psoは既にnbt_to_litematic.pyに dense YZX + pack_litematica_bits を持つので、生成段(dem_to_blocks)からdense numpy配列に直書きすればCompound列挙と.nbt中間生成を両方廃せる(must, effort medium)。(B) 機能ギャップ: flood_psoはnative Anvil(.mca)書き出しを持たず歩けるワールドをTellus mod依存で得ている。arnis java.rsのcreate_chunk_nbt/build_section_value/compute_heightmapsが1.18+ chunkスキーマの完全な実装で、flood_psoが既にimport済のnbt.region.RegionFile(write_chunk)で移植可能。これによりTellus非依存・ビューア上限撤廃で御坊全域の歩けるワールドを直接出力できる(must, effort high)。補助として region並列(rayon→multiprocessing)・stream-to-disk eviction(背景writer threadでピークRAM抑制)・タイル境界のauth+halo merge(現状『連結窓』のアドホックを置換し建物/樹冠のシーム解消)が中優先で効く。RustからPythonへはBlockStorage/pack/heightmap/merge は全てnumpyベクトル化で素直に落とせる一方、Bedrock/Luanti/lighting bake/road-aware/UUIDはflood_psoの文脈外。御坊はOSMでなくDEM5A+FGD+ortho由来だが、書出層(section/chunk/region/palette)は地形ソースに非依存なので適合性は高い。

---

### 軸6: ブロックパレット/土地被覆/色マッピング

**flood_pso 現状**: block_palette.py が約80色のバニラブロックを key→(minecraft名,(r,g,b),role) で単一真実源として保持し、MATCH_KEYS を最近傍色マッチのアンカーにする。ortho_surface.py は GSI オルソを enhance_rgb(チャンネル別パーセンタイル・ストレッチ+彩度) で霞補正後、classify_rgb_to_palette が素のRGBユークリッド距離(L2)で各ピクセルを最近傍ブロックへ写像(写真モザイク)。terrain_render.py の classify_surface_block_grid / _esa は DEM由来量(slope/convexity/海岸距離)+ESA WorldCover クラスを「1クラス=1ブロック(フラット)」で割当て、coarse_dirt/grass/water/gravel等に上書き。ESA cover_patch のダウンサンプルは np.median(カテゴリ値)。道路/水域/海岸ベルトは二値マスクで上書き。建物壁/屋根は FG-GML/PLATEAU の type→1ブロックの固定辞書(BUILDING_WALL_BY_TYPE/ROOF_BY_TYPE)。弱点: (1)色距離が知覚非一致でroad等が誤マッチ、(2)1クラス=1色のため広い地表が単調・のっぺり、(3)カテゴリ median/二値マスクで境界がESA10m/タイル格子のブロック状ギザギザ、(4)建物色がtypeごとに均一で多様性なし、(5)バイオーム概念なし(草/水のtintが付かない)。

**arnis 読込ファイル**: `src/colors.rs`, `src/biome.rs`, `src/land_cover.rs`, `src/ground_generation.rs`, `src/block_definitions.rs`

#### 1. [🔴 must / low] 色の最近傍マッチをRGBユークリッドからOklab知覚距離に置換する。arnisはsrgb→linear→LMS→cbrt→Oklabの行列変換でΔを取り、テスト(oklab_prefers_perceptual_neighbor)で『鉄茶色ターゲットに対し純赤より茶系を正しく近いと判定』することを保証している。

- **なぜ御坊に効くか**: ortho_surface.py classify_rgb_to_palette(64-82)の素のL2は、道路の灰や畑の緑がスペクトル的に遠いブロックへ飛ぶ誤マッチを起こす。Oklabにすると御坊オルソの路面・植生・屋根の写像が知覚的に自然になり、写真モザイクの『色化け』が減る。flood_psoの最大の即効改善点。
- **arnis参照**: `colors.rs:114 oklab_distance / colors.rs:124 srgb_to_linear / colors.rs:134 rgb_to_oklab(テスト colors.rs:162)`
- **適合方法（移植）**: rgb_to_oklab/srgb_to_linearをnumpyベクトル化で移植(_ANCHOR_RGBを起動時にOklabへ前計算、px側も(P,3)で一括変換)。距離はL/a/bの二乗和。enhance_rgb後段に挟むだけで既存パイプライン互換。cbrtはnp.cbrt。

#### 2. [🔴 must / medium] 決定論的座標ハッシュ coord_hash(x,z) による地表の確率的ディザリング。1クラスを単一ブロックでなく重み付きブロック混合で塗る(例: built-up=72%stone_bricks/15%cracked/5%stone/8%cobble、急斜面scree=ANDESITE/TUFF/STONE/COBBLE/GRAVELを%配分)。

- **なぜ御坊に効くか**: terrain_render.py classify_surface_block_grid_esa(232-268)は cropland→coarse_dirt, built-up→stone のように1クラス=1色で、広域が単調なベタ塗りになる(現状弱点2)。座標ハッシュ%Nで数種を混ぜると、PSO本体に影響せずレンダだけで地表のテクスチャ感・情報量が上がる。
- **arnis参照**: `land_cover.rs:1123 coord_hash / ground_generation.rs:444-557(slope>6,>4 と LC_BUILT_UP/LC_BARE の h%N 分岐)`
- **適合方法（移植）**: coord_hashをnumpy uint64で移植(wrapping_mulは&0xFFFFFFFFFFFFFFFFでエミュレート、x/z格子のmeshgridに一括適用)。各ESAクラスに(block,確率)テーブルを定義し、np.whereやnp.select で閾値割当て。タイル分割でも(x,z)が世界座標なら境界整合する。

#### 3. [🔴 must / low] 二値水マスクをσ=3セルのガウシアンでぼかし0..1の water-ness 場を作り、0.5アイソラインを水際にする(ESA10m矩形格子のギザギザを~3ブロック軟化)。

- **なぜ御坊に効くか**: terrain_render.py は water_mask/sea_mask/dist_shore を二値で扱い、河川・海岸・内陸水の縁がブロック格子状に角張る(現状弱点3)。flood_psoは既にscipy.gaussian_filterを import 済みなので、水マスクをぼかして0.5で再二値化するだけで水際が滑らかになる。
- **arnis参照**: `land_cover.rs:104 compute_water_blend_smooth(SIGMA_CELLS=3、binaryマスク→gaussian_blur_grid→0.5閾値; 利用 ground_generation.rs:384,607)`
- **適合方法（移植）**: make_sea_mask/water_mask の bool を float化→gaussian_filter(sigma=3)→>0.5。砂浜帯(beach/coast_g)判定もこの場の勾配で出せる。海岸のっぺり対策(terrain_render.py:902-910)と統合可能。

#### 4. [🟡 nice / medium] value_noise_01: 整数格子4隅の coord_hash 値をsmoothstep(3t^2-2t^3)+双線形補間した0..1ノイズ。ディザを単一ピクセル散布でなく有機的なブロブにまとめる(閾値0.4で約20%、0.45で約30%被覆)。

- **なぜ御坊に効くか**: 純ハッシュ%Nだと畑/裸地のcoarse_dirtが塩胡椒状に散ってチカチカする。value_noiseで~5-6ブロック解像度の塊にすると、御坊の田畑・河川敷の裸地が自然なパッチに見える。ディザ(must項目)の品質を一段上げる上位互換。
- **arnis参照**: `ground_generation.rs:1227 value_noise_01(使用箇所 ground_generation.rs:479 LC_SHRUBLAND, :537 LC_BARE)`
- **適合方法（移植）**: div_euclidはnp.floor_divide、smoothstep/双線形はnumpyで素直に移植。scaleは5-6ブロック。ディザ閾値ロジックと組合せて使う。

#### 5. [🟡 nice / medium] カテゴリLCグリッドの境界だけをガウシアン重み多数決で平滑化(snapshotに対し各境界セルが近傍クラスのガウシアン投票で最頻クラスへ)。内部セルはスキップして高速化。

- **なぜ御坊に効くか**: terrain_render.py:805 の cover_patch ダウンサンプルは np.median で、ESA分類のノイズ(孤立画素)や境界のガタつきがそのまま残る(現状弱点3)。多数決平滑化で田畑/森/市街の境界が整い、孤立誤分類画素が消える。arnisが LC_BARE 孤立画素を除去する発想(ground_generation.rs:509-526)も同趣旨。
- **arnis参照**: `land_cover.rs:1005 smooth_class_boundaries(SIGMA_CELLS=2, is_boundary判定で境界のみ処理)`
- **適合方法（移植）**: median ダウンサンプルの代わり/後段に、カテゴリ配列へガウシアン重み多数決を適用。scipyなら各クラスのone-hotをgaussian_filterしてargmaxで近似実装でき、Rustの手書きカーネルより簡潔。

#### 6. [🟡 nice / medium] アンカー色→『視覚的に近い複数ブロックのリスト』を持ち、最近傍アンカーを選んだ後その候補から乱数(または建物ID/座標ハッシュ)で1つ選ぶ DEFINED_COLORS 方式。

- **なぜ御坊に効くか**: terrain_render.py の BUILDING_WALL_BY_TYPE/ROOF_BY_TYPE(360-394)は type→1ブロック固定で、同type建物が全部同色になり街区が単調(現状弱点4)。flood_psoは既に color_building_roofs/roof_color_tol を持つので、オルソ屋根色→最近傍アンカー→候補から建物IDハッシュで選択にすると、写真色を尊重しつつ棟ごとに微妙な色差が出て街が生き生きする。
- **arnis参照**: `block_definitions.rs:1320-1459 DEFINED_COLORS テーブル / block_definitions.rs:1462 get_building_wall_block_for_color / :1475 get_fallback_building_block`
- **適合方法（移植）**: アンカーRGB→候補keyリストの辞書を block_palette に追加。選択は乱数でなく building_id の coord_hash で決定論化(litematic再現性のため)。屋根の per-building 集約(build_building_maps の id)と相性良。

#### 7. [🟡 nice / high] ESA LCクラス+緯度+水距離→Minecraftバイオーム割当てと、Anvil1.18+の4x4サンプリング・パレット+ビット詰めbiomes生成。

- **なぜ御坊に効くか**: flood_pso BLOCKSのgrass/water色は固定RGBで、実Minecraftでは草・葉・水はバイオームtintで色が変わる。anvil_loader.py で歩けるワールド(region)を書く経路ではバイオームを正しく入れないと、御坊(温帯 abs_lat≈33.9→forest/plains/river)の地表色がデフォルトtintになる。water_distance≥8でocean/未満でriverの判定は日高川と海の塗り分けに直接効く。
- **arnis参照**: `biome.rs:13 biome_for_class / biome.rs:47 build_chunk_biome_nbt / biome.rs:103 bits_per_index / biome.rs:112 pack_biome_indices`
- **適合方法（移植）**: biome_for_class は単純なmatchなのでPython辞書で移植容易。pack_biome_indices(post-1.16の境界跨ぎ無しパッキング)は anvil 書き出し経路にのみ必要。litematic/NBT structure 経路はバイオームを持たないので対象外。effortは Anvil 書き出しの有無次第。

#### 8. [🟡 nice / low] 急斜面が土地被覆を上書きするカスケード(slope>8崖=deepslate縦縞, >6=石主体+cobble/andesiteバンド, >4=scree gravel少数派)。崖は列全体を1材質にして下方充填と一致させ縦縞を出す。

- **なぜ御坊に効くか**: terrain_render.py classify_surface_block は SLOPE_SCREE/VERY_STEEP/STEEP で stone/gravel を出すが2値的で岩肌が均一。arnisの%配分+崖縦縞は山地の岩肌を立体的に見せる。御坊周辺の急斜面・護岸の見栄え向上。slopeは既にcompute_slopeで計算済み。
- **arnis参照**: `ground_generation.rs:429-464(slopeしきい値 8/6/4 ≈45/37/27°と材質配分のコメント)`
- **適合方法（移植）**: classify_surface_block_grid の slope分岐に coord_hash ディザ(must項目)を併用するだけ。崖の『列全体1材質』は flood_pso の地盤柱(deep_ground)生成で under_block を surface と揃える形で再現。

**この軸で流用しない/不要なもの**:

- colors.rs:52 color_name_to_rgb_tuple(OSM colour=*の英名/HEXパーサ)— FG-GML/PLATEAUに自由記述の色タグは無く、御坊は type分類+オルソ実色で塗るため不要。
- land_cover.rs:146-870 ESA COG の HTTP Range取得・IFDパース・LZW/Deflate展開一式 — flood_pso は独自の ESA/cover 取得と GSI オルソ(tellus_data)を持つため再実装不要。
- biome.rs の Anvil1.18 biomesビット詰め(pack_biome_indices) — flood_pso の主経路は NBT structure / litematic でバイオームを保持しないため、region書き出し(anvil_loader)を使わない限り不要。
- bedrock_block_map.rs / luanti_block_map.rs — Java以外のターゲット変換で、flood_pso(NBT/litematic)には無関係。
- arnis の rgb_distance(colors.rs:102)単体 — Oklab(oklab_distance)で置換すべきなので、L2版そのものは流用しない。

> 軸まとめ: flood_pso の色軸の弱点は『知覚非一致のRGB最近傍』『1クラス=1色の単調地表』『二値マスク/medianによる角張った境界』の3点に集約され、arnis にはそれぞれへの直球の対策がある。最優先(must)は3つ: (1)colors.rs の Oklab距離を numpy 移植して ortho_surface の最近傍マッチへ差し込む(low/即効)、(2)land_cover.rs coord_hash + ground_generation の確率配分テーブルで地表クラスをディザ混合化(medium)、(3)compute_water_blend_smooth のガウシアン water-ness+0.5アイソラインで水際を軟化(flood_psoは既にscipy gaussian_filter利用、low)。次点(nice)で value_noise_01 によるディザのブロブ化、smooth_class_boundaries の境界多数決、DEFINED_COLORS方式の建物色多様化、急斜面カスケード、(region書き出しを採るなら)biome.rs のバイオーム割当て。Rust→Python移植は色変換・ハッシュ・ガウシアンとも numpy/scipy で素直に表現でき、最小工数で品質(写像精度・テクスチャ感・境界の滑らかさ)を底上げできる。決定論性(coord_hashを世界座標で)はタイル分割litematicの再現性に必須なので、乱数ではなく座標/IDハッシュで選択する点を守ること。

---

### 軸7: OSM/橋/道路など要素処理

**flood_pso 現状**: 橋はOSM由来のみ。bridge_osm.py の load_bridges() が Overpass の `out geom tags` JSON を手動curlで取得したキャッシュ(data_cache/osm/gobo_bridges_geom.json)を読み、highway→road_class(main/normal/dirt)・width・layer を付けた dict リスト化。bbox は単純な min/max 重なり判定で橋全体を採否(部分クリップ無し)。レンダリングは terrain_render.py:548 add_bridge_blocks() が各橋wayを独立に立体化: end_base()で両端の陸地形を外向き25blockのmedianサンプル→両岸線形補間 baseline、ramp(端0→中央最大の4:1勾配)、水を渡る区間だけ y_sea_surface+clearance(main6/normal5/dirt3m)を最低デッキ高に、PIER_SPACING_M=16mで橋脚、デッキ2層(上面=オルソ路面色/下面=andesite)、両端に欄干、layer×arch_rise_mのアーチ持ち上げ、橋下を洪水水位までwaterで充填。道路自体はOSMではなくラスタ road_mask/road_major_mask から描画。橋スタイルは andesite 単一(構造装飾なし)。複数wayに分割された1本の橋を束ねる仕組みが無く、隣接way間でデッキYが食い違いうる。

**arnis 読込ファイル**: `src/retrieve_data.rs`, `src/osm_parser.rs`, `src/element_processing/bridges.rs`, `src/element_processing/bridge_styles.rs`, `src/element_processing/highways.rs`, `src/element_processing/mod.rs`

#### 1. [🔴 must / medium] 複数の橋wayをUnionFindで1構造にグルーピングし、グループ単位で単一デッキY(terrain_max+clearance)を決める。union条件は(1)同一effective_layerで端点共有(2)同一bridge:nameかつ重心が近接(3)平行one-way対=デュアルカーブウェイ。

- **なぜ御坊に効くか**: flood_psoは橋wayを独立にadd_bridge_blocksしているため、OSMで1本の橋が複数wayに分割されている場合(御坊の長い橋や交差点で切れる橋)、隣接way端でend_base/ramp計算が別々になりデッキYが段差・不連続になる。グループ内で1つのdeck_yを共有すれば「歩ける」連続デッキになり、端点整合(タイル境界の連結窓と同じ思想)も取れる。
- **arnis参照**: `BridgeStructureMap::build (bridges.rs:97-391) のStep1-3 (bridges.rs:143-217)、UnionFind (bridges.rs:705-739)、deck_y算出 (bridges.rs:281-304)`
- **適合方法（移植）**: Rust UnionFindをPythonクラスに移植(20行程度)。bridge_osmのdictにway id相当(OSM element id)とlayerを保持させ、端点は格子座標 _lonlat_to_grid_xy 後の(round x,z)でキー化。bridge:name は現状未取得なのでOverpassのtagsにbridge:name追加が必要。deck_yはend_base群のmaxを採用するようadd_bridge_blocksのbaseline決定をグループ前計算に切り出す。

#### 2. [🔴 must / low] デッキ持ち上げ(clearance)を『平坦地形のときだけ』適用する判定。グループのcenterline_samplesで terrain_max-terrain_min=dip を測り、dip<閾値(4block) かつ 橋が一定長以上のときのみ layer段差/clearanceを足す。自然な谷(dipが大)があれば地形に沿わせる。

- **なぜ御坊に効くか**: 現状flood_psoは『水を渡る区間は常に y_sea_surface+clearance』で一律持ち上げる。日高川のように川幅・河床高が変化する場所では、自然に低い谷でも一律持ち上げると橋台手前で不自然な瘤や過大なrampが出やすい。dipベース判定なら地形が掘れている所は地形追従、平坦に水面だけある所だけ持ち上げ、と橋の挙動が自然になる。
- **arnis参照**: `centerline_samples (bridges.rs:635-659)、dip/clearance分岐 (bridges.rs:282-304)`
- **適合方法（移植）**: has_water判定の代わりに、橋中心線上の y_surf_land を等間隔サンプルしdip算出。dip<閾値かつtotal>=一定長のときのみ clear_full を min_deck に反映。閾値はscale_land(1.5)を掛けて調整。numpyで数行。

#### 3. [🟡 nice / low] Overpassの自動取得: 公式3+フォールバック2サーバを shuffle し、500/429/timeout を分類して指数的でないが段階的(primary3s/fallback5s)バックオフで順次リトライ、user-agent付与、空レスポンス/remark(out of memory)判定。

- **なぜ御坊に効くか**: 現状bridge_osm.pyはdocstringの手動curlワンライナーに依存し、Overpassの不安定(429/混雑)で手作業再実行が必要。御坊bboxの橋取得を main.py から再現可能・堅牢にでき、研究の再現性が上がる。
- **arnis参照**: `fetch_data_from_overpass (retrieve_data.rs:125-301)、サーバ群 (retrieve_data.rs:135-144)、リトライ/遅延 (retrieve_data.rs:248-282)、remark処理 (retrieve_data.rs:313-353)`
- **適合方法（移植）**: Python requests + 複数エンドポイントの逐次try/except、status_codeでメッセージ分岐、time.sleepで段階遅延。キャッシュが在れば取得スキップ。秘匿情報は無いがログにkey混入させない方針(MEMORY)に沿いURLそのままで可。bridgesだけならクエリは `way["bridge"]["highway"](bbox);out geom tags;` のまま。

#### 4. [🟡 nice / high] 橋スタイル(Beam/Arch/Truss/Suspension/CableStayed/Covered/Boardwalk)をbridge:structure/bridgeタグから解決し、上部構造を手続き生成。アーチは放物線 rise=max*4t(1-t)、吊橋ケーブルは懸垂線近似 dip*4t(1-t)、トラスはWarren鋸歯0..4..0、塔/横梁/ハンガー。

- **なぜ御坊に効くか**: flood_psoの橋はandesite桁+欄干のみで、天田橋・野口橋など特徴的な橋の見た目が出ない。Minecraft可視化(Layer C)の説得力・地物同定性が上がる。アーチ/懸垂線の数式はnumpyに直移植でき、橋脚配置(pillar_interval)も流用可。
- **arnis参照**: `resolve_bridge_style (bridge_styles.rs:213-244)、place_arch_spandrel_cell放物線 (bridge_styles.rs:380-410)、decorate_suspension懸垂線 (bridge_styles.rs:514-587, dip式570行)、decorate_truss (bridge_styles.rs:464-512)`
- **適合方法（移植）**: bridge:structure/bridgeタグをOverpass取得tagsに追加。スタイル別decorateをadd_bridge_blocks後段でデッキ中心線samples(既にループ内にcx,cz,dy,法線ox,ozがある)に対し実装。block_paletteにiron_block/chain/dark_oak等のkey追加が要る。座標系はox,oz(法線)が既存なのでside_offsetsはそのまま移植可。御坊で実在するスタイルだけ先行実装(arch/truss優先)。

#### 5. [🟡 nice / low] 路面/デッキの『横断方向medianで水平化+進行方向3タップmedianで1セル穴埋め』。横断ストリップのground中央値を採れば斜面でも左右に傾かない平面、長手3タップで …1 1 0 1 1… を …1 1 1 1 1… に補正(単調勾配は不変)。

- **なぜ御坊に効くか**: 『歩ける御坊』ではラスタroad_maskやDEM由来の路面が1セルの凹凸でガタつき、歩行・スケール1.5で段差が目立つ。横断median化と3タップ穴埋めは軽量で、橋アプローチや幅広路面を平滑にし歩行体験を改善する。flood_psoの道路はラスタなので幅方向ループに直接適用しやすい。
- **arnis参照**: `perpendicular_median_ground_y (highways.rs:122-166)、perpendicular_median_raw (highways.rs:29-54)、precompute_row_medians (highways.rs:69-97)`
- **適合方法（移植）**: numpyで実装容易: 路面マスク上で各中心セルの横断幅ぶんの y_surf_land を取りmedian、長手方向に scipy.ndimage.median_filter(size=3) を1Dで掛ける。橋デッキのput前に dy 配列へ適用。block_range相当=half_w。

#### 6. [🟡 nice / low] 幅の解決規則: width=* タグ優先、無ければ lanes×3.5m から half-width 算出し型別既定値とmax、上限クランプ、scale<1で縮約だが最低1。

- **なぜ御坊に効くか**: flood_psoは width 無し時に road_class 既定(main9/normal5.5/dirt3m)固定で、実際の車線数を無視。lanesタグを使えば御坊の片側2車線県道などで橋幅がより実寸に近づき、半幅half_wの過小/過大を減らせる。
- **arnis参照**: `highway_block_range (highways.rs:1588-1630)、highway_default_lanes (highways.rs:1579-1584)`
- **適合方法（移植）**: _parse_width のフォールバックを lanes 基準に拡張: width無→tags.get('lanes')→lanes*3.5m。Overpass tags に lanes 追加。h_res_block_m で block 化する既存式に半幅を渡すだけ。

#### 7. [🟡 nice / medium] way単位のbboxクリップ(端点間を線分でbboxに切り、はみ出しノードを境界交点に置換)で部分的にbbox内の橋を保持。完全外のみ除外。

- **なぜ御坊に効くか**: 現状flood_psoのload_bridgesはbbox重なりがあれば橋coords全体を渡し、無ければ橋ごと捨てる二択。タイル分割(scale1.5でタイル化)時、橋がタイル境界を跨ぐとタイル外のノードまで描画ループに入り、別タイルと整合させづらい。線分クリップすればタイル境界で橋デッキがきれいに切れ、連結窓方式と整合する。
- **arnis参照**: `parse_osm_data 内の clip_way_to_bbox 呼び出し (osm_parser.rs:325-339, relation側412)`
- **適合方法（移植）**: Cohen–Sutherland等の線分クリップをlat/lon→grid後の座標で実装(20-30行)。ただしデッキ連続性のためグループ化はクリップ前のフル形状で行い、描画だけクリップ範囲に限定する二段構えが望ましい。

#### 8. [🟡 nice / low] デュアルカーブウェイ検出: 平行/反平行(heading差±20°)かつ中点間距離<=12block の one-way 橋way対を同一構造に統合。

- **なぜ御坊に効くか**: 御坊近辺で上下線が別wayの橋(分離橋)がある場合、独立描画だと2つの欄干付き桁が並び不自然/Y食い違い。統合すれば一体のデッキとして高さ整合できる。
- **arnis参照**: `are_dual_carriageway_pair (bridges.rs:661-683)、heading_deg (bridges.rs:685-697)`
- **適合方法（移植）**: グループ化(must項目)のStep3として追加。one-wayはタグ取得が要る。heading/midpointはnumpyで簡単。御坊に分離橋が無ければskip可なのでpriority nice。

#### 9. [🟡 nice / low] 欄干/ケーブル等の対角ライン4連結化: bresenham対角ステップで生じるL字隙間を stair_fill_cells で4近傍補完し連続させる。

- **なぜ御坊に効くか**: flood_psoの欄干は各サンプルの両端(abs(w)==half_w)に置くだけで、橋が斜めだと欄干セルが対角に飛んで隙間が出る(歩行時に落下/見た目の途切れ)。4連結補完で連続した手すりになる。
- **arnis参照**: `stair_fill_cells (highways.rs:215-232)、レール配置で利用 (highways.rs:1083-1141)、draw_cableのL角補完 (bridge_styles.rs:776-785)`
- **適合方法（移植）**: 欄干putの前段で前サンプルのレールセルと現セルを stair_fill_cells で繋ぐ。put関数とseenで重複は既に吸収。スタイル装飾を入れる場合のcableにも同ロジック流用。

**この軸で流用しない/不要なもの**:

- Overpassの全要素クエリ(building/landuse/water/highway/...一括取得とrelation member展開, retrieve_data.rs:150-194)は、flood_psoが建物・道路・水域をFGD地物/DEM/オルソから生成しOSMからは橋のみ使う方針のため不要。橋に限れば既存の単純クエリで足りる。
- WebMercator/Local projection と CoordTransformer(osm_parser.rs:246-260)は、flood_psoが独自の _lonlat_to_grid_xy + patch_bbox_latlon 格子を持つため不要(座標変換思想は同じだが置換不要)。
- tag間引き IGNORED_TAGS/PREFIXES(osm_parser.rs:12-69)は、flood_psoが橋 way だけ少量を読むためメモリ最適化のメリットが薄い。
- street_lamp/bus_stop/traffic_signal などノード地物の立体化(highways.rs:347-519)は洪水可視化に不要な街路装飾でskip。
- ESA WorldCoverによる海岸線/海洋OSM除外ロジック(retrieve_data.rsクエリのnatural!=coastline等)は、flood_psoがDEM由来のsea_mask/y_sea_surfaceで海陸を判定済みのため不要。
- multipolygon/building relation組み立て(osm_parser.rs:342-437, merge_way_segments mod.rs:29-125)は橋(単純polyline way)には不要。ただし将来FGDの面地物整形に転用余地はある。

> 軸まとめ: arnisで最も価値が高いのは『橋way群のUnionFindグルーピング+グループ単位の単一デッキY』(bridges.rs:97-391)で、flood_psoが橋wayを独立描画して隣接way端でデッキYが食い違う弱点を直接埋め、歩ける連続デッキ/タイル境界整合に効く(must)。併せて『dipベースのclearance適用』(bridges.rs:282-304)が一律持ち上げによる橋台手前の瘤を解消(must)。次点で、Overpass自動取得の堅牢化(retrieve_data.rs:125-301)、橋スタイル装飾(bridge_styles.rs全体, アーチ放物線/懸垂線の数式は直移植可)、路面の横断median+3タップ穴埋め平滑化(highways.rs:122-166)が品質・歩行体験を底上げ(nice)。RustからPythonへはUnionFind/幾何/補間がいずれも数十行で移植可能で、座標は既存の _lonlat_to_grid_xy 後の格子で扱える。OSM全要素取得や投影系・relation組立はflood_psoがFGD/DEM/オルソ主体のため不要。前提として、グルーピングやスタイル判定に必要な bridge:name / layer / lanes / bridge:structure を Overpass の取得tagsへ追加する小改修が要る。

---

## すぐ効く小改善（quick wins, low cost）

- 反復3x3平均NaN補間（`postprocess.rs:1061`）を `dem_parser`/`wakayama_pcd` 出口へ共通関数で挿入 — NaN海誤認・方向性アーティファクト解消
- IQR×3外れ値クリップ（`postprocess.rs:1117`）を前処理先頭へ
- Oklab距離（`colors.rs:114`）を `classify_rgb_to_palette` へ差し込み — オルソ色化け即改善
- 水マスク gaussian+0.5（`land_cover.rs:104`）— 海岸線/河川縁の角張り軟化（既存scipy利用）
- 近傍最低Y可変アンダーフィル（`ground_generation.rs:716-758`）— 固定 deep_ground=8 の +77% 無駄と底抜けを同時解消
- floodfill を `binary_propagation` へ置換（`floodfill.rs:115-197`）— PSO内側ループ高速化、1コール化
- floodfill スタック前計算キャッシュ（`floodfill_cache.rs`発想）— `simulate_flood_hd` の water_level/dh 非依存な gaussian 再計算を PSO 開始前に1回
- dip ベース clearance（`bridges.rs:282-304`）— 橋台手前の瘤解消

## 腰を据えた大きめの作業（med〜high cost）

- **`{pos,state}`列挙→密numpy配列**（`world_editor/common.rs:84-364`）— `dem_to_blocks` 改修、.nbt中間生成廃止。8-12GB問題の根治
- **native Anvil(.mca)書き出し器**（`world_editor/java.rs:744-999`）— Tellus非依存・ビューア上限撤廃。最終目標の歩けるワールド直接出力（high）
- **タイル整数ブロック整列+halo描画**（`tile.rs:56-260`）— `752`マジック/×0.985隙間バグの原理的解消
- 中央値+MAD地形異常修復（`postprocess.rs:22`）— docs/07自認の最大欠落を埋める
- 解像度順provider選択+NaN率フォールバック（`selector.rs:26` / `mod.rs:133`）— GSI+Mapzenハイブリッド設計の実体化、ZeroDivisionErrorガード
- FGDポリゴン scanline ラスタライズで水源精緻化（`water_areas.rs:274-387`）— 損失精度の底上げ
- 橋way UnionFindグルーピング（`bridges.rs:97-391`）— 連続デッキ
- value_noise/ディザ混合（`ground_generation.rs:1227` / `land_cover.rs:1123`）— 単調地表の緩和（決定論ハッシュ必須）

## 流用しない/不要なもの（全体）

- **rayon並列**（postprocess全域）— numpyベクトル化で同等以上、明示並列不要
- **f64→f32ダウンキャスト**（`mod.rs:29,265`）— 単一都市タイル数千²でメモリ律速でなく得が薄い
- **symlink安全キャッシュ削除**（`cache.rs:63-156`）— OSS配布のセキュリティ要件、研究用ローカルには過剰。shutil/pathlibで十分
- **Web Mercator 主DEM経路導入** — 御坊は南北15km・歪み<0.1%、既存lat/lon等間隔格子とミスアライン。`geo_distance`等の精度補正のみ流用
- **map回転/translate**（`transform_map.rs`）— 御坊は真北upでそのまま出す
- **snow_line**（`ground.rs:48-72`）— 御坊は温暖湿潤Cfaで雪不要
- **OSM全要素取得/relation組立/multipolygon** — flood_psoはFGD/DEM/オルソ主体、橋のみOSM
- **Bedrock/Luanti出力・bedrock_block_map** — Java NBT/litematic専用で対象外
- **lighting bake**（`java.rs:612-709`）— 本体/litematicaは読込時再ライティング、Webビューア自前描画時のみ
- **DDAメッシュボクセル化**（`voxelize.rs`）— ランドマーク建物限定の高コスト任意機能。歩行地形は heightfield で足りる

## 推奨着手順序

1. **DEM後処理3点（quick win）を前処理段へ挿入** — IQRクリップ→中央値+MAD修復→反復NaN補間の順（arnis同順）。`dem_parser`/`wakayama_pcd` 出口の共通関数化。docs/07自認の穴を最小コストで埋め、以降の全レンダ品質の土台になる。
2. **color/water/underfill のレンダ即効改善** — Oklab距離・水マスクgaussian・近傍最低Y可変アンダーフィルを投入。PSO本体に不干渉でNBT軽量化と見栄えを同時に得る。座標ハッシュは必ず世界座標で（litematic再現性）。
3. **密numpy配列化** — `dem_to_blocks` を `{pos,state}` 列挙から密配列へ。8-12GBメモリ問題を根治し、これが次のタイル整合・native Anvilの前提インフラになる。
4. **タイル整数ブロック整列+halo + floodfillのフルドメイン1回計算** — 継ぎ目を構造的に解消。浸水はグローバル計算→タイル純粋スライスに分離し、`binary_propagation`化でPSOも高速化。
5. **(最終目標) native Anvil書き出し器** — heightmap生成込みでTellus依存を脱却、御坊全域の歩けるワールドを直接出力。最も大きいがLayer Cの本丸。

## 移植の現実性

arnisはRustだが、流用対象のコアは色変換 / 座標ハッシュ / ガウシアン / 距離変換(EDT) / 中央値フィルタ / scanlineラスタライズ / UnionFind / bit-pack で、すべて numpy・scipy.ndimage・skimage・rasterio・pyproj の既存関数か数十行の純numpyに落ちる。Rust固有の所有権共有・rayon・f32最適化・symlink処理は移植時に捨てるのが正解で、誇張なく低〜中コストで等価実装できる。

---
*出典: マルチエージェント調査ワークフロー（7軸×抽出→統合, 2026-06-26）。arnis v2.9 / flood_pso 現行。file:line はarnis側の参照位置。*
