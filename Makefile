# ============================================================================
#  flood_pso — 御坊全域 walkable Anvil ワールド生成 Makefile
# ----------------------------------------------------------------------------
#  実証済レシピ(2026-06-26, 18図郭 --tiles 6x12 --anvil-world, 当時 完走5h11m/出力2.2GB)に
#  最新フル機能(トンネル / 橋v8 / OSM電線・駐車場 / 鉄道 / 道路curb / 連絡通路 / gap補完)を統合。
#  フル機能を載せた現行構成の実測は 5.29h(下の world-full を参照)。
#
#  使い方:
#    make help          … ターゲット一覧
#    make crop          … 中心400m角を1タイルだけ生成(実測: 初回 5分35秒[洪水sim込] /
#                          2回目以降 26.2秒 / W=200 なら 15.3秒)。
#                          橋/トンネル等のコード修正 → 目視確認を高速反復するためのターゲット。
#    make world-test    … 南部4図郭の中央2km角を2x2タイルで生成し **オプション検証**
#                          (実測 13分34秒[sim キャッシュ有り], 出力 139MB)。
#    make world-full    … 御坊全域18図郭の本番生成(実測 5.29h, 洪水sim ~2h含む, 出力 2.2GB)。
#    make osm-check     … world-full が要求する OSM geom JSON の実在チェック(world-full が自動実行)
#    make osm-fetch     … 上で欠けている JSON を Overpass から取得(要ネット)
#
#  オプション上書き(必要時のみ):
#    make crop CLAT=33.8385 CLON=135.1800 W=600      … crop の中心/一辺[m]を変更
#    make world-full LEGEND=--legend-layer            … 地下にLayer C可視化層(土地利用/洪水/樹木)を埋込
#    make world-full ORTHO_LAYER=seamlessphoto        … オルソが ort で欠ける場合のフォールバック
#    make world-full EVAC='--evac --evac-xml <P20.xml>' … 避難施設マーカー(※P20 xmlは現在欠落)
#    make world-full POWER= PARKING=                  … 持っていないOSMフィーチャを明示的に無効化
#    make crop KEEP_NBT=1                             … 中間 structure .nbt も残す(既定は Anvil のみ)
# ============================================================================

PY        := PYTHONUNBUFFERED=1 .venv/bin/python
LID       := data_cache/wakayama_lidar
OSM       := data_cache/osm
D561      := ../kennkyuu20260114/地形データ/FG-GML-503561-ALL-20251001
D551      := ../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101

# --- FGD(国土地理院 FG-GML): 全域は lat33.8333 を跨ぐので 503561(北)+503551(南) の両メッシュ union ---
FGD_BLD   := $(D561)/FG-GML-503561-BldA-20251001-0001.xml,$(D551)/FG-GML-503551-BldA-20260101-0001.xml
FGD_RDEDG := $(D561)/FG-GML-503561-RdEdg-20251001-0001.xml,$(D551)/FG-GML-503551-RdEdg-20260101-0001.xml
FGD_WA    := $(D561)/FG-GML-503561-WA-20251001-0001.xml,$(D561)/FG-GML-503561-WStrA-20251001-0001.xml,$(D551)/FG-GML-503551-WA-20260101-0001.xml,$(D551)/FG-GML-503551-WStrA-20260101-0001.xml
FGD_RAIL  := $(D561)/FG-GML-503561-RailCL-20251001-0001.xml,$(D551)/FG-GML-503551-RailCL-20260101-0001.xml

# --- LiDAR 18図郭(御坊全域 6km×10.5km)。grd=地表DEM, org=DSM建物実高さ ---
#     ※grdは結合mosaic npz(427MB)が, orgは図郭別npz(18枚)が既にキャッシュ済 → 高速ロード(txt欠落でも可)
GRD18 := $(LID)/06RC701_grd.txt,$(LID)/06RC702_grd.txt,$(LID)/06RC703_grd.txt,$(LID)/06RC704_grd.txt,$(LID)/06RC711_grd.txt,$(LID)/06RC713_grd.txt,$(LID)/06RC801_grd.txt,$(LID)/06RC802_grd.txt,$(LID)/06RC803_grd.txt,$(LID)/06RC804_grd.txt,$(LID)/06RC811_grd.txt,$(LID)/06RC813_grd.txt,$(LID)/06RC902_grd.txt,$(LID)/06RC904_grd.txt,$(LID)/06RC911_grd.txt,$(LID)/06RC913_grd.txt,$(LID)/06SC002_grd.txt,$(LID)/06SC011_grd.txt
ORG18 := $(LID)/06RC701_org.txt,$(LID)/06RC702_org.txt,$(LID)/06RC703_org.txt,$(LID)/06RC704_org.txt,$(LID)/06RC711_org.txt,$(LID)/06RC713_org.txt,$(LID)/06RC801_org.txt,$(LID)/06RC802_org.txt,$(LID)/06RC803_org.txt,$(LID)/06RC804_org.txt,$(LID)/06RC811_org.txt,$(LID)/06RC813_org.txt,$(LID)/06RC902_org.txt,$(LID)/06RC904_org.txt,$(LID)/06RC911_org.txt,$(LID)/06RC913_org.txt,$(LID)/06SC002_org.txt,$(LID)/06SC011_org.txt

# --- LiDAR 南部4図郭(御坊市南部 4.0km×3.0km, lat 33.8201-33.8474 / lon 135.1571-135.2006)。
#     結合mosaic npz(0.666667m, 4556×6030)がキャッシュ済なので world-test/crop はここから即ロード。
GRD4 := $(LID)/06RC904_grd.txt,$(LID)/06RC913_grd.txt,$(LID)/06SC002_grd.txt,$(LID)/06SC011_grd.txt
ORG4 := $(LID)/06RC904_org.txt,$(LID)/06RC913_org.txt,$(LID)/06SC002_org.txt,$(LID)/06SC011_org.txt

# --- OSM 由来フィーチャ(FG-GMLに無いものを補完。値行に末尾コメントを書くと空白が混入するので別行) ---
#     ※パスを渡したのにファイルが無いと make_nbt_hd.py は **停止する**（旧: 無言で0本のワールドが
#       出来ていた）。この4本は world-full が既定で全部渡すので、持っていないフィーチャは
#       空文字にして明示的に無効化すること: make world-full POWER= PARKING=
#       欠落したまま world-full を叩いた場合は osm-check が「何が無くてどう取るか」を出して止まる。
# 橋(全御坊144本full_geom, make_nbt_hd既定と同一) / トンネル(tunnel=yes, 同既定)
BRIDGES ?= $(OSM)/gobo_bridges_full_geom.json
TUNNELS ?= $(OSM)/gobo_tunnels_geom.json
# 送電線+鉄塔 / 駐車場舗装+白線 (make_nbt_hd 側の既定は空だが world-full ではこのパスを渡す)
POWER   ?= $(OSM)/gobo_power_geom.json
PARKING ?= $(OSM)/gobo_parking_geom.json
OSM_FULL := --bridges-json "$(BRIDGES)" --tunnels-json "$(TUNNELS)" \
            --power-json "$(POWER)" --parking-json "$(PARKING)"
# osm-check / osm-fetch 用: kind(osm_cache.GEOM_QUERIES のキー):出力パス。空パス=無効化。
OSM_KINDS := bridge:$(BRIDGES) tunnel:$(TUNNELS) power:$(POWER) parking:$(PARKING)
# 変数名の対応（欠落時にどれを空にすれば良いか案内するため）
OSM_VARS  := BRIDGES:$(BRIDGES) TUNNELS:$(TUNNELS) POWER:$(POWER) PARKING:$(PARKING)
# osm-fetch の取得 bbox "S,W,N,E"。御坊全域18図郭(6km×10.5km, --tiles 6x12)を内包する広めの矩形。
#   厳密でなくてよい: 各 loader が patch bbox で絞るため superset は無害（範囲外は捨てられる）ので
#   足りないより広い方を採る。南端/東端は南部4図郭 mosaic の実測値(33.8201 / 135.2006)が根拠。
OSM_BB    ?= 33.812,135.125,33.935,135.210

# 南部4図郭用: 橋は図郭別 JSON を結合して使う(全御坊 full_geom が無い環境でも動く)。
# トンネル/電線/駐車場の JSON は南部では未取得のため空文字で無効化。
BR_PARTS := $(OSM)/gobo_bridges_904_geom.json $(OSM)/gobo_bridges_913_geom.json \
            $(OSM)/gobo_bridges_002_geom.json $(OSM)/gobo_bridges_011_geom.json
BR_SOUTH := $(OSM)/gobo_bridges_south4_geom.json
OSM_SOUTH := --bridges-json "$(BR_SOUTH)" --tunnels-json "" --power-json "" --parking-json ""

# --- 上書き可能トグル(既定: 純粋な歩けるワールド) ---
# 地表写真: ort=整備済オルソ(高精細) / seamlessphoto=最新シームレス(被覆広い)
ORTHO_LAYER ?= ort
# 地下Layer C可視化層: 入れるなら make ... LEGEND=--legend-layer
LEGEND      ?=
# 避難施設: 入れるなら make ... EVAC='--evac --evac-xml <P20.xml>' (現状 P20 xml 欠落)
EVAC        ?=
# Anvilワールドの最下ブロックを置く world Y。負値で世界を下げ高い山が build limit(319)で
# 切れないよう頭上余裕を作る。-50=山対策(MCは-64まで可。さらに切れるなら -64 に)。
WORLD_BASE_Y ?= -50
# 中間 structure .nbt: world-test/crop は既定で書かない(1タイル約60MB×枚数を節約)。
# KEEP_NBT=1 で従来どおり results/nbt/hd に残す。world-full は常に残す(本番成果物)。
KEEP_NBT ?= 0
NBT_OPT  := $(if $(filter 1 yes true on,$(KEEP_NBT)),,--no-intermediate-nbt)
# 中間 .nbt の gzip 圧縮レベル。6 が最速かつ最小(L9比 9-15倍速・同等以下のサイズ)。
NBTZ ?= 6

# --- crop(高速反復)の中心と一辺[m]。橋/トンネルの目視確認したい場所へ移して使う ---
#   既定は南部4図郭mosaicの中心。例: make crop CLAT=33.8385 CLON=135.1800 W=600
CLAT ?= 33.8337
CLON ?= 135.1789
W    ?= 400
H    ?= $(W)
CROP_OUT ?= results/anvil/gobo_crop
# world-test の一辺[m]とタイル分割(中心は CLAT/CLON 共通)。2000/2x2 で 1km タイル×4枚。
TEST_W     ?= 2000
TEST_TILES ?= 2x2

# --- 中間 structure .nbt の置き場と陳腐化対策 ---
#   make_nbt_hd.py の出力名は "gobo_hd_...<tsuffix>_<tag-suffix>[_rNcM].nbt"（--tag-suffix _crop
#   なら "__crop"）。Anvil を毎回作り直すターゲット(crop / world-test)は **同名 .nbt を
#   上書きも削除もしない** ため、KEEP_NBT=1 で一度書いた .nbt が以後の実行を生き延び、
#   Anvil と食い違う古い世界を指し続ける。Anvil 側と同じ理由でこちらも毎回消す。
#   (world-full の .nbt は本番成果物なので対象外)
NBT_DIR   := results/nbt/hd
CROP_TAG  := _crop
TEST_TAG  := _test4
# make_nbt_hd.py:554-561 の _ttag() は --tiles に応じて 3 形式を出す:
#   Nx1 → _c{ci} / 1xN → _r{ri} / NxM(両方>1) → _r{ri}c{ci}
# `_r*c*` だけだと Nx1/1xN の中間 .nbt が残って陳腐化するので `_[rc]*` で全形式を拾う。
CROP_NBT   = $(NBT_DIR)/*_$(CROP_TAG).nbt $(NBT_DIR)/*_$(CROP_TAG)_[rc]*.nbt
TEST_NBT   = $(NBT_DIR)/*_$(TEST_TAG).nbt $(NBT_DIR)/*_$(TEST_TAG)_[rc]*.nbt

# --- 全機能共通フラグ(本番/検証で共有。OSM JSON はターゲット毎に OSM_FULL / OSM_SOUTH) ---
#   methods gt = 地形ベース / use-fgd = FGD建物・道路 / road-curb は既定ON(連絡通路1F抜きを含む)
#   trees sparse = 間引き個別樹木 / scale 1.5 = 0.667m/block
COMMON := --K 16 --seed 0 --preset gobo_walk_1km --methods gt --scale 1.5 \
          --use-fgd --fgd-bld "$(FGD_BLD)" --fgd-rdedg "$(FGD_RDEDG)" \
          --fgd-wa "$(FGD_WA)" --fgd-rail "$(FGD_RAIL)" \
          --surface-ortho --ortho-layer $(ORTHO_LAYER) \
          --trees --tree-mode sparse \
          --world-base-y $(WORLD_BASE_Y) --nbt-compresslevel $(NBTZ) \
          --fill-gap-osm $(LEGEND) $(EVAC)

.PHONY: help world-full world-test crop crop-clean osm-check osm-fetch
.DEFAULT_GOAL := help

help:
	@echo "flood_pso 御坊ワールド生成:"
	@echo "  make crop         … 中心400m角×1タイル(実測 初回5分35秒[sim込] / 2回目以降26.2秒 / W=200なら15.3秒)"
	@echo "                      橋/トンネル修正の目視反復用"
	@echo "                      → $(CROP_OUT) (中心/サイズ変更: make crop CLAT=.. CLON=.. W=..)"
	@echo "  make world-test   … 南部4図郭の中央2km角 2x2タイル(実測13分34秒/出力139MB)。本番前のオプション検証"
	@echo "  make world-full   … 御坊全域18図郭 本番生成(実測5.29h, 出力~2.2GB to results/anvil/gobo_zeniki_v2)"
	@echo "  make osm-check    … world-full が要求する OSM geom JSON 4本の実在チェック(world-full が自動実行)"
	@echo "  make osm-fetch    … 欠けている OSM geom JSON を Overpass から取得(要ネット, bbox=$(OSM_BB))"
	@echo "  make crop-clean   … crop の出力ワールドと中間 .nbt を削除"
	@echo "  上書き例: make world-full LEGEND=--legend-layer ORTHO_LAYER=seamlessphoto"
	@echo "            make world-full POWER= PARKING=   (持っていないOSMフィーチャを明示的に無効化)"

# 図郭別の橋 JSON を1本に結合(重複 way は id で排除)。world-test/crop の前提データ。
$(BR_SOUTH): $(BR_PARTS)
	@$(PY) -c "import json,sys;els=[];[els.extend(json.load(open(p)).get('elements',[])) for p in sys.argv[1:]];u={e.get('id',i):e for i,e in enumerate(els)};json.dump({'version':0.6,'generator':'flood_pso Makefile (merge of 図郭別 bridges)','elements':list(u.values())},open('$@','w'));print('  [osm] merged %d ways -> $@' % len(u))" $(BR_PARTS)

# OSM geom JSON の事前チェック。make_nbt_hd.py は「明示指定したパスが無ければ停止」するが、
#   それだけだと *何を* どう取れば良いか分からないので、make 側で欠落一覧と取得方法を出す。
#   「無言で0本」に戻さないため、欠落があれば **必ず非ゼロで終了する**（勝手に外さない）。
#   空文字にした変数は「意図的に無効化」なのでチェック対象外。
osm-check:
	@miss=""; empty=""; \
	for spec in $(OSM_VARS); do \
	  var=$${spec%%:*}; f=$${spec#*:}; \
	  if [ -n "$$f" ] && [ ! -f "$$f" ]; then miss="$$miss $$var"; \
	    echo "  [osm] 欠落: $$f  ($$var)"; \
	  elif [ -n "$$f" ]; then \
	    n=$$($(PY) -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('elements',[])))" "$$f" 2>/dev/null || echo ERR); \
	    if [ "$$n" = "ERR" ]; then miss="$$miss $$var"; echo "  [osm] 壊れ: $$f  ($$var) — JSON として読めません"; \
	    elif [ "$$n" = "0" ]; then empty="$$empty $$var"; echo "  [osm] 0件: $$f  ($$var) — 取得時の bbox が違う可能性"; \
	    fi; \
	  fi; \
	done; \
	if [ -n "$$empty" ] && [ -z "$$miss" ]; then \
	  echo ""; \
	  echo "*** world-full を中止しました（elements が 0 件の JSON があります）"; \
	  echo "  このまま回すと 5.29時間かけて そのフィーチャが0本の世界が出来ます。"; \
	  echo "  取り直す: 該当ファイルを消してから make osm-fetch OSM_BB=S,W,N,E"; \
	  printf "  0件を承知で生成する: make world-full"; for v in $$empty; do printf " %s=" "$$v"; done; echo ""; \
	  exit 1; \
	fi; \
	if [ -z "$$miss" ]; then echo "  [osm] ok: world-full が要求する OSM geom JSON はすべて実在し 1件以上"; exit 0; fi; \
	echo ""; \
	echo "*** world-full を中止しました（無言で0本のワールドが出来るのを防ぐため）"; \
	echo "  取得する(要ネット, bbox=$(OSM_BB)):"; \
	echo "      make osm-fetch                    … 欠落分だけ Overpass から取得して上のパスへ保存"; \
	echo "      make osm-fetch OSM_BB=S,W,N,E     … bbox を変える場合"; \
	echo "  取得せずに生成する(そのフィーチャは0件になります):"; \
	printf "      make world-full"; for v in $$miss $$empty; do printf " %s=" "$$v"; done; echo ""; \
	exit 1

# 欠落している OSM geom JSON を Overpass から取得する（osm_cache 経由 = bbox量子化キャッシュ+
#   ミラー再試行+FLOOD_PSO_OFFLINE ガード付き。クエリは src/osm_cache.py の GEOM_QUERIES が正）。
#   既存ファイルは上書きしない（手で結合した full_geom を潰さないため）。
#   実測: 4本で 5分25秒（Overpass が 504 を返すとミラーを巡回して待つため。ハングではない）。
#   0件で返ってきた場合は WARN を出す（ファイルは出来るが、そのフィーチャは0本のままになる）。
osm-fetch:
	@mkdir -p $(OSM)
	@for spec in $(OSM_KINDS); do \
	  kind=$${spec%%:*}; out=$${spec#*:}; \
	  if [ -z "$$out" ]; then echo "  [osm] $$kind: 変数が空 → スキップ(意図的に無効化)"; continue; fi; \
	  if [ -f "$$out" ]; then echo "  [osm] $$kind: 既存 $$out → スキップ"; continue; fi; \
	  $(PY) -c "import sys,json,pathlib;sys.path.insert(0,'src');from osm_cache import fetch_overpass_geom;k,o=sys.argv[1],sys.argv[2];S,W,N,E=[float(x) for x in sys.argv[3].split(',')];d=fetch_overpass_geom(k,S,N,W,E);n=len(d.get('elements',[]));pathlib.Path(o).write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8');print('  [osm] %s -> %s (%d elements)' % (k,o,n));print('  [osm][WARN] %s は0件です。bbox=%s が対象域を覆っているか確認して下さい(このままだと%sは0本のワールドになります)' % (k,sys.argv[3],k)) if n==0 else None" \
	    $$kind "$$out" "$(OSM_BB)" || exit 1; \
	done

# 本番: 御坊全域18図郭 → 1 Anvil world(--tiles 6x12=72タイルを実座標mergeで密着)。litematicは巨大化するため抑止。
#   洪水simは ~2h。--reuse-inundation で2回目以降(オプション微調整)は sim をスキップして再利用。
#   実測 5.29h / 出力 2.2GB。中間 .nbt は本番成果物として残す(KEEP_NBT 非依存)。
world-full: osm-check
	@mkdir -p data_cache/inund_zeniki results/anvil
	$(PY) src/make_nbt_hd.py $(COMMON) $(OSM_FULL) \
	  --wakayama-grd "$(GRD18)" --wakayama-org "$(ORG18)" \
	  --tiles 6x12 --anvil-world results/anvil/gobo_zeniki_v2 --no-litematic \
	  --reuse-inundation data_cache/inund_zeniki --tag-suffix _zeniki

# 検証: 南部4図郭(904/913/002/011)の中央 2km 角を --tiles 2x2 で生成し、全フィーチャの
#   オプションが通るか・タイル境界が密着するかを確認する。実測13分34秒 / 出力139MB
#   (1kmタイル×4枚、洪水sim キャッシュ有り)。sim 未計算の初回は +5分ほど。旧コメントの
#   「数分」は誤り。※旧版は 06RC701/702 を指していたが両図郭は未取得で即 FileNotFoundError。
#   crop と同様、出力ワールドと中間 .nbt($(TEST_TAG)) を毎回消してから作り直す。
world-test: $(BR_SOUTH)
	@rm -rf results/anvil/gobo_test_south4
	@rm -f $(TEST_NBT)
	@mkdir -p data_cache/inund_south4 results/anvil
	$(PY) src/make_nbt_hd.py $(COMMON) $(OSM_SOUTH) $(NBT_OPT) \
	  --wakayama-grd "$(GRD4)" --wakayama-org "$(ORG4)" \
	  --center-lat $(CLAT) --center-lon $(CLON) --width $(TEST_W) --depth $(TEST_W) \
	  --tiles $(TEST_TILES) --anvil-world results/anvil/gobo_test_south4 --no-litematic \
	  --reuse-inundation data_cache/inund_south4 --tag-suffix $(TEST_TAG)

# 高速反復: 1タイルだけを小さく切り出す。橋/トンネル/道路のコードを直して目視確認する用。
#   洪水sim は data_cache/inund_south4 に共有キャッシュ(world-test と同一)。初回は sim 込みで
#   実測 5分35秒、2回目以降は実測 26.2秒(W=200 なら 15.3秒)。
#   --no-intermediate-nbt(既定) で中間 .nbt を書かず Anvil だけ出す。
#   ※ $(CROP_OUT) と中間 .nbt($(CROP_TAG)) は毎回作り直す(既存を削除)。CLAT/CLON/W を動かした
#      とき前回の chunk や .nbt が残って「直したはずの橋が古いまま見える」事故を防ぐため。
#      保存したい世界は別名でコピーを。
crop: $(BR_SOUTH)
	@rm -rf $(CROP_OUT)
	@rm -f $(CROP_NBT)
	@mkdir -p data_cache/inund_south4 results/anvil
	$(PY) src/make_nbt_hd.py $(COMMON) $(OSM_SOUTH) $(NBT_OPT) \
	  --wakayama-grd "$(GRD4)" --wakayama-org "$(ORG4)" \
	  --center-lat $(CLAT) --center-lon $(CLON) --width $(W) --depth $(H) \
	  --anvil-world $(CROP_OUT) --no-litematic \
	  --reuse-inundation data_cache/inund_south4 --tag-suffix $(CROP_TAG)

crop-clean:
	rm -rf $(CROP_OUT)
	rm -f $(CROP_NBT)
