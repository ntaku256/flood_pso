# ============================================================================
#  flood_pso — 御坊全域 walkable Anvil ワールド生成 Makefile
# ----------------------------------------------------------------------------
#  実証済レシピ(2026-06-26, 18図郭 --tiles 6x12 --anvil-world, 完走5h11m, 出力2.2GB)に
#  最新フル機能(トンネル / 橋v8 / OSM電線・駐車場 / 鉄道 / 道路curb / 連絡通路 / gap補完)を統合。
#
#  使い方:
#    make help          … ターゲット一覧
#    make world-test    … 2図郭(701+702)で **オプション検証**(数分)。本番前に必ず実行推奨。
#    make world-full    … 御坊全域18図郭の本番生成(初回 ~5h, 洪水sim ~2h含む)。
#
#  オプション上書き(必要時のみ):
#    make world-full LEGEND=--legend-layer            … 地下にLayer C可視化層(土地利用/洪水/樹木)を埋込
#    make world-full ORTHO_LAYER=seamlessphoto        … オルソが ort で欠ける場合のフォールバック
#    make world-full EVAC='--evac --evac-xml <P20.xml>' … 避難施設マーカー(※P20 xmlは現在欠落)
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

# --- OSM 由来フィーチャ(FG-GMLに無いものを補完。値行に末尾コメントを書くと空白が混入するので別行) ---
# 橋(全御坊144本full_geom, make_nbt_hd既定と同一) / トンネル(tunnel=yes, 同既定)
BRIDGES := $(OSM)/gobo_bridges_full_geom.json
TUNNELS := $(OSM)/gobo_tunnels_geom.json
# 送電線+鉄塔 / 駐車場舗装+白線 (どちらも明示必須=既定空文字)
POWER   := $(OSM)/gobo_power_geom.json
PARKING := $(OSM)/gobo_parking_geom.json

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

# --- 全機能共通フラグ(本番/検証で共有) ---
#   methods gt = 地形ベース / use-fgd = FGD建物・道路 / road-curb は既定ON(連絡通路1F抜きを含む)
#   trees sparse = 間引き個別樹木 / scale 1.5 = 0.667m/block
COMMON := --K 16 --seed 0 --preset gobo_walk_1km --methods gt --scale 1.5 \
          --use-fgd --fgd-bld "$(FGD_BLD)" --fgd-rdedg "$(FGD_RDEDG)" \
          --fgd-wa "$(FGD_WA)" --fgd-rail "$(FGD_RAIL)" \
          --surface-ortho --ortho-layer $(ORTHO_LAYER) \
          --trees --tree-mode sparse \
          --bridges-json "$(BRIDGES)" --tunnels-json "$(TUNNELS)" \
          --power-json "$(POWER)" --parking-json "$(PARKING)" \
          --world-base-y $(WORLD_BASE_Y) \
          --fill-gap-osm $(LEGEND) $(EVAC)

.PHONY: help world-full world-test
.DEFAULT_GOAL := help

help:
	@echo "flood_pso 御坊全域ワールド生成:"
	@echo "  make world-test   … 2図郭(701+702)でオプション検証(数分, 本番前推奨)"
	@echo "  make world-full   … 御坊全域18図郭 本番生成(初回~5h, 出力~2.2GB to results/anvil/gobo_zeniki_v2)"
	@echo "  上書き例: make world-full LEGEND=--legend-layer ORTHO_LAYER=seamlessphoto"

# 本番: 御坊全域18図郭 → 1 Anvil world(--tiles 6x12=72タイルを実座標mergeで密着)。litematicは巨大化するため抑止。
#   洪水simは ~2h。--reuse-inundation で2回目以降(オプション微調整)は sim をスキップして再利用。
world-full:
	@mkdir -p data_cache/inund_zeniki results/anvil
	$(PY) src/make_nbt_hd.py $(COMMON) \
	  --wakayama-grd "$(GRD18)" --wakayama-org "$(ORG18)" \
	  --tiles 6x12 --anvil-world results/anvil/gobo_zeniki_v2 --no-litematic \
	  --reuse-inundation data_cache/inund_zeniki --tag-suffix _zeniki

# 検証: 北西2図郭(701+702)を --tiles 2x2 で速く生成し、全フィーチャのオプションが通るか確認。
world-test:
	@mkdir -p data_cache/inund_zeniki_test results/anvil
	$(PY) src/make_nbt_hd.py $(COMMON) \
	  --wakayama-grd "$(LID)/06RC701_grd.txt,$(LID)/06RC702_grd.txt" \
	  --wakayama-org "$(LID)/06RC701_org.txt,$(LID)/06RC702_org.txt" \
	  --tiles 2x2 --anvil-world results/anvil/gobo_test_2fig --no-litematic \
	  --reuse-inundation data_cache/inund_zeniki_test --tag-suffix _test2
