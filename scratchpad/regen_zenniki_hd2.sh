#!/bin/bash
# Full 47km² regen at scale 1.5 with ALL fixes + 503551 southern mesh + reclaim(plant ground).
set +e
R=$HOME/web-app/flood_pso; cd "$R" || exit 1
export PYTHONUTF8=1
export TREE_STEP=5
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
GD=$R/../kennkyuu20260114/地形データ
D561="$GD/FG-GML-503561-ALL-20251001"
D551="$GD/FG-GML-503551-ALL-20260701"
BLD="$D551/FG-GML-503551-BldA-20260701-0001.xml,$D561/FG-GML-503561-BldA-20251001-0001.xml"
RDEDG="$D551/FG-GML-503551-RdEdg-20260701-0001.xml,$D561/FG-GML-503561-RdEdg-20251001-0001.xml"
CST="$D551/FG-GML-503551-Cstline-20260701-0001.xml,$D561/FG-GML-503561-Cstline-20251001-0001.xml"
WSTRL="$D551/FG-GML-503551-WStrL-20260701-0001.xml,$D561/FG-GML-503561-WStrL-20251001-0001.xml"
RAIL="$D561/FG-GML-503561-RailCL-20251001-0001.xml"
LID=data_cache/wakayama_lidar
ORG="$LID/06RC703_org.txt,$LID/06RC713_org.txt,$LID/06RC801_org.txt,$LID/06RC802_org.txt,$LID/06RC904_org.txt"
rm -rf results/anvil/gobo_zenniki_hd results/nbt/hd/*__zennikihd*.nbt
.venv/bin/python src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt --scale 1.5 \
  --tiles 3x7 \
  --use-fgd --use-osm --fgd-bld "$BLD" --fgd-rdedg "$RDEDG" \
  --surface-ortho --ortho-layer ort --trees --tree-mode sparse --trees-esa \
  --world-base-y -50 --nbt-compresslevel 6 --fill-gap-osm --dem-gsi-tiles \
  --wakayama-org "$ORG" \
  --center-lat 33.86833 --center-lon 135.17063 --width 4503 --depth 10460 \
  --no-flood \
  --waterways-fetch --power-poles --signals --bridges-json "" --bridges-fetch --tunnels-fetch \
  --railway --fgd-rail "$RAIL" --barriers --busstops \
  --fgd-wstrl "$WSTRL" --fgd-cstline "$CST" --manmade --landmarks \
  --anvil-world results/anvil/gobo_zenniki_hd --no-litematic --tag-suffix _zennikihd 2>&1 \
  | grep -iE 'blk-offset|tile _r|canopy|樹木セル|trees-esa|外洋void|reclaim|工場発電|osm-bld|power-pole|shadow-lift|bridge|anvil. world|Saved|Structure size|Error|Traceback' \
  | grep -viE '404|warn' | tee results/zenniki_hd2_build.log
echo "ZENNIKI_HD2_DONE rc=$?"
