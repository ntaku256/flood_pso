#!/bin/bash
# 全域 v2 更新: 10図郭 × 4分割、最新フル機能（樹木/水域/海岸/建物/道路/避難所/橋/橋下water）
cd /home/moriken/web-app/flood_pso
source .venv/bin/activate 2>/dev/null
L=data_cache/wakayama_lidar; O=data_cache/osm
D551="../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101"
D561="../kennkyuu20260114/地形データ/FG-GML-503561-ALL-20251001"
B="$D551/FG-GML-503551-BldA-20260101-0001.xml,$D561/FG-GML-503561-BldA-20251001-0001.xml"
R="$D551/FG-GML-503551-RdEdg-20260101-0001.xml,$D561/FG-GML-503561-RdEdg-20251001-0001.xml"
WA="$D551/FG-GML-503551-WA-20260101-0001.xml,$D551/FG-GML-503551-WStrA-20260101-0001.xml,$D561/FG-GML-503561-WA-20251001-0001.xml,$D561/FG-GML-503561-WStrA-20251001-0001.xml"
for code in 703 704 801 802 804 902 904 913 002 011; do
  if [[ $code == 002 || $code == 011 ]]; then gc="06SC${code}"; else gc="06RC${code}"; fi
  grd="$L/${gc}_grd.txt"
  if [[ $code == 802 ]]; then br="$O/gobo_bridges_geom.json"; else br="$O/gobo_bridges_${code}_geom.json"; fi
  echo "########## 図郭 $code ##########"
  if [[ $code == 002 || $code == 011 ]]; then
    PYTHONUNBUFFERED=1 python3 src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt \
      --use-fgd --fgd-bld "$B" --fgd-rdedg "$R" --fgd-wa "$WA" \
      --wakayama-grd "$grd" --surface-ortho --trees --tree-mode sparse --evac \
      --bridges-json "$br" --scale 1.5 --tiles 4x1 --tag-suffix "$code" 2>&1 \
      | grep -E "tile _c|Structure size|litematic\] saved|All done|Killed|Traceback"
  else
    PYTHONUNBUFFERED=1 python3 src/make_nbt_hd.py --K 16 --seed 0 --preset gobo_walk_1km --methods gt \
      --use-fgd --wakayama-grd "$grd" --surface-ortho --trees --tree-mode sparse --evac \
      --bridges-json "$br" --scale 1.5 --tiles 4x1 --tag-suffix "$code" 2>&1 \
      | grep -E "tile _c|Structure size|litematic\] saved|All done|Killed|Traceback"
  fi
done
echo "ALL FIGURES DONE"
