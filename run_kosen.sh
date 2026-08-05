#!/usr/bin/env bash
# 和歌山高専 現況キャンパス(約400m角)ワールド生成の再現スクリプト。
#   usage: ./run_kosen.sh [SCALE]   (既定 2.0 = 0.5m/block。1.5=0.667m/block が旧ライブ)
# 地形/建物高=和歌山LiDAR(grd=DEM, org=DSM veg class3除外), 建物footprint=FGD(503561/503551),
# 旧体育館3棟を除去+総合体育館を跡地(元テニスコート)へ追加(campus_*.geojson), ortho=ort, 樹木=sparse。
set -euo pipefail
cd "$(dirname "$0")"
SCALE="${1:-2.0}"
LID=data_cache/wakayama_lidar
GRD="$LID/06RC904_grd.txt,$LID/06RC913_grd.txt,$LID/06SC002_grd.txt,$LID/06SC011_grd.txt"
ORG="$LID/06RC904_org.txt,$LID/06RC913_org.txt,$LID/06SC002_org.txt,$LID/06SC011_org.txt"
FG=../kennkyuu20260114/地形データ
D61="$FG/FG-GML-503561-ALL-20251001"; D51="$FG/FG-GML-503551-ALL-20260101"
FGD_BLD="$D61/FG-GML-503561-BldA-20251001-0001.xml,$D51/FG-GML-503551-BldA-20260101-0001.xml"
FGD_RDEDG="$D61/FG-GML-503561-RdEdg-20251001-0001.xml,$D51/FG-GML-503551-RdEdg-20260101-0001.xml"
FGD_WA="$D61/FG-GML-503561-WA-20251001-0001.xml,$D61/FG-GML-503561-WStrA-20251001-0001.xml,$D51/FG-GML-503551-WA-20260101-0001.xml,$D51/FG-GML-503551-WStrA-20260101-0001.xml"
# 現況キャンパスの建物補正 geojson（tizucra-walk リポジトリ側にある）。
# 別マシンでは配置が違うので CAMPUS=... で上書きできるようにしてある。
CAMPUS="${CAMPUS:-../tizucra-walk/tools/campus}"
if [ ! -d "$CAMPUS" ]; then
  echo "campus geojson が見つかりません: $CAMPUS" >&2
  echo "  tizucra-walk を隣に clone するか CAMPUS=<path> を渡してください" >&2
  exit 1
fi

PY=".venv/bin/python"; export PYTHONUNBUFFERED=1
"$PY" src/make_nbt_hd.py --K 16 --seed 0 --preset kosen_campus --methods gt --scale "$SCALE" \
  --wakayama-grd "$GRD" --wakayama-org "$ORG" \
  --center-lat 33.8332 --center-lon 135.1774 --terrain-skirt 16 \
  --use-fgd --fgd-bld "$FGD_BLD" --fgd-rdedg "$FGD_RDEDG" --fgd-wa "$FGD_WA" \
  --remove-bld-geojson "$CAMPUS/campus_remove.geojson" \
  --add-bld-geojson "$CAMPUS/campus_add.geojson" \
  --surface-ortho --ortho-layer ort --trees --tree-mode sparse \
  --world-base-y -50 --anvil-world results/anvil/kosen_campus --no-litematic \
  --anvil-level-template results/anvil/gobo_zeniki_v2/level.dat --tag-suffix _kosen
