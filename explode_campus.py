#!/usr/bin/env python3
# 学科棟ブロックの各建物を「間隔をあけて」平面に整列配置した専用ワールドを生成する（パンフレット風）。
#   入力 : results/nbt/hd/..._kosen.nbt（生成済み 0.5m/block 構造・palette=global順）
#   手法 : FGD フットプリントで各建物の bbox を得 → 生成ワールドから建物+周辺を切り出し
#          → 地面を共通 base Y に揃えてグリッド整列 → anvil 出力
#   出力 : results/anvil/kosen_explode
import sys, numpy as np, nbtlib
sys.path.insert(0, "src")
from anvil_export import write_anvil_world, DATA_VERSION
from block_palette import PALETTE_KEYS

STRUCT = "results/nbt/hd/gobo_hd_K16_seed0_kosen_campus_gt_fgd_ortho__kosen.nbt"
OUT = "results/anvil/kosen_explode"
LEVEL_TMPL = "results/anvil/gobo_zeniki_v2/level.dat"
FGD_BLD = ("../kennkyuu20260114/地形データ/FG-GML-503561-ALL-20251001/FG-GML-503561-BldA-20251001-0001.xml,"
           "../kennkyuu20260114/地形データ/FG-GML-503551-ALL-20260101/FG-GML-503551-BldA-20260101-0001.xml")
# 新 georef（patch_bbox / grid798×800）
LAT0, LAT1 = 33.8314015, 33.8349947      # south, north
LON0, LON1 = 135.1752433, 135.1795583    # west, east
NX, NZ = 798, 800
RESLON = (LON1 - LON0) / NX
RESLAT = (LAT1 - LAT0) / NZ
BASE_Y = -50                             # world_base_y
# 学科棟ブロック（floor マップと同じ範囲）
AX0, AX1, AZ0, AZ1 = 325, 566, 110, 400
MARGIN = 3        # 建物 bbox の周囲に足す地面リング [blocks]
GAP = 14          # タイル間の間隔 [blocks]
COMMON_BASE = 6   # 全建物を揃える共通地表 Y（world 座標）
BASESLAB = 4      # 各タイル下に敷く土台の厚み [blocks]

def ll2xz(lat, lon):
    return (lon - LON0) / RESLON, (LAT0 + (LAT1 - LAT0)) and (LAT1 - lat) / RESLAT
def ll2x(lon): return (lon - LON0) / RESLON
def ll2z(lat): return (LAT1 - lat) / RESLAT

print("[1/5] read structure NBT ...", flush=True)
f = nbtlib.load(STRUCT)
NY = int(f['size'][1])
blocks = f['blocks']
# 高速 dense 構築: pos/state を一括抽出
n = len(blocks)
pos = np.empty((n, 3), np.int32); st = np.empty(n, np.uint16)
for i, b in enumerate(blocks):
    p = b['pos']; pos[i, 0] = int(p[0]); pos[i, 1] = int(p[1]); pos[i, 2] = int(p[2]); st[i] = int(b['state'])
dense = np.zeros((NY, NZ, NX), np.uint16)          # (Y,Z,X) air=0
dense[pos[:, 1], pos[:, 2], pos[:, 0]] = st
print(f"    dense {dense.shape}  nonair={n:,}", flush=True)

# palette 分類（樹木=葉/幹, 空気）
names = [str(p['Name']) for p in f['palette']]
AIR = 0
TREE = np.array([i for i, nm in enumerate(names) if ('leaves' in nm or nm.endswith('_log')
                 or 'log' in nm or 'leaf' in nm or 'bamboo' in nm)], np.uint16)
is_tree = np.zeros(len(names), bool);  is_tree[TREE] = True
print("    tree palette idx:", list(TREE))

print("[2/5] load FGD academic buildings ...", flush=True)
sys.path.insert(0, "src")
from fgd_vector import load_fgd_buildings_roads
fgd = load_fgd_buildings_roads(FGD_BLD, None, lat_min=LAT0, lat_max=LAT1,
                               lon_min=LON0, lon_max=LON1, verbose=False)
blds = fgd["buildings"]
tiles = []
for b in blds:
    co = np.asarray(b["coords"], float)              # [[lat,lon],...]
    xs = np.array([ll2x(lo) for _, lo in co]); zs = np.array([ll2z(la) for la, _ in co])
    cx, cz = xs.mean(), zs.mean()
    if not (AX0 <= cx <= AX1 and AZ0 <= cz <= AZ1):   # 学科棟ブロック内のみ
        continue
    x0 = max(0, int(xs.min()) - MARGIN); x1 = min(NX, int(xs.max()) + MARGIN + 1)
    z0 = max(0, int(zs.min()) - MARGIN); z1 = min(NZ, int(zs.max()) + MARGIN + 1)
    area = (x1 - x0) * (z1 - z0)
    if x1 - x0 < 6 or z1 - z0 < 6 or area < 150:  # 微小構造(小屋/塀/断片)は除外
        continue
    tiles.append(dict(x0=x0, x1=x1, z0=z0, z1=z1, cx=cx, cz=cz,
                      h=float(b["tags"].get("height_m", 6.0))))
print(f"    academic buildings (area>=150): {len(tiles)}", flush=True)

# 重複 bbox（近接建物が同一 crop に入る）を大きい順に貪欲マージ回避 → そのまま個別 crop
# 各 crop の地表 Y（周辺リング=地面の代表値）と屋根 Y を求める
print("[3/5] crop + measure each building ...", flush=True)
crops = []
for t in tiles:
    sub = dense[:, t['z0']:t['z1'], t['x0']:t['x1']]          # (Y,dz,dx)
    solid = sub != AIR
    if not solid.any():
        continue
    # 各列の最上 solid
    ys = np.where(solid.any(axis=0),
                  (NY - 1 - solid[::-1].argmax(axis=0)), -1)    # (dz,dx) top solid Y or -1
    valid = ys[ys >= 0]
    ground_y = int(np.percentile(valid, 25))                   # 地面代表（下側クラスタ）
    roof_y = int(valid.max())
    ylo = max(0, ground_y - BASESLAB)
    yhi = min(NY - 1, roof_y + 2)
    crops.append(dict(t=t, sub=sub[ylo:yhi + 1].copy(), ylo=ylo, ground_y=ground_y,
                      dy=yhi - ylo + 1, dz=t['z1'] - t['z0'], dx=t['x1'] - t['x0']))
print(f"    crops: {len(crops)}", flush=True)

# [4/5] シェルフパッキング: 高さ(dz)降順で行に詰め、行幅が目標を超えたら折返す
crops.sort(key=lambda c: -c['dz'])
tot_area = sum((c['dx'] + GAP) * (c['dz'] + GAP) for c in crops)
TARGET_W = int((tot_area * 1.5) ** 0.5)               # 全体を横:縦≈3:2 に
TARGET_W = max(TARGET_W, max(c['dx'] for c in crops) + 2 * GAP)
shelves = []; cur = []; cur_w = GAP
for c in crops:
    if cur and cur_w + c['dx'] + GAP > TARGET_W:
        shelves.append(cur); cur = []; cur_w = GAP
    cur.append(c); cur_w += c['dx'] + GAP
if cur:
    shelves.append(cur)
shelf_d = [max(c['dz'] for c in sh) for sh in shelves]
Wx = max(sum(c['dx'] + GAP for c in sh) + GAP for sh in shelves)
Wz = sum(shelf_d) + GAP * (len(shelves) + 1)
Hy = max(c['dy'] for c in crops) + COMMON_BASE + 8
print(f"[5/5] assemble  X={Wx} Z={Wz} Y={Hy}  shelves={len(shelves)}  tiles={len(crops)}", flush=True)
out = np.zeros((Hy, Wz, Wx), np.uint16)
GRASS = names.index('minecraft:grass_block') if 'minecraft:grass_block' in names else 4
DIRT = names.index('minecraft:dirt') if 'minecraft:dirt' in names else 5
positions = []                                          # (name?, ox, oz, dx, dz) ラベル用
zc = GAP
for si, sh in enumerate(shelves):
    xc = GAP
    for c in sh:
        ox, oz = xc, zc
        sub = c['sub']
        g_in = c['ground_y'] - c['ylo']
        oy = COMMON_BASE - g_in
        out[0:COMMON_BASE + 1, oz:oz + c['dz'], ox:ox + c['dx']] = DIRT
        out[COMMON_BASE, oz:oz + c['dz'], ox:ox + c['dx']] = GRASS
        dy, dz, dx = sub.shape
        y0 = max(0, oy); ys0 = y0 - oy
        y1 = min(Hy, oy + dy); ys1 = ys0 + (y1 - y0)
        seg = sub[ys0:ys1]
        tgt = out[y0:y1, oz:oz + dz, ox:ox + dx]
        m = seg != AIR
        tgt[m] = seg[m]
        positions.append(dict(ox=ox, oz=oz, dx=dx, dz=dz,
                              cx=c['t']['cx'], cz=c['t']['cz']))
        xc += c['dx'] + GAP
    zc += shelf_d[si] + GAP
import json as _json
_json.dump(positions, open("results/anvil/kosen_explode_positions.json", "w"))

# air 以外の総数
print("    nonair out:", int((out != AIR).sum()), flush=True)
write_anvil_world(out, [Wx, Hy, Wz], OUT, y_offset=BASE_Y, level_name="kosen_explode",
                  level_template=LEVEL_TMPL, verbose=True)
# spawn ~ 中央上空
print(f"DONE explode → {OUT}  size X{Wx} Y{Hy} Z{Wz}  spawn~({Wx//2},{COMMON_BASE+BASE_Y+20},{Wz//2})")
