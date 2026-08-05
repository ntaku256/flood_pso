#!/usr/bin/env python3
"""新エリア LiDAR 図郭(grd.txt)の lat/lon bbox を抽出し、2次メッシュ別に図郭を割り当てる。
系VI(EPSG:6674)の easting/northing(各行 class,E,N,z)を awk で min/max → pyproj で WGS84。"""
import glob, subprocess
from pyproj import Transformer

tr = Transformer.from_crs("EPSG:6674", "EPSG:4326", always_xy=True)
AWK = (r'NR==1{a=b=$2;c=d=$3} {if($2<a)a=$2; if($2>b)b=$2; '
       r'if($3<c)c=$3; if($3>d)d=$3} END{print a,b,c,d}')
rows = []
for f in sorted(glob.glob('data_cache/wakayama_lidar/06R[BC][012]*_grd.txt')):
    e0, e1, n0, n1 = map(float, subprocess.check_output(["awk", "-F,", AWK, f]).split())
    lons, lats = tr.transform([e0, e1, e0, e1], [n0, n0, n1, n1])
    name = f.split('/')[-1].replace('_grd.txt', '')
    rows.append((name, min(lats), max(lats), min(lons), max(lons)))
    print(f"{name}: lat[{min(lats):.4f},{max(lats):.4f}] lon[{min(lons):.4f},{max(lons):.4f}]")

# 2次メッシュ範囲
MESH = {
    "513500": (34.0000, 34.0833, 135.0000, 135.1250),
    "513501": (34.0000, 34.0833, 135.1250, 135.2500),
    "513502": (34.0000, 34.0833, 135.2500, 135.3750),
    "513510": (34.0833, 34.1667, 135.0000, 135.1250),
    "513511": (34.0833, 34.1667, 135.1250, 135.2500),
    "513512": (34.0833, 34.1667, 135.2500, 135.3750),
}
print("\n=== メッシュ別カバー図郭(bboxが重なるもの) ===")
for m, (la0, la1, lo0, lo1) in MESH.items():
    hit = [r[0] for r in rows if r[2] >= la0 and r[1] <= la1 and r[4] >= lo0 and r[3] <= lo1]
    print(f"{m} lat[{la0},{la1}] lon[{lo0},{lo1}]: {len(hit)}図郭  {','.join(hit)}")
