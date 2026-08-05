import numpy as np, math
from pathlib import Path
from scipy.ndimage import label as nd_label
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data.npz")
dem, src, gtd = d["dem"], d["src"], d["gt_depth"]
lat_min, lat_max, lon_min, lon_max, res_lat, res_lon = d["meta"]
gt = gtd > 0
H, W = dem.shape
nan = np.isnan(dem)
print(f"DEM {H}x{W}  res_lat={res_lat:.3e}deg res_lon={res_lon:.3e}deg")
m_lat = res_lat*111320.0
m_lon = res_lon*111320.0*math.cos(math.radians((lat_min+lat_max)/2))
print(f"cell size: {m_lat:.2f} m (N-S) x {m_lon:.2f} m (E-W)")
print(f"domain: {H*m_lat/1000:.2f} km x {W*m_lon/1000:.2f} km")
for K in (8,16,24,32,64,128):
    print(f"  K={K:3d}: dh block = {H/K*m_lat:7.1f} m x {W/K*m_lon:7.1f} m   (D={1+K*K})")
print()
print(f"NaN cells {nan.sum()} ({100*nan.mean():.1f}%)")
print(f"GT cells {gt.sum()} ({100*gt.mean():.2f}%)  area={gt.sum()*m_lat*m_lon/1e6:.2f} km2")
print(f"GT & NaN(dem) = {int((gt&nan).sum())}  <- permanently FN (sea/nodata)")
print(f"src cells {int(src.sum())} area={src.sum()*m_lat*m_lon/1e6:.2f} km2")
print(f"src & ~GT = {int((src&~gt).sum())}  src&GT={int((src&gt).sum())}")
v = dem[~nan]
print(f"dem range {v.min():.1f}..{v.max():.1f}")
# GT depth ranks
r, c = np.unique(gtd[gt], return_counts=True)
for rr, cc in zip(r, c):
    print(f"   rank depth={rr:6.2f} m : {cc:8d} ({100*cc/gt.sum():5.2f}%)")
# GT elevation distribution
g = dem[gt & ~nan]
print("GT dem percentiles:", np.round(np.percentile(g, [1,5,25,50,75,90,95,99,100]),2))
for th in (5,8,10,12,15,20,30):
    print(f"   GT cells with dem > {th:4.1f} m : {int((g>th).sum()):8d} ({100*(g>th).mean():5.2f}%)")
# non-GT low land
ng = dem[~gt & ~nan]
for th in (5,8,10):
    print(f"   nonGT cells with dem < {th:4.1f} m : {int((ng<th).sum()):8d}")
