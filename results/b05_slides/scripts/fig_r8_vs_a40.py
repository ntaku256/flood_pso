"""fig_r8_vs_a40.png: R8 復号 最大浸水域 (hmax>=0.3) と H25 A40-16 (国交省 津波浸水想定) の重ね図 + IoU。"""
import json, numpy as np, rasterio.features
from affine import Affine
from shapely.geometry import shape
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch
from r8common import *
d, g = load_r8(); hmax, dem = d["hmax"], d["dem"]; H, W = dem.shape
gj = json.load(open("/home/ntaku/laravel-project/tizucra-walk/data_cache/a40/A40-16_30_GML/A40-16_30.geojson"))
print("crs:", gj.get("crs")); labels = sorted({f["properties"]["A40_003"] for f in gj["features"]}); print(labels)
lat_min = g["lat_max"] - H * g["res_lat"]; lon_max = g["lon_min"] + W * g["res_lon"]
tf = Affine(g["res_lon"], 0, g["lon_min"], 0, -g["res_lat"], g["lat_max"])
def in_bbox(geom):
    rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    xs = [c[0] for ring in rings for c in ring]; ys = [c[1] for ring in rings for c in ring]
    return not (max(xs) < g["lon_min"] or min(xs) > lon_max or max(ys) < lat_min or min(ys) > g["lat_max"])
def ras(pred):
    sh = [(f["geometry"], 1) for f in gj["features"] if in_bbox(f["geometry"]) and pred(f["properties"]["A40_003"])]
    return rasterio.features.rasterize(sh, out_shape=(H, W), transform=tf, fill=0, all_touched=False, dtype="uint8").astype(bool)
valid = np.isfinite(dem)
a40_all = ras(lambda s: True) & valid
a40 = ras(lambda s: not s.startswith("0.01m")) & valid     # 主比較: A40 の 0.3 m 以上区分 (R8 側の閾値 0.3 m と揃える)
def iou(a, b): return float((a & b).sum() / (a | b).sum())
variants = {}
for thr in [0.01, 0.3, 1.0]:
    r = (hmax >= thr) & valid
    variants[f"R8 depth>={thr} vs A40 all(>=0.01m)"] = iou(r, a40_all)
    variants[f"R8 depth>={thr} vs A40 >=0.3m"] = iou(r, a40)
# README の 0.803 は動画フレーム格子 (11 m/px) で η>=0.3 (水位, DEM 差し引き前) と a40_on_frame を比べた値。再現:
fr = np.load(f"{R8}/decoded/gobo_r8_tsunami_frames.npz"); af = np.load(f"{R8}/decoded/a40_on_frame.npy")
variants["frame grid: eta>=0.3 vs a40_on_frame>0 (README 0.803 の再現)"] = iou(fr["hmax"] >= 0.3, af > 0)
shapes = []
for f in gj["features"]:
    geom = f["geometry"]
    xs = [c[0] for ring in ([geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]) for c in ring]
    ys = [c[1] for ring in ([geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]) for c in ring]
    if max(xs) < g["lon_min"] or min(xs) > lon_max or max(ys) < lat_min or min(ys) > g["lat_max"]: continue
    shapes.append((geom, 1))
print("A40 polygons in bbox:", len(shapes))
r8 = (hmax >= 0.3) & valid; a40v = a40
inter = (r8 & a40v).sum(); union = (r8 | a40v).sum()
iou = inter / union
DX, DY = cell_dims(g, H); cell_ha = DX * DY / 1e4
stats = dict(IoU=float(iou), r8_cells=int(r8.sum()), a40_cells=int(a40v.sum()), inter=int(inter), union=int(union),
             r8_ha=float(r8.sum() * cell_ha), a40_ha=float(a40v.sum() * cell_ha),
             r8_only_ha=float((r8 & ~a40v).sum() * cell_ha), a40_only_ha=float((a40v & ~r8).sum() * cell_ha),
             recall_of_a40=float(inter / a40v.sum()), precision_vs_a40=float(inter / r8.sum()),
             a40_cells_on_sea_nan=int((ras(lambda s: True) & ~valid).sum()), variants=variants)
json.dump(stats, open(f"{OUT}/fig_r8_vs_a40_stats.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(stats, indent=1))

ext = [g["lon_min"], lon_max, lat_min, g["lat_max"]]; asp = 1 / np.cos(np.radians((g["lat_max"] + lat_min) / 2))
ls = LightSource(azdeg=315, altdeg=45); hs = ls.hillshade(np.where(valid, dem, 0.0).astype(float), vert_exag=1.0, dx=DX, dy=DY)
base = np.dstack([hs * 0.55 + 0.45] * 3); base[~valid] = np.array([0.80, 0.87, 0.93])
cls = np.zeros((H, W), np.uint8); cls[r8 & a40v] = 1; cls[r8 & ~a40v] = 2; cls[a40v & ~r8] = 3
cm = ListedColormap(["#00000000", C["violet"], C["orange"], C["aqua"]])
fig, ax = plt.subplots(figsize=(10, 7.4))
ax.imshow(base, extent=ext, aspect=asp, interpolation="nearest"); ax.grid(False)
ax.imshow(np.ma.masked_where(cls == 0, cls), cmap=cm, vmin=0, vmax=3, extent=ext, aspect=asp, interpolation="nearest", alpha=0.85)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
leg = [Patch(color=C["violet"], label=f"両方 (共通)  {stats['inter']*cell_ha:,.0f} ha"),
       Patch(color=C["orange"], label=f"R8 復号のみ  {stats['r8_only_ha']:,.0f} ha"),
       Patch(color=C["aqua"], label=f"H25 A40-16 (≥0.3 m) のみ  {stats['a40_only_ha']:,.0f} ha")]
ax.legend(handles=leg, loc="lower left", frameon=True, framealpha=0.9, fontsize=10, title=f"IoU = {iou:.3f}", title_fontsize=12)
ax.set_title("最大浸水域の比較: 県 R8 (2026) 津波動画の復号 (深さ ≥0.3 m)  vs  国交省 A40-16 (H25 津波浸水想定, 浸水深 0.3 m 以上の区分)\n"
             f"5 m 格子 {H}×{W} セル, GSI DEM5A の陸域のみで評価 (陰影 = DEM)。A40 全区分 (0.01 m 以上) に対しては IoU {variants['R8 depth>=0.3 vs A40 all(>=0.01m)']:.3f}", fontsize=10.5, loc="left")
fig.savefig(f"{OUT}/fig_r8_vs_a40.png", dpi=170)
print("saved")
