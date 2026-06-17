"""
hazard_gt.py
国土地理院ハザードマップポータルの「洪水浸水想定区域（想定最大規模）浸水深」
ラスタタイル（XYZ / Web Mercator, EPSG:3857）を取得し、
DEM 格子（緯度経度）上の正解浸水深マップ（GT）に変換する。

実データ校正（Phase1）のターゲット。合成 GT（make_synthetic_ground_truth）の
差し替え先として benchmark.py から利用する。

タイル:
  統合（国+都道府県）: https://disaportaldata.gsi.go.jp/raster/01_flood_l2_shinsuishin_data/{z}/{x}/{y}.png
  都道府県管理（県コード付）: .../01_flood_l2_shinsuishin_pref_data/{pref}/{z}/{x}/{y}.png
  ※ GSI タイルは「データ無し」を HTTP 404 で返す（=非浸水/区域外）。

色 → 水深ランク（標準 MLIT 凡例。ランク中央値で連続化, doc/12 §4.2）:
  (247,245,169) 0–0.5m   -> 0.25
  (255,216,192) 0.5–3m   -> 1.75
  (255,183,183) 3–5m     -> 4.0
  (255,145,145) 5–10m    -> 7.5
  (242,133,201) 10–20m   -> 15.0
  (220,122,220) 20m–     -> 25.0
（凡例 RGB のうち先頭4色は御坊範囲のタイルで実測確認済み。残り2色は標準凡例値。）
"""

import math
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage import map_coordinates

TILE_URL = "https://disaportaldata.gsi.go.jp/raster/{layer}/{z}/{x}/{y}.png"
DEFAULT_LAYER = "01_flood_l2_shinsuishin_data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "disaportal"

# 凡例 RGB -> 代表水深 [m]（ランク中央値）
LEGEND = [
    ((247, 245, 169), 0.25),
    ((255, 216, 192), 1.75),
    ((255, 183, 183), 4.0),
    ((255, 145, 145), 7.5),
    ((242, 133, 201), 15.0),
    ((220, 122, 220), 25.0),
]
_LEG_RGB = np.array([c for c, _ in LEGEND], dtype=np.float64)
_LEG_D = np.array([d for _, d in LEGEND], dtype=np.float64)


def _fetch_tile(layer: str, z: int, x: int, y: int, retries: int = 2):
    """タイルPNGをキャッシュ付きで取得。404(データ無し)は None。"""
    cache = CACHE_DIR / layer.replace("/", "_") / str(z) / str(x) / f"{y}.png"
    if cache.exists():
        return cache if cache.stat().st_size > 0 else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = TILE_URL.format(layer=layer, z=z, x=x, y=y)
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flood_pso-research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            cache.write_bytes(data)
            return cache
        except urllib.error.HTTPError as e:
            if e.code == 404:
                cache.write_bytes(b"")  # 空マーカ（再取得回避）
                return None
            if k == retries:
                raise
            time.sleep(0.5)
        except Exception:
            if k == retries:
                raise
            time.sleep(0.5)
    return None


def _decode_tile(path: Path, tol: float = 45.0) -> np.ndarray:
    """タイルPNG -> (256,256) 代表水深[m]。非浸水/透明=0.0。"""
    a = mpimg.imread(str(path))
    a = a.astype(np.float64)
    if a.max() <= 1.0:
        a = a * 255.0
    H, W = a.shape[:2]
    rgb = a[:, :, :3].reshape(-1, 3)
    alpha = a[:, :, 3].reshape(-1) if a.shape[2] >= 4 else np.full(rgb.shape[0], 255.0)
    # 各画素を凡例色に最近傍マッチ
    d2 = ((rgb[:, None, :] - _LEG_RGB[None, :, :]) ** 2).sum(axis=2)  # (N, 6)
    idx = np.argmin(d2, axis=1)
    mind = np.sqrt(d2[np.arange(len(idx)), idx])
    depth = _LEG_D[idx]
    valid = (alpha > 30) & (mind < tol)
    return np.where(valid, depth, 0.0).reshape(H, W)


def load_hazard_gt(dem_info: dict, zoom: int = 15, layer: str = DEFAULT_LAYER,
                   verbose: bool = True):
    """
    DEM 格子に整合した正解浸水深マップ GT を返す。

    Parameters
    ----------
    dem_info : dict  (dem, lat_min/max, lon_min/max, res_lat/lon)  ※dem_parser/downsample 出力
    zoom     : XYZ タイルのズーム（z15≈4m/px。25m DEM 格子に対し十分）

    Returns
    -------
    gt_depth : (H, W) float32  正解浸水深[m]（非浸水=0）
    gt_mask  : (H, W) bool      浸水域マスク
    """
    dem = dem_info["dem"]
    H, W = dem.shape
    lat_max = dem_info["lat_max"]
    lon_min = dem_info["lon_min"]
    res_lat = dem_info["res_lat"]
    res_lon = dem_info["res_lon"]

    # DEM 各セル中心の lat/lon（row0=北端）
    lats = lat_max - np.arange(H) * res_lat
    lons = lon_min + np.arange(W) * res_lon
    LON, LAT = np.meshgrid(lons, lats)

    # lat/lon -> グローバル Web Mercator ピクセル（zoom z）
    n = 2 ** zoom
    PX = (LON + 180.0) / 360.0 * n * 256.0
    PY = (1.0 - np.arcsinh(np.tan(np.radians(LAT))) / np.pi) / 2.0 * n * 256.0

    tx0, tx1 = int(np.floor(PX.min() / 256)), int(np.floor(PX.max() / 256))
    ty0, ty1 = int(np.floor(PY.min() / 256)), int(np.floor(PY.max() / 256))
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    if verbose:
        print(f"[hazard_gt] layer={layer} zoom={zoom}  x[{tx0}..{tx1}] y[{ty0}..{ty1}] = {n_tiles} tiles")

    # mercator ピクセル空間のモザイク（非浸水=0）
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), dtype=np.float32)
    got = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            p = _fetch_tile(layer, zoom, tx, ty)
            if p is None:
                continue
            d = _decode_tile(p)
            r0 = (ty - ty0) * 256
            c0 = (tx - tx0) * 256
            mosaic[r0:r0 + 256, c0:c0 + 256] = d
            got += 1
    if verbose:
        print(f"[hazard_gt] fetched {got}/{n_tiles} non-empty tiles  "
              f"(inundated px in mosaic: {int(np.sum(mosaic > 0))})")

    # モザイク（mercator px）-> DEM 格子へ最近傍サンプリング
    local_y = (PY - ty0 * 256).ravel()
    local_x = (PX - tx0 * 256).ravel()
    sampled = map_coordinates(mosaic, [local_y, local_x], order=0, mode="constant", cval=0.0)
    gt_depth = sampled.reshape(H, W).astype(np.float32)
    gt_mask = gt_depth > 0
    return gt_depth, gt_mask


# ─────────────────────────────────────────────────────────────
# 単体動作確認 + 可視化（ジオレファレンス検証）
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from dem_parser import mosaic_tiles, downsample

    REPO = Path(__file__).resolve().parent.parent
    DEM_DIR = os.environ.get(
        "FLOOD_PSO_DEM_DIR",
        str(REPO.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-DEM5A-20250620"),
    )
    ZOOM = int(os.environ.get("FLOOD_PSO_HAZARD_ZOOM", "15"))

    print("[setup] DEM 読み込み (downsample x5 = 25m)...")
    dem_info = downsample(mosaic_tiles(DEM_DIR), 5)
    dem = dem_info["dem"]

    gt_depth, gt_mask = load_hazard_gt(dem_info, zoom=ZOOM)

    n_in = int(gt_mask.sum())
    print(f"\nGT 浸水セル: {n_in} / {gt_mask.size}  ({100*n_in/gt_mask.size:.1f}%)")
    if n_in:
        print(f"GT 浸水深 [m]: min={gt_depth[gt_mask].min():.2f} "
              f"mean={gt_depth[gt_mask].mean():.2f} max={gt_depth[gt_mask].max():.2f}")
        ranks, cnts = np.unique(gt_depth[gt_mask], return_counts=True)
        for r, c in zip(ranks.tolist(), cnts.tolist()):
            print(f"   depth={r:>5.2f} m : {c} cells")

    out = REPO / "results" / "hazard_gt_check.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    demv = np.where(np.isnan(dem), np.nan, dem)
    axes[0].imshow(demv, cmap="terrain", origin="upper")
    axes[0].imshow(np.where(gt_mask, gt_depth, np.nan), cmap="Blues", alpha=0.7,
                   origin="upper", vmin=0, vmax=20)
    axes[0].set_title(f"DEM + GT inundation overlay (z{ZOOM})")
    im = axes[1].imshow(np.where(gt_mask, gt_depth, np.nan), cmap="viridis",
                        origin="upper", vmin=0, vmax=20)
    axes[1].set_title(f"GT depth [m]  ({n_in} cells)")
    plt.colorbar(im, ax=axes[1], fraction=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")
