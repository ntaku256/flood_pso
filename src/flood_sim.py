"""
flood_sim.py
画像処理ベースの洪水シミュレーション。

【アルゴリズム】
バスタブモデル + 連結成分ラベリング (Connected-Component Flood Fill)

1. DEM をグレースケール画像として扱い、各セルの標高を画素値に対応させる
2. 水面標高 (water_level) 以下の全セルを「浸水候補」とする 2値画像を生成
3. scipy.ndimage.label (= 4連結または8連結の画像処理ラベリング) で
   連結成分を抽出し、水源と接続している成分のみを洪水域とする
4. DEM の不確かさ / 地表粗度を考慮する追加パラメータとして
   Gaussian blur によるDEM平滑化 (sigma) を適用可能にする

最適化パラメータ (PSO で探索):
  x[0]: water_level  [m]    : 洪水水面標高。範囲 [0, 15]
  x[1]: sigma        [cells]: DEM 平滑化度。範囲 [0, 3]
                              (大きいほど緩やかな勾配→水が障壁を越えやすい)
"""

import numpy as np
from scipy.ndimage import label as nd_label, gaussian_filter


# ─────────────────────────────────────────────────────────────
# 水源マスク
# ─────────────────────────────────────────────────────────────

def make_river_source(dem: np.ndarray, lat_max: float, res_lat: float,
                      lon_min: float, res_lon: float,
                      river_bbox: dict,
                      elev_max: float = 5.0) -> np.ndarray:
    """
    日高川の河道に対応するセルを水源マスクとして返す。
    river_bbox: {'lat_min','lat_max','lon_min','lon_max'}
    elev_max  : 河道として認める最大標高 [m]
    """
    H, W = dem.shape
    rows = np.arange(H)
    cols = np.arange(W)
    lats = lat_max - rows * res_lat
    lons = lon_min + cols * res_lon

    lat_mask = (lats >= river_bbox["lat_min"]) & (lats <= river_bbox["lat_max"])
    lon_mask = (lons >= river_bbox["lon_min"]) & (lons <= river_bbox["lon_max"])

    source = np.zeros_like(dem, dtype=bool)
    source[np.ix_(lat_mask, lon_mask)] = True
    source = source & ~np.isnan(dem) & (dem <= elev_max)
    return source


# ─────────────────────────────────────────────────────────────
# 洪水シミュレーション（画像処理ベース）
# ─────────────────────────────────────────────────────────────

def simulate_flood(dem: np.ndarray, source_mask: np.ndarray,
                   water_level: float, sigma: float = 0.0,
                   connectivity: int = 2) -> np.ndarray:
    """
    バスタブ+連結成分ラベリングによる洪水シミュレーション。

    Parameters
    ----------
    dem          : 標高配列 [m], NaN=海/NoData
    source_mask  : 水源セルのbool マスク
    water_level  : 洪水水面標高 [m]
    sigma        : DEMに適用するGaussian blur の標準偏差 [cells]
                   (地表粗度・DEM誤差のモデリング)
    connectivity : 1=4連結, 2=8連結

    Returns
    -------
    inundation   : 浸水深マップ [m] (非浸水セルは 0.0)
    """
    # NaN を高標高で穴埋め（境界処理）
    land = np.where(np.isnan(dem), 9999.0, dem).astype(np.float64)

    # Gaussian blur で DEM を平滑化（sigma=0 は恒等変換）
    if sigma > 0:
        land_smooth = gaussian_filter(land, sigma=sigma)
    else:
        land_smooth = land

    # 浸水候補マスク: 平滑化DEM が water_level 以下
    candidate = land_smooth < water_level

    # 8連結 (structure=[[1,1,1],[1,1,1],[1,1,1]]) または 4連結 で連結成分ラベリング
    struct = np.ones((3, 3), dtype=int) if connectivity == 2 else None
    labeled, _ = nd_label(candidate, structure=struct)

    # 水源と接続している連結成分のラベルを取得
    source_valid = source_mask & candidate
    if not np.any(source_valid):
        # 水源が浸水候補に含まれない → 全セル非浸水
        return np.zeros_like(dem, dtype=np.float32)

    flood_labels = set(labeled[source_valid].tolist())
    flood_labels.discard(0)

    # 洪水域マスク
    flood_mask = np.isin(labeled, list(flood_labels))

    # 浸水深 = water_level - 元のDEM (平滑化前)
    inundation = np.where(flood_mask, water_level - land, 0.0)
    inundation = np.maximum(inundation, 0.0)

    return inundation.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 参照マスク・評価指標
# ─────────────────────────────────────────────────────────────

def make_reference_mask(dem: np.ndarray, elev_threshold: float) -> np.ndarray:
    """
    標高閾値ベースの参照浸水域マスク。
    実際の洪水ハザードマップ画像が使える場合はそちらに差し替え可能。
    """
    return (~np.isnan(dem)) & (dem <= elev_threshold)


def iou_loss(sim_inundation: np.ndarray, ref_mask: np.ndarray,
             sim_threshold: float = 0.05) -> float:
    """
    シミュレーション結果と参照マスクの 1 - IoU を返す（PSO は最小化）。
    """
    sim_mask = sim_inundation > sim_threshold
    intersection = np.sum(sim_mask & ref_mask)
    union = np.sum(sim_mask | ref_mask)
    if union == 0:
        return 1.0
    return 1.0 - intersection / union


# ─────────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = 200, 300
    y, x = np.meshgrid(np.linspace(0, 10, H), np.linspace(0, 15, W), indexing="ij")
    dem_test = y + 0.5 * x + np.random.rand(H, W) * 0.5
    dem_test[80:120, 50:80] = -1.0   # 疑似河道

    source = np.zeros((H, W), dtype=bool)
    source[80:120, 50:80] = True

    inundation = simulate_flood(dem_test, source, water_level=4.0, sigma=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(dem_test, cmap="terrain", origin="upper")
    axes[0].set_title("DEM (synthetic)")
    axes[1].imshow(inundation, cmap="Blues", origin="upper")
    axes[1].set_title("Flood inundation (bathtub + label)")
    plt.tight_layout()
    plt.savefig("test_flood_sim.png", dpi=100)
    print("Saved test_flood_sim.png")
    print(f"Flooded cells: {np.sum(inundation > 0.05)}")
