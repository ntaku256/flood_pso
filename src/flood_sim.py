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
from scipy.ndimage import label as nd_label, gaussian_filter, zoom as nd_zoom


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
# 高次元版：ブロック単位の水位補正マップ Δh(x,y) をサポート
# ─────────────────────────────────────────────────────────────

def upsample_dh(dh_map: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    K×K の補正マップをバイリニア補間で target_shape (H, W) にアップサンプル。
    """
    K_y, K_x = dh_map.shape
    H, W = target_shape
    zoom_y = H / K_y
    zoom_x = W / K_x
    return nd_zoom(dh_map.astype(np.float64), zoom=(zoom_y, zoom_x), order=1, mode="nearest")


def simulate_flood_hd(dem: np.ndarray, source_mask: np.ndarray,
                      water_level_global: float,
                      dh_map: np.ndarray,
                      sigma: float = 0.0,
                      sigma_map: np.ndarray = None,
                      connectivity: int = 2) -> np.ndarray:
    """
    高次元版バスタブ+連結成分洪水シミュレーション。
    水位は water_level_global + Δh(x,y) として空間分布する。

    Parameters
    ----------
    dem                : 標高配列 [m]
    source_mask        : 水源セル
    water_level_global : 大局水位 [m]
    dh_map             : K×K のブロック単位水位補正 [m]
    sigma              : DEM 平滑化（既存と同じ）。sigma_map が None のときのみ参照。
    sigma_map          : K_s×K_s の局所平滑化 sigma マップ（None なら既存挙動）。
                         非 None の場合、5 段階 sigma の合成で場所別平滑化を近似する
                         （sigma_levels = [0, 0.5, 1.0, 2.0, 4.0]、各ピクセルで最近傍を採用）。
    """
    land = np.where(np.isnan(dem), 9999.0, dem).astype(np.float64)
    if sigma_map is not None:
        # K_s × K_s をフル解像度にアップサンプル
        sigma_full = upsample_dh(sigma_map, dem.shape)
        sigma_full = sigma_full[:dem.shape[0], :dem.shape[1]]
        if sigma_full.shape != dem.shape:
            pad_y = dem.shape[0] - sigma_full.shape[0]
            pad_x = dem.shape[1] - sigma_full.shape[1]
            sigma_full = np.pad(sigma_full, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")
        # 5 段階 sigma で gaussian_filter、各ピクセルで最近傍 lvl を選択
        sigma_levels = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)
        stack = np.stack([
            land.copy() if s == 0.0 else gaussian_filter(land, sigma=float(s))
            for s in sigma_levels
        ], axis=0)  # shape: (n_levels, H, W)
        idx = np.argmin(np.abs(sigma_full[None, :, :] - sigma_levels[:, None, None]), axis=0)
        H, W = land.shape
        land_smooth = stack[idx, np.arange(H)[:, None], np.arange(W)[None, :]]
    elif sigma > 0:
        land_smooth = gaussian_filter(land, sigma=sigma)
    else:
        land_smooth = land

    dh_full = upsample_dh(dh_map, dem.shape)
    # アップサンプル誤差で形状ズレが生じうるのでクロップ
    dh_full = dh_full[:dem.shape[0], :dem.shape[1]]
    if dh_full.shape != dem.shape:
        # 不足分はゼロパディング
        pad_y = dem.shape[0] - dh_full.shape[0]
        pad_x = dem.shape[1] - dh_full.shape[1]
        dh_full = np.pad(dh_full, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")

    water_field = water_level_global + dh_full
    candidate = land_smooth < water_field

    struct = np.ones((3, 3), dtype=int) if connectivity == 2 else None
    labeled, _ = nd_label(candidate, structure=struct)

    source_valid = source_mask & candidate
    if not np.any(source_valid):
        return np.zeros_like(dem, dtype=np.float32)

    flood_labels = set(labeled[source_valid].tolist())
    flood_labels.discard(0)

    flood_mask = np.isin(labeled, list(flood_labels))
    inundation = np.where(flood_mask, water_field - land, 0.0)
    return np.maximum(inundation, 0.0).astype(np.float32)


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


def depth_loss(sim_inundation: np.ndarray, gt_inundation: np.ndarray,
               sim_threshold: float = 0.05) -> float:
    """
    浸水深ベースの損失。マスク IoU と異なり、各ブロックの局所水位差にも感度を持つ。
    sim_mask ∪ gt_mask 上での平均絶対誤差 (MAE) を返す。
    両方とも非浸水のセルは集計しない（全領域0で割られるのを避けるため）。
    """
    sim_mask = sim_inundation > sim_threshold
    gt_mask  = gt_inundation > sim_threshold
    union = sim_mask | gt_mask
    n = int(np.sum(union))
    if n == 0:
        return 0.0
    diff = np.abs(sim_inundation - gt_inundation)
    return float(np.sum(diff[union]) / n)


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
