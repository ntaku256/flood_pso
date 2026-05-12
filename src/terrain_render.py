"""
terrain_render.py
NBT 化のための「地形レンダリング」改善モジュール。

`Tellus` Fabric mod（Earth-scale terrain Minecraft mod）の地形生成パイプラインから
**flood_pso の用途（御坊海岸＋日高川）に効く部分だけ**を Python に移植する。

参考にした Tellus 実装：
  - `worldgen/MountainSurfaceRules.java` — 標高/斜度/凸性で地表ブロックを決定
  - `worldgen/OceanClassification.java`  — 海域判定
  - `worldgen/WaterSurfaceResolver.java` — 海岸距離・shoreline blend
  - `worldgen/TerrainAnomalyRepair.java` — 9近傍でタイル継ぎ目修復（参考、本モジュール未移植）

flood_pso の従来 `nbt_export.dem_to_blocks` の改善点：
  1. **海岸線・海の表現**：海岸からの距離で段階的に水深を増やし、海底に砂/砂利を敷く
  2. **緑斜面の階段化抑制**：cliff-aware gaussian smoothing（崖は保ったまま緩斜面だけ平滑化）
  3. **地表ブロック判定**：勾配・凸性・海岸距離で sand/gravel/stone/grass を判定
  4. **陸/海で垂直誇張を分離**：陸は v_exag_land、水中は v_exag_sea
  5. **地盤柱を深く**：従来 3 ブロック → 既定 8 ブロックで視点による底抜けを抑制
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
import nbtlib

# nbt_export 側のパレットを再利用（同じ block_id/PALETTE 定義）
from nbt_export import block_id


# ─────────────────────────────────────────────────────────────
# DEM 由来の地形特徴量
# ─────────────────────────────────────────────────────────────

def compute_slope(dem: np.ndarray, h_res_m: float) -> np.ndarray:
    """
    中央差分による勾配の大きさ [m/m] を返す。NaN は 0 とみなす。
    急斜面検出に使う（45° = slope 1.0、22.5° ≈ 0.41）。
    """
    d = np.where(np.isnan(dem), 0.0, dem).astype(np.float64)
    gy, gx = np.gradient(d, h_res_m)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def compute_convexity(dem: np.ndarray) -> np.ndarray:
    """
    ラプラシアン（4 近傍）による凸性指標 [m]。山頂で +、谷で - 。
    NaN は 0 として扱う。`MountainSurfaceRules` の `convexity` の役割。
    """
    d = np.where(np.isnan(dem), 0.0, dem).astype(np.float64)
    lap = (
        np.roll(d,  1, axis=0) + np.roll(d, -1, axis=0)
        + np.roll(d,  1, axis=1) + np.roll(d, -1, axis=1)
        - 4.0 * d
    )
    # 端は 0 にクランプ（roll で wrap した値の影響を消す）
    lap[0, :] = 0; lap[-1, :] = 0; lap[:, 0] = 0; lap[:, -1] = 0
    return lap.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Cliff-aware smoothing（階段化対策）
# ─────────────────────────────────────────────────────────────

def cliff_aware_smooth(
    dem: np.ndarray,
    h_res_m: float,
    sigma_cells: float = 1.0,
    cliff_threshold_m_per_m: float = 0.4,
) -> np.ndarray:
    """
    緩斜面だけを gaussian smoothing し、崖（slope ≥ threshold）は元の値を保持する。

    `Tellus` README §Settings の "Limit Shoreline Blend on Cliffs" の発想を一般化。
    PSO のシミュレーション本体には影響を与えない（NBT 化時のレンダ専用）。

    Returns: 平滑化後の DEM（NaN は元のまま温存）
    """
    if sigma_cells <= 0:
        return dem.copy()
    nan_mask = np.isnan(dem)
    d = np.where(nan_mask, 0.0, dem).astype(np.float64)
    smoothed = gaussian_filter(d, sigma=sigma_cells)
    slope = compute_slope(dem, h_res_m)
    cliff = slope >= cliff_threshold_m_per_m
    # 崖は元の値、それ以外は平滑値。NaN は NaN のまま戻す。
    out = np.where(cliff, d, smoothed)
    out = np.where(nan_mask, np.nan, out)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 海岸線・海域
# ─────────────────────────────────────────────────────────────

def make_sea_mask(dem: np.ndarray, sea_level_m: float = 0.0) -> np.ndarray:
    """
    海域マスク：NaN（NoData）または `dem <= sea_level_m` のセル。
    `Tellus.OceanClassification.isOcean` の簡易版（land mask が無いので NaN を ocean hint として扱う）。
    """
    return np.isnan(dem) | (np.where(np.isnan(dem), 0.0, dem) <= sea_level_m)


def distance_to_shore(land_mask: np.ndarray, h_res_m: float) -> np.ndarray:
    """
    各陸セルから最も近い海セルまでの距離 [m] を返す。
    海セルは 0、陸セルは正値。`scipy.ndimage.distance_transform_edt`。
    """
    if not np.any(~land_mask):
        # 海セル無し
        return np.full_like(land_mask, np.inf, dtype=np.float32)
    # land_mask=True を「変換対象」、False を「シード(海)」として扱う
    return (distance_transform_edt(land_mask) * h_res_m).astype(np.float32)


def make_ocean_depth(
    dem: np.ndarray,
    sea_level_m: float = 0.0,
    max_depth_m: float = 8.0,
    depth_per_m: float = 0.04,
) -> np.ndarray:
    """
    海セルについて、海岸からの距離で段階的に水深を返す。

    - 沿岸（distance=0）: 0 m（水面 = 海底）
    - 沖合: 線形に増加して `max_depth_m` で頭打ち
    - `depth_per_m`: 海岸 1m あたり水深 [m]。0.04 なら 200m 沖で 8m 水深

    陸セルは 0 を返す。
    """
    sea_mask = make_sea_mask(dem, sea_level_m)
    if not np.any(sea_mask):
        return np.zeros_like(dem, dtype=np.float32)
    # 海セルから「陸セルまでの距離」を測る（陸が seed）
    dist_from_land = distance_transform_edt(sea_mask)  # cells
    # 単位を m に直す（distance_transform_edt は cell 数なので h_res_m を別途かける必要があるが、
    # ここでは relative depth の形にしておき、呼び出し側の h_res_m と一緒に乗算する設計にする）
    return dist_from_land.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 地表ブロック分類（Tellus MountainSurfaceRules 簡易版）
# ─────────────────────────────────────────────────────────────

# 標高閾値 [m]（Tellus は 200/120 m を使用）
HIGHLAND_M = 120.0
ALPINE_M   = 200.0

# 斜度閾値 [m/m]
SLOPE_STEEP        = 0.40   # ~22°
SLOPE_VERY_STEEP   = 0.70   # ~35°
SLOPE_SCREE        = 1.00   # ~45°

# 海岸距離 [m]
SHORE_SAND_M       = 8.0
SHORE_GRAVEL_M     = 25.0


def classify_surface_block(
    elev_m: float,
    slope_m_per_m: float,
    convexity_m: float,
    dist_to_shore_m: float,
    sea_level_m: float = 0.0,
) -> str:
    """
    1 セルあたりの地表ブロック決定。`Tellus.MountainSurfaceRules` の判定順を圧縮し、
    土地被覆データ（ESA WorldCover）が無い前提で **DEM 由来量だけ** で分類する。

    優先順位（上から順に判定）：
      1. 海岸 ≤ SHORE_SAND_M → sand
      2. 海岸 ≤ SHORE_GRAVEL_M かつ低標高 (< 1.5m) → gravel
      3. 急斜面 (slope ≥ SLOPE_SCREE) → stone（rocky scree）
      4. 急斜面 (slope ≥ SLOPE_VERY_STEEP) かつ凸性 ≥ 0 → gravel（talus）
      5. 高標高 (≥ ALPINE_M) → stone
      6. 高標高 (≥ HIGHLAND_M) かつ急斜面 (slope ≥ SLOPE_STEEP) → stone
      7. それ以外 → grass
    """
    # 海岸付近：先に判定
    if dist_to_shore_m <= SHORE_SAND_M:
        return "sand"
    if dist_to_shore_m <= SHORE_GRAVEL_M and elev_m < 1.5:
        return "gravel"
    # 斜度ベース
    if slope_m_per_m >= SLOPE_SCREE:
        return "stone"
    if slope_m_per_m >= SLOPE_VERY_STEEP and convexity_m >= 0:
        return "gravel"
    # 標高ベース
    if elev_m >= ALPINE_M:
        return "stone"
    if elev_m >= HIGHLAND_M and slope_m_per_m >= SLOPE_STEEP:
        return "stone"
    return "grass"


def classify_surface_block_grid(
    dem_ds: np.ndarray,
    slope_ds: np.ndarray,
    convex_ds: np.ndarray,
    dist_shore_ds: np.ndarray,
    sea_level_m: float = 0.0,
) -> np.ndarray:
    """
    `classify_surface_block` のベクトル化版。各セルのブロック種別文字列の配列を返す。
    """
    out = np.full(dem_ds.shape, "grass", dtype=object)
    is_nan = np.isnan(dem_ds)
    elev = np.where(is_nan, 0.0, dem_ds)

    # 7 → 6 → 5 → 4 → 3 → 2 → 1 の逆順で上書きしていくと前段の結果が残る
    # 高標高 + 急斜面 → stone
    high_steep = (elev >= HIGHLAND_M) & (slope_ds >= SLOPE_STEEP)
    out[high_steep] = "stone"
    # 高標高アルパイン → stone
    out[elev >= ALPINE_M] = "stone"
    # 急斜面 (talus) → gravel
    talus = (slope_ds >= SLOPE_VERY_STEEP) & (convex_ds >= 0)
    out[talus] = "gravel"
    # 急斜面 (scree) → stone
    out[slope_ds >= SLOPE_SCREE] = "stone"
    # 海岸付近 (低標高+中距離) → gravel
    coast_mid = (dist_shore_ds <= SHORE_GRAVEL_M) & (elev < 1.5)
    out[coast_mid] = "gravel"
    # 海岸最近 → sand
    out[dist_shore_ds <= SHORE_SAND_M] = "sand"
    return out


def classify_surface_block_grid_esa(
    dem_ds: np.ndarray,
    slope_ds: np.ndarray,
    convex_ds: np.ndarray,
    dist_shore_ds: np.ndarray,
    cover_ds: np.ndarray,
    sea_level_m: float = 0.0,
) -> np.ndarray:
    """
    ESA WorldCover 2021 cover_class を主軸にした地表ブロック分類。
    `Tellus.MountainSurfaceRules.classifyApproximateSurface` の動作（cover をベースに
    山岳・急斜面・海岸ベルトで上書き）を Python に圧縮。
    """
    out = np.full(dem_ds.shape, "grass", dtype=object)
    is_nan = np.isnan(dem_ds)
    elev = np.where(is_nan, 0.0, dem_ds)

    # ESA 主導（後段で上書きされる）
    out[cover_ds == 80]  = "water"        # 内陸水面
    out[cover_ds == 70]  = "blue_ice"     # snow/ice
    out[cover_ds == 50]  = "stone"        # built-up（cobblestone 風に stone で代替）
    out[cover_ds == 60]  = "gravel"       # bare
    out[cover_ds == 90]  = "gravel"       # wetland
    out[cover_ds == 95]  = "grass"        # mangrove
    out[cover_ds == 10]  = "grass"        # tree cover
    out[cover_ds == 20]  = "grass"        # shrubland
    out[cover_ds == 30]  = "grass"        # grassland
    out[cover_ds == 40]  = "grass"        # cropland
    out[cover_ds == 100] = "gravel"       # moss/lichen

    # Tellus.qualifiesForMountainPalette 相当：高標高 + 急斜面 → stone へ強制
    out[(elev >= HIGHLAND_M) & (slope_ds >= SLOPE_STEEP)] = "stone"
    out[elev >= ALPINE_M] = "stone"
    out[slope_ds >= SLOPE_SCREE] = "stone"
    # 海岸最近：ESA より sand を優先（ESA は陸/海の境界で粗いことがある）
    out[dist_shore_ds <= SHORE_SAND_M] = "sand"
    return out


# ─────────────────────────────────────────────────────────────
# OSM (建物・道路) → ブロック grid 上のラスタ mask
# ─────────────────────────────────────────────────────────────

def _lonlat_to_grid_xy(
    lat: float, lon: float,
    patch_bbox_latlon: tuple,  # (lat_min, lat_max, lon_min, lon_max)
    grid_h: int, grid_w: int,
) -> tuple[float, float]:
    """経緯度 → ブロック grid の (col=x, row=z)。北が z=0、東が x 増加。"""
    lat_min, lat_max, lon_min, lon_max = patch_bbox_latlon
    x = (lon - lon_min) / (lon_max - lon_min) * grid_w
    z = (lat_max - lat) / (lat_max - lat_min) * grid_h
    return x, z


def polygon_mask_from_latlon(
    coords_latlon: list,
    patch_bbox_latlon: tuple,
    grid_h: int, grid_w: int,
) -> np.ndarray:
    """polygon (closed ring, [lat, lon] のリスト) の内部セル bool mask。
    matplotlib.path.Path で contains_points。"""
    import matplotlib.path as mpath
    if len(coords_latlon) < 3:
        return np.zeros((grid_h, grid_w), dtype=bool)
    pts = np.array([_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, grid_h, grid_w)
                     for la, lo in coords_latlon])
    path = mpath.Path(pts)
    xs, zs = np.meshgrid(np.arange(grid_w) + 0.5, np.arange(grid_h) + 0.5)
    grid_pts = np.column_stack([xs.ravel(), zs.ravel()])
    return path.contains_points(grid_pts).reshape(grid_h, grid_w)


def polyline_buffer_mask_from_latlon(
    coords_latlon: list,
    patch_bbox_latlon: tuple,
    grid_h: int, grid_w: int,
    buffer_cells: float,
) -> np.ndarray:
    """polyline (open, [lat, lon] のリスト) を buffer_cells 半径で太らせた bool mask。
    Bresenham でセル列にラスタ化 → distance_transform_edt で buffer。"""
    if len(coords_latlon) < 2:
        return np.zeros((grid_h, grid_w), dtype=bool)
    pts = [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, grid_h, grid_w)
            for la, lo in coords_latlon]
    line = np.zeros((grid_h, grid_w), dtype=bool)
    for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
        n = int(max(abs(x1 - x0), abs(z1 - z0))) + 1
        for t in range(n + 1):
            f = t / n if n > 0 else 0.0
            xi = int(round(x0 + (x1 - x0) * f))
            zi = int(round(z0 + (z1 - z0) * f))
            if 0 <= zi < grid_h and 0 <= xi < grid_w:
                line[zi, xi] = True
    if buffer_cells > 0:
        dist = distance_transform_edt(~line)
        return dist <= buffer_cells
    return line


def build_osm_masks(
    osm: dict,
    patch_bbox_latlon: tuple,
    grid_h: int, grid_w: int,
    h_res_block_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """OSM dict (fetch_osm_buildings_roads の戻り値) からブロック grid 上の
    (building_mask, road_mask) を生成する。"""
    building_mask = np.zeros((grid_h, grid_w), dtype=bool)
    road_mask     = np.zeros((grid_h, grid_w), dtype=bool)
    for b in osm.get("buildings", []):
        m = polygon_mask_from_latlon(b["coords"], patch_bbox_latlon, grid_h, grid_w)
        building_mask |= m
    for r in osm.get("roads", []):
        # buffer 半径 = 道路幅/2 をブロック単位に
        buf = max(1.0, float(r.get("width_m", 4)) / 2.0 / max(h_res_block_m, 0.1))
        m = polyline_buffer_mask_from_latlon(r["coords"], patch_bbox_latlon,
                                              grid_h, grid_w, buffer_cells=buf)
        road_mask |= m
    return building_mask, road_mask


# ─────────────────────────────────────────────────────────────
# Enhanced ブロック化
# ─────────────────────────────────────────────────────────────

def dem_to_blocks_enhanced(
    dem_patch: np.ndarray,
    inundation_patch: np.ndarray,
    h_res_dem: float,
    h_res_block: float,
    *,
    v_res_land: float = 1.0,
    v_exag_land: float = 1.5,
    v_exag_sea:  float = 0.5,
    sea_level_m: float = 0.0,
    ocean_max_depth_m: float = 8.0,
    smooth_sigma_cells: float = 1.0,
    cliff_threshold_m_per_m: float = 0.4,
    deep_ground: int = 8,
    flood_threshold: float = 0.05,
    cover_patch: np.ndarray | None = None,
    building_mask: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    building_height_m: float = 6.0,
) -> tuple[list, list[int]]:
    """
    `nbt_export.dem_to_blocks` の置き換え。Tellus 風の改善 5 点を適用：

      1. ダウンサンプル前に **cliff-aware smoothing**（緩斜面の階段化抑制）
      2. ダウンサンプル後に **海/陸を sea_level で分離**
      3. 海セルは **海岸からの距離で段階的水深**、海底に砂/砂利
      4. 地表ブロックは **slope/convexity/海岸距離** で sand/gravel/stone/grass を判定
      5. 地盤柱は `deep_ground` ブロック（既定 8、従来 3）

    Returns: (blocks_list, [nx, max_y+1, nz])
    """
    # ─── 1) Cliff-aware smoothing（フル解像度のまま） ───
    dem_smooth = cliff_aware_smooth(
        dem_patch, h_res_m=h_res_dem,
        sigma_cells=smooth_sigma_cells,
        cliff_threshold_m_per_m=cliff_threshold_m_per_m,
    )

    # ─── 2) ダウンサンプル ───
    factor = max(1, round(h_res_block / h_res_dem))
    H, W = dem_smooth.shape
    nz = H // factor
    nx = W // factor

    d = dem_smooth[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
    i = inundation_patch[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dem_ds = np.nanmean(d, axis=(1, 3))   # (nz, nx)
        idn_ds = np.nanmax(i,  axis=(1, 3))   # (nz, nx)

    # ESA cover を持っていれば、最近傍ダウンサンプル（カテゴリ値なので mean ではなく中央値）
    cover_ds = None
    if cover_patch is not None:
        cp = cover_patch[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
        cover_ds = np.median(cp, axis=(1, 3)).astype(np.uint8)

    # ─── 3) 海/陸マスク + 地形特徴量（ダウンサンプル後の解像度で計算） ───
    sea_mask  = make_sea_mask(dem_ds, sea_level_m)
    land_mask = ~sea_mask

    h_res_block_m = h_res_block
    slope_ds  = compute_slope(dem_ds, h_res_block_m)
    convex_ds = compute_convexity(dem_ds)
    dist_shore = distance_to_shore(land_mask, h_res_block_m)

    # 海セルの水深（cell 単位 → m）
    if np.any(sea_mask):
        # 海岸からの距離（cells）→ m に
        dist_from_land_cells = distance_transform_edt(sea_mask)
        dist_from_land_m = dist_from_land_cells * h_res_block_m
        # 線形に増えて max_depth で頭打ち
        # 200m 沖で max_depth_m に達する（depth_per_m = max / 200）
        depth_per_m = ocean_max_depth_m / 200.0
        ocean_depth = np.minimum(dist_from_land_m * depth_per_m, ocean_max_depth_m)
        ocean_depth = np.where(sea_mask, ocean_depth, 0.0).astype(np.float32)
    else:
        ocean_depth = np.zeros_like(dem_ds, dtype=np.float32)

    # ─── 4) y 座標変換（陸 v_exag、海 v_exag を分離） ───
    scale_land = v_exag_land / v_res_land
    scale_sea  = v_exag_sea  / v_res_land

    # 陸地表 y（最低 1）
    elev_land = np.where(np.isnan(dem_ds), 0.0, dem_ds)
    y_surf_land = np.maximum(1, (elev_land * scale_land).astype(int))
    # 海面 y（sea_level + 1 が水面ブロック）。地盤柱の起点として使う海底 y は sea - depth
    y_sea_surface = max(1, int((sea_level_m + 1.0) * scale_sea))
    y_sea_floor   = (np.maximum(0.0, sea_level_m - ocean_depth) * scale_sea).astype(int)
    y_sea_floor   = np.maximum(0, y_sea_floor)

    # 浸水深 → 浸水水柱の天井（陸セルのみ）
    y_flood_top = np.where(idn_ds > flood_threshold,
                           ((elev_land + idn_ds) * scale_land).astype(int),
                           y_surf_land)

    # ─── 5) 地表ブロック決定（ESA WorldCover があれば優先） ───
    if cover_ds is not None:
        surf_block = classify_surface_block_grid_esa(
            dem_ds, slope_ds, convex_ds, dist_shore, cover_ds, sea_level_m=sea_level_m,
        )
    else:
        surf_block = classify_surface_block_grid(
            dem_ds, slope_ds, convex_ds, dist_shore, sea_level_m=sea_level_m,
        )

    # OSM 道路は地表を gravel で上書き（陸セルのみ、建物より優先順位は低い）
    if road_mask is not None and road_mask.shape == surf_block.shape:
        land_for_road = ~np.isnan(dem_ds) & ~(np.where(np.isnan(dem_ds), 0.0, dem_ds) <= sea_level_m)
        surf_block[road_mask & land_for_road] = "gravel"

    valid_elevs = dem_ds[~np.isnan(dem_ds)]
    max_elev_y = int(valid_elevs.max() * scale_land) if len(valid_elevs) > 0 else 1
    max_y = min(max(max_elev_y + 5, y_sea_surface + 2), 500)

    # ─── 6) ブロック生成（numpy ベクトルで append） ───
    BZ, BX = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    blocks: list = []

    # --- 海セル：海底 stone/sand 柱 + 水柱 ---
    sea_idx = np.argwhere(sea_mask)
    for j, i_ in sea_idx.tolist():
        bx_v = int(BX[j, i_]); bz_v = int(BZ[j, i_])
        floor_y = int(y_sea_floor[j, i_])
        # 海底ブロック（深いほど砂利、浅いほど砂）
        floor_kind = "sand" if ocean_depth[j, i_] < 1.5 else "gravel"
        blocks.append(nbtlib.Compound({
            "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(floor_y), nbtlib.Int(bz_v)]),
            "state": block_id(floor_kind),
        }))
        # 海底直下に少しの stone 地盤
        for dy in range(max(0, floor_y - 3), floor_y):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(dy), nbtlib.Int(bz_v)]),
                "state": block_id("stone"),
            }))
        # 水柱（floor+1 ～ sea_surface）
        for fy in range(floor_y + 1, y_sea_surface + 1):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                "state": block_id("water"),
            }))

    # --- 陸セル：deep_ground ブロックの stone 地盤柱 + 地表 ---
    land_idx = np.argwhere(land_mask)
    for j, i_ in land_idx.tolist():
        bx_v = int(BX[j, i_]); bz_v = int(BZ[j, i_])
        y_top = int(y_surf_land[j, i_])
        # 地盤柱
        for dy in range(max(0, y_top - deep_ground), y_top):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(dy), nbtlib.Int(bz_v)]),
                "state": block_id("stone"),
            }))
        # 地表
        kind = surf_block[j, i_]
        blocks.append(nbtlib.Compound({
            "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(y_top), nbtlib.Int(bz_v)]),
            "state": block_id(kind),
        }))

    # --- 浸水ブロック（陸セルで idn > threshold） ---
    flood_mask = land_mask & (idn_ds > flood_threshold)
    flood_idx = np.argwhere(flood_mask)
    for j, i_ in flood_idx.tolist():
        bx_v = int(BX[j, i_]); bz_v = int(BZ[j, i_])
        y_s  = int(y_surf_land[j, i_])
        y_ft = int(y_flood_top[j, i_])
        for fy in range(y_s + 1, min(y_ft + 1, y_s + 30)):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                "state": block_id("water"),
            }))

    # --- OSM 建物（陸セルのみ、地表柱の上に stone を building_height_m 分積む） ---
    if building_mask is not None and building_mask.shape == dem_ds.shape:
        bh_blocks = max(2, int(round(building_height_m * scale_land)))
        b_idx = np.argwhere(building_mask & land_mask)
        b_max_y = 0
        for j, i_ in b_idx.tolist():
            bx_v = int(BX[j, i_]); bz_v = int(BZ[j, i_])
            y_top = int(y_surf_land[j, i_])
            top_y = y_top + bh_blocks
            if top_y > b_max_y:
                b_max_y = top_y
            for fy in range(y_top + 1, top_y + 1):
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                    "state": block_id("stone"),
                }))
        max_y = max(max_y, b_max_y + 2)

    return blocks, [nx, max_y + 1, nz]
