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
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
import nbtlib

# nbt_export 側のパレットを再利用（同じ block_id/PALETTE 定義）
from nbt_export import block_id


class _DenseBlockSink:
    """`blocks.append(Compound)` インターフェースを保ったまま、各 Compound の
    pos/state を密3D配列 ``arr[y,z,x]``（uint16, 0=air, 値=palette index）へ書き込む
    シム（施策③: {pos,state}個別Compound列挙 → 密numpy配列）。

    16 箇所の append サイトを**無改変**で密配列化できる。座標は既存 Compound から
    取り出すので x↔z 取り違えが原理的に起きず、重なりは後勝ち（代入の上書き）で
    自然に再現される。Compound は append のたびに即捨てされ保持しないため、
    数千万ボクセルでもメモリは密配列分に収まる（8-12GB 問題の根治）。
    範囲外/上限超の書き込みは捨てる（litematic の valid フィルタと等価な境界クランプ）。
    """
    __slots__ = ("nx", "nz", "arr", "max_y")

    def __init__(self, nx: int, nz: int, y_cap: int = 501):
        self.nx = int(nx)
        self.nz = int(nz)
        self.arr = np.zeros((int(y_cap), int(nz), int(nx)), dtype=np.uint16)
        self.max_y = 0

    def append(self, compound) -> None:
        p = compound["pos"]
        x = int(p[0]); y = int(p[1]); z = int(p[2])
        if 0 <= x < self.nx and 0 <= z < self.nz and 0 <= y < self.arr.shape[0]:
            self.arr[y, z, x] = int(compound["state"])
            if y > self.max_y:
                self.max_y = y

    def array(self, height: int) -> np.ndarray:
        """size の Y（=max_y+1）まで切り詰めた密配列 (Y,Z,X) を返す。copy を返すので
        呼び出し後に y_cap 分の大きい内部バッファは解放できる（書き出し時メモリ削減）。"""
        h = max(1, min(int(height), self.arr.shape[0]))
        return self.arr[:h].copy()


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

def make_sea_mask(dem: np.ndarray, sea_level_m: float = 0.0,
                  smooth_sigma: float = 0.0) -> np.ndarray:
    """
    海域マスク：NaN（NoData）または `dem <= sea_level_m` のセル。
    `Tellus.OceanClassification.isOcean` の簡易版（land mask が無いので NaN を ocean hint として扱う）。

    smooth_sigma>0 で、二値マスクを σ ガウシアンで平滑化し 0.5 アイソラインで再二値化
    （arnis land_cover.rs:104 compute_water_blend_smooth 移植）。海岸線の 1 セルの
    ギザギザ（角張り）を曲線化する。σ は小さめ推奨（1.0〜1.5）。大きいと小島/入り江が消える。
    """
    raw = np.isnan(dem) | (np.where(np.isnan(dem), 0.0, dem) <= sea_level_m)
    if smooth_sigma and smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(raw.astype(np.float32), sigma=float(smooth_sigma))
        return blurred >= 0.5
    return raw


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


# ── 軸6-2: 決定論的座標ハッシュによる地表ディザ混合 ──
# 1クラス=1ブロックの単調地表を、世界座標ハッシュで重み付きブロック混合に散らし、
# 岩肌/礫/耕地に自然な斑（テクスチャ感）を与える。乱数でなく座標ハッシュなので
# タイル分割しても同じ世界セルは常に同じブロック＝litematic/Anvil 再現性を保つ。
def coord_hash01(gx: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """世界ブロック座標 (gx, gz) → 決定論的 [0,1) ノイズ（splitmix64 系, ベクトル化）。
    小さい構造化座標(隣接整数)でも均一になるよう avalanche の強い finalizer を使い、
    よく撹拌された高ビット側を採用する。"""
    x = (np.asarray(gx, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    z = (np.asarray(gz, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    h = x * np.uint64(0x9E3779B97F4A7C15)
    h = h ^ (z * np.uint64(0xC2B2AE3D27D4EB4F))
    h = h ^ (h >> np.uint64(30)); h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27)); h = h * np.uint64(0x94D049BB133111EB)
    h = h ^ (h >> np.uint64(31))
    return (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)   # 高24ビット

# クラス → [(ブロックキー, 重み), ...]（重みは正規化される）。全キーは block_palette に実在。
SURFACE_DITHER = {
    "stone":       [("stone", 0.55), ("andesite", 0.16), ("cobblestone", 0.12),
                    ("tuff", 0.09), ("gravel", 0.08)],
    "gravel":      [("gravel", 0.68), ("coarse_dirt", 0.16), ("stone", 0.10),
                    ("cobblestone", 0.06)],
    "coarse_dirt": [("coarse_dirt", 0.76), ("dirt", 0.16), ("rooted_dirt", 0.08)],
}


def apply_surface_dither(surf_block: np.ndarray, cell_offset: tuple) -> None:
    """surf_block(class文字列 grid, nz×nx) を世界座標ハッシュで in-place にディザ混合。
    cell_offset=(gx0, gz0) はこのパッチ左上の世界ブロック座標（タイル間整合の基準）。"""
    nz, nx = surf_block.shape
    gx0, gz0 = int(cell_offset[0]), int(cell_offset[1])
    gx = gx0 + np.arange(nx, dtype=np.int64)[None, :]
    gz = gz0 + np.arange(nz, dtype=np.int64)[:, None]
    h = coord_hash01(np.broadcast_to(gx, (nz, nx)), np.broadcast_to(gz, (nz, nx)))
    orig = surf_block.copy()   # クラス膜は元配列から取る（ディザ出力="gravel"等を後段が再ディザしない）
    for cls, mix in SURFACE_DITHER.items():
        m = (orig == cls)
        if not m.any():
            continue
        keys = [k for k, _ in mix]
        ws = np.array([w for _, w in mix], dtype=np.float64)
        cum = np.cumsum(ws) / ws.sum()
        idx = np.clip(np.searchsorted(cum, h, side="right"), 0, len(keys) - 1)
        for ki, k in enumerate(keys):
            surf_block[m & (idx == ki)] = k


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
    road_major_mask = np.zeros((grid_h, grid_w), dtype=bool)  # 幹線(幅≥4.5m)のみ→舗装
    for b in osm.get("buildings", []):
        m = polygon_mask_from_latlon(b["coords"], patch_bbox_latlon, grid_h, grid_w)
        building_mask |= m
    for r in osm.get("roads", []):
        # buffer 半径 = 道路幅/2 をブロック単位に
        w = float(r.get("width_m", 4))
        buf = max(1.0, w / 2.0 / max(h_res_block_m, 0.1))
        m = polyline_buffer_mask_from_latlon(r["coords"], patch_bbox_latlon,
                                              grid_h, grid_w, buffer_cells=buf)
        road_mask |= m
        if w >= 4.5:                       # 真幅道路/幹線 → 舗装(andesite)
            road_major_mask |= m
    return building_mask, road_mask, road_major_mask


# FG-GML type → 壁/屋根ブロック（屋根は ortho 無効時 or 集約不能時の fallback）。
# 木造住宅=白壁 / RC=コンクリ灰 / 無壁舎(倉庫・車庫)=石 で見た目を3分化。
BUILDING_WALL_BY_TYPE = {
    "普通建物":     "white_concrete",
    "堅ろう建物":   "light_gray_concrete",
    "高層建物":     "light_gray_concrete",
    "普通無壁舎":   "stone",
    "堅ろう無壁舎": "stone",
    # PLATEAU 用途別（①）。壁材を用途・構造で変えて色面を多様化
    "商業ビル":     "light_gray_concrete",   # 商業/業務/複合
    "宿泊":         "white_concrete",        # ホテル等
    "マンション":   "white_terracotta",      # 共同住宅（クリーム系）
    "住宅":         "sandstone",             # 戸建（暖色タン）
    "木造住宅":     "spruce_planks",         # 木造（木質）
    "工場":         "andesite",              # 工場（灰）
    "倉庫":         "gray_concrete",         # 倉庫/供給処理（金属灰）
    "公共":         "diorite",               # 官公庁/文教（白御影風）
    "農業施設":     "oak_planks",            # 農林漁業用（木）
    "ランドマーク": "white_concrete",        # 城・高いLOD2建物（白漆喰）
}
BUILDING_ROOF_BY_TYPE = {
    "普通建物":     "gray_concrete",
    "堅ろう建物":   "light_gray_concrete",
    "高層建物":     "light_gray_concrete",
    "普通無壁舎":   "gray_concrete",
    "堅ろう無壁舎": "gray_concrete",
    "商業ビル":     "light_gray_concrete",
    "宿泊":         "light_gray_concrete",
    "マンション":   "gray_concrete",
    "住宅":         "gray_concrete",
    "木造住宅":     "deepslate",
    "工場":         "gray_concrete",
    "倉庫":         "gray_concrete",
    "公共":         "gray_concrete",
    "農業施設":     "deepslate",
    "ランドマーク": "deepslate",
}
# 寄棟風の勾配屋根にする用途（戸建・木造）。ビル/工場/倉庫/マンションは陸屋根のまま
HIP_ROOF_TYPES = ("普通建物", "住宅", "木造住宅", "農業施設")

# 建物スタイル（窓の量・内部構造を変える）: house=戸建(窓少なめ) / building=ビル(窓多め) / factory=工場・倉庫
_FACTORY_TYPES = ("普通無壁舎", "堅ろう無壁舎", "工場", "倉庫", "農業施設")
_BUILDING_TYPES = ("堅ろう建物", "高層建物", "商業ビル", "マンション", "宿泊", "公共", "ランドマーク")


def building_style_for_type(tp: str) -> str:
    if tp in _FACTORY_TYPES:
        return "factory"
    if tp in _BUILDING_TYPES:
        return "building"
    return "house"   # 普通建物 / 住宅 / 木造住宅 / 不明


def _is_window(style: str, r: int, fh: int, run: int) -> bool:
    """周壁セルが窓(ガラス)になるか。r=階内の高さ位置(0=床ライン), fh=階高,
    run=壁が走る向きの座標。窓は3マスおき＝窓と窓の間に最低2マスの壁を必ず残す。
    家=各階に窓 / ビル=全階に連窓 / 工場=高所のハイサイド窓。"""
    if r <= 0 or r >= fh:                       # 床/天井ラインには窓を置かない
        return False
    if run % 3 != 0:                            # 横方向は3マスおき（窓の間に最低2マスの壁）
        return False
    if style == "building":
        return True                             # 全階に窓
    if style == "factory":
        return r in (fh - 2, fh - 3)            # 高所のハイサイド窓
    return r in (1, 2)                          # 戸建: 各階に窓


def _rasterize_lod2_roof(roof3d, patch_bbox_latlon, grid_h, grid_w, x0, y0, x1, y1):
    """PLATEAU LOD2 の屋根ポリゴン群を block-grid にラスタ化し、各セルの屋根上面の標高 z を返す。
    各セル = そのセルを覆うポリゴンの平面 z の最大値（=最上面=屋根）。覆われないセルは NaN。
    壁（grid 投影が直線=退化）はスキップ。城など複雑な屋根形状をそのまま高さに反映できる。"""
    import matplotlib.path as mpath
    sz = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float64)
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    gp = np.column_stack([gx.ravel(), gy.ravel()])
    for poly in roof3d:
        if len(poly) < 3:
            continue
        P = np.array([_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, grid_h, grid_w)
                      for la, lo, _ in poly], dtype=np.float64)
        if (P[:, 0].max() - P[:, 0].min()) < 1e-6 or (P[:, 1].max() - P[:, 1].min()) < 1e-6:
            continue   # 壁（直線投影）はスキップ
        z = np.array([zz for _, _, zz in poly], dtype=np.float64)
        try:
            A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        except Exception:
            continue
        inside = mpath.Path(P).contains_points(gp).reshape(gy.shape)
        if not inside.any():
            continue
        zest = coef[0] * gx + coef[1] * gy + coef[2]
        sz = np.where(inside & (np.isnan(sz) | (zest > sz)), zest, sz)
    return sz
DEFAULT_WALL_KEY = "white_concrete"
DEFAULT_ROOF_KEY = "gray_concrete"


def build_building_maps(
    buildings: list,
    dsm_h_block: np.ndarray | None,
    patch_bbox_latlon: tuple,
    grid_h: int, grid_w: int,
    *,
    pct: int = 75,
    type_floor_frac: float = 0.6,
    min_h_m: float = 2.0,
    roof_slope: float = 0.35,
    roof_cap: float = 3.0,
) -> dict:
    """FG-GML 各建物を1棟単位でラスタ化し、描画に必要な block-grid マップ一式を返す。

    - height : footprint 内 DSM-DEM の pct パーセンタイル（既定 p75）を **1棟1値** でフラット化（P1）。
      per-cell 拾いの屋根凸凹・植生スパイク・切株化を解消。type 高さ×type_floor_frac を下限に。
    - id     : 建物ごとの整数ラベル（-1=非建物）。屋根色を1棟で均一化する集約に使う（P2）。
    - wall_keys / roof_keys : 建物 id → FG-GML type 由来の壁/屋根ブロックキー（P2 fallback）。
    interior（中庭）は除外して空洞に保つ。dsm_h_block=None なら type 高さのみでフラット化。

    returns dict(mask:bool, height:float32 NaN外, id:int32 -1外, wall_keys:list, roof_keys:list)。
    """
    import matplotlib.path as mpath
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    hmap = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    idmap = np.full((grid_h, grid_w), -1, dtype=np.int32)
    wall_keys: list[str] = []
    roof_keys: list[str] = []
    style_keys: list[str] = []
    bid = 0
    for b in buildings:
        ext = b.get("coords")
        if not ext or len(ext) < 3:
            continue
        pts = np.array([_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, grid_h, grid_w)
                        for la, lo in ext])
        x0 = max(0, int(np.floor(pts[:, 0].min())))
        x1 = min(grid_w, int(np.ceil(pts[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(pts[:, 1].min())))
        y1 = min(grid_h, int(np.ceil(pts[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        gp = np.column_stack([gx.ravel(), gy.ravel()])
        ins = mpath.Path(pts).contains_points(gp).reshape(gy.shape)
        for hole in (b.get("holes") or []):
            if len(hole) < 3:
                continue
            hpts = np.array([_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, grid_h, grid_w)
                             for la, lo in hole])
            ins &= ~mpath.Path(hpts).contains_points(gp).reshape(gy.shape)
        if not ins.any():
            continue
        tp = b.get("tags", {}).get("fgd_type", "")
        _hm = b.get("tags", {}).get("height_m")
        floor = float(_hm if _hm is not None else 6.0) * type_floor_frac
        h = floor
        if dsm_h_block is not None:
            vals = dsm_h_block[y0:y1, x0:x1][ins]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                h = max(float(np.percentile(vals, pct)), floor)
        h = max(h, min_h_m)
        sub_m = mask[y0:y1, x0:x1]; sub_m[ins] = True
        # 屋根形状: 普通建物(住宅)は寄棟風の勾配屋根(縁=壁top, 内側ほど高い)。
        # 堅ろう建物(RC)・無壁舎(倉庫)は陸屋根(フラット)のまま。
        sub_h = hmap[y0:y1, x0:x1]
        roof3d = b.get("roof3d")
        if roof3d:
            # PLATEAU LOD2: 屋根面を per-cell でラスタ化し、城などの屋根形状をそのまま高さに反映
            sz = _rasterize_lod2_roof(roof3d, patch_bbox_latlon, grid_h, grid_w, x0, y0, x1, y1)
            bz = float(min(zz for p in roof3d for (_, _, zz) in p))
            cov = np.isfinite(sz) & ins
            sub_h[ins] = h
            if cov.any():
                sub_h[cov] = np.maximum(sz[cov] - bz, min_h_m).astype(np.float32)
        elif roof_slope > 0 and tp in HIP_ROOF_TYPES and ins.sum() >= 4:
            d = distance_transform_edt(ins).astype(np.float32)   # 縁からの内側距離(block)
            rise = np.minimum(np.clip(d - 1.0, 0, None) * roof_slope, roof_cap)
            sub_h[ins] = h + rise[ins]                            # 重なりは後勝ち
        else:
            sub_h[ins] = h
        sub_i = idmap[y0:y1, x0:x1]; sub_i[ins] = bid
        wall_keys.append(BUILDING_WALL_BY_TYPE.get(tp, DEFAULT_WALL_KEY))
        roof_keys.append(BUILDING_ROOF_BY_TYPE.get(tp, DEFAULT_ROOF_KEY))
        style_keys.append(building_style_for_type(tp))
        bid += 1
    return {"mask": mask, "height": hmap, "id": idmap,
            "wall_keys": wall_keys, "roof_keys": roof_keys, "style_keys": style_keys}


def add_bridge_blocks(blocks, bridges, patch_bbox_latlon, nz, nx, *,
                      y_surf_land, sea_mask, y_sea_surface, y_sea_floor,
                      scale_land, h_res_block_m, surf_block=None,
                      deck_key="andesite", pier_key="andesite",
                      cap_key="andesite", rail_key="andesite",
                      arch_rise_m=0.0, y_flood_top=None) -> int:
    """OSM 橋（polyline + layer + road_class + width）を Tellus 流に立体化して blocks へ追加。

    桁Y(station) = max(両岸補間 baseline + ramp(layer×arch_rise_m),  局所地形/水面 + ramp(clearance))
      ramp は端0→中央最大の 4:1 勾配（=アプローチ坂）。clearance(main6/normal5/dirt3 m)が
      layer 情報無しでも川を跨がせる。arch_rise_m=0 なら両岸補間に沿う平坦橋（天田橋等は両端高さに）。
    デッキは2層: 上面=surf_block（衛星写真の路面色）／下面+橋脚+欄干+笠=deck_key(安山岩)。
    既存ブロックより後に置く（litematic は後勝ち）ので水上でデッキが優先される。
    返り値: 置いた最大 y（max_y 更新用）。
    """
    import math
    MAX_RISE_M, RAMP_HV = 10.0, 4.0
    CLEAR_M = {"main": 6.0, "normal": 5.0, "dirt": 3.0}
    PIER_SPACING_M = 16.0
    seen: set = set()
    ymax = [0]

    def put(ix, iy, iz, key):
        if not (0 <= ix < nx and 0 <= iz < nz) or iy < 0 or iy > 500:
            return
        k = (ix, iy, iz)
        if k in seen:
            return
        seen.add(k)
        if iy > ymax[0]:
            ymax[0] = iy
        blocks.append(nbtlib.Compound({
            "pos": nbtlib.List[nbtlib.Int]([nbtlib.Int(ix), nbtlib.Int(iy), nbtlib.Int(iz)]),
            "state": block_id(key),
        }))

    def col(x, z):
        return int(round(x)), int(round(z))

    def terrain_y(x, z):
        i, j = col(x, z)
        return int(y_surf_land[j, i]) if (0 <= j < nz and 0 <= i < nx) else 1

    def ground_y(x, z):     # 水なら水面、陸なら地表
        i, j = col(x, z)
        if 0 <= j < nz and 0 <= i < nx:
            return int(y_sea_surface) if sea_mask[j, i] else int(y_surf_land[j, i])
        return 1

    def floor_y(x, z):      # 橋脚の底（川底 or 地表）
        i, j = col(x, z)
        if 0 <= j < nz and 0 <= i < nx:
            return int(y_sea_floor[j, i]) if sea_mask[j, i] else int(y_surf_land[j, i])
        return 0

    def ramp(station, total, full):
        if full <= 0 or total <= 1e-6:
            return 0.0
        rl = full * RAMP_HV
        s = min(max(station, 0.0), total)
        if total >= rl * 2.0:
            if s < rl:
                return full * (s / rl)
            if s > total - rl:
                return full * ((total - s) / rl)
            return full
        half = total * 0.5
        if half <= 1e-6:
            return full
        return full * (s / half) if s <= half else full * ((total - s) / half)

    def end_base(end_pt, in_pt):
        # 橋端 → 外向き（橋の延長＝奥に続く道路の方向）へ陸地形をサンプルし中央値で安定化。
        # 端ピンポイント依存をやめ、データ境界/水面/穴での baseline 破綻や橋台手前の瘤を防ぐ。
        dx, dz = end_pt[0] - in_pt[0], end_pt[1] - in_pt[1]
        L = math.hypot(dx, dz)
        if L > 1e-6:
            dx, dz = dx / L, dz / L
        ys = []
        for d in range(0, 26):    # 端から外側 0..25 block（奥に続く道路方向）の陸地形
            i, j = col(end_pt[0] + dx * d, end_pt[1] + dz * d)
            if 0 <= j < nz and 0 <= i < nx and not sea_mask[j, i] and np.isfinite(y_surf_land[j, i]):
                ys.append(int(y_surf_land[j, i]))
        return int(np.median(ys)) if ys else terrain_y(*end_pt)

    for b in bridges:
        pts = [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, nz, nx) for la, lo in b["coords"]]
        rc = b.get("road_class", "normal")
        half_w = max(0, int(round((float(b.get("width_m") or 5.5) / max(h_res_block_m, 0.1)) / 2.0)))
        layer = int(b.get("layer", 1))
        seg, total = [], 0.0
        for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
            L = math.hypot(x1 - x0, z1 - z0); seg.append(L); total += L
        if total < 2.0:
            continue
        startS = end_base(pts[0], pts[1] if len(pts) > 1 else pts[0])
        endS = end_base(pts[-1], pts[-2] if len(pts) > 1 else pts[-1])
        rise_full = min(layer * arch_rise_m, MAX_RISE_M) * scale_land
        clear_full = CLEAR_M.get(rc, 5.0) * scale_land
        # スパンが水を渡るか（渡る時だけ水面+clearance を最低デッキ高に）
        has_water = False
        for (xa, za), (xb, zb) in zip(pts, pts[1:]):
            for tt in (0.2, 0.4, 0.6, 0.8):
                ci, cj = int(round(xa + (xb - xa) * tt)), int(round(za + (zb - za) * tt))
                if 0 <= cj < nz and 0 <= ci < nx and sea_mask[cj, ci]:
                    has_water = True
                    break
            if has_water:
                break
        min_deck = (int(y_sea_surface) + clear_full) if has_water else -1.0e9
        pier_step = max(4.0, PIER_SPACING_M / max(h_res_block_m, 0.1))
        next_pier = pier_step
        s_acc = 0.0
        for si in range(len(seg)):
            (x0, z0), (x1, z1) = pts[si], pts[si + 1]
            L = seg[si]
            if L < 1e-6:
                continue
            tx, tz = (x1 - x0) / L, (z1 - z0) / L
            ox, oz = -tz, tx
            n = max(1, int(L / 0.5))
            for k in range(n + 1):
                t = k / n
                cx, cz = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
                station = s_acc + L * t
                base = startS + (endS - startS) * (station / total)
                # 水面クリアランス: 両端の道路高から 4:1 で立ち上がり min_deck で頭打ち。
                # 端付近の地形に依存しないので橋台手前の不自然な瘤が出ない。
                lift = min(min_deck,
                           startS + station / RAMP_HV,
                           endS + (total - station) / RAMP_HV)
                dy = int(round(max(base + ramp(station, total, rise_full), lift)))
                for w in range(-half_w, half_w + 1):
                    ix, iz = col(cx + ox * w, cz + oz * w)
                    top_key = deck_key
                    if surf_block is not None and 0 <= iz < nz and 0 <= ix < nx:
                        sk = surf_block[iz, ix]
                        if sk and sk != "water":    # 水域(河川/海)の色は橋の路面に使わない
                            top_key = sk            # 上面=衛星写真の路面色
                    put(ix, dy, iz, top_key)
                    put(ix, dy - 1, iz, deck_key)   # 下面=安山岩(構造)
                    if abs(w) == half_w and half_w >= 1:
                        put(ix, dy + 1, iz, rail_key)
                if station >= next_pier and not (si == 0 and k == 0):
                    next_pier += pier_step
                    fy = floor_y(cx, cz)
                    if dy - 2 > fy:
                        shafts = (-half_w + 1, half_w - 1) if (rc == "main" and half_w >= 2) else (0,)
                        for w in shafts:
                            ix, iz = col(cx + ox * w, cz + oz * w)
                            for yy in range(fy, dy - 1):
                                put(ix, yy, iz, pier_key)
                            put(ix, dy - 1, iz, cap_key)
                # 橋下を「橋を抜いた高さ（周辺の洪水水位/水面）」まで water で埋める。
                # デッキ・橋脚は既に put 済み(seen)なので上書きされず、橋脚の間だけ水が入る。
                if y_flood_top is not None:
                    wt = -1
                    for w in range(-half_w - 3, half_w + 4):
                        ix2, iz2 = col(cx + ox * w, cz + oz * w)
                        if 0 <= iz2 < nz and 0 <= ix2 < nx:
                            wt = max(wt, int(y_flood_top[iz2, ix2]))
                            if sea_mask[iz2, ix2]:
                                wt = max(wt, int(y_sea_surface))
                    g = floor_y(cx, cz)
                    for wy in range(g + 1, min(int(dy) - 1, wt) + 1):
                        for w in range(-half_w, half_w + 1):
                            ix2, iz2 = col(cx + ox * w, cz + oz * w)
                            put(ix2, wy, iz2, "water")
            s_acc += L
    return ymax[0]


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
    sea_smooth_sigma: float = 1.0,
    smooth_sigma_cells: float = 1.0,
    cliff_threshold_m_per_m: float = 0.4,
    deep_ground: int = 8,
    flood_threshold: float = 0.05,
    cover_patch: np.ndarray | None = None,
    building_mask: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    building_height_m: float = 6.0,
    building_height_patch: np.ndarray | None = None,
    building_height_block: np.ndarray | None = None,
    tree_height_patch: np.ndarray | None = None,
    tree_mode: str = "canopy",
    building_id: np.ndarray | None = None,
    building_wall_keys: list | None = None,
    building_roof_keys: list | None = None,
    roof_color_tol: float = 55.0,
    color_building_roofs: bool = False,
    wall_block: str = "white_concrete",
    window_block: str = "glass",
    floor_height: int = 5,
    floor_block: str = "light_gray_concrete",
    interior_light: str = "sea_lantern",
    building_style_keys: list | None = None,
    hollow_buildings: bool = True,
    legend_layer: bool = False,
    surface_grid_override: np.ndarray | None = None,
    bridges: list | None = None,
    patch_bbox_latlon: tuple | None = None,
    road_block: str = "andesite",
    road_major_mask: np.ndarray | None = None,
    road_minor_block: str = "gravel",
    water_mask: np.ndarray | None = None,
    water_block: str = "water",
    evac_facilities: list | None = None,
    cell_offset: tuple = (0, 0),
    dither_surface: bool = True,
) -> tuple[list, list[int]]:
    """
    `nbt_export.dem_to_blocks` の置き換え。Tellus 風の改善 5 点を適用：

      1. ダウンサンプル前に **cliff-aware smoothing**（緩斜面の階段化抑制）
      2. ダウンサンプル後に **海/陸を sea_level で分離**
      3. 海セルは **海岸からの距離で段階的水深**、海底に砂/砂利
      4. 地表ブロックは **slope/convexity/海岸距離** で sand/gravel/stone/grass を判定
      5. 地盤柱は **可変アンダーフィル**（arnis 移植）：8近傍の最低地表 Y まで
         stone で埋め、深さは `deep_ground` を上限にクランプ。平地は 2 ブロックで
         済みブロック数が激減し、崖面は隣接セルの底まで埋めて見える穴を塞ぐ。
         （従来は全セル一律 `deep_ground` 本＝地下 stone を無駄に増やしていた）

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

    # 建物高さ[m]。building_height_block（per-building 集約済みのフラット高さ, block grid）が
    # あればそれを優先（屋根フラット化）。無ければ従来どおり DSM patch を per-cell ダウンサンプル。
    bh_ds = None
    if building_height_block is not None and building_height_block.shape == (nz, nx):
        bh_ds = building_height_block
    elif building_height_patch is not None:
        bp = building_height_patch[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bh_ds = np.nanmean(bp, axis=(1, 3))

    # 樹冠高[m]（LiDAR class3 由来）をダウンサンプル
    tree_ds = None
    if tree_height_patch is not None:
        tp = tree_height_patch[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tree_ds = np.nanmean(tp, axis=(1, 3))

    # ─── 3) 海/陸マスク + 地形特徴量（ダウンサンプル後の解像度で計算） ───
    sea_mask  = make_sea_mask(dem_ds, sea_level_m, smooth_sigma=sea_smooth_sigma)
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

    # 世界全体を _lift 持ち上げ、最下段(y=_lift-1)に土台層を敷く（砂/砂利がブロック更新で落ちないよう支える）。
    # 凡例3層(地下データ層)有効時はさらに底に凡例用の空間も空ける。
    LEGEND_YS = (0, 2, 4)        # 地下データ層の高さ: y=0 土地利用 / y=2 洪水 / y=4 樹木（間隔をあけて）
    _lift = (LEGEND_YS[-1] + 2) if legend_layer else 1   # 地形/構造物の最低 y。y=_lift-1 が土台層

    # 陸地表 y（最低 1）。凡例有効時は _lift 持ち上げ
    elev_land = np.where(np.isnan(dem_ds), 0.0, dem_ds)
    y_surf_land = np.maximum(1, (elev_land * scale_land).astype(int)) + _lift
    # 海面 y（sea_level + 1 が水面ブロック）。地盤柱の起点として使う海底 y は sea - depth
    y_sea_surface = max(1, int((sea_level_m + 1.0) * scale_sea)) + _lift
    y_sea_floor   = (np.maximum(0.0, sea_level_m - ocean_depth) * scale_sea).astype(int)
    y_sea_floor   = np.maximum(0, y_sea_floor) + _lift

    # 浸水深 → 浸水水柱の天井（陸セルのみ）
    y_flood_top = np.where(idn_ds > flood_threshold,
                           ((elev_land + idn_ds) * scale_land).astype(int) + _lift,
                           y_surf_land)

    # ─── 5) 地表ブロック決定 ───
    # 優先順位: surface_grid_override (Tellus 直接) > ESA cover > slope/convex/海岸距離
    if surface_grid_override is not None:
        if surface_grid_override.shape == dem_ds.shape:
            surf_block = surface_grid_override.copy()
        elif surface_grid_override.shape == dem_patch.shape:
            # フル解像度を渡されたら nearest でダウンサンプル
            sg = surface_grid_override[:nz*factor, :nx*factor].reshape(nz, factor, nx, factor)
            # オブジェクト dtype の最頻値は遅いので中央位置サンプリング
            surf_block = sg[:, factor // 2, :, factor // 2].copy()
        else:
            raise ValueError(
                f"surface_grid_override shape {surface_grid_override.shape} != "
                f"dem_ds {dem_ds.shape} nor dem_patch {dem_patch.shape}"
            )
    elif cover_ds is not None:
        surf_block = classify_surface_block_grid_esa(
            dem_ds, slope_ds, convex_ds, dist_shore, cover_ds, sea_level_m=sea_level_m,
        )
    else:
        surf_block = classify_surface_block_grid(
            dem_ds, slope_ds, convex_ds, dist_shore, sea_level_m=sea_level_m,
        )

    # ESA WorldCover の土地利用を ortho 地表に重ねて田畑・草地・内陸水を明確化
    # （写真色の上に意味カテゴリを反映。森/市街は ortho の細かい色のまま残す）
    if cover_ds is not None and cover_ds.shape == surf_block.shape:
        land_esa = ~np.isnan(dem_ds) & ~(np.where(np.isnan(dem_ds), 0.0, dem_ds) <= sea_level_m)
        surf_block[(cover_ds == 40) & land_esa] = "coarse_dirt"  # cropland 田畑(耕地)
        surf_block[(cover_ds == 30) & land_esa] = "grass"        # grassland 草地
        surf_block[(cover_ds == 80) & land_esa] = "water"        # 内陸水面

    # 海岸: ortho 地表でも海岸線(dist_shore 小・低地)を砂浜/礫浜/護岸に（海岸ののっぺり解消）
    if dist_shore is not None and dist_shore.shape == surf_block.shape:
        elev0 = np.where(np.isnan(dem_ds), 999.0, dem_ds)
        gentle = slope_ds < SLOPE_STEEP
        coast_g = land_mask & (dist_shore <= SHORE_GRAVEL_M) & (elev0 < 3.5) & gentle
        surf_block[coast_g] = "gravel"                       # 礫浜（中距離）
        beach = land_mask & (dist_shore <= SHORE_SAND_M) & (elev0 < 3.0)
        surf_block[beach & gentle] = "sand"                  # 砂浜（最近・緩斜面）
        surf_block[beach & ~gentle] = "stone"                # 護岸/磯（最近・急斜面）

    # 道路を地表に上書き（陸セルのみ）。細道=road_minor_block(砂利)、幹線=road_block(舗装)
    if road_mask is not None and road_mask.shape == surf_block.shape:
        land_for_road = ~np.isnan(dem_ds) & ~(np.where(np.isnan(dem_ds), 0.0, dem_ds) <= sea_level_m)
        surf_block[road_mask & land_for_road] = road_minor_block
        if road_major_mask is not None and road_major_mask.shape == surf_block.shape:
            surf_block[road_major_mask & land_for_road] = road_block

    # FG-GML 水域(WA/WStrA: 河川・池等)を地表に水面として上書き（陸セルのみ。海は別途 sea_mask）
    if water_mask is not None and water_mask.shape == surf_block.shape:
        land_for_water = ~np.isnan(dem_ds) & ~(np.where(np.isnan(dem_ds), 0.0, dem_ds) <= sea_level_m)
        surf_block[water_mask & land_for_water] = water_block

    # 軸6-2: 単調な岩/礫/耕地クラスを世界座標ハッシュで重み付き混合にディザ（道路/水域/砂浜
    # など意図的な上書きの後に適用し、それらは混ぜない）。cell_offset で世界座標に整合。
    if dither_surface:
        apply_surface_dither(surf_block, cell_offset)

    valid_elevs = dem_ds[~np.isnan(dem_ds)]
    max_elev_y = (int(valid_elevs.max() * scale_land) if len(valid_elevs) > 0 else 1) + _lift
    max_y = min(max(max_elev_y + 5, y_sea_surface + 2), 500)

    # ─── 6) ブロック生成（施策③: append を密配列シムへ。後勝ちは上書きで自然再現） ───
    BZ, BX = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    blocks = _DenseBlockSink(nx, nz, y_cap=501)

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
        # 海底直下に少しの stone 地盤（凡例層より下には伸ばさない＝_lift で下限）
        base_y = max(_lift, floor_y - 3)
        for dy in range(base_y, floor_y):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(dy), nbtlib.Int(bz_v)]),
                "state": block_id("stone"),
            }))
        # 一番下に土台層（各柱の最下の1個下＝地形の起伏に沿う。海底の砂/砂利が浮かないよう支える）
        if base_y - 1 >= _lift:
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(base_y - 1), nbtlib.Int(bz_v)]),
                "state": block_id("deepslate"),
            }))
        # 水柱（floor+1 ～ sea_surface）
        for fy in range(floor_y + 1, y_sea_surface + 1):
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                "state": block_id("water"),
            }))

    # --- 陸セル：可変アンダーフィル(arnis ground_generation.rs:716-758 移植) + 地表 ---
    # 固定 deep_ground 本ではなく、8近傍の最低地表 Y までを stone で埋める。平地は
    # neigh_min≈y_top → 2 ブロックで済みブロック数が激減し、崖面は隣接セルの底まで
    # 埋めて見える穴を塞ぐ。深さは [2, deep_ground] にクランプ（従来より常に少ない）。
    from scipy.ndimage import minimum_filter as _min_filter
    _yfm = np.where(land_mask, y_surf_land.astype(np.float32), np.float32(1e9))
    _neigh_min = _min_filter(_yfm, size=3, mode="nearest")
    _under_depth = np.clip(
        y_surf_land.astype(np.int32) - _neigh_min.astype(np.int32) + 1,
        2, int(max(2, deep_ground))).astype(np.int32)
    land_idx = np.argwhere(land_mask)
    for j, i_ in land_idx.tolist():
        bx_v = int(BX[j, i_]); bz_v = int(BZ[j, i_])
        y_top = int(y_surf_land[j, i_])
        # 地盤柱（凡例層より下には伸ばさない＝_lift で下限。深さは近傍最低Yまで可変）
        for dy in range(max(_lift, y_top - int(_under_depth[j, i_])), y_top):
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

    # --- 建物（陸セルのみ）。高さは DSM 由来 building_height_patch があれば実測、無ければ一律。
    #     壁=wall_block + 階ごとの窓 window_block、屋根トップ=オルソ色（color_building_roofs）。 ---
    if building_mask is not None and building_mask.shape == dem_ds.shape:
        default_bh = max(2, int(round(building_height_m * scale_land)))
        fh = max(2, int(floor_height))
        # --- 空洞化の事前計算: 周壁/角/ドア + 棟ごとの軒高（寄棟の屋根裏が筒抜けになるのを塞ぐ） ---
        # 各セルの建物高さ[block]（屋根勾配で内側が高い）
        if bh_ds is not None:
            _bbg = np.clip(np.round(bh_ds * scale_land), 2, 60)
            bh_blocks_grid = np.where(np.isfinite(bh_ds), _bbg, default_bh).astype(np.int32)
        else:
            bh_blocks_grid = np.full(building_mask.shape, default_bh, dtype=np.int32)
        eave_blocks_by_bid: dict = {}
        base_y_arr = None
        if hollow_buildings and building_id is not None:
            _pad = np.pad(building_id, 1, constant_values=-2)
            _c = _pad[1:-1, 1:-1]
            _diff_y = (_pad[:-2, 1:-1] != _c) | (_pad[2:, 1:-1] != _c)
            _diff_x = (_pad[1:-1, :-2] != _c) | (_pad[1:-1, 2:] != _c)
            wall_cell = building_mask & (_diff_x | _diff_y)      # 外周＝壁
            corner_cell = building_mask & _diff_x & _diff_y       # 角＝窓を置かず柱に
            wall_x = building_mask & _diff_x & ~_diff_y           # E/W面（壁はz方向に走る）
            wall_y = building_mask & _diff_y & ~_diff_x           # N/S面（壁はx方向に走る）
            # 棟ごとの軒高 = 周壁セルの最小高さ。これより上は中実の屋根にして屋根裏の筒抜けを塞ぐ
            for _j, _i in np.argwhere(wall_cell).tolist():
                _b = int(building_id[_j, _i]); _v = int(bh_blocks_grid[_j, _i])
                if _b not in eave_blocks_by_bid or _v < eave_blocks_by_bid[_b]:
                    eave_blocks_by_bid[_b] = _v
            # 棟ごとの基準地盤 = footprint 内の最高 y_surf_land。平らな床に載せ、傾斜地の段差を基礎で埋める
            _maxid = int(building_id.max())
            if _maxid >= 0:
                base_y_arr = np.zeros(_maxid + 1, dtype=np.int32)
                _bs = building_mask & land_mask & (building_id >= 0)
                np.maximum.at(base_y_arr, building_id[_bs], y_surf_land[_bs].astype(np.int32))
            # ドア: 内側(空洞)を持つ棟だけ、可能なら道路側の周壁に1枚
            _has_interior = set(building_id[building_mask & ~wall_cell].tolist())
            door_cell = np.zeros_like(building_mask)              # 1棟1枚の出入口
            _seen: set = set()
            _passes = []
            if road_mask is not None and road_mask.shape == building_mask.shape:
                _passes.append(binary_dilation(road_mask) & wall_cell & ~corner_cell)
            _passes.append(wall_cell & ~corner_cell)
            for _m in _passes:
                for _j, _i in np.argwhere(_m).tolist():
                    _b = int(building_id[_j, _i])
                    if _b in _has_interior and _b not in _seen:
                        _seen.add(_b); door_cell[_j, _i] = True
        else:
            wall_cell = building_mask
            corner_cell = np.zeros_like(building_mask)
            wall_x = np.zeros_like(building_mask)
            wall_y = np.zeros_like(building_mask)
            door_cell = np.zeros_like(building_mask)
        # P2: 屋根を1棟の代表色に寄せる。ただし単色だと不自然なので、代表色から
        #     RGB 距離 roof_color_tol 以内（=同系統の濃淡）はセルの色を残し、外れ色
        #     （木の緑・隣家の別色など speckle）だけ代表色へスナップする。
        #     color_building_roofs 無効/未集約は type 由来の屋根キー（単色）。
        roof_by_id = None          # 各建物の代表屋根キー
        roof_dom_rgb = None        # 代表屋根キーの RGB（同系統判定用）
        if building_id is not None and building_roof_keys is not None:
            roof_by_id = list(building_roof_keys)
            if color_building_roofs:
                from collections import Counter
                from block_palette import BLOCKS as _BP
                bsel = (building_id >= 0) & building_mask & land_mask
                acc: dict[int, Counter] = {}
                for _id, _c in zip(building_id[bsel].tolist(),
                                   np.asarray(surf_block)[bsel].tolist()):
                    acc.setdefault(_id, Counter())[_c] += 1
                for _id, c in acc.items():
                    if 0 <= _id < len(roof_by_id):
                        roof_by_id[_id] = c.most_common(1)[0][0]
                roof_dom_rgb = [(_BP[k][1] if k in _BP else (128, 128, 128))
                                for k in roof_by_id]
        _tol2 = float(roof_color_tol) * float(roof_color_tol)
        from block_palette import BLOCKS as _BP2
        b_idx = np.argwhere(building_mask & land_mask)
        b_max_y = 0
        for j, i_ in b_idx.tolist():
            bx_v = int(BX[j, i_]); bz_v = int(BZ[j, i_])
            y_top = int(y_surf_land[j, i_])
            bh_blocks = int(bh_blocks_grid[j, i_])
            bid_c = int(building_id[j, i_]) if building_id is not None else -1
            # 棟を平らな基準面に載せる（傾斜地でも段差なし＝壁/屋根の筒抜けを防ぐ）
            y_base = (int(base_y_arr[bid_c]) if (base_y_arr is not None and 0 <= bid_c < len(base_y_arr))
                      else y_top)
            top_y = y_base + bh_blocks
            ceil_y = y_base + int(eave_blocks_by_bid.get(bid_c, bh_blocks))   # 軒高（上は中実の屋根）
            if top_y > b_max_y:
                b_max_y = top_y
            if roof_dom_rgb is not None and 0 <= bid_c < len(roof_by_id):
                # color_building_roofs 経路: 同系統の濃淡は残し、外れ色だけ代表色へ
                cell_k = surf_block[j, i_]
                dom_k = roof_by_id[bid_c]
                if cell_k == dom_k:
                    roof_kind = dom_k
                else:
                    cr = _BP2[cell_k][1] if cell_k in _BP2 else (128, 128, 128)
                    dr = roof_dom_rgb[bid_c]
                    dist2 = (cr[0]-dr[0])**2 + (cr[1]-dr[1])**2 + (cr[2]-dr[2])**2
                    roof_kind = cell_k if dist2 <= _tol2 else dom_k
            elif roof_by_id is not None and 0 <= bid_c < len(roof_by_id):
                roof_kind = roof_by_id[bid_c]                     # type 屋根（単色）
            else:
                roof_kind = surf_block[j, i_] if color_building_roofs else "stone"
            if building_wall_keys is not None and 0 <= bid_c < len(building_wall_keys):
                wall_kind = building_wall_keys[bid_c]             # type 別の壁
            else:
                wall_kind = wall_block
            # 屋根/軒に重力ブロック(砂利/砂/赤砂)が来ると落ちるので、非重力の同系色へ置換
            roof_kind = {"gravel": "andesite", "sand": "sandstone",
                         "red_sand": "orange_terracotta"}.get(roof_kind, roof_kind)
            style = (building_style_keys[bid_c]
                     if (building_style_keys is not None and 0 <= bid_c < len(building_style_keys))
                     else "house")
            is_wall = bool(wall_cell[j, i_])
            is_corner = bool(corner_cell[j, i_])
            is_door = bool(door_cell[j, i_])
            light_here = (bx_v % 5 == 2) and (bz_v % 5 == 2)   # 疎な内側格子に照明
            attic = hollow_buildings and (ceil_y < top_y)        # 寄棟で軒より上に屋根裏がある棟
            # 基礎: 地盤(y_top)から基準面(y_base)まで壁材で充填（傾斜地の段差を塞ぐ）
            for fy in range(y_top + 1, y_base + 1):
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                    "state": block_id(wall_kind),
                }))
            for fy in range(y_base + 1, top_y + 1):
                rel = fy - (y_base + 1)
                r = rel % fh                       # 階内位置 0..fh-1（0=床スラブ位置）
                is_slab = (r == 0 and fy != y_base + 1)
                if fy == top_y:
                    kind = roof_kind                                      # 屋根面（全 footprint）
                elif attic and fy > ceil_y:
                    kind = roof_kind                                      # 軒より上＝屋根裏を中実化(筒抜け防止)
                elif is_slab or (attic and fy == ceil_y):
                    # 各階の床 / 軒の天井（全 footprint）。疎な内側格子は天井灯
                    kind = interior_light if (not is_wall and light_here) else floor_block
                elif not hollow_buildings:
                    kind = window_block if (r == 2 and fy < top_y - 1) else wall_kind  # 旧ソリッド
                elif is_wall:
                    if is_door and fy <= y_base + 2:
                        continue                                         # 出入口（接地2マス開口）
                    _run = bz_v if wall_x[j, i_] else bx_v                # 壁の走る向きに沿って窓を間引く
                    kind = (window_block if (not is_corner and _is_window(style, r, fh, _run))
                            else wall_kind)                              # ガラス窓 or 壁
                else:
                    # 内側＝空洞。平屋根の最上階だけ屋根下に灯を吊る
                    if light_here and fy == top_y - 1:
                        kind = interior_light
                    else:
                        continue
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                    "state": block_id(kind),
                }))
            # 軒(庇): 屋根レベルを footprint 外1ブロックに張り出す（壁より屋根が出る家らしさ）
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i_ + di, j + dj
                if 0 <= ni < nx and 0 <= nj < nz and \
                        (building_id is None or building_id[nj, ni] < 0):
                    blocks.append(nbtlib.Compound({
                        "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(int(BX[nj, ni])),
                                                          nbtlib.Int(top_y), nbtlib.Int(int(BZ[nj, ni]))]),
                        "state": block_id(roof_kind),
                    }))
        max_y = max(max_y, b_max_y + 2)

    # --- 樹木（LiDAR class3 由来）。建物・道路・水域・海(land_mask)には立てない。
    #     tree_mode: "canopy"=セル毎に幹+葉(密な森) / "sparse"=間引いた個別樹木(球状樹冠) ---
    tree_cells = None                                # 凡例レイヤ用に「木のセル」を保持
    if tree_ds is not None and tree_ds.shape == dem_ds.shape:
        # 建物＋軒(8近傍1ます外周)を樹木禁止に（軒/角の張り出しに葉がめり込むのを防ぐ＝壁優先）
        _bld_dil = (binary_dilation(building_mask, structure=np.ones((3, 3), bool), iterations=1)
                    if building_mask is not None else np.zeros(dem_ds.shape, dtype=bool))
        no_tree = _bld_dil.copy()
        if road_mask is not None and road_mask.shape == dem_ds.shape:
            no_tree |= road_mask
        if water_mask is not None and water_mask.shape == dem_ds.shape:
            no_tree |= water_mask
        cand = (tree_ds >= 2.0) & land_mask & ~no_tree
        # 空中写真(surf_block)で周囲5ブロックに緑系(草/葉/苔)が無い所には木を置かない
        # （class3 が建物影・ノイズで誤検出した非植生に木が立つのを防ぐ）
        from block_palette import BLOCKS as _BPg
        green = np.zeros(dem_ds.shape, dtype=bool)
        for key in np.unique(surf_block):
            v = _BPg.get(key)
            if v is not None:
                r0, g0, b0 = v[1]
                if g0 > r0 and g0 > b0 and g0 > 55:   # 緑っぽい地表
                    green |= (surf_block == key)
        if green.any():
            cand &= binary_dilation(green, iterations=5)
        tree_cells = cand
        t_max_y = 0

        def _putt(ix, iy, iz, key):
            if 0 <= ix < nx and 0 <= iz < nz and 0 <= iy <= 500:
                if _bld_dil[iz, ix]:
                    return                                   # 壁優先: 建物・軒には樹木(葉/幹)を置かない
                blocks.append(nbtlib.Compound({
                    "pos": nbtlib.List[nbtlib.Int]([nbtlib.Int(int(ix)), nbtlib.Int(int(iy)), nbtlib.Int(int(iz))]),
                    "state": block_id(key)}))

        def _species(th_m):
            # 高さで樹種: 低木=明るい茂み(birch) / 中木=広葉(oak) / 高木=針葉(spruce, 円錐)
            if th_m < 4.0:
                return "oak_log", "birch_leaves", "bush"
            if th_m < 8.0:
                return "oak_log", "oak_leaves", "round"
            return "spruce_log", "spruce_leaves", "cone"

        if tree_mode == "sparse":
            step = max(2, int(round(2.5 / h_res_block)))   # ~2.5m 間隔（森林を密に）
            rows = np.arange(nz)[:, None]; cols = np.arange(nx)[None, :]
            # 行帯ごとに半ステップずらして隣列を斜めにずらす（千鳥配置で自然な森に）
            offset = ((rows // step) % 2) * (step // 2)
            sel = cand & ((rows % step) == 0) & (((cols - offset) % step) == 0)
            for j, i_ in np.argwhere(sel).tolist():
                th_m = float(tree_ds[j, i_])
                th = max(2, min(int(round(th_m * scale_land)), 30))
                log_k, leaf_k, shape = _species(th_m)
                y0 = int(y_surf_land[j, i_]); top = y0 + th
                if shape == "bush":                          # 低木: 地表から葉を接地(幹なし, canopy風)
                    for fy in range(y0 + 1, top + 1):        # 中心は地表から樹冠高まで葉柱
                        _putt(i_, fy, j, leaf_k)
                    for dj in (-1, 0, 1):                    # 上部を横に広げて隣と繋ぐ
                        for di in (-1, 0, 1):
                            _putt(i_ + di, top, j + dj, leaf_k)
                            if th >= 3:
                                _putt(i_ + di, top - 1, j + dj, leaf_k)
                    t_max_y = max(t_max_y, top)
                elif shape == "cone":                        # 針葉: 幹+円錐樹冠
                    ch = max(3, th * 2 // 3); base = top - ch
                    for fy in range(y0 + 1, base + 1):
                        _putt(i_, fy, j, log_k)
                    for li, cy in enumerate(range(base, top + 1)):
                        rr = max(0, int(round((1.0 - li / max(1, ch)) * 2)))
                        for dj in range(-rr, rr + 1):
                            for di in range(-rr, rr + 1):
                                if di*di + dj*dj <= rr*rr + 1:
                                    _putt(i_ + di, cy, j + dj, leaf_k)
                    _putt(i_, top + 1, j, leaf_k)
                    t_max_y = max(t_max_y, top + 1)
                else:                                        # 中木: 幹+球状樹冠
                    r = 2 if th >= 8 else 1
                    for fy in range(y0 + 1, top - r + 1):
                        _putt(i_, fy, j, log_k)
                    cyc = top - r
                    for dj in range(-r, r + 1):
                        for di in range(-r, r + 1):
                            for dy in range(-r, r + 1):
                                if di*di + dj*dj + dy*dy <= r*r + 1:
                                    _putt(i_ + di, cyc + dy, j + dj, leaf_k)
                    t_max_y = max(t_max_y, top + 1)
        else:  # canopy（既定）: セル毎に幹+葉。葉/幹は高さの樹種で
            for j, i_ in np.argwhere(cand).tolist():
                th_m = float(tree_ds[j, i_])
                th = max(2, min(int(round(th_m * scale_land)), 30))
                log_k, leaf_k, _ = _species(th_m)
                y0 = int(y_surf_land[j, i_]); top = y0 + th
                trunk_top = y0 + max(1, th // 2)
                for fy in range(y0 + 1, top + 1):
                    _putt(i_, fy, j, log_k if fy <= trunk_top else leaf_k)
                t_max_y = max(t_max_y, top)
        max_y = max(max_y, t_max_y + 2)

    # --- 橋（OSM bridge を Tellus 流に立体化）。最後に置いて水上で優先させる。 ---
    if bridges and patch_bbox_latlon is not None:
        bridge_ymax = add_bridge_blocks(
            blocks, bridges, patch_bbox_latlon, nz, nx,
            y_surf_land=y_surf_land, sea_mask=sea_mask,
            y_sea_surface=y_sea_surface, y_sea_floor=y_sea_floor,
            scale_land=scale_land, h_res_block_m=h_res_block,
            surf_block=surf_block, y_flood_top=y_flood_top,
        )
        max_y = max(max_y, bridge_ymax + 2)

    # --- 避難所マーカー（国土数値情報 P20）。地表から緑柱+発光で遠くから視認 ---
    if evac_facilities and patch_bbox_latlon is not None:
        EVAC_H = 28
        for ev in evac_facilities:
            x_, z_ = _lonlat_to_grid_xy(ev["lat"], ev["lon"], patch_bbox_latlon, nz, nx)
            ix, iz = int(round(x_)), int(round(z_))
            if not (0 <= ix < nx and 0 <= iz < nz):
                continue
            y0v = y_surf_land[iz, ix]
            if not np.isfinite(y0v):
                continue
            y0 = int(y0v)
            for fy in range(y0 + 1, y0 + EVAC_H):
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(ix), nbtlib.Int(fy), nbtlib.Int(iz)]),
                    "state": block_id("lime_concrete"),
                }))
            for ty in (y0 + EVAC_H, y0 + EVAC_H + 1):   # 頂部の発光2段
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(ix), nbtlib.Int(ty), nbtlib.Int(iz)]),
                    "state": block_id("sea_lantern"),
                }))
            max_y = max(max_y, y0 + EVAC_H + 2)

    # --- 土台層: 全セルの y=_lift-1 に1段（海底の砂/砂利等がブロック更新で落ちないよう下から支える）---
    #     deepslate（割れる）なので、これより下の凡例層は掘って到達できる。
    _found_y = _lift - 1
    if _found_y >= 0:
        for j in range(nz):
            for i_ in range(nx):
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(int(i_)), nbtlib.Int(int(_found_y)), nbtlib.Int(int(j))]),
                    "state": block_id("deepslate"),
                }))

    # --- 地下データ層: 土地利用の解釈を色付きガラスで表面化（最後に置いて後勝ちで露出）。光源なし。---
    #     重なる洪水・樹木は層を分けて別の高さに置く（コマンドで各層を独立に読み取れる）。間隔をあけて配置:
    #       y=0 土地利用ベース（建物/道路/海/河川/橋/地表）  y=2 洪水  y=4 樹木。地形は _lift で上に退避。
    if legend_layer:
        WHITE = "white_stained_glass"

        def _emit(layer_y, grid, full):
            """grid(object 配列, None=置かない) を y=layer_y に敷く。full=Trueは全セル。"""
            it = (np.argwhere(grid != None).tolist() if not full          # noqa: E711
                  else ((j, i_) for j in range(nz) for i_ in range(nx)))
            for j, i_ in it:
                k = grid[j, i_]
                if k is None:
                    continue
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(int(i_)), nbtlib.Int(int(layer_y)), nbtlib.Int(int(j))]),
                    "state": block_id(k)}))

        # y=0 土地利用ベース（全セル。洪水・樹木は含めない）
        base = np.full((nz, nx), WHITE, dtype=object)            # 既定=地表(白)
        if road_mask is not None and road_mask.shape == (nz, nx):
            base[road_mask] = "black_stained_glass"
        if water_mask is not None and water_mask.shape == (nz, nx):
            base[water_mask & ~sea_mask] = "light_blue_stained_glass"   # 河川/池
        base[sea_mask] = "blue_stained_glass"
        if building_mask is not None and building_mask.shape == (nz, nx):
            base[building_mask] = "red_stained_glass"
        if bridges and patch_bbox_latlon is not None:
            _bm = np.zeros((nz, nx), dtype=bool)
            for _b in bridges:
                _c = _b.get("coords")
                if not _c or len(_c) < 2:
                    continue
                _buf = max(1.0, (float(_b.get("width_m") or 5.5) / max(h_res_block, 0.1)) / 2.0)
                _bm |= polyline_buffer_mask_from_latlon(_c, patch_bbox_latlon, nz, nx, buffer_cells=_buf)
            base[_bm] = "orange_stained_glass"
        _emit(LEGEND_YS[0], base, full=True)

        # 洪水（全セル: 浸水=水色 / 非浸水=白）。全敷きで地形ブロックを上書きし純粋な二値マップに
        flood_grid = np.full((nz, nx), WHITE, dtype=object)
        flood_grid[(idn_ds > flood_threshold) & ~sea_mask] = "light_blue_stained_glass"
        _emit(LEGEND_YS[1], flood_grid, full=True)

        # 樹木（全セル: 木=緑 / それ以外=白）
        tree_grid = np.full((nz, nx), WHITE, dtype=object)
        if tree_cells is not None and tree_cells.shape == (nz, nx):
            tree_grid[tree_cells] = "green_stained_glass"
        _emit(LEGEND_YS[2], tree_grid, full=True)

    return blocks.array(max_y + 1), [nx, max_y + 1, nz]
