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


# ── 調整可能な既定値（呼出側から引数で上書き可） ─────────────────
# トンネル: コア(OSM way 本体)区間の被覆判定を何 block 甘くするか。
# 0(既定) = 「構造(壁+アーチ天井)の頂部が地表以下」なら密閉。延長部の判定
# (tc > fy+CLEAR+SHELL) より 1block だけ緩い。大きくすると密閉を維持しやすくなるが、
# その分シェル頂部が地表に露出する（合成データ実測: slack 0→1 で山越えトンネルの
# 地表露出ブロックが 0→108、2 で 308）。
TUNNEL_CORE_COVER_SLACK = 0
# トンネル: 被覆判定を station 方向へ closing する長さ[block]。密閉に挟まれた
# これ未満の非密閉ギャップは密閉へ戻す（山中の小さな谷/DEMノイズ対策）。0 で無効。
TUNNEL_COVER_CLOSE_BLOCKS = 8
# トンネル直上(被覆部=山の中)の“地表道路”を消す閾値[block]。トンネルは地下なので、
# 地表が床グレード(startF/endF or 坑口地表を長さ内挿)より この値以上 高いセルを「被覆部」
# とみなし road_mask から除去する（=山の上に道路を描かない・木も生える）。坑口/開削部は
# 地表≈床なので残り、トンネル回廊外の別の道路も残る。負値で無効。CLEAR+SHELL 相当。
TUNNEL_SURFACE_ROAD_COVER_MARGIN = 9
# 上記の回廊半幅に足す余裕[block]。実際の路面(オルソ舗装+路肩+RdEdg帯)は OSM の width_m
# より広いので、道路半幅に この値を足して回廊を広げ、路面全体を確実に覆う。片側車道どうしの
# 間隔(中央分離帯)より小さく保つこと(大きすぎると分離帯の森も消える)。
TUNNEL_SURFACE_ROAD_CORRIDOR_PAD = 8
# 橋デッキ直下の地表道路除去。True で「高架の路面が地面に二重に出る」のを消す。
# デッキ足元(半幅=道路半幅+PAD)のオルソ/road_mask を除くが、その足元にある別の地上道路
# (OSM 非bridge道路=road_curb_osm_mask のうち橋中心線から外れるもの)は残す。FGD RdEdg は
# 高架/地上を区別しないため OSM way 同定で切り分ける。
BRIDGE_UNDERROAD_REMOVE = True
BRIDGE_UNDERROAD_PAD = 4
# 地盤アンダーフィル: 隣接セルとの段差に応じた可変深さの下限/上限[block]。
# 上限は「段差 + UNDERFILL_EXTRA」でも足りない崖のための安全弁で、既定は
# 従来の deep_ground クランプを超えて崖の穴を塞げるよう十分大きく取る。
UNDERFILL_MIN = 2
UNDERFILL_EXTRA = 1
UNDERFILL_HARD_CAP = 96
# トンネル坑口(出入口)“周り”の下方向 増し厚。坑口は地表が道路レベルまで局所的に下げられ、
# 可変アンダーフィル(近傍最低段差ベース)では床下がほぼ埋まらずシェル内部の空洞(すきま)が
# 残りやすい。各トンネルの coords 両端(=坑口)を grid に投影し、坑口を中心に**軸方向**へ内側
# (トンネル側)REACH_IN・外側(進入路側)REACH_OUT[block] 伸ばした半径 RADIUS[block] のカプセル
# 内の陸セルのアンダーフィル深さを最低 DEPTH[block] へ引き上げて床下を stone で塞ぐ
# (UNDERFILL_HARD_CAP でクランプ)。傾斜が緩く「坑門(覆われる端)と入口(開削端)」が離れる区間も
# 軸方向に伸ばして覆う。刳り貫きは後段なので増し厚は坑口の周り/床下だけに残る。RADIUS=0 で無効。
TUNNEL_PORTAL_UNDERFILL_RADIUS = 22
TUNNEL_PORTAL_UNDERFILL_DEPTH = 18
TUNNEL_PORTAL_UNDERFILL_REACH_IN = 32   # 坑口から内側(トンネル側)へ延ばす長さ[block]
TUNNEL_PORTAL_UNDERFILL_REACH_OUT = 12  # 坑口から外側(進入路側)へ延ばす長さ[block]


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


def build_transverse_crossing_mask(
    bridges: list, roads: list, patch_bbox_latlon: tuple,
    grid_h: int, grid_w: int, h_res_block_m: float,
    deck_pad: float = 4.0, min_angle_deg: float = 35.0,
) -> np.ndarray:
    """橋デッキ足元(deck)で、**橋軸に対して横断方向(≥min_angle_deg)** に走る道路セルの
    bool マスクを返す。高架自身の並走 footprint(軸と平行)は落ち、真下/斜め下を横切る別道路
    (FGD 農道・庭園路や OSM 生活道路)だけが残る。roads は [{"coords":[[lat,lon],..],
    "width_m":w}, ..]。deck 近傍のセルのみ評価するので軽い。"""
    nz, nx = grid_h, grid_w
    JJ, II = np.mgrid[0:nz, 0:nx]
    JJ = JJ.astype(np.float32); II = II.astype(np.float32)

    def _pts(coords):
        return [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, nz, nx) for la, lo in coords]

    bang = np.full((nz, nx), np.nan, np.float32)   # 橋軸角[rad]（deck上）
    deck = np.zeros((nz, nx), bool)
    for b in (bridges or []):
        c = b.get("coords") or []
        if len(c) < 2:
            continue
        wm = float(b.get("width_m") or 5.5)
        hw = max(1.5, (wm / max(h_res_block_m, 0.1)) / 2.0 + deck_pad); hw2 = hw * hw
        bp = _pts(c)
        for k in range(len(bp) - 1):
            ax, az = bp[k]; bx, bz = bp[k + 1]
            dx, dz = bx - ax, bz - az; l2 = dx * dx + dz * dz
            if l2 < 1e-6:
                continue
            t = np.clip(((II - ax) * dx + (JJ - az) * dz) / l2, 0.0, 1.0)
            d2 = (II - (ax + t * dx)) ** 2 + (JJ - (az + t * dz)) ** 2
            m = d2 <= hw2
            deck |= m
            bang[m] = np.arctan2(dz, dx)
    if not deck.any():
        return np.zeros((nz, nx), bool)
    zs, xs = np.where(deck)
    z0, z1 = zs.min() - 6, zs.max() + 6
    x0, x1 = xs.min() - 6, xs.max() + 6

    rang = np.full((nz, nx), np.nan, np.float32)    # 道路角[rad]（deck上）
    for r in (roads or []):
        c = r.get("coords") or []
        if len(c) < 2:
            continue
        wm = float(r.get("width_m") or 4.0)
        hw = max(1.0, (wm / max(h_res_block_m, 0.1)) / 2.0); hw2 = hw * hw
        rp = _pts(c)
        for k in range(len(rp) - 1):
            ax, az = rp[k]; bx, bz = rp[k + 1]
            if max(ax, bx) < x0 or min(ax, bx) > x1 or max(az, bz) < z0 or min(az, bz) > z1:
                continue                              # deck から遠い区間は skip（高速化）
            dx, dz = bx - ax, bz - az; l2 = dx * dx + dz * dz
            if l2 < 1e-6:
                continue
            t = np.clip(((II - ax) * dx + (JJ - az) * dz) / l2, 0.0, 1.0)
            d2 = (II - (ax + t * dx)) ** 2 + (JJ - (az + t * dz)) ** 2
            m = (d2 <= hw2) & deck
            rang[m] = np.arctan2(dz, dx)
    have = (~np.isnan(rang)) & (~np.isnan(bang))
    diff = np.abs(rang - bang)
    diff = np.mod(diff, np.pi)                       # 無向線の角度差 → [0, pi/2]
    diff = np.minimum(diff, np.pi - diff)
    return have & (diff >= np.deg2rad(min_angle_deg))


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


# ── 建物アーキタイプ（外壁装飾用）─────────────────────────────────────────────
#   御坊は OSM に階数/種別/roof情報がほぼ無い(building=yes 97%)。そこで
#   屋根形状(=LiDAR DSM の起伏 relief) × 高さ × footprint面積 × FGD種別 を
#   「建物種別の代理特徴」にしてアーキタイプを推定し、種別ごとの外壁スタイル
#   (壁材/トリム/窓/窓パターン/parapet/店頭/床ライン)を割り当てる。屋根スラブ・
#   内装は現状維持(ユーザ方針)。arnis の facade 表現(角柱/床帯/連窓/parapet)を移植。
ARCHETYPE_STYLE = {
    # archetype:    wall                     trim                   window                       style           parapet shopfront floor_band
    "wood_house":   dict(wall="white_terracotta",     trim="stripped_oak_log",    window="glass",                  style="wood_house",    parapet=0, shopfront=False, floor_band=False),
    "apartment":    dict(wall="white_concrete",       trim="light_gray_concrete", window="light_blue_stained_glass", style="apartment",   parapet=1, shopfront=False, floor_band=True),
    "shop":         dict(wall="light_gray_terracotta",trim="gray_terracotta",     window="glass",                  style="shop",          parapet=1, shopfront=True,  floor_band=True),
    "rc_building":  dict(wall="light_gray_concrete",  trim="white_concrete",      window="light_blue_stained_glass", style="rc",          parapet=2, shopfront=False, floor_band=True),
    "warehouse":    dict(wall="gray_concrete",        trim="andesite",            window="glass",                  style="warehouse",     parapet=1, shopfront=False, floor_band=False),
    "institutional":dict(wall="white_concrete",       trim="light_gray_terracotta",window="glass",                 style="institutional", parapet=1, shopfront=False, floor_band=True),
}

_WAREHOUSE_FGD = ("普通無壁舎", "堅ろう無壁舎", "工場", "倉庫", "農業施設")
_RC_FGD = ("堅ろう建物", "高層建物", "商業ビル", "公共", "ランドマーク", "マンション", "宿泊")

# 御坊の約95%を占める wood_house の外壁/トリムが white_terracotta + stripped_oak_log 一色に
# なるのを防ぐため、落ち着いた住宅外壁の候補から **座標決定的に** 多様化する。
#   - 決定的: footprint 重心(緯度経度)のハッシュで選ぶ → 同じ土地・建物なら毎回同結果。
#   - 反偏り: splitmix64 ミックス + 均等 mod でほぼ一様分布。
#   - 反隣接(ゆるめ): 重心ハッシュは非相関なので隣家同士は概ね別材になる(絶対回避はしない)。
#   - 壁とトリムは別ソルトで独立に選ぶ → 万一壁が一致しても見た目が被りにくい。
# 他アーキタイプ(apartment/shop/rc/warehouse/institutional)は対象外（方針: 変更不要）。
_HOUSE_WALL_VARIANTS = ("white_terracotta", "light_gray_terracotta", "white_concrete",
                        "sandstone", "light_gray_concrete", "clay")
_HOUSE_TRIM_VARIANTS = ("stripped_oak_log", "oak_log", "spruce_log", "dark_oak_planks")


def _det_hash64(*vals) -> int:
    """プロセス非依存の決定的整数ハッシュ（splitmix64 ミックス）。Python の hash() は
    文字列がプロセス毎ソルトで非決定なので使わない。"""
    M = (1 << 64) - 1
    h = 0x9E3779B97F4A7C15
    for v in vals:
        h = ((h ^ (int(v) & M)) * 0xBF58476D1CE4E5B9) & M
        h ^= h >> 30
        h = (h * 0x94D049BB133111EB) & M
        h ^= h >> 27
    return h & M


def _house_facade_variant(spec: dict, ext) -> dict:
    """wood_house の壁/トリムを footprint 重心(緯度経度)から決定的に選んだ spec を返す。"""
    arr = np.asarray(ext, dtype=np.float64)
    qlat = int(round(float(arr[:, 0].mean()) * 1e5))   # ≈1m 量子化した重心緯度
    qlon = int(round(float(arr[:, 1].mean()) * 1e5))   # 〃 経度
    wall = _HOUSE_WALL_VARIANTS[_det_hash64(qlat, qlon, 0xA17) % len(_HOUSE_WALL_VARIANTS)]
    trim = _HOUSE_TRIM_VARIANTS[_det_hash64(qlat, qlon, 0xB29) % len(_HOUSE_TRIM_VARIANTS)]
    return {**spec, "wall": wall, "trim": trim}


def building_archetype(tp: str, h_m: float, relief_m: float, area_m2: float) -> str:
    """屋根形状(relief)×高さ×規模×FGD種別から建物アーキタイプを推定。
    relief_m = 屋根面の起伏(DSM p90-p10): 小さい=陸屋根, 大きい=勾配(切妻/寄棟)。"""
    flat = relief_m < 1.5
    if tp in _WAREHOUSE_FGD:
        return "warehouse"
    if tp in _RC_FGD or h_m >= 11.0:
        return "institutional" if (area_m2 >= 700 and h_m < 13) else "rc_building"
    # ここから先は 普通建物/住宅系(=御坊の約95%)を 形状×規模×高さ で細分
    if area_m2 >= 600 and flat:
        return "institutional"
    if flat and area_m2 >= 300 and h_m < 8.0:
        return "warehouse"                 # 大きく平らで低い=倉庫/作業場
    if flat and area_m2 >= 100:
        return "shop"
    if h_m >= 6.5 and area_m2 >= 110:
        return "apartment"
    return "wood_house"


def _is_window(style: str, r: int, fh: int, run: int) -> bool:
    """周壁セルが窓(ガラス)になるか。r=階内の高さ位置(0=床ライン), fh=階高,
    run=壁が走る向きの座標。窓と窓の間に最低1〜2マスの壁を残す。アーキタイプ別:
    rc=全高の縦連窓 / apartment=各階3行 / shop・institutional・wood_house=各階2行 /
    warehouse=高所ハイサイド窓のみ。"""
    if r <= 0 or r >= fh:                        # 床/天井ラインには窓を置かない
        return False
    if style in ("rc", "building"):              # ビル: 縦に連なるガラス列(3マスおき・全高)
        return run % 3 == 0
    if style in ("warehouse", "factory"):        # 倉庫/工場: 高所のハイサイド窓(疎)
        return (run % 4 == 0) and (r in (fh - 2, fh - 3))
    if style == "apartment":                     # 集合住宅: 各階 下3行に窓
        return (run % 3 == 0) and (r in (1, 2, 3))
    # shop / institutional / wood_house / house(既定): 各階 下2行に窓
    return (run % 3 == 0) and (r in (1, 2))


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
    facade_keys: list[dict] = []          # 建物 id → 外壁装飾スペック(アーキタイプ由来)
    roof_solid_keys: list[bool] = []      # 建物 id → 屋根を型単色にしオルソ焼込を無効化(新設建物用)
    # 1セルの実寸(㎡)= アーキタイプ判定の footprint 面積に使う
    _la0, _la1, _lo0, _lo1 = patch_bbox_latlon
    _cell_ns = (_la1 - _la0) * 111000.0 / max(grid_h, 1)
    _cell_ew = (_lo1 - _lo0) * 111000.0 * float(np.cos(np.radians(0.5 * (_la0 + _la1)))) / max(grid_w, 1)
    _cell_m2 = max(_cell_ns * _cell_ew, 1e-6)
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
        relief = 2.0 if tp in HIP_ROOF_TYPES else 0.6   # DSM無し時の推定(勾配/陸屋根)
        if dsm_h_block is not None:
            vals = dsm_h_block[y0:y1, x0:x1][ins]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                h = max(float(np.percentile(vals, pct)), floor)
            if vals.size >= 4:                          # 屋根面の起伏=形状の代理(陸屋根/勾配)
                relief = float(np.percentile(vals, 90) - np.percentile(vals, 10))
        h = max(h, min_h_m)
        area_m2 = float(int(ins.sum()) * _cell_m2)
        arch = building_archetype(tp, h, relief, area_m2)
        spec = ARCHETYPE_STYLE[arch]
        if arch == "wood_house":                 # 壁/トリムを座標決定的に多様化（他種別は不変）
            spec = _house_facade_variant(spec, ext)
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
        # 壁材・窓スタイルはアーキタイプ由来。屋根材は従来どおり type 由来(屋根は現状維持)。
        wall_keys.append(spec["wall"])
        roof_keys.append(BUILDING_ROOF_BY_TYPE.get(tp, DEFAULT_ROOF_KEY))
        style_keys.append(spec["style"])
        facade_keys.append(spec)
        roof_solid_keys.append(bool(b.get("tags", {}).get("roof_solid")))
        bid += 1
    return {"mask": mask, "height": hmap, "id": idmap,
            "wall_keys": wall_keys, "roof_keys": roof_keys, "style_keys": style_keys,
            "facade": facade_keys, "roof_solid": roof_solid_keys}


def assign_global_power_anchors(lines, dem_full, lat_max, lon_min, res_lat, res_lon,
                                *, scale_land, lift, search_cells: int = 8):
    """各送電線の頂点（=実鉄塔位置）の地表Y を **全域DEM** から計算して
    ``L["ground_y"] = [int|None, ...]``（頂点数と同数）として付与する（in-place）。

    add_power_blocks の ground_y はタイルローカル grid を見るため、--tiles 分割で径間の
    端点（鉄塔）がタイル外に出ると端点高が「線とタイルの交差区間の端の地形高」に化ける。
    ところが径間内部の位置決めは**全長基準**の媒介変数 f なので、タイル毎に別の直線を
    引くことになり、傾斜地では継ぎ目に段差が出て対地クリアランスも狂う（本番 halo は
    16 DEMセルしか無く、実鉄塔径間は数百 block なのでほぼ必ずクリップされる）。
    本関数は分割前に全域DEMで各頂点の地表Yを1回だけ計算し、全タイルが**同一の径間端点高**を
    参照できるようにする（橋の assign_global_bridge_anchors と同じ手当て）。
    y_surf_land = max(1, 標高×scale_land) + lift（terrain_render の地表Yと同式）。

    search_cells : 頂点セルの DEM が欠損(NaN)のとき、この半径[DEMセル]まで近傍を探す。
                   見つからなければ None を入れ、add_power_blocks はタイルローカルへ
                   フォールバックする（後方互換）。
    """
    H, W = dem_full.shape

    def ysl(row, col):
        if 0 <= row < H and 0 <= col < W:
            v = dem_full[row, col]
            if np.isfinite(v):
                return max(1, int(v * scale_land)) + lift
        return None

    def anchor(lat, lon):
        r = int(round((lat_max - lat) / res_lat))
        c = int(round((lon - lon_min) / res_lon))
        v = ysl(r, c)
        if v is not None:
            return int(v)
        for rad in range(1, int(max(1, search_cells)) + 1):     # NaN 穴は近傍リングで補完
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    v = ysl(r + dr, c + dc)
                    if v is not None:
                        return int(v)
        return None

    for L in (lines or []):
        cs = L.get("coords") or []
        L["ground_y"] = [anchor(float(la), float(lo)) for la, lo in cs]


def add_power_blocks(blocks, lines, towers, patch_bbox_latlon, nz, nx, *,
                     y_surf_land, sea_mask, scale_land,
                     wire_key="iron_bars", pylon_key="iron_bars",
                     clip_spans_to_grid: bool = True) -> int:
    """OSM 送電線（power=line/minor_line）+ 鉄塔/電柱（power=tower/pole）を立体化。

    - 電線: voltage → 基準高さ(実m×scale_land)。径間（頂点間）ごとにカテナリ（垂れ）で
      iron_bars 架線。両端の鉄塔頂を結ぶ直線から sag を引いて配置。
    - 鉄塔: 線の各頂点（=実鉄塔位置）に iron_bars ラティス柱+頂部クロスアーム。
      power=tower/pole の単独ノードも柱（tower=高,pole=低）として立てる（頂点と重複時は省略）。
    返り値: 置いた最大 y（max_y 更新用）。FG-GML に電力設備が無いため OSM を入力に使う。

    clip_spans_to_grid : True(既定) で、径間端点がタイル外のとき「線とタイルの交差区間の
        端」の地形高で代用し、タイル内に入る区間だけを描く。False で旧挙動（端点が
        タイル外の径間は丸ごと捨てる＝タイルを貫く送電線が消える）。
        ※ これは assign_global_power_anchors が使えないとき用のフォールバックで、
          タイルローカル高を端点に使う以上、傾斜地では継ぎ目に段差が残る。
          ``L["ground_y"]``（全域DEMアンカー）があればそちらが優先される。
    """
    import math
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

    def ground_y(x, z):
        i, j = int(round(x)), int(round(z))
        if 0 <= j < nz and 0 <= i < nx and np.isfinite(y_surf_land[j, i]):
            return int(y_surf_land[j, i])
        return None

    def span_ground_y(x, z, xo, zo):
        """径間端点 (x,z) の地形高。端点がタイル外/無効セルなら、対向端点 (xo,zo) 方向へ
        進んで**最初に有効になる点＝線とタイルの交差区間の端**の高さで代用する。

        これが無いと「タイルを貫くが両端の鉄塔がタイル外」の径間が丸ごと消える。
        power_osm._bbox_hit は線全体の bbox 重なりで拾うだけでジオメトリをクリップ
        しないため、端点がタイル外の coords がそのまま渡ってくるのが原因。
        径間内部のサンプルは f（全長基準の媒介変数）で位置決めしており、タイルを
        分割しても同じ f→同じ高さになるので継ぎ目で架線が繋がる。"""
        g = ground_y(x, z)
        if g is not None or not clip_spans_to_grid:
            return g
        dx, dz = xo - x, zo - z
        d = math.hypot(dx, dz)
        if d < 1e-9:
            return None
        n = max(1, min(int(d * 2) + 1, 4096))       # 0.5block 刻み（上限 4096 サンプル）
        for t in range(1, n + 1):
            f = t / n
            g = ground_y(x + dx * f, z + dz * f)
            if g is not None:
                return g
        return None

    def base_h_m(volt):
        if volt >= 500000:
            return 50.0
        if volt >= 220000:
            return 40.0
        if volt >= 110000:
            return 30.0
        if volt >= 60000:
            return 22.0
        if volt > 0:
            return 14.0
        return 12.0

    def pylon(ix, iz, gy, top, kind="pole"):
        """arnis 風の送電構造物（src/element_processing/power.rs 移植）。
        kind と実高さで形状・寸法を変える：高い/tower=テーパー格子鉄塔、低い/pole=単柱。
        パレット制約で IRON_BLOCK→light_gray_concrete, 格子/碍子→iron_bars, 基礎→gray_concrete,
        腕木→stripped_oak_log に対応付け。"""
        H = max(int(top - gy), 1)                 # 構造高さ[block]
        h_m = H / max(scale_land, 1e-6)            # 実高さ[m]
        lattice = (kind == "tower") or (h_m >= 16.0)
        if lattice:
            bw = int(round(min(max(h_m / 9.0, 1.0), 4.0)))   # 基部ハーフ幅(高いほど広い)
            tw = 1                                            # 頂部ハーフ幅
            arm_len = int(round(min(max(h_m / 7.0, 2.0), 6.0)))
            brace = max(int(round(5 * scale_land)), 3)        # 水平ブレース間隔
            arm_h = top - max(int(round(4 * scale_land)), 2)  # 上段アーム高
            # テーパー4脚 + 水平/中央対角ブレース
            for y in range(gy + 1, top + 1):
                p = (y - gy) / H
                cw = int(round(bw - (bw - tw) * p))
                if cw < 1:
                    put(ix, y, iz, "light_gray_concrete")     # 頂部付近は単柱化
                    continue
                for dx, dz in ((-cw, -cw), (cw, -cw), (-cw, cw), (cw, cw)):
                    put(ix + dx, y, iz + dz, "light_gray_concrete")
                lvl = y - gy
                if lvl % brace == 0 and y < top - 2:          # 水平ブレース
                    for d in range(-cw, cw + 1):
                        put(ix + d, y, iz - cw, "iron_bars"); put(ix + d, y, iz + cw, "iron_bars")
                        put(ix - cw, y, iz + d, "iron_bars"); put(ix + cw, y, iz + d, "iron_bars")
                elif lvl % brace in (1, brace - 1):           # 中央対角(ラティス感)
                    put(ix, y, iz, "iron_bars")
            # クロスアーム2段 + 端部碍子
            for ah, al in ((arm_h, arm_len),
                           (arm_h - max(int(round(5 * scale_land)), 3), max(arm_len - 1, 1))):
                if ah <= gy + 2:
                    continue
                for d in range(-al, al + 1):
                    put(ix + d, ah, iz, "light_gray_concrete")
                put(ix - al, ah + 1, iz, "iron_bars"); put(ix + al, ah + 1, iz, "iron_bars")
            put(ix, top, iz, "iron_bars")                     # 頂部(避雷針代替)
            for dx in range(-bw, bw + 1):                     # 基礎
                for dz in range(-bw, bw + 1):
                    put(ix + dx, gy, iz + dz, "gray_concrete")
        else:
            for y in range(gy + 1, top + 1):                  # コンクリート単柱
                put(ix, y, iz, "light_gray_concrete")
            al = 2                                            # 腕木(横木)
            for d in range(-al, al + 1):
                put(ix + d, top, iz, "stripped_oak_log")
            put(ix - al, top + 1, iz, "iron_bars")            # 端部+中央碍子
            put(ix + al, top + 1, iz, "iron_bars")
            put(ix, top + 1, iz, "iron_bars")

    pylon_cells: set = set()

    # ── 送電線 + 線頂点の鉄塔 ──
    for L in (lines or []):
        pts = [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, nz, nx) for la, lo in L["coords"]]
        bh = int(round(base_h_m(int(L.get("voltage", 0))) * scale_land))
        # 全域DEMアンカー（assign_global_power_anchors が付与）。頂点数が合うときだけ使う。
        ganch = L.get("ground_y")
        if not (isinstance(ganch, (list, tuple)) and len(ganch) == len(pts)):
            ganch = None
        # 各頂点（実鉄塔位置）に柱。柱の**足元**はタイルローカル地形（地面から浮かせない）、
        # **頂部**はアンカー基準（架線の端点と必ず一致させる）。
        vert_top = {}
        for vi, (x, z) in enumerate(pts):
            ix, iz = int(round(x)), int(round(z))
            gy = ground_y(x, z)
            if gy is None:
                continue
            ga = ganch[vi] if ganch is not None else None
            top = (int(ga) if ga is not None else gy) + bh
            vert_top[(ix, iz)] = top
            if (ix, iz) not in pylon_cells:
                pylon_cells.add((ix, iz))
                pylon(ix, iz, gy, top,
                      kind=("tower" if (bh / max(scale_land, 1e-6)) >= 16.0 else "pole"))
        # 径間ごとに架線（カテナリ）。両端鉄塔頂を線形補間し sag を引く
        for si in range(len(pts) - 1):
            (x0, z0), (x1, z1) = pts[si], pts[si + 1]
            g0 = ganch[si] if ganch is not None else None
            g1 = ganch[si + 1] if ganch is not None else None
            if g0 is None:
                # アンカー欠損時のみ: 端点がタイル外なら交差区間の端の地形高で代用
                g0 = span_ground_y(x0, z0, x1, z1)
            if g1 is None:
                g1 = span_ground_y(x1, z1, x0, z0)
            if g0 is None or g1 is None:
                continue
            g0, g1 = int(g0), int(g1)
            if ganch is not None:
                # 全域アンカー使用時は径間ジオメトリを量子化して浮動小数のタイル依存を消す。
                # 隣接タイルの (x,z) は整数オフセットだけ違うので、丸めると dist/f/xi が
                # bit 一致し、どのタイルで描いても同じ world 位置に同じ y が出る。
                x0, z0, x1, z1 = (round(v, 6) for v in (x0, z0, x1, z1))
            dist = math.hypot(x1 - x0, z1 - z0)
            n = int(dist) + 1
            max_sag = min(max(dist / 15.0, 1.0), 6.0)
            for t in range(n + 1):
                f = t / n if n > 0 else 0.0
                xi, zi = x0 + (x1 - x0) * f, z0 + (z1 - z0) * f
                gy = ground_y(xi, zi)
                if gy is None:
                    continue
                top = (g0 + bh) * (1.0 - f) + (g1 + bh) * f       # 両端鉄塔頂の直線
                wy = int(round(top - 4.0 * max_sag * f * (1.0 - f)))
                if wy <= gy:
                    wy = gy + 1
                put(int(round(xi)), wy, int(round(zi)), wire_key)

    # ── 単独ノード（線頂点に無い鉄塔/電柱） ──
    for tw in (towers or []):
        x, z = _lonlat_to_grid_xy(tw["lat"], tw["lon"], patch_bbox_latlon, nz, nx)
        ix, iz = int(round(x)), int(round(z))
        if (ix, iz) in pylon_cells:
            continue
        gy = ground_y(x, z)
        if gy is None:
            continue
        pylon_cells.add((ix, iz))
        is_tower = tw.get("kind") == "tower"
        h_m = 18.0 if is_tower else 10.0          # 単独柱: tower=小型格子鉄塔/pole=単柱
        pylon(ix, iz, gy, gy + int(round(h_m * scale_land)),
              kind=("tower" if is_tower else "pole"))

    return ymax[0]


def add_rail_blocks(blocks, rails, patch_bbox_latlon, nz, nx, *,
                    y_surf_land, sea_mask=None,
                    ballast_key="gravel", sleeper_key="spruce_log",
                    half_width: int = 1, sleeper_step: int = 2) -> int:
    """FG-GML RailCL（鉄道中心線）を **道床(gravel)＋枕木(spruce_log)＋本物のレール(minecraft:rail)** で
    立体化。arnis railways.rs の at-grade rail を 0.667m/block 用に幅を持たせて移植：
    中心線セル列に沿って幅 (2*half_width+1) の道床を敷き、枕木を sleeper_step セル毎に渡し、
    **中心線上に minecraft:rail を1本通す**（鉄格子=縦柵ではレール感が無いため実ブロックに変更）。
    レール向き(shape)は前後の連結セル方向から決め、直線(ns/ew)・曲線(ne/nw/se/sw)を出し分ける。
    曲線で連結が切れないよう中心線は 4 近傍連結に整える。種別分岐は持たず地表敷設のみ。
    返り値: 置いた最大 y。"""
    import math
    seen: set = set()
    ymax = [0]
    # 近傍セル方向（grid: +x=東 / +z=南）→ 方位名。rail の shape 名は連結 2 方向を表す。
    _DIRNAME = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}
    _SHAPE_KEY = {
        frozenset({"north", "south"}): "rail_ns", frozenset({"east", "west"}): "rail_ew",
        frozenset({"north"}): "rail_ns", frozenset({"south"}): "rail_ns",
        frozenset({"east"}): "rail_ew",  frozenset({"west"}): "rail_ew",
        frozenset({"north", "east"}): "rail_ne", frozenset({"north", "west"}): "rail_nw",
        frozenset({"south", "east"}): "rail_se", frozenset({"south", "west"}): "rail_sw",
    }

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

    def ground_y(x, z):
        i, j = int(round(x)), int(round(z))
        if 0 <= j < nz and 0 <= i < nx and np.isfinite(y_surf_land[j, i]):
            return int(y_surf_land[j, i])
        return None

    def four_connect(cs):
        """連続セル列を 4 近傍連結へ（斜め/飛びは x→z の順に 1 セルずつ階段で埋める）。
        これで隣接レールが必ず辺で接し、curved shape と連結が成立する。"""
        path: list = []
        for c in cs:
            if not path:
                path.append(c); continue
            px, pz = path[-1]
            cx, cz = c
            if (cx, cz) == (px, pz):
                continue
            sgx = 1 if cx > px else (-1 if cx < px else 0)
            sgz = 1 if cz > pz else (-1 if cz < pz else 0)
            while (px, pz) != (cx, cz):
                if px != cx:
                    px += sgx
                else:
                    pz += sgz
                path.append((px, pz))
        return path

    for R in (rails or []):
        pts = [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, nz, nx) for la, lo in R["coords"]]
        # 中心線を連続セル列へ（セグメントごとに線形補間し重複除去）
        cells: list = []
        for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
            dist = math.hypot(x1 - x0, z1 - z0)
            n = int(dist) + 1
            for t in range(n + 1):
                f = t / n if n > 0 else 0.0
                c = (int(round(x0 + (x1 - x0) * f)), int(round(z0 + (z1 - z0) * f)))
                if not cells or cells[-1] != c:
                    cells.append(c)
        path = four_connect(cells)
        for idx, (ix, iz) in enumerate(path):
            gy = ground_y(ix, iz)
            if gy is None:
                continue
            if sea_mask is not None and sea_mask[iz, ix]:
                continue
            # 進行方向（次優先, 端は前）→ 垂直ベクトル（道床/枕木の横展開に使う）
            if idx + 1 < len(path):
                jx, jz = path[idx + 1]
            elif idx > 0:
                jx, jz = path[idx - 1]
            else:
                jx, jz = ix + 1, iz
            ddx, ddz = jx - ix, jz - iz
            mag = math.hypot(ddx, ddz) or 1.0
            px, pz = -ddz / mag, ddx / mag
            # 道床（枕木セルは木、それ以外は砂利）を幅いっぱいに
            bed = sleeper_key if (idx % sleeper_step == 0) else ballast_key
            for w in range(-half_width, half_width + 1):
                put(int(round(ix + px * w)), gy, int(round(iz + pz * w)), bed)
            # レール本体（中心線上に minecraft:rail を1本, 道床の1段上）。
            # shape は前後の連結セル方向から決定 → 直線/曲線を自動で出し分け。
            dirs = []
            if idx > 0:
                dirs.append(_DIRNAME.get((path[idx - 1][0] - ix, path[idx - 1][1] - iz)))
            if idx + 1 < len(path):
                dirs.append(_DIRNAME.get((path[idx + 1][0] - ix, path[idx + 1][1] - iz)))
            rkey = _SHAPE_KEY.get(frozenset(d for d in dirs if d), "rail_ns")
            put(ix, gy + 1, iz, rkey)
    return ymax[0]


def assign_global_bridge_anchors(bridges, dem_full, lat_max, lon_min, res_lat, res_lon,
                                 *, h_res_block_m, scale_land, lift, sea_level_m=0.0):
    """各橋の端アンカー高 b["startS"]/["endS"] を **全域DEM** から計算して付与する（in-place）。

    add_bridge_blocks の end_base はタイルローカル grid を見るため、--tiles 分割で長い高架が複数
    タイルにまたがると、橋端点を含まない中間タイルで端点が範囲外→ terrain_y=1 にフォールバックし、
    デッキが地表付近まで降下していた。本関数は分割前に全域DEMで端アンカー(端点から橋の外向き
    0..45block の陸地形 median 標高→ブロックY)を1回だけ計算し、全タイルが同一の高さを参照できる
    ようにする。y_surf_land = max(1, 標高×scale_land) + lift（terrain_render の地表Yと同式）。
    """
    import math
    H, W = dem_full.shape
    M_PER_DEG_LAT = 111320.0
    h_res_dem = max(res_lat * M_PER_DEG_LAT, 1e-6)        # DEM 1セルの m（lat方向）
    # 端アンカーは端点から外側(奥へ続く道路方向)へ ≈45m の陸地形 median を取る。距離は必ず
    # メートル基準にする: DEM が粗い(GSI 5m→約4m/セル)と 1セル=数block になり、旧実装の
    # 「45セル」方式では 45×4≈180m 先まで届いて『トンネルの先の山』を拾い、坑口手前でデッキが
    # 高いまま浮く不具合が出ていた。LiDAR(1m)では 45セル=45m と実質同じ挙動を保つ。
    n_out_anchor = max(1, int(round(45.0 / h_res_dem)))   # ≈45m を DEMセル数へ換算

    def cell(lat, lon):
        return int(round((lat_max - lat) / res_lat)), int(round((lon - lon_min) / res_lon))  # (row,col)

    def ysl(row, col):
        if 0 <= row < H and 0 <= col < W:
            v = dem_full[row, col]
            if np.isfinite(v) and v > sea_level_m:        # 陸のみ（水上は橋台高に使わない）
                return max(1, int(v * scale_land)) + lift
        return None

    def anchor(p_end, p_in):
        r0, c0 = cell(p_end[0], p_end[1]); r1, c1 = cell(p_in[0], p_in[1])
        dr, dc = r0 - r1, c0 - c1                          # 外向き（橋の延長＝奥の道路方向）
        L = math.hypot(dr, dc)
        if L > 1e-6:
            dr, dc = dr / L, dc / L
        ys = []
        for d in range(0, n_out_anchor + 1):              # 端から外側 ≈45m の陸地形（DEMセル単位）
            v = ysl(int(round(r0 + dr * d)), int(round(c0 + dc * d)))
            if v is not None:
                ys.append(v)
        if not ys:
            v = ysl(r0, c0)
            return int(v) if v is not None else None       # None = 生成DEM範囲外/水上で高さ不明
        return int(round(np.median(ys)))

    for b in bridges:
        c = b.get("coords") or []
        if len(c) < 2:
            continue
        s = anchor(c[0], c[1])
        e = anchor(c[-1], c[-2])
        # 片端が生成DEMの範囲外/水上で高さ不明(None)なら、もう片方の端の高さで代用する。
        # 橋が生成範囲で途切れる所で、端アンカーが地表(1)へ落ちてデッキ全体が地面まで
        # 降下するのを防ぐ（＝範囲外の端は反対端の高さに合わせて平坦に飛ばす）。両端不明なら 1。
        if s is None:
            s = e
        if e is None:
            e = s
        b["startS"] = int(s) if s is not None else 1
        b["endS"] = int(e) if e is not None else 1


def add_bridge_blocks(blocks, bridges, patch_bbox_latlon, nz, nx, *,
                      y_surf_land, sea_mask, y_sea_surface, y_sea_floor,
                      scale_land, h_res_block_m, surf_block=None,
                      deck_key="andesite", pier_key="andesite",
                      cap_key="andesite", rail_key="andesite",
                      arch_rise_m=0.0, y_flood_top=None, road_mask=None) -> int:
    """OSM 橋（polyline + layer + road_class + width）を Tellus 流に立体化して blocks へ追加。

    桁Y(station) = max(両岸補間 baseline + ramp(layer×arch_rise_m),  局所地形/水面 + ramp(clearance))
      ramp は端0→中央最大の 4:1 勾配（=アプローチ坂）。clearance(main6/normal5/dirt3 m)が
      layer 情報無しでも川を跨がせる。arch_rise_m=0 なら両岸補間に沿う平坦橋（天田橋等は両端高さに）。
    デッキは2層: 上面=surf_block（衛星写真の路面色）／下面+橋脚+欄干+笠=deck_key(安山岩)。
    既存ブロックより後に置く（litematic は後勝ち）ので水上でデッキが優先される。
    返り値: 置いた最大 y（max_y 更新用）。
    """
    import math
    _BDBG = bool(__import__("os").environ.get("BRIDGE_DEBUG"))
    _BDUMP = __import__("os").environ.get("BRIDGE_DUMP")   # 指定で高さプロファイルを npz 出力
    _dump_recs = []
    MAX_RISE_M, RAMP_HV = 10.0, 4.0
    CLEAR_M = {"main": 6.0, "normal": 5.0, "dirt": 3.0}
    PIER_SPACING_M = 16.0
    # ── 橋端すり付け/支持の自動判別パラメータ（デッキが道路/地面から浮くのを防ぐ）──
    BRIDGE_END_LAND_TH = 2      # デッキが直下地表からこの差以内なら「着地済み」とみなす[block]
    BRIDGE_END_LOOK = 24        # 端の外向きに水/範囲外(=支持ケース)を判定する走査距離[block]
    BRIDGE_END_RAMP = 120       # 端を地表へ降ろすすり付けの最大長[block]（4:1で既存プロファイルに合流）
    BRIDGE_SUPPORT_MIN = 6      # デッキが直下床からこれ以上浮く所は橋脚で必ず支える[block]
    BRIDGE_SUPPORT_STEP = 8     # 追加橋脚の最小間隔[block]（宙吊りの空洞を無くす）
    # デッキ直下に地上道路(横断道・農道・生活道など road_mask で残った道)がある列には橋脚を
    # 立てない＝道路を跨ぐ。1セル膨張して路肩ギリギリに柱が刺さるのも避ける。橋自身の路面は
    # road_mask から除去済みなので中心線上の柱は従来どおり立つ。
    _no_pier = None
    if road_mask is not None:
        try:
            _no_pier = binary_dilation(road_mask, iterations=1)
        except Exception:
            _no_pier = road_mask
    def _pier_blocked(iz, ix):
        return (_no_pier is not None and 0 <= iz < nz and 0 <= ix < nx
                and bool(_no_pier[iz, ix]))
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
        # 橋端 → 外向き（橋の延長＝奥に続く道路の方向）へ陸地形をサンプルしてバンク高を決める。
        # 端ピンポイント依存をやめ、データ境界/水面/穴での baseline 破綻や橋台手前の瘤を防ぐ。
        # 「もっと奥まで見て高さを決める」: 0..45 block(≈30m) まで見て、橋が着地すべき高い陸地
        # （堤防/丘の道路高）を拾えるよう中央値だけでなく上側(75%tile)も考慮して安定化。
        dx, dz = end_pt[0] - in_pt[0], end_pt[1] - in_pt[1]
        L = math.hypot(dx, dz)
        if L > 1e-6:
            dx, dz = dx / L, dz / L
        ys = []
        for d in range(0, 46):    # 端から外側 0..45 block（奥に続く道路方向）の陸地形
            i, j = col(end_pt[0] + dx * d, end_pt[1] + dz * d)
            if 0 <= j < nz and 0 <= i < nx and not sea_mask[j, i] and np.isfinite(y_surf_land[j, i]):
                ys.append(int(y_surf_land[j, i]))
        if not ys:
            return terrain_y(*end_pt)
        # 奥に続く道路の代表地形高（中央値）。チェーン両端の着地高に使うので過大評価しない。
        return int(round(np.median(ys)))

    # ── 軸7-1: 各橋の幾何を前計算（pts/seg/total/baseline/has_water/min_deck） ──
    infos = []
    for b in bridges:
        pts = [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, nz, nx) for la, lo in b["coords"]]
        rc = b.get("road_class", "normal")
        _wm = float(b.get("width_m") or 5.5)
        if "link" in (b.get("highway") or ""):     # IC ランプは本線と同じ9m指定でも実態は単車線
            _wm = min(_wm, 5.5)
        half_w = max(0, int(round((_wm / max(h_res_block_m, 0.1)) / 2.0)))
        layer = int(b.get("layer", 1))
        seg, total = [], 0.0
        for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
            L = math.hypot(x1 - x0, z1 - z0); seg.append(L); total += L
        if total < 2.0:
            infos.append(None)
            continue
        # 端アンカー高。全域で事前計算した b["startS"]/["endS"] があれば優先（タイル分割で橋端点が
        # タイル外に出ても降下しない＝高架が複数タイルにまたがっても一貫した高さで連続平坦飛行）。
        # 無ければ従来どおりタイルローカル地形から end_base（単一タイル内に収まる橋はこれで十分）。
        _gs = b.get("startS"); _ge = b.get("endS")
        startS = int(_gs) if _gs is not None else end_base(pts[0], pts[1] if len(pts) > 1 else pts[0])
        endS = int(_ge) if _ge is not None else end_base(pts[-1], pts[-2] if len(pts) > 1 else pts[-1])
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
        infos.append(dict(pts=pts, seg=seg, total=total, startS=startS, endS=endS,
                          rise_full=rise_full, clear_full=clear_full, has_water=has_water,
                          min_deck=min_deck, half_w=half_w, rc=rc, layer=layer,
                          highway=(b.get("highway") or ""),
                          name=(b.get("name") or "").strip()))

    # ── 橋wayをチェーン再構成（ほぼ直線で連なる way 群を1本の連続橋に）。
    #   方針: 分割/途切れた橋群でも「ほぼ直線でつながる」なら1チェーンとみなし、
    #   **チェーン両端の高さだけ**を見て端から端へ線形にデッキを通す（途中で地形に追従して
    #   川底へ落とさない）。実橋も高所→低所へ緩く傾斜するので両端線形は違和感が小さく、
    #   恩恵(谷で浮き続ける)を受ける橋は少数。並走する上下線は端点間ベクトルが進行方向と
    #   直交するため別チェーンに分かれる。──
    ENDPOINT_TOL = max(2.0, 3.0 / max(h_res_block_m, 0.1))
    MAX_LINK = 130.0 / max(h_res_block_m, 0.1)        # 連結する隙間上限(block, ≈130m)
    MAX_LINK_NAME = 400.0 / max(h_res_block_m, 0.1)   # 同一道路名は橋間の地上区間(OSM非bridge)越えで連結
    COS_LINK = math.cos(math.radians(30.0))           # 直線継続とみなす角度許容(過剰グループ化抑制)

    def _udir(p_from, p_to):
        dx, dz = p_to[0] - p_from[0], p_to[1] - p_from[1]
        L = math.hypot(dx, dz)
        return (dx / L, dz / L) if L > 1e-6 else (0.0, 0.0)

    def _is_link(i):                                   # IC ランプ(motorway_link 等)か
        return "link" in (infos[i]["highway"] or "")

    def _name_tokens(i):
        return {s.strip() for s in (infos[i]["name"] or "").replace("；", ";").split(";") if s.strip()}

    def _same_road(i, j):
        # 「阪和自動車道」と「阪和自動車道;湯浅御坊道路」のような併記差も同一路線として扱う。
        a, b = _name_tokens(i), _name_tokens(j)
        return bool(a and b and (a & b))

    valid = [i for i in range(len(infos)) if infos[i] is not None]
    def _epos(i, e):
        return infos[i]["pts"][0] if e == 0 else infos[i]["pts"][-1]
    def _eout(i, e):                                  # 端点の外向き単位ベクトル
        P = infos[i]["pts"]
        return _udir(P[1], P[0]) if e == 0 else _udir(P[-2], P[-1])

    eps = [(i, e) for i in valid for e in (0, 1)]
    epidx = {ep: k for k, ep in enumerate(eps)}
    cands = []
    for a in range(len(eps)):
        ia, ea = eps[a]
        for b in range(a + 1, len(eps)):
            ib, eb = eps[b]
            if ia == ib or infos[ia]["layer"] != infos[ib]["layer"]:
                continue
            if _is_link(ia) != _is_link(ib):           # 本線とランプは別チェーンに分ける
                continue
            pa, pb = _epos(ia, ea), _epos(ib, eb)
            gv = (pb[0] - pa[0], pb[1] - pa[1])
            gl = math.hypot(*gv)
            # 同一道路名(阪和道など本線)は橋区間の間の地上区間を越えて連結。角度(直線継続)は維持する
            # ので上下線(進行方向と直交)・分岐は連結しない。
            _same = (not _is_link(ia)) and _same_road(ia, ib)
            if gl > (MAX_LINK_NAME if _same else MAX_LINK):
                continue
            if gl > ENDPOINT_TOL:                     # 近接共有でなければ直線継続を要求
                gu = (gv[0] / gl, gv[1] / gl)
                da, db = _eout(ia, ea), _eout(ib, eb)
                if da[0] * gu[0] + da[1] * gu[1] < COS_LINK:
                    continue
                if db[0] * (-gu[0]) + db[1] * (-gu[1]) < COS_LINK:
                    continue
            cands.append((gl, a, b))
    cands.sort(key=lambda c: c[0])
    linked = {}
    for gl, a, b in cands:                            # 端点1対1で貪欲連結（近い順）
        if a in linked or b in linked:
            continue
        linked[a] = b; linked[b] = a

    # チェーン構築: 自由端(リンク無し端点)から辿る → (way, flip) の順序列
    seen = set()
    chains = []
    def _chain_from(startk):
        order = []
        k = startk
        while True:
            i, e = eps[k]
            if i in seen:
                break
            seen.add(i)
            order.append((i, e == 1))                 # 端(1)から入ったら反転して通す
            nb = linked.get(epidx[(i, 1 - e)])        # 反対の端から出る
            if nb is None:
                break
            k = nb
        return order
    for k, (i, e) in enumerate(eps):
        if i not in seen and k not in linked:         # 自由端から
            ch = _chain_from(k)
            if ch:
                chains.append(ch)
    for k, (i, e) in enumerate(eps):                  # 残り(リンクが閉路)
        if i not in seen:
            ch = _chain_from(k)
            if ch:
                chains.append(ch)

    if _BDBG:
        _dropped = [b.get("name", "") for b, inf in zip(bridges, infos) if inf is None]
        print(f"[bridge] ways={len(bridges)} valid={len(valid)} "
              f"dropped(total<2)={len(_dropped)} chains={len(chains)} "
              f"links={len(linked)//2}")

    # ── デッキ高の解析式（_render_span と「派生元の橋の高さ」問い合わせで共有） ──
    def _deck_dy(station, total, startS, endS, rise_full, min_deck):
        base = startS + (endS - startS) * (station / total) if total > 1e-6 else startS
        # 水面クリアランス: 両端の道路高から 4:1 で立ち上がり min_deck で頭打ち。
        lift = min(min_deck,
                   startS + station / RAMP_HV,
                   endS + (total - station) / RAMP_HV)
        return int(round(max(base + ramp(station, total, rise_full), lift)))

    def _chain_profile(mpts, seglen, startS, endS, rise_full, min_deck, kind):
        # 検証用: 実レンダと同じ _deck_profile の deck 高＋直下地形/水面/橋脚底をサンプル。
        total = sum(seglen)
        cxs, czs, sts, dys = _deck_profile(mpts, seglen, total, startS, endS,
                                           rise_full, min_deck)
        terr = [terrain_y(cxs[i], czs[i]) for i in range(len(sts))]
        flr = [floor_y(cxs[i], czs[i]) for i in range(len(sts))]
        wss = []
        for i in range(len(sts)):
            ci, cj = col(cxs[i], czs[i])
            _sea = (0 <= cj < nz and 0 <= ci < nx and sea_mask[cj, ci])
            wss.append(int(y_sea_surface) if _sea else -9999)
        return dict(kind=kind, station=np.array(sts), x=np.array(cxs), z=np.array(czs),
                    dy=np.array(dys), terr=np.array(terr), floor=np.array(flr),
                    wsurf=np.array(wss),
                    startS=float(startS), endS=float(endS), total=float(total),
                    min_deck=float(min_deck))

    # 橋デッキ/路面に使われる灰色系(衛星が橋を写し込むと橋下に出る色)。補完では避けたい。
    _ROADISH = {"andesite", "gray_concrete", "light_gray_concrete", "stone", "smooth_stone",
                "black_concrete", "cyan_concrete", "polished_andesite", "gravel", "cobblestone"}

    def _surf_at(ix, iz):                              # 地表色(水/海でない自然色)を返す, 無ければ None
        if surf_block is None or not (0 <= iz < nz and 0 <= ix < nx):
            return None
        sk = surf_block[iz, ix]
        if sk and sk != "water" and not sea_mask[iz, ix]:
            return sk
        return None

    def _fill_color(cx, cz, ox, oz, lhw):
        # 橋下の乾いた地表に塗る「周囲色」。橋脇を外側へ探索し、道路灰色でない自然色を優先採用。
        fallback = None
        for off in range(lhw + 2, lhw + 12):
            for sgn in (1, -1):
                s = _surf_at(*col(cx + ox * off * sgn, cz + oz * off * sgn))
                if s is None:
                    continue
                if s not in _ROADISH:
                    return s                            # 自然色(草/土/砂等)が見つかれば即採用
                if fallback is None:
                    fallback = s
        return fallback                                 # 自然色が無ければ最寄りの灰色

    def _road_halfw(cx, cz, ox, oz, fallback):
        # デッキ半幅を「直下の地表道路レンダ(road_mask)の実幅」に合わせる。中心(±2)に道路が無ければ
        # width_m 由来 fallback。連続走査で道路の途切れまでを左右に測り、交差点での膨張は 2×fallback で頭打ち。
        if road_mask is None:
            return fallback
        ci, cj = col(cx, cz)
        on_road = (0 <= cj < nz and 0 <= ci < nx and road_mask[cj, ci])
        if not on_road:
            for s in (1, -1, 2, -2):
                ix, iz = col(cx + ox * s, cz + oz * s)
                if 0 <= iz < nz and 0 <= ix < nx and road_mask[iz, ix]:
                    on_road = True
                    break
            if not on_road:
                return fallback
        cap = fallback * 2 + 2
        rp = rm = 0
        for w in range(1, cap + 1):
            ix, iz = col(cx + ox * w, cz + oz * w)
            if 0 <= iz < nz and 0 <= ix < nx and road_mask[iz, ix]:
                rp = w
            else:
                break
        for w in range(1, cap + 1):
            ix, iz = col(cx - ox * w, cz - oz * w)
            if 0 <= iz < nz and 0 <= ix < nx and road_mask[iz, ix]:
                rm = w
            else:
                break
        hw = (rp + rm + 1) // 2
        # road_mask は「細くする方向のみ」採用(nominal=width_m半幅を超えない)。交差点の膨張を防ぐ。
        return max(1, min(hw, fallback)) if hw > 0 else fallback

    # ── デッキ高プロファイル: 解析デッキ raw=_deck_dy を、直下フットプリント(幅方向)の最高
    #    ground(水面 or 地形トップ)より下げない＝埋没/水没を防ぐ。ただし「超過分 excess=ground-raw」
    #    だけを 1/RAMP_HV 勾配で envelope し raw に足す（dy=raw+excess）。これにより:
    #      ・端や平坦高所橋では ground<=raw → excess0 → raw のまま着地/平坦飛行を保持（コブ/+1段差なし）
    #      ・谷/低地/境界に潰れていた橋(従来 Y1)は excess が滑らかに持ち上げて連続・可視化
    #    境界外(OOB)サンプルは ground 評価から除外（terrain_y の OOB=1 で低く潰れるのを防ぐ）。
    def _deck_profile(mpts, seg, total, startS, endS, rise_full, min_deck, lhw=1):
        cxs, czs, sts, oxs, ozs = [], [], [], [], []
        s_acc = 0.0
        for si in range(len(seg)):
            (x0, z0), (x1, z1) = mpts[si], mpts[si + 1]
            L = seg[si]
            if L < 1e-6:
                continue
            ox, oz = -(z1 - z0) / L, (x1 - x0) / L
            n = max(1, int(L / 0.5))
            for k in range(n + 1):
                t = k / n
                cxs.append(x0 + (x1 - x0) * t); czs.append(z0 + (z1 - z0) * t)
                sts.append(s_acc + L * t); oxs.append(ox); ozs.append(oz)
            s_acc += L
        m = len(sts)
        raw = [float(_deck_dy(sts[i], total, startS, endS, rise_full, min_deck))
               for i in range(m)]
        excess = [0.0] * m
        for i in range(m):
            g = None
            for w in (-lhw, -lhw * 0.5, 0.0, lhw * 0.5, lhw):
                ci, cj = col(cxs[i] + oxs[i] * w, czs[i] + ozs[i] * w)
                if 0 <= cj < nz and 0 <= ci < nx:
                    gw = int(y_sea_surface) if sea_mask[cj, ci] else int(y_surf_land[cj, ci])
                    g = gw if g is None else max(g, gw)
            if g is not None:
                excess[i] = max(0.0, g - raw[i])
        # excess(床超過分)だけを 1/RAMP_HV 勾配で envelope（前後2パス）。raw は素のまま。
        for i in range(1, m):
            excess[i] = max(excess[i], excess[i - 1] - abs(sts[i] - sts[i - 1]) / RAMP_HV)
        for i in range(m - 2, -1, -1):
            excess[i] = max(excess[i], excess[i + 1] - abs(sts[i + 1] - sts[i]) / RAMP_HV)
        return cxs, czs, sts, [int(round(raw[i] + excess[i])) for i in range(m)]

    # ── レンダリング本体: 1スパン=連続ポリラインを startS→endS 線形デッキで立体化。
    #    head_ext/tail_ext = 端の延長区間長(station)。延長部は柵を生成しない(道路へ自然に接続)。──
    def _render_span(mpts, startS, endS, rise_full, min_deck, half_w, rc,
                     head_ext=0.0, tail_ext=0.0, treat_head=True, treat_tail=True):
        seg = [math.hypot(mpts[s + 1][0] - mpts[s][0], mpts[s + 1][1] - mpts[s][1])
               for s in range(len(mpts) - 1)]
        total = sum(seg)
        if total < 2.0:
            return [], []
        pier_step = max(4.0, PIER_SPACING_M / max(h_res_block_m, 0.1))
        next_pier = pier_step
        next_support = 0.0
        # デッキ幅は「直下道路実幅の一番大きい部分」で全長統一(#1)。事前スキャンで最大半幅を求める
        # (nominal=width_m半幅が上限)。くびれ/ジャギを無くし均一幅にする。
        uniform_hw = 1
        for si in range(len(seg)):
            (x0, z0), (x1, z1) = mpts[si], mpts[si + 1]
            L = seg[si]
            if L < 1e-6:
                continue
            ox, oz = -(z1 - z0) / L, (x1 - x0) / L
            n = max(1, int(L / 0.5))
            for k in range(n + 1):
                t = k / n
                uniform_hw = max(uniform_hw, _road_halfw(
                    x0 + (x1 - x0) * t, z0 + (z1 - z0) * t, ox, oz, half_w))
        lhw = max(1, uniform_hw)
        _cxs, _czs, _prof_sts, dyt = _deck_profile(mpts, seg, total, startS, endS,
                                                   rise_full, min_deck, lhw)
        # ── 端すり付け/支持の自動判別: 各自由端で「奥に地上道路が続く」ならデッキを地表高へ
        #    勾配で降ろして道路へ接続（すり付け）。範囲外/水/谷で切れる端はデッキ高を据置き、
        #    後段の橋脚で地面まで支える（どちらの場合も宙吊りを作らない）。──
        _m = len(dyt)

        def _grd(i):
            return ground_y(_cxs[i], _czs[i])

        def _treat_end(term, inw, sgn):
            if not (0 <= term < _m and 0 <= inw < _m):
                return
            g_term = _grd(term)
            if dyt[term] - g_term <= BRIDGE_END_LAND_TH:
                return                                   # 既に地表へ着地している端
            dxo, dzo = _cxs[term] - _cxs[inw], _czs[term] - _czs[inw]
            Lo = math.hypot(dxo, dzo) or 1.0
            dxo, dzo = dxo / Lo, dzo / Lo                 # 端の外向き（奥へ続く道路方向）
            nwater = ntot = 0
            for d in range(1, BRIDGE_END_LOOK + 1):
                ci, cj = col(_cxs[term] + dxo * d, _czs[term] + dzo * d)
                if not (0 <= cj < nz and 0 <= ci < nx):
                    return                               # 生成範囲外で切断＝支持ケース（据置）
                ntot += 1
                if sea_mask[cj, ci]:
                    nwater += 1
            if ntot and nwater * 2 >= ntot:              # 外向きが主に水＝川/海を渡る端＝支持（据置）
                return
            # 降ろすケース（奥が陸: 地上道路の続き / トンネル坑口へ下る等）: 端を g_term まで下げ、
            # 内側へ 4:1 勾配で復帰する（min で既存プロファイルに自然合流・持ち上げはしない）。
            for j in range(0, BRIDGE_END_RAMP + 1):
                i = term + sgn * j
                if not (0 <= i < _m):
                    break
                dyt[i] = max(_grd(i), min(dyt[i], int(round(g_term + j / RAMP_HV))))
        if treat_head:
            _treat_end(0, 1, +1)
        if treat_tail:
            _treat_end(_m - 1, _m - 2, -1)
        idx = 0
        s_acc = 0.0
        for si in range(len(seg)):
            (x0, z0), (x1, z1) = mpts[si], mpts[si + 1]
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
                in_ext = (station < head_ext - 1e-6) or (station > total - tail_ext + 1e-6)
                dy = dyt[idx]; idx += 1
                # 斜め橋でデッキに 1×1 ピンホールが空くのを防ぐため、法線方向を 0.5 step で走査。
                half_steps = max(1, int(round(lhw / 0.5)))
                for wi in range(-half_steps, half_steps + 1):
                    w = wi * 0.5
                    ix, iz = col(cx + ox * w, cz + oz * w)
                    top_key = deck_key
                    if surf_block is not None and 0 <= iz < nz and 0 <= ix < nx:
                        sk = surf_block[iz, ix]
                        if sk and sk != "water":    # 水域(河川/海)の色は橋の路面に使わない
                            top_key = sk            # 上面=衛星写真の路面色
                    put(ix, dy, iz, top_key)
                    put(ix, dy - 1, iz, deck_key)   # 下面=安山岩(構造)
                    if abs(wi) == half_steps and lhw >= 1 and not in_ext:
                        put(ix, dy + 1, iz, rail_key)   # 延長部(in_ext)は柵なし
                if station >= next_pier and not (si == 0 and k == 0):
                    next_pier += pier_step
                    fy = floor_y(cx, cz)
                    if dy - 2 > fy:
                        shafts = (-lhw + 1, lhw - 1) if (rc == "main" and lhw >= 2) else (0,)
                        for w in shafts:
                            ix, iz = col(cx + ox * w, cz + oz * w)
                            if _pier_blocked(iz, ix):
                                continue          # 直下に道路 → 橋脚を立てず跨ぐ
                            for yy in range(fy, dy - 1):
                                put(ix, yy, iz, pier_key)
                            put(ix, dy - 1, iz, cap_key)
                # 追加支持: デッキが直下床から高く浮く区間（端の据置・谷/窪みの跨ぎ）は、
                # 通常橋脚(pier_step)より細かい間隔で中央橋脚を必ず立てて宙吊りの空洞を無くす。
                _fy2 = floor_y(cx, cz)
                if dy - 1 - _fy2 >= BRIDGE_SUPPORT_MIN and station >= next_support:
                    next_support = station + BRIDGE_SUPPORT_STEP
                    shafts2 = (-lhw + 1, lhw - 1) if (rc == "main" and lhw >= 2) else (0,)
                    for w in shafts2:
                        ix, iz = col(cx + ox * w, cz + oz * w)
                        if _pier_blocked(iz, ix):
                            continue              # 直下に道路 → 橋脚を立てず跨ぐ
                        for yy in range(_fy2, dy - 1):
                            put(ix, yy, iz, pier_key)
                        put(ix, dy - 1, iz, cap_key)
                # 橋下の処理（列ごと）: 実際に水がある列のみ「その列の水面まで」水柱、
                #   乾いた陸の列は衛星が橋を写し込んだ路面色を、橋脇の周囲色で上書き補完する。
                #   水位はバンド最大でなく列ごとの実水面で決める(端の高地に引っ張られた異常水位を防ぐ)。
                if y_flood_top is not None:
                    fill_col = _fill_color(cx, cz, ox, oz, lhw)
                    for w in range(-lhw, lhw + 1):
                        ix2, iz2 = col(cx + ox * w, cz + oz * w)
                        if not (0 <= iz2 < nz and 0 <= ix2 < nx):
                            continue
                        if sea_mask[iz2, ix2]:
                            gcol = int(y_sea_floor[iz2, ix2])
                            wt = max(int(y_flood_top[iz2, ix2]), int(y_sea_surface))
                        elif int(y_flood_top[iz2, ix2]) > int(y_surf_land[iz2, ix2]):
                            gcol = int(y_surf_land[iz2, ix2])
                            wt = int(y_flood_top[iz2, ix2])
                        else:
                            gcol = int(y_surf_land[iz2, ix2]); wt = -1   # 乾いた陸
                        top_w = min(int(dy) - 2, wt)
                        if top_w > gcol:
                            for wy in range(gcol + 1, top_w + 1):
                                put(ix2, wy, iz2, "water")
                        elif fill_col is not None and gcol < int(dy) - 1:
                            put(ix2, gcol, iz2, fill_col)   # 橋下地表トップを周囲色で補完
            s_acc += L
        return _prof_sts, dyt          # 実デッキ高プロファイル(ランプ接続=_parent_at で参照)

    # ── チェーン → マージ済みポリライン / 属性 / 端延長 / 射影 のヘルパ ──
    def _seglens(mpts):
        return [math.hypot(mpts[s + 1][0] - mpts[s][0], mpts[s + 1][1] - mpts[s][1])
                for s in range(len(mpts) - 1)]

    def _build_mpts(ch):
        # チェーン内の way を順に連結（flip）してマージ済みポリラインへ。ギャップは直線セグメント。
        mpts = []
        for (i, flip) in ch:
            P = infos[i]["pts"]
            sp = P[::-1] if flip else P
            if mpts and math.hypot(mpts[-1][0] - sp[0][0], mpts[-1][1] - sp[0][1]) < 1e-6:
                mpts += sp[1:]
            else:
                mpts += sp
        return mpts

    def _chain_attrs(ch):
        has_water = any(infos[i]["has_water"] for (i, _) in ch)
        clear_full = max(infos[i]["clear_full"] for (i, _) in ch)
        rise_full = max(infos[i]["rise_full"] for (i, _) in ch)
        half_w = max(infos[i]["half_w"] for (i, _) in ch)
        rc = "main" if any(infos[i]["rc"] == "main" for (i, _) in ch) else infos[ch[0][0]]["rc"]
        min_deck = (int(y_sea_surface) + clear_full) if has_water else -1.0e9
        return rise_full, half_w, rc, min_deck

    EXTEND_MAX = 50                                    # 端の最大延長(block)
    # 最高点へ着地したあと、直進方向に道路(road_mask)が続く限り更に延ばす柵なしすり付けの
    # 上限[block]。道路が途切れる/T字で直進の先に道路が無い所で手前停止する。0 で無効。
    BRIDGE_APPROACH_EXTRA = 10
    def _extend_to_highest(end_pt, in_pt):
        # 端点から外向き(奥に続く道路方向)へ d=0..EXTEND_MAX block 走査し、その区間の最高地形(=道路)点に着地。
        # 「元データから延長して一番高い道路地点から橋をはやす」。チェーン分割後に適用するので
        # この延長部はグループ化の対象外（既に linked/chains 確定後）。川底/低所への降下を防ぐ。
        # 着地後は直進の道路が続く限り BRIDGE_APPROACH_EXTRA block まで更に延長し、柵なし
        # すり付けを道路へ長く馴染ませる（直進の先に道路が無い=終端/T字なら手前で停止）。
        dx, dz = end_pt[0] - in_pt[0], end_pt[1] - in_pt[1]
        L = math.hypot(dx, dz)
        if L < 1e-6:
            return end_pt, terrain_y(*end_pt)
        dx, dz = dx / L, dz / L
        best_pt, best_h, best_d = None, None, 0
        for d in range(0, EXTEND_MAX + 1):
            x, z = end_pt[0] + dx * d, end_pt[1] + dz * d
            i, j = col(x, z)
            if not (0 <= j < nz and 0 <= i < nx):
                break
            if sea_mask[j, i] or not np.isfinite(y_surf_land[j, i]):
                continue
            h = int(y_surf_land[j, i])
            if best_h is None or h > best_h:
                best_h, best_pt, best_d = h, (x, z), d
        if best_h is None:
            return end_pt, terrain_y(*end_pt)
        # 最高点の先へ、直進方向に道路が続く限り更に延長（T字/道路終端で手前停止）。
        if road_mask is not None and BRIDGE_APPROACH_EXTRA > 0:
            for d in range(best_d + 1, best_d + int(BRIDGE_APPROACH_EXTRA) + 1):
                x, z = end_pt[0] + dx * d, end_pt[1] + dz * d
                i, j = col(x, z)
                if not (0 <= j < nz and 0 <= i < nx):
                    break
                if sea_mask[j, i] or not np.isfinite(y_surf_land[j, i]):
                    break
                if not road_mask[j, i]:              # 直進の先に道路が無い=終端/T字 → 手前で停止
                    break
                best_pt, best_h = (x, z), int(y_surf_land[j, i])
        return best_pt, best_h

    def _project(mpts, seglen, q):
        # q をポリラインへ射影 → (station, 距離^2)
        best_s, best_d2, s_acc = 0.0, 1.0e18, 0.0
        for s in range(len(mpts) - 1):
            (x0, z0), (x1, z1) = mpts[s], mpts[s + 1]
            L = seglen[s]
            if L < 1e-6:
                continue
            t = ((q[0] - x0) * (x1 - x0) + (q[1] - z0) * (z1 - z0)) / (L * L)
            t = min(1.0, max(0.0, t))
            px, pz = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
            d2 = (q[0] - px) ** 2 + (q[1] - pz) ** 2
            if d2 < best_d2:
                best_d2, best_s = d2, s_acc + L * t
            s_acc += L
        return best_s, best_d2

    def _ext_head(mpts):                               # 始端を延長 → (新mpts, 延長長, 高さ)
        e, h = _extend_to_highest(mpts[0], mpts[1])
        d = math.hypot(e[0] - mpts[0][0], e[1] - mpts[0][1])
        return (([e] + mpts) if d > 1e-6 else mpts), d, h

    def _ext_tail(mpts):                               # 終端を延長 → (新mpts, 延長長, 高さ)
        e, h = _extend_to_highest(mpts[-1], mpts[-2])
        d = math.hypot(e[0] - mpts[-1][0], e[1] - mpts[-1][1])
        return ((mpts + [e]) if d > 1e-6 else mpts), d, h

    # ── 本線高架(motorway/trunk 等)を先に描き、各チェーンのデッキ高プロファイルを保持。
    #    その後 IC ランプ(motorway_link 等)を描き、本線へ接続する側の端を「派生元の橋の高さ」に
    #    合わせる（反対端は最高点へ延長）。──
    main_specs = []
    for ch in chains:
        if any(_is_link(i) for (i, _) in ch):
            continue
        raw_mpts = _build_mpts(ch)
        if len(raw_mpts) < 2:
            continue
        rise_full, half_w, rc, min_deck = _chain_attrs(ch)
        join_head, join_tail = raw_mpts[0], raw_mpts[-1]
        join_head_dir = _udir(raw_mpts[1], raw_mpts[0])
        join_tail_dir = _udir(raw_mpts[-2], raw_mpts[-1])
        mpts = list(raw_mpts)
        mpts, head_ext, h0 = _ext_head(mpts)
        mpts, tail_ext, h1 = _ext_tail(mpts)
        # チェーン両端の高さ。_extend_to_highest はタイルローカル y_surf_land を見るため、--tiles 分割で
        # チェーン端点がタイル外に出ると地表(terrain_y=1)にフォールバックし、デッキ全体が地表へ降下する
        # （チェーン内 way 同士は線形補間で連続するが、グループ全体が地表に沈む）。チェーン端 way の
        # 全域アンカー（assign_global_bridge_anchors が infos に付与した startS/endS, flip 考慮）を優先。
        _i0, _f0 = ch[0]; _iL, _fL = ch[-1]
        _g0 = infos[_i0]["endS"] if _f0 else infos[_i0]["startS"]   # チェーン始端 way の自由端アンカー
        _g1 = infos[_iL]["startS"] if _fL else infos[_iL]["endS"]   # チェーン終端 way の自由端アンカー
        startS = int(_g0) if _g0 is not None else h0                # 全域アンカー優先(タイル間で一貫＝境界段差を防止)
        endS = int(_g1) if _g1 is not None else h1
        seglen = _seglens(mpts)
        main_specs.append(dict(ch=ch, mpts=mpts, seglen=seglen, total=sum(seglen),
                               startS=startS, endS=endS, rise_full=rise_full,
                               min_deck=min_deck, half_w=half_w, rc=rc,
                               head_ext=head_ext, tail_ext=tail_ext,
                               join_head=join_head, join_tail=join_tail,
                               join_head_dir=join_head_dir, join_tail_dir=join_tail_dir,
                               layer=infos[ch[0][0]]["layer"],
                               tokens=set().union(*(_name_tokens(i) for (i, _) in ch))))

    # チェーン自体を連結できなかった接続点でも、同一路線の本線端が近ければ端高を揃える。
    # OSM の bridge=yes が IC/JCT や短い地上区間で複数 way に分かれると、片側だけ地形medianに
    # 落ちて「橋が切れた段差」に見える。ジオメトリは分けたまま、接続点の高さだけ高い方へ寄せる。
    HEIGHT_JOIN_TOL = 60.0 / max(h_res_block_m, 0.1)
    COS_HEIGHT_LINK = math.cos(math.radians(75.0))
    endpoints = []
    for si, sp in enumerate(main_specs):
        endpoints.append((si, "startS", sp["join_head"], sp["join_head_dir"]))
        endpoints.append((si, "endS", sp["join_tail"], sp["join_tail_dir"]))
    for a in range(len(endpoints)):
        ia, ka, pa, da = endpoints[a]
        for b in range(a + 1, len(endpoints)):
            ib, kb, pb, db = endpoints[b]
            if ia == ib:
                continue
            A, B = main_specs[ia], main_specs[ib]
            if A["layer"] != B["layer"] or not (A["tokens"] and B["tokens"] and (A["tokens"] & B["tokens"])):
                continue
            gv = (pb[0] - pa[0], pb[1] - pa[1])
            gl = math.hypot(*gv)
            if gl > HEIGHT_JOIN_TOL:
                continue
            if gl > ENDPOINT_TOL:
                gu = (gv[0] / gl, gv[1] / gl)
                if da[0] * gu[0] + da[1] * gu[1] < COS_HEIGHT_LINK:
                    continue
                if db[0] * (-gu[0]) + db[1] * (-gu[1]) < COS_HEIGHT_LINK:
                    continue
            # 分岐点では両チェーン端が同じ地形median(地表)に落ち、max(A,B) では更新されないことがある。
            # 各チェーンの反対端(高架側)も見て、接続点を高架高へ引き上げる(地表に沈む接続を防ぐ)。
            _ao = A["endS"] if ka == "startS" else A["startS"]
            _bo = B["endS"] if kb == "startS" else B["startS"]
            h = max(int(A[ka]), int(B[kb]), int(_ao), int(_bo))
            if h > A[ka] or h > B[kb]:
                if _BDBG:
                    print(f"[bridge] JOIN height {ia}.{ka}<->{ib}.{kb} "
                          f"dist={gl:.0f}b {A[ka]}/{B[kb]} -> {h}")
                A[ka] = B[kb] = h

    # 同一路線の上下線は横並びで別チェーンに残す必要があるが、橋台の高さは揃っていないと
    # 片側だけ地表アンカーへ落ちて大きな段差に見える。ジオメトリは連結せず、近接・平行な
    # main チェーン端だけ高い方へ同期する。
    COS_PARALLEL_HEIGHT = math.cos(math.radians(30.0))
    for a in range(len(endpoints)):
        ia, ka, pa, da = endpoints[a]
        for b in range(a + 1, len(endpoints)):
            ib, kb, pb, db = endpoints[b]
            if ia == ib:
                continue
            A, B = main_specs[ia], main_specs[ib]
            if A["rc"] != "main" or B["rc"] != "main":
                continue
            if A["layer"] != B["layer"] or not (A["tokens"] and B["tokens"] and (A["tokens"] & B["tokens"])):
                continue
            gl = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if gl > HEIGHT_JOIN_TOL:
                continue
            if abs(da[0] * db[0] + da[1] * db[1]) < COS_PARALLEL_HEIGHT:
                continue
            h = max(int(A[ka]), int(B[kb]))
            if h > A[ka] or h > B[kb]:
                if _BDBG:
                    print(f"[bridge] PARALLEL height {ia}.{ka}<->{ib}.{kb} "
                          f"dist={gl:.0f}b {A[ka]}/{B[kb]} -> {h}")
                A[ka] = B[kb] = h

    main_params = []
    for sp in main_specs:
        ch = sp["ch"]
        mpts, seglen = sp["mpts"], sp["seglen"]
        startS, endS = int(sp["startS"]), int(sp["endS"])
        rise_full, min_deck = sp["rise_full"], sp["min_deck"]
        half_w, rc = sp["half_w"], sp["rc"]
        head_ext, tail_ext = sp["head_ext"], sp["tail_ext"]
        psts, pdy = _render_span(mpts, startS, endS, rise_full, min_deck, half_w, rc,
                                 head_ext, tail_ext)
        main_params.append(dict(mpts=mpts, seglen=seglen, total=sp["total"],
                                startS=startS, endS=endS, rise_full=rise_full,
                                min_deck=min_deck, psts=psts, pdy=pdy))
        if _BDBG:
            _tot = sp["total"]
            _names = sorted({infos[i]["name"] for (i, _) in ch if infos[i]["name"]})
            _steps = [abs(pdy[k + 1] - pdy[k]) for k in range(len(pdy) - 1)] or [0]
            _samp = [pdy[int(round(j * (len(pdy) - 1) / 10))] for j in range(11)] if pdy else []
            print(f"[bridge] MAIN ways={[i for i,_ in ch]} layer={infos[ch[0][0]]['layer']} "
                  f"rc={rc} L={_tot:.0f}b startS={startS} endS={endS} min_deck={min_deck:.0f} "
                  f"has_water={min_deck>-1e8} head_ext={head_ext:.0f} tail_ext={tail_ext:.0f} "
                  f"maxstep={max(_steps):.0f} dy@={_samp} names={_names}")
        if _BDUMP:
            _dump_recs.append(_chain_profile(mpts, seglen, startS, endS, rise_full,
                                             min_deck, "main"))

    JOIN_TOL = 40.0 / max(h_res_block_m, 0.1)         # ランプ端が本線へ接続とみなす距離(block, ≈40m)

    def _parent_at(q):
        # q に最も近い本線高架の「実デッキ高」（床/勾配で嵩上げ後の psts/pdy を補間）。
        #   解析値 _deck_dy ではなく実レンダ高を返すことで、嵩上げされた本線へランプが段差なく接続。
        best_dy, best_d2 = None, JOIN_TOL * JOIN_TOL
        for pp in main_params:
            s, d2 = _project(pp["mpts"], pp["seglen"], q)
            if d2 < best_d2:
                best_d2 = d2
                if pp["psts"]:
                    best_dy = int(round(float(np.interp(s, pp["psts"], pp["pdy"]))))
                else:
                    best_dy = _deck_dy(s, pp["total"], pp["startS"], pp["endS"],
                                       pp["rise_full"], pp["min_deck"])
        return best_dy, best_d2

    for ch in chains:
        if not any(_is_link(i) for (i, _) in ch):
            continue
        mpts = _build_mpts(ch)
        if len(mpts) < 2:
            continue
        rise_full, half_w, rc, min_deck = _chain_attrs(ch)
        dy0, d0 = _parent_at(mpts[0])
        dy1, d1 = _parent_at(mpts[-1])
        head_ext = tail_ext = 0.0
        treat_head = treat_tail = True                # 本線接続端は降ろさない(接続が切れるため)
        if dy0 is not None and (dy1 is None or d0 <= d1):
            startS = dy0                              # 接続端=派生元の橋の高さ（延長しない）
            mpts, tail_ext, endS = _ext_tail(mpts)
            treat_head = False                        # head=本線接続端 → 高さ据置
        elif dy1 is not None:
            endS = dy1
            mpts, head_ext, startS = _ext_head(mpts)
            treat_tail = False                        # tail=本線接続端 → 高さ据置
        else:                                         # 本線未接続: 通常チェーン同様に両端延長
            mpts, head_ext, startS = _ext_head(mpts)
            mpts, tail_ext, endS = _ext_tail(mpts)
        _render_span(mpts, startS, endS, rise_full, min_deck, half_w, rc, head_ext, tail_ext,
                     treat_head=treat_head, treat_tail=treat_tail)
        if _BDBG or _BDUMP:
            _sl = _seglens(mpts); _tot = sum(_sl)
            if _BDBG:
                _pr = [_deck_dy(s, _tot, startS, endS, rise_full, min_deck)
                       for s in np.linspace(0, _tot, 11)]
                _stp = max(abs(_pr[k + 1] - _pr[k]) for k in range(len(_pr) - 1)) if _tot else 0
                print(f"[bridge] RAMP ways={[i for i,_ in ch]} L={_tot:.0f}b "
                      f"startS={startS} endS={endS} head_ext={head_ext:.0f} "
                      f"tail_ext={tail_ext:.0f} maxstep={_stp:.0f} dy@={_pr}")
            if _BDUMP:
                _dump_recs.append(_chain_profile(mpts, _sl, startS, endS, rise_full,
                                                 min_deck, "ramp"))
    if _BDUMP and _dump_recs:
        np.savez(_BDUMP, recs=np.array(_dump_recs, dtype=object))
        print(f"[bridge] dumped {len(_dump_recs)} chains → {_BDUMP}")
    return ymax[0]


def assign_global_tunnel_anchors(tunnels, dem_full, lat_max, lon_min, res_lat, res_lon,
                                 *, h_res_block_m, scale_land, lift, portal_blocks: int = 8):
    """各トンネルの両坑口の床高 ``t["startF"]``/``t["endF"]`` を **全域DEM** から計算して
    付与する（in-place）。

    add_tunnel_blocks の end_floor/portal_floor はタイルローカル grid を走査するため、
    --tiles 分割でトンネルが複数タイルにまたがると、坑口を含まないタイルでは「そのタイル内で
    最初に in-grid になった点」の地形高が坑口高に化ける。床高は両坑口の線形補間なので、
    タイルごとに違う床勾配になり境界で床が段差になる（実測で最大 29 block）。
    本関数は分割前に全域DEMで両坑口の床高を1回だけ計算し、全タイルが同一の値を参照
    できるようにする（橋の assign_global_bridge_anchors と同じ手当て）。

    坑口床高 = 坑口から**外向き**（トンネル外＝道路側）0..portal_blocks block の地形Y の最小値
    （add_tunnel_blocks.portal_floor と同式）。
    y_surf_land = max(1, 標高×scale_land) + lift。
    """
    import math
    H, W = dem_full.shape
    M_PER_DEG_LAT = 111320.0
    h_res_dem = max(res_lat * M_PER_DEG_LAT, 1e-6)        # DEM 1セルの m（lat方向）
    step = max(1, int(round(h_res_block_m / h_res_dem)))  # 1 block = step DEMセル

    def cell(lat, lon):
        return int(round((lat_max - lat) / res_lat)), int(round((lon - lon_min) / res_lon))

    def ysl(row, col):
        if 0 <= row < H and 0 <= col < W:
            v = dem_full[row, col]
            if np.isfinite(v):
                return max(1, int(v * scale_land)) + lift
        return None

    def portal(p_end, p_in):
        r0, c0 = cell(p_end[0], p_end[1]); r1, c1 = cell(p_in[0], p_in[1])
        dr, dc = r0 - r1, c0 - c1                          # 外向き（坑口の先＝道路側）
        L = math.hypot(dr, dc)
        if L > 1e-6:
            dr, dc = dr / L, dc / L
        ys = []
        for d in range(0, int(max(0, portal_blocks)) + 1):
            v = ysl(int(round(r0 + dr * d * step)), int(round(c0 + dc * d * step)))
            if v is not None:
                ys.append(v)
        if ys:
            return int(min(ys))
        v = ysl(r0, c0)
        return int(v) if v is not None else None

    for t in (tunnels or []):
        c = t.get("coords") or []
        if len(c) < 2:
            continue
        t["startF"] = portal(c[0], c[1])
        t["endF"] = portal(c[-1], c[-2])


def add_tunnel_blocks(blocks, tunnels, patch_bbox_latlon, nz, nx, *,
                      y_surf_land, h_res_block_m, surf_block=None, sea_mask=None,
                      road_mask=None,
                      floor_key="gray_concrete", base_key="stone",
                      light_key="sea_lantern", line_key="white_concrete",
                      core_always_covered: bool = False,
                      core_cover_slack: int | None = None,
                      cover_close_blocks: int | None = None) -> int:
    """OSM トンネル（tunnel=yes の highway/railway polyline）を地形に刳り貫いて生成。
    橋の逆処理: 両坑口の道路高を線形補間した床高に、道路幅×アーチ断面の空気トンネルを
    掘る（=air を後勝ちで上書き）。地形に埋まる区間のみ密閉(床+路盤+アーチ天井(石2厚)+
    壁(石2厚)+灯り)し、地表に出る区間は開削(cut-and-cover)にする。
    路面に白線。上の地表に写り込んだ道路色を周囲色で消す(#4)。坑口の手前 EXT block を延長
    し道路との障害物を除去(#3)、ただし交叉点に当たる手前で止める。床傾斜は範囲外含む全way端
    で決める(#5)。返り値: 触れた最大 y。

    core_always_covered : True で旧挙動（OSM way 本体は地形の有無に関係なく常に密閉）。
                          既定 False＝平坦地の tunnel=yes が地表に石の箱を生やさない。
    core_cover_slack    : コア区間の被覆判定を何 block 甘くするか（既定
                          TUNNEL_CORE_COVER_SLACK）。大きいほど密閉を維持しやすい。
    cover_close_blocks  : 密閉判定を station 方向に closing する長さ[block]（既定
                          TUNNEL_COVER_CLOSE_BLOCKS）。山中の小さな谷/DEMノイズで
                          密閉が途切れて穴が開くのを防ぐ。0 で無効。
    """
    import math
    core_slack = (TUNNEL_CORE_COVER_SLACK if core_cover_slack is None
                  else int(core_cover_slack))
    close_w = (TUNNEL_COVER_CLOSE_BLOCKS if cover_close_blocks is None
               else int(cover_close_blocks))
    WALL_H = 4                      # アーチの直壁部の高さ(block)
    CLEAR = 8                       # アーチ頂部(中央)の内空高(block)
    EXT = 20                        # 坑口より手前へ延長する長さ(block, 交叉点手前で短縮)
    SHELL = 2                       # 壁・天井・路盤の厚さ(block)
    seen: set = set()
    seen_light: set = set()
    ymax = [0]
    ROADISH = {"andesite", "gray_concrete", "light_gray_concrete", "stone", "smooth_stone",
               "black_concrete", "cyan_concrete", "polished_andesite", "gravel", "cobblestone"}

    def put(ix, iy, iz, key):
        if not (0 <= ix < nx and 0 <= iz < nz) or iy < 0 or iy > 500:
            return
        k = (ix, iy, iz)
        if k in seen:
            return
        seen.add(k)
        if key != "air" and iy > ymax[0]:
            ymax[0] = iy
        blocks.append(nbtlib.Compound({
            "pos": nbtlib.List[nbtlib.Int]([nbtlib.Int(ix), nbtlib.Int(iy), nbtlib.Int(iz)]),
            "state": block_id(key),
        }))

    def col(x, z):
        return int(round(x)), int(round(z))

    def terr(x, z):
        i, j = col(x, z)
        if 0 <= j < nz and 0 <= i < nx and np.isfinite(y_surf_land[j, i]):
            return int(y_surf_land[j, i])
        return None

    def arch_h(w, half_w):
        # 断面: |w|=0で CLEAR、端で WALL_H の半楕円アーチ天井高
        if half_w <= 0:
            return CLEAR
        r = min(1.0, abs(w) / (half_w + 1e-9))
        return WALL_H + int(round((CLEAR - WALL_H) * math.sqrt(max(0.0, 1.0 - r * r))))

    def fill_color(cx, cz, ox, oz, half_w):
        # 地表補完色: 走廊の外側を探索し道路灰色(ROADISH)でない自然色を優先（橋の _fill_color 相当）
        if surf_block is None:
            return None
        fb = None
        for off in range(half_w + 2, half_w + 12):
            for sgn in (1, -1):
                i, j = col(cx + ox * off * sgn, cz + oz * off * sgn)
                if not (0 <= j < nz and 0 <= i < nx):
                    continue
                sk = surf_block[j, i]
                if not sk or sk == "water" or (sea_mask is not None and sea_mask[j, i]):
                    continue
                if sk not in ROADISH:
                    return sk
                if fb is None:
                    fb = sk
        return fb

    def portal_floor(end_pt, in_pt):
        # 坑口側(トンネル外=道路側)へ 0..8 block の地形高 min（=道路/谷床）を床高に
        dx, dz = end_pt[0] - in_pt[0], end_pt[1] - in_pt[1]
        L = math.hypot(dx, dz)
        if L > 1e-6:
            dx, dz = dx / L, dz / L
        ys = [h for d in range(0, 9)
              if (h := terr(end_pt[0] + dx * d, end_pt[1] + dz * d)) is not None]
        if ys:
            return int(min(ys))
        t0 = terr(*end_pt)
        return t0 if t0 is not None else 1

    def end_floor(pts, from_start):
        # #5: 範囲外を含む全way端から、その端方向で最初に in-grid となる点で床高を採る
        rng = range(len(pts)) if from_start else range(len(pts) - 1, -1, -1)
        for i in rng:
            if terr(*pts[i]) is not None:
                inn = (pts[i + 1] if from_start and i + 1 < len(pts)
                       else pts[i - 1] if not from_start and i - 1 >= 0 else pts[i])
                return portal_floor(pts[i], inn)
        return 1

    def udir(a, b):
        dx, dz = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dz)
        return (dx / L, dz / L) if L > 1e-6 else (0.0, 0.0)

    def safe_ext_len(end_pt, dx, dz, half_w):
        # 坑口外側へ d=1..EXT 進め、走廊の**両側**(法線 half_w+2..+7 の左右どちらにも)に道路が
        # ある地点=横断路(交叉点)の手前で停止する。自路の直進継続や片側のみの並走車線では
        # 止めない(片側だけの検出は無視)。これが無いと坑口直近で常時0になり延長が効かない。
        if road_mask is None:
            return EXT
        ox, oz = -dz, dx
        for d in range(1, EXT + 1):
            px, pz = end_pt[0] + dx * d, end_pt[1] + dz * d
            sides = 0
            for side in (1, -1):
                # 自路(半幅±half_w)より遠い側方のみ見る＝横断路だけが当たる
                for off in range(half_w + 9, half_w + 16):
                    i, j = col(px + ox * off * side, pz + oz * off * side)
                    if 0 <= j < nz and 0 <= i < nx and road_mask[j, i]:
                        sides += 1
                        break
            if sides == 2:                  # 左右両側遠方に道路 = 横断路(交叉点)
                return max(0, d - 3)         # 交叉点の少し手前で止める
        return EXT

    def is_line(w, ostat, half_w):
        # 路面の白線（てきとうに）: 中央=破線(車線分離), 端寄り=実線(車道外側線)
        if w == 0:
            return (int(round(ostat)) // 3) % 2 == 0
        return half_w >= 3 and abs(w) == half_w - 1

    def open_approach(start_pt, dx, dz, fy_start, half_w):
        # 密閉延長の端から外側へ「開削」(壁/天井なし)して地表道路へ接続する。
        # 坑口を出た先〜道路の間に残る地形コブ(=道路との障害物)を路面＋頭上クリア
        # ランス分だけ air で除去し、床を1block/blockで地表(道路面)へ擦り付ける。
        # 中央列が地表道路(road_mask)に達したらそこで打ち切り(交叉点の手前まで掘る)。
        ox, oz = -dz, dx
        fy_prev = fy_start
        appr = min(EXT, 16)                  # 開削は密閉延長端→地表道路の最終接続のみ(短く)
        for d in range(1, appr + 1):
            cx, cz = start_pt[0] + dx * d, start_pt[1] + dz * d
            tc = terr(cx, cz)
            if tc is None:
                break
            fy = min(fy_prev + 1, tc) if tc > fy_prev else tc   # ≤45°で地表へ擦り上げ/下げ
            fy_prev = fy
            clear_top = max(tc, fy + CLEAR)
            ci, cj = col(cx, cz)
            on_road = (road_mask is not None and 0 <= cj < nz and 0 <= ci < nx
                       and bool(road_mask[cj, ci]))
            for w in range(-half_w, half_w + 1):
                wx, wz = cx + ox * w, cz + oz * w
                ix, iz = col(wx, wz)
                put(ix, fy, iz, line_key if is_line(w, d, half_w) else floor_key)
                put(ix, fy - 1, iz, base_key)
                twc = terr(wx, wz)
                yb = fy - 2
                while twc is not None and yb > twc and yb > fy - 10:
                    put(ix, yb, iz, base_key)
                    yb -= 1
                for yy in range(fy + 1, clear_top + 1):
                    put(ix, yy, iz, "air")          # 頭上の地形コブを除去（天井なし＝開削）
            if on_road:
                break

    n_t = 0
    for b in tunnels:
        pts = [_lonlat_to_grid_xy(la, lo, patch_bbox_latlon, nz, nx) for la, lo in b["coords"]]
        if len(pts) < 2:
            continue
        _sf, _ef = b.get("startF"), b.get("endF")
        if _sf is not None or _ef is not None:
            # 全域アンカー使用時はジオメトリを量子化して浮動小数のタイル依存を消す。
            # 隣接タイルの (x,z) は整数オフセットだけ違うので、丸めると seg/total/ostat が
            # bit 一致し、床高 fy がどのタイルで描いても同じになる。
            pts = [(round(x, 6), round(z, 6)) for x, z in pts]
        _wm = float(b.get("width_m") or 5.5)
        if "link" in (b.get("highway") or ""):
            _wm = min(_wm, 5.5)
        half_w = max(1, int(round((_wm / max(h_res_block_m, 0.1)) / 2.0)))
        seg0 = [math.hypot(b1[0] - a1[0], b1[1] - a1[1]) for a1, b1 in zip(pts, pts[1:])]
        total = sum(seg0)
        if total < 2.0:
            continue
        # トンネル長に応じて天井高/延長を可変化（短い=低天井(≈アーチ化前の高さ)・延長ほぼ無し,
        # 長い=高アーチ・延長大）。WALL_H/CLEAR/EXT を再代入＝閉包(arch_h/safe_ext_len/
        # open_approach)が呼出時にこの値を参照する。
        Lm = total * h_res_block_m                          # トンネル長[m]
        CLEAR = int(min(9, max(5, round(4.0 + Lm / 80.0)))) # 内空頂高: 短~5(≈アーチ前) → 長~8-9
        WALL_H = min(4, CLEAR - 1)                          # 直壁高(=端の内空高)
        EXT = int(min(70, max(1, round((Lm - 30.0) / 6.0)))) # 延長: 短~1-2(ほぼ無) → 長~50-55
        # 坑口床高: 全域DEMアンカー（assign_global_tunnel_anchors）を優先。無ければ
        # 従来のタイルローカル走査へフォールバック（後方互換）。
        _sf, _ef = b.get("startF"), b.get("endF")
        f0 = int(_sf) if _sf is not None else end_floor(pts, True)
        f1 = int(_ef) if _ef is not None else end_floor(pts, False)
        # #3: 両端を延長（交叉点手前で短縮）。延長部も壁・天井で密閉。station は元始点基準。
        d0 = udir(pts[1], pts[0]); d1 = udir(pts[-2], pts[-1])
        head_len = safe_ext_len(pts[0], d0[0], d0[1], half_w)
        tail_len = safe_ext_len(pts[-1], d1[0], d1[1], half_w)
        head = (pts[0][0] + d0[0] * head_len, pts[0][1] + d0[1] * head_len)
        tail = (pts[-1][0] + d1[0] * tail_len, pts[-1][1] + d1[1] * tail_len)
        epts = [head] + pts + [tail]
        seg = [math.hypot(b1[0] - a1[0], b1[1] - a1[1]) for a1, b1 in zip(epts, epts[1:])]
        n_t += 1
        # ── パス1: サンプル列を先に構築（被覆判定を station 方向へ平滑化するため2パス化）──
        samples = []            # (cx, cz, ox, oz, ostat, fy, tc, in_core)
        s_acc = 0.0
        for si in range(len(seg)):
            (x0, z0), (x1, z1) = epts[si], epts[si + 1]
            L = seg[si]
            if L < 1e-6:
                continue
            tx, tz = (x1 - x0) / L, (z1 - z0) / L
            ox, oz = -tz, tx
            n = max(1, int(L / 0.5))
            for k in range(n + 1):
                t = k / n
                cx, cz = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
                ostat = (s_acc + L * t) - head_len      # 元始点基準の station（延長部は負/total超）
                fy = (int(round(f0 + (f1 - f0) * (ostat / total)))
                      if total > 1e-6 else f0)
                tc = terr(cx, cz)
                if tc is None:
                    continue
                samples.append((cx, cz, ox, oz, ostat, fy, tc,
                                -1e-6 <= ostat <= total + 1e-6))
            s_acc += L
        # ── 被覆判定 ──
        # 旧: コア(OSMトンネル本体, 0≤ostat≤total)は tunnel=yes なので**無条件**に密閉。
        #     → 直上に地形が無い平坦地の tunnel=yes が地表に石の箱を生やしていた
        #       (way長400blockで地表より上の非air 2万個超)。
        # 新: コアも延長部と同じく「構造(壁+アーチ天井 SHELL厚)が地形に埋まるか」で判定し、
        #     埋まらない区間は開削(cut-and-cover)にする。コアは延長部より core_slack block
        #     だけ緩い閾値を使い、さらに station 方向の closing を掛けることで、
        #     山中の小さな谷や DEM ノイズで密閉が途切れて穴が開く退行を防ぐ。
        #     延長部の閾値(tc > fy+CLEAR+SHELL)は従来と完全に同じ。
        cov = [(s[6] + (core_slack if s[7] else 0)) >= s[5] + CLEAR + SHELL + (0 if s[7] else 1)
               for s in samples]
        if core_always_covered:                          # 旧挙動へのエスケープハッチ
            cov = [c or s[7] for c, s in zip(cov, samples)]
        elif close_w > 0:
            # station 方向 closing: 密閉に挟まれた close_w block 未満の非密閉区間は密閉に戻す
            # (サンプル間隔は 0.5 block なので窓は 2*close_w サンプル)
            wmax = max(1, int(round(close_w / 0.5)))
            i0 = 0
            ncov = len(cov)
            while i0 < ncov:
                if cov[i0]:
                    i0 += 1
                    continue
                j0 = i0
                while j0 < ncov and not cov[j0]:
                    j0 += 1
                if i0 > 0 and j0 < ncov and (j0 - i0) <= wmax:
                    for q in range(i0, j0):
                        cov[q] = True                    # 両側が密閉な短いギャップを埋める
                i0 = j0
        # ── パス2: レンダリング ──
        for (cx, cz, ox, oz, ostat, fy, tc, _in_core), covered in zip(samples, cov):
            for w in range(-half_w, half_w + 1):
                ix, iz = col(cx + ox * w, cz + oz * w)
                # 路面（白線 or 路盤色）＋路盤(SHELL厚)＋地表が床より低い延長部は床下を石で支持
                put(ix, fy, iz, line_key if is_line(w, ostat, half_w) else floor_key)
                for s in range(1, SHELL + 1):
                    put(ix, fy - s, iz, base_key)
                yb = fy - SHELL - 1
                while yb > tc and yb > fy - 12:
                    put(ix, yb, iz, base_key)
                    yb -= 1
                ah = arch_h(w, half_w)
                if covered:
                    for yy in range(fy + 1, fy + ah + 1):
                        put(ix, yy, iz, "air")           # 内空をアーチ状に刳り貫く
                    for s in range(1, SHELL + 1):
                        put(ix, fy + ah + s, iz, base_key)   # アーチ天井(SHELL厚)
                else:
                    # 開削: 路面上〜地表まで空に(頭上の地形コブ除去, 天井なし)
                    twc = terr(cx + ox * w, cz + oz * w)
                    top = max(fy + ah, twc if twc is not None else fy + ah)
                    for yy in range(fy + 1, top + 1):
                        put(ix, yy, iz, "air")
            if covered:
                # 壁(両側 SHELL厚)を床下〜天井上まで石で密閉
                for w in [(-half_w - s) for s in range(1, SHELL + 1)] + \
                         [(half_w + s) for s in range(1, SHELL + 1)]:
                    ix, iz = col(cx + ox * w, cz + oz * w)
                    for yy in range(fy - SHELL, fy + CLEAR + SHELL + 1):
                        put(ix, yy, iz, base_key)
            # #4: トンネル上に地形(山)がある被覆部は、地表トップの道路色を周囲色で消す
            if tc > fy + CLEAR + SHELL + 1:
                fc = fill_color(cx, cz, ox, oz, half_w)
                if fc is not None:
                    for w in range(-half_w, half_w + 1):
                        ty = terr(cx + ox * w, cz + oz * w)
                        if ty is not None and ty > fy + arch_h(w, half_w) + SHELL:
                            ix, iz = col(cx + ox * w, cz + oz * w)
                            put(ix, ty, iz, fc)
            # 照明: 刳り貫き後に直接配置(last-winsでairを上書き; put/seenだと隣接サンプルのairに先取り
            #       されて消えるため)。天井直下中央に一定間隔。被覆(密閉)部のみ。
            if covered and int(round(ostat)) % 6 == 0:
                ix, iz = col(cx, cz)
                ly = fy + CLEAR
                if 0 <= ix < nx and 0 <= iz < nz and (ix, ly, iz) not in seen_light:
                    seen_light.add((ix, ly, iz))
                    blocks.append(nbtlib.Compound({
                        "pos": nbtlib.List[nbtlib.Int](
                            [nbtlib.Int(ix), nbtlib.Int(ly), nbtlib.Int(iz)]),
                        "state": block_id(light_key)}))
        # #3b: 密閉延長端から地表道路まで開削して接続（坑口外の障害物を除去）
        fy_h = (int(round(f0 + (f1 - f0) * (-head_len / total))) if total > 1e-6 else f0)
        fy_t = (int(round(f0 + (f1 - f0) * ((total + tail_len) / total))) if total > 1e-6 else f0)
        open_approach(head, d0[0], d0[1], fy_h, half_w)
        open_approach(tail, d1[0], d1[1], fy_t, half_w)
    if n_t:
        print(f"  [tunnel] OSMトンネル {n_t} 本を密閉刳り貫き (内空頂高/延長は長さ依存:"
              f"短5block・ほぼ延長無→長8-9block・延長~40, 壁天井{SHELL}厚, "
              f"交叉点手前で停止, 白線+照明+地表道路消去)")
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
    # 地盤アンダーフィル深さの上限[block]。None(既定)＝隣接セルとの段差に応じて可変
    # （崖面のすきまを塞ぐ）。int を渡すとその値で一律クランプ＝旧挙動
    # （underfill_cap=deep_ground で従来と完全一致）。
    underfill_cap: int | None = None,
    # 旧挙動エスケープハッチ（add_tunnel_blocks / add_power_blocks へそのまま中継）
    tunnel_core_always_covered: bool = False,
    tunnel_core_cover_slack: int | None = None,
    tunnel_cover_close_blocks: int | None = None,
    power_clip_spans_to_grid: bool = True,
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
    building_roof_solid: list | None = None,   # 建物 id → 屋根を型単色にしオルソ焼込無効
    roof_color_tol: float = 55.0,
    color_building_roofs: bool = False,
    terrain_skirt_cells: int = 0,   # >0: ワールド外周この幅を斜面で下ろし境界の崖を無くす(単一タイル前提)
    wall_block: str = "white_concrete",
    window_block: str = "glass",
    floor_height: int = 5,
    floor_block: str = "light_gray_concrete",
    interior_light: str = "sea_lantern",
    building_style_keys: list | None = None,
    building_facade_by_id: list | None = None,    # 建物 id → 外壁装飾スペック(アーキタイプ由来)
    hollow_buildings: bool = True,
    legend_layer: bool = False,
    surface_grid_override: np.ndarray | None = None,
    bridges: list | None = None,
    tunnels: list | None = None,
    powerlines: list | None = None,
    power_towers: list | None = None,
    rails: list | None = None,
    parkings: list | None = None,
    ortho_rgb: np.ndarray | None = None,
    patch_bbox_latlon: tuple | None = None,
    road_block: str = "andesite",
    road_major_mask: np.ndarray | None = None,
    road_minor_block: str = "gravel",
    # 道路の「一番外側」に引く 1 ブロック境界線（普通=灰色コンクリ / 小路=青緑テラコッタ）
    road_edge_major_block: str = "gray_concrete",
    road_edge_minor_block: str = "cyan_terracotta",
    road_edge_close_iter: int = 9,
    road_edge_hole_fill_cells: int = 800,         # closing 残穴のうち極小穴(交差点ダイヤ)を埋める閾値
    road_curb_osm_mask: np.ndarray | None = None,  # OSM道路センターライン塗りつぶし回廊(交差点判定用)
    road_ground_other_mask: np.ndarray | None = None,  # 非高架の地上道路のみ(橋直下で残す別道路の同定)
    road_cross_extra_mask: np.ndarray | None = None,   # 橋軸に横断する道(FGD農道/OSM生活道)＝橋直下でも残す
    road_unpaved_mask: np.ndarray | None = None,       # 未舗装道(service/track/農道/庭園路/歩道)→砂利+土の小径
    road_unpaved_block: str = "coarse_dirt",
    road_path_block: str = "dirt_path",
    road_under_building: bool = True,             # 道路が端から端まで横断する建物(連絡通路)の1Fを抜いて通す
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
         stone で埋め、平地は 2 ブロックで済みブロック数が激減し、崖面は隣接セルの
         底まで埋めて見える穴を塞ぐ。
         （従来は全セル一律 `deep_ground` 本＝地下 stone を無駄に増やしていた）
         上限は `underfill_cap`（None=段差に応じて可変, 既定）。旧来の固定クランプに
         戻すには `underfill_cap=deep_ground` を渡す。

    旧挙動エスケープハッチ（すべて既定=新挙動。CLI からは make_nbt_hd の
    `--underfill-cap` / `--tunnel-core-always-covered` / `--tunnel-core-cover-slack` /
    `--tunnel-cover-close-blocks` / `--power-no-clip-spans` で到達できる）:
      underfill_cap              : int でアンダーフィル深さを一律クランプ（旧挙動）
      tunnel_core_always_covered : True で OSM way 本体を無条件密閉（旧挙動）
      tunnel_core_cover_slack    : コア被覆判定の緩さ[block]（None=既定値）
      tunnel_cover_close_blocks  : 被覆判定の station 方向 closing 長[block]（0 で無効）
      power_clip_spans_to_grid   : False で端点がタイル外の径間を丸ごと捨てる（旧挙動）

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
    # terrain skirt: ワールド外周 N セルを端に向け斜面で下ろし、境界の垂直な崖を無くす。
    #   単一タイル(campus)前提＝dem_ds の外周がそのままワールド端。--tiles 併用時は継ぎ目にも
    #   斜面が出るため使わないこと。陸セルのみ対象(海/水面は既に傾斜)。
    if terrain_skirt_cells and terrain_skirt_cells > 0:
        _N = int(terrain_skirt_cells)
        _dz = np.minimum(np.arange(nz), nz - 1 - np.arange(nz))[:, None]
        _dx = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))[None, :]
        _d = np.minimum(_dz, _dx)                       # 最寄り端までのセル距離
        _frac = np.clip(_d / float(_N), 0.0, 1.0)
        _y_skirt = (_lift + (y_surf_land - _lift) * _frac).round().astype(y_surf_land.dtype)
        _inzone = (_d < _N) & land_mask
        y_surf_land = np.where(_inzone, np.minimum(y_surf_land, _y_skirt), y_surf_land)
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

    # 駐車場(OSM amenity=parking)を先に描く。道路をこの後に上書きすることで「領域競合=道路優先」
    # （駐車場は粒度が粗いので、駐車場内でも道路の安山岩/砂利を生成する）。オルソ有=写真アスファルト色を
    # 残し行ベースライン/境界を合成、無=black_concrete。車は立体化せず色のみ。
    # オルソRGBを surf_block 解像度(nz,nx)へ整合（駐車場と屋根色集約で共用）
    _ortho_ds = ortho_rgb
    if _ortho_ds is not None and _ortho_ds.shape[:2] != (nz, nx):
        if _ortho_ds.shape[:2] == dem_patch.shape:
            _ortho_ds = _ortho_ds[:nz * factor, :nx * factor].reshape(
                nz, factor, nx, factor, 3)[:, factor // 2, :, factor // 2]
        else:
            _ortho_ds = None
    _parking_cars = []
    parking_area = None
    parking_boundary = None
    _road_filled = None                  # closing済みの塗り潰し道路(連絡通路の貫通判定/1F削り用)
    _pk_boundary_key = "gray_concrete"   # 駐車場境界線の色（render_parking の boundary_key と一致）
    if parkings and patch_bbox_latlon is not None:
        from parking_render import render_parking
        land_for_pk = ~np.isnan(dem_ds) & ~(np.where(np.isnan(dem_ds), 0.0, dem_ds) <= sea_level_m)
        _parking_cars, parking_area, parking_boundary = render_parking(
            parkings, patch_bbox_latlon, nz, nx,
            surf_block=surf_block, y_surf_land=y_surf_land, land_mask=land_for_pk,
            ortho_rgb=_ortho_ds, h_res_block_m=h_res_block, boundary_key=_pk_boundary_key,
        )

    # 道路を地表に上書き（陸セルのみ）。RdEdg=道路縁バッファなので「左右2本の帯＋中央オルソ」。
    # 細道=road_minor_block(砂利)、幹線=road_block(andesite舗装)。駐車場の後＝道路優先。
    if road_mask is not None and road_mask.shape == surf_block.shape:
        land_for_road = ~np.isnan(dem_ds) & ~(np.where(np.isnan(dem_ds), 0.0, dem_ds) <= sea_level_m)
        # トンネル直上(被覆部=山の中)の地表道路を消す。トンネルは地下なので山の上に道路を
        # 描かない（木も生える）。各トンネルの coords を grid 投影し、床グレード(startF/endF、
        # 無ければ坑口セルの地表)を区間長で内挿。回廊(±(道路半幅+CORRIDOR_PAD))内で地表が
        # 床+COVER_MARGIN より高いセル=被覆部として road_mask/road_major_mask から除去し、
        # さらにオルソ由来の路面色を最近傍の回廊外地表で埋め直す。坑口/開削部は地表≈床なので
        # 残り、回廊外の別道路も残る。
        if (tunnels and patch_bbox_latlon is not None
                and TUNNEL_SURFACE_ROAD_COVER_MARGIN >= 0):
            _rjj, _rii = np.mgrid[0:nz, 0:nx]
            _rjj = _rjj.astype(np.float32); _rii = _rii.astype(np.float32)
            _tcov = np.zeros((nz, nx), dtype=bool)
            _cmarg = float(TUNNEL_SURFACE_ROAD_COVER_MARGIN)
            for _tb in tunnels:
                _co = _tb.get("coords") or []
                if len(_co) < 2:
                    continue
                _p = [_lonlat_to_grid_xy(_la, _lo, patch_bbox_latlon, nz, nx)
                      for _la, _lo in _co]

                def _floor_ref(_gp, _fb):
                    if _fb is not None:
                        return float(_fb)
                    _c = int(round(_gp[0])); _r = int(round(_gp[1]))
                    if 0 <= _r < nz and 0 <= _c < nx:
                        return float(y_surf_land[_r, _c])
                    return 0.0

                _fs = _floor_ref(_p[0], _tb.get("startF"))
                _fe = _floor_ref(_p[-1], _tb.get("endF"))
                _sl = [((_p[_k + 1][0] - _p[_k][0]) ** 2 + (_p[_k + 1][1] - _p[_k][1]) ** 2) ** 0.5
                       for _k in range(len(_p) - 1)]
                _tot = sum(_sl) or 1.0
                _fv = [_fs]; _acc = 0.0
                for _s in _sl:
                    _acc += _s
                    _fv.append(_fs + (_fe - _fs) * (_acc / _tot))
                _wm = float(_tb.get("width_m") or 5.5)
                _hw = (max(1.0, (_wm / max(h_res_block, 0.1)) / 2.0)
                       + float(TUNNEL_SURFACE_ROAD_CORRIDOR_PAD))
                _hw2 = _hw * _hw
                for _k in range(len(_p) - 1):
                    _ax, _az = _p[_k]; _bx, _bz = _p[_k + 1]
                    _dx, _dz = _bx - _ax, _bz - _az
                    _l2 = _dx * _dx + _dz * _dz
                    if _l2 < 1e-6:
                        continue
                    _t = np.clip(((_rii - _ax) * _dx + (_rjj - _az) * _dz) / _l2, 0.0, 1.0)
                    _d2 = (_rii - (_ax + _t * _dx)) ** 2 + (_rjj - (_az + _t * _dz)) ** 2
                    _fl = _fv[_k] + _t * (_fv[_k + 1] - _fv[_k])
                    _tcov |= (_d2 <= _hw2) & (y_surf_land > _fl + _cmarg)
            if _tcov.any():
                # ① 道路帯(road_mask/road_major_mask)から除去 → 地表に道路を塗らない
                road_mask = road_mask & (~_tcov)
                if road_major_mask is not None and road_major_mask.shape == (nz, nx):
                    road_major_mask = road_major_mask & (~_tcov)
                # ② オルソ由来の路面色(道路中央=衛星画像の舗装。road_mask ではないので①では
                #    消えない)も、被覆部の回廊内を最近傍の“回廊外”地表で埋め直して山肌に
                #    道路が見えないようにする（周囲の森林地表に馴染ませる）。
                from scipy.ndimage import distance_transform_edt as _edt
                _, (_iy, _ix) = _edt(_tcov, return_indices=True)
                surf_block[_tcov] = surf_block[_iy[_tcov], _ix[_tcov]]
        # ── 橋デッキ直下の地表道路を除去（高架の路面が地面に二重に出るのを防ぐ）。ただし
        #    デッキ足元にある「別の地上道路」（OSM 非bridge道路 = road_curb_osm_mask のうち
        #    橋中心線から外れるもの）は残す/塗る。FGD は高架/地上を区別しないので OSM で同定。──
        if (bridges and patch_bbox_latlon is not None and BRIDGE_UNDERROAD_REMOVE):
            _bjj, _bii = np.mgrid[0:nz, 0:nx]
            _bjj = _bjj.astype(np.float32); _bii = _bii.astype(np.float32)
            _bcov = np.zeros((nz, nx), dtype=bool)     # デッキ足元(広)＝処理領域
            _bnar = np.zeros((nz, nx), dtype=bool)     # 橋中心線(狭)＝橋自身の車線
            for _bb in bridges:
                _bc = _bb.get("coords") or []
                if len(_bc) < 2:
                    continue
                _bp = [_lonlat_to_grid_xy(_la, _lo, patch_bbox_latlon, nz, nx) for _la, _lo in _bc]
                _wm = float(_bb.get("width_m") or 5.5)
                _hw = max(1.5, (_wm / max(h_res_block, 0.1)) / 2.0 + BRIDGE_UNDERROAD_PAD)
                _hw2 = _hw * _hw
                _nw2 = max(1.2, (_wm / max(h_res_block, 0.1)) / 2.0) ** 2   # 狭(車線幅)
                for _k in range(len(_bp) - 1):
                    _ax, _az = _bp[_k]; _bx, _bz = _bp[_k + 1]
                    _dx, _dz = _bx - _ax, _bz - _az
                    _l2 = _dx * _dx + _dz * _dz
                    if _l2 < 1e-6:
                        continue
                    _t = np.clip(((_bii - _ax) * _dx + (_bjj - _az) * _dz) / _l2, 0.0, 1.0)
                    _d2 = (_bii - (_ax + _t * _dx)) ** 2 + (_bjj - (_az + _t * _dz)) ** 2
                    _bcov |= (_d2 <= _hw2)
                    _bnar |= (_d2 <= _nw2)
            if _bcov.any():
                # 「別の地上道路」＝非高架 OSM 道路(road_ground_other_mask)がデッキ足元に有る所。
                # tags(bridge/layer)で高架 way を除いてあるので、橋自身の路面は含まれない＝
                # 橋の路面だけを地面から消し、真下/斜め下を通る別道路は way 単位で保持できる。
                _grd = (road_ground_other_mask if (road_ground_other_mask is not None
                        and road_ground_other_mask.shape == (nz, nx)) else None)
                if _grd is not None:
                    _keep = _grd & _bcov
                    _mode = "way-id(非高架OSM)"
                else:
                    # フォールバック(OSM回廊取得不可時): 狭い中心線だけ残すヒューリスティック
                    _osmr = (road_curb_osm_mask if (road_curb_osm_mask is not None
                             and road_curb_osm_mask.shape == (nz, nx)) else np.zeros((nz, nx), bool))
                    _keep = _osmr & (~_bnar)
                    _mode = "fallback(中心線)"
                # 橋軸に横断する道(FGD農道/OSM生活道)を追加で残す。並走(高架自身)は角度で落ちる。
                _cross = (road_cross_extra_mask if (road_cross_extra_mask is not None
                          and road_cross_extra_mask.shape == (nz, nx)) else None)
                _ncross = 0
                if _cross is not None:
                    _cross = _cross & _bcov
                    _ncross = int((_cross & (~_keep)).sum())
                    _keep = _keep | _cross
                # 横断道が除去で細切れにならないよう、deck 内で keep を軽く連結(closing)。
                # ただし膨張が高架自身の帯を埋め戻さないよう、_keep の近傍1セルのみ対象。
                if _keep.any():
                    from scipy.ndimage import binary_closing as _bclose_k
                    _keep = _bclose_k(_keep, iterations=1, border_value=0) & _bcov
                _brem = _bcov & (~_keep)                # 除去 = デッキ足元 − 別道路
                print(f"  [bridge-underroad] deck足元={int(_bcov.sum())} "
                      f"別道路keep={int(_keep.sum())}(内 横断追加={_ncross}) 除去={int(_brem.sum())} [{_mode}]")
                road_mask = road_mask & (~_brem)
                if road_major_mask is not None and road_major_mask.shape == (nz, nx):
                    road_major_mask = road_major_mask & (~_brem)
                from scipy.ndimage import distance_transform_edt as _edt2
                _, (_iy2, _ix2) = _edt2(_brem, return_indices=True)
                surf_block[_brem] = surf_block[_iy2[_brem], _ix2[_brem]]  # 高架写り込みを周囲色に
                import os as _os_ur
                if _os_ur.environ.get("BRIDGE_UNDERROAD_DUMP"):
                    np.savez(_os_ur.environ["BRIDGE_UNDERROAD_DUMP"],
                             bcov=_bcov, brem=_brem, keep=_keep,
                             bnar=_bnar,
                             osmr=(road_curb_osm_mask if road_curb_osm_mask is not None
                                   else np.zeros((nz, nx), bool)),
                             bbox=np.array(patch_bbox_latlon, dtype=np.float64))
                    print(f"  [bridge-underroad] dump -> {_os_ur.environ['BRIDGE_UNDERROAD_DUMP']}")
        surf_block[road_mask & land_for_road] = road_minor_block
        if road_major_mask is not None and road_major_mask.shape == surf_block.shape:
            surf_block[road_major_mask & land_for_road] = road_block
        # 未舗装道(service/track/農道/庭園路/歩道) → 砂利+土の小径。舗装(幹線)は上書きしない。
        if road_unpaved_mask is not None and road_unpaved_mask.shape == surf_block.shape:
            _unp = road_unpaved_mask & land_for_road
            if road_major_mask is not None and road_major_mask.shape == surf_block.shape:
                _unp = _unp & (~road_major_mask)
            if _unp.any():
                _rj, _ri = np.mgrid[0:nz, 0:nx]
                _h = (_ri.astype(np.int64) * 17 + _rj.astype(np.int64) * 31) % 100
                surf_block[_unp] = road_unpaved_block                 # 主: coarse_dirt
                surf_block[_unp & (_h < 40)] = road_path_block        # 小径感: dirt_path 4割
                surf_block[_unp & (_h >= 88)] = "gravel"              # 砂利アクセント 1割強

        # ── 道路の「一番外側」に 1 ブロックの境界線（curb）を引く ──
        #   道路中央(オルソ)の細い隙間だけ closing で埋めて左右の帯を一体化 → その外周 1 セル。
        #   中央は埋めず line セルにのみ描くので、オルソ路面はそのまま残る。
        #   色: 普通道路(幹線)=road_edge_major_block / 小路=road_edge_minor_block。
        from scipy.ndimage import (binary_closing as _bclose, binary_erosion as _berode,
                                   binary_fill_holes as _bfill, label as _blabel)
        it = max(1, int(road_edge_close_iter))
        band = _bclose(road_mask, iterations=it, border_value=0)
        # 交差点の偽枠線対策: closing が広い道路の交差点中心に残す「閉じ穴(ダイヤ)」の外周が
        #   道路を横切る余分な線になる。閉じ穴のうち ①OSM道路センターライン回廊の上にある穴
        #   (=交差点で連続する同一道路) ②極小穴 だけを埋め、実街区の内側境界は残す。
        holes = _bfill(band) & ~band
        if holes.any():
            hlab, hn = _blabel(holes)
            hsz = np.bincount(hlab.ravel())
            osm_c = (road_curb_osm_mask if (road_curb_osm_mask is not None
                     and road_curb_osm_mask.shape == band.shape) else None)
            thr = max(0, int(road_edge_hole_fill_cells))
            for k in range(1, hn + 1):
                hk = (hlab == k)
                area = int(hsz[k])
                on_road = (int((hk & osm_c).sum()) / max(area, 1)) if osm_c is not None else 0.0
                if area < thr or on_road > 0.3:
                    band |= hk
        _road_filled = band                       # 連絡通路の貫通判定/1F削りに使う塗り潰し道路
        edge_line = band & ~_berode(band) & land_for_road
        # 駐車場領域内には curb を引かない（駐車場自身の境界線と二重になり見にくいため）。
        if parking_area is not None and parking_area.shape == edge_line.shape:
            edge_line = edge_line & ~parking_area
        rmaj = (road_major_mask if (road_major_mask is not None
                and road_major_mask.shape == surf_block.shape)
                else np.zeros_like(road_mask))
        near_major = binary_dilation(_bclose(rmaj, iterations=it, border_value=0), iterations=2)
        line_minor = edge_line & ~near_major
        line_major = edge_line & near_major
        # 未舗装道(農道/小径)には縁石(cyan)を引かない：素地に馴染ませる。
        if road_unpaved_mask is not None and road_unpaved_mask.shape == surf_block.shape:
            line_minor = line_minor & ~binary_dilation(road_unpaved_mask, iterations=2)
        surf_block[line_minor] = road_edge_minor_block
        surf_block[line_major] = road_edge_major_block

    # 駐車場の境界線を道路の後に再描画して保護（道路優先で路面は道路だが、駐車場の縁取りは消さない）。
    if parking_boundary is not None and parking_boundary.shape == surf_block.shape:
        surf_block[parking_boundary] = _pk_boundary_key

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
    # 埋めて見える穴を塞ぐ。
    #
    # 上限クランプ:
    #   旧 = deep_ground(既定8)で固定 → 8block(=5.3m)超の段差で崖面に横から見える
    #        すきまが開いていた（20m崖で最大20block・穴800個を実測）。
    #   新 = 隣接セルとの段差そのもの(+UNDERFILL_EXTRA)を上限にする＝段差が大きいセル
    #        だけ深く埋める。平地の柱の深さは旧と変わらない(2block)ので総ブロック増は
    #        崖の面積分だけ。UNDERFILL_HARD_CAP は暴走防止の安全弁。
    #   underfill_cap に int を渡すと旧挙動（その値で一律クランプ）に戻せる。
    from scipy.ndimage import minimum_filter as _min_filter
    _yfm = np.where(land_mask, y_surf_land.astype(np.float32), np.float32(1e9))
    _neigh_min = _min_filter(_yfm, size=3, mode="nearest")
    _drop = y_surf_land.astype(np.int32) - _neigh_min.astype(np.int32)
    _umin = int(max(1, UNDERFILL_MIN))
    if underfill_cap is None:
        # 自動: 上限は UNDERFILL_HARD_CAP。deep_ground はこれより大きい値を渡したときだけ
        # 上限を「引き上げる」方向に効く（小さくして崖に穴を開け直したい場合は
        # underfill_cap を明示すること）。
        _cap = int(max(_umin, UNDERFILL_HARD_CAP, int(deep_ground)))
        _under_depth = np.clip(_drop + int(UNDERFILL_EXTRA), _umin, _cap).astype(np.int32)
    else:                                    # 旧挙動: 一律クランプ
        _under_depth = np.clip(_drop + 1, 2, int(max(2, underfill_cap))).astype(np.int32)
    # --- トンネル坑口“周り”の下方向 増し厚（坑口壁際/床下のすきま対策） ---
    # 坑口では地表が道路レベルまで下げられ、その真下にシェル内部の空洞(すきま)が残りやすい。
    # 可変アンダーフィルは近傍最低段差で決まるため、坑口のような“局所的に低い平面”は
    # 深さ≈最小(2)しか埋まらず床下に穴が開く。coords 両端(=坑口)と内隣接点から軸方向を求め、
    # 坑口を中心に内側 REACH_IN・外側 REACH_OUT 伸ばした半径 R のカプセル内の陸セルの
    # アンダーフィル深さを最低 DEPTH[block] へ引き上げて床下を stone で塞ぐ
    # (UNDERFILL_HARD_CAP でクランプ)。刳り貫きは後段なので増し厚は坑口の周り/床下だけに残る。
    if (tunnels and patch_bbox_latlon is not None
            and int(TUNNEL_PORTAL_UNDERFILL_RADIUS) > 0):
        _pr2 = float(TUNNEL_PORTAL_UNDERFILL_RADIUS) ** 2
        _pdep = int(min(max(1, TUNNEL_PORTAL_UNDERFILL_DEPTH), UNDERFILL_HARD_CAP))
        _rin = float(max(0, TUNNEL_PORTAL_UNDERFILL_REACH_IN))
        _rout = float(max(0, TUNNEL_PORTAL_UNDERFILL_REACH_OUT))
        _pjj, _pii = np.mgrid[0:nz, 0:nx]
        _pjj = _pjj.astype(np.float32); _pii = _pii.astype(np.float32)
        _pmask = np.zeros((nz, nx), dtype=bool)

        def _portal_capsule(_p_ll, _n_ll):
            # 坑口 _p_ll と内隣接 _n_ll(トンネル内側)から軸方向 u を求め、坑口を中心に
            # [-REACH_OUT, +REACH_IN]u 伸ばした線分から半径 R 以内の grid セルを返す。
            _px, _pz = _lonlat_to_grid_xy(_p_ll[0], _p_ll[1], patch_bbox_latlon, nz, nx)
            _nxg, _nzg = _lonlat_to_grid_xy(_n_ll[0], _n_ll[1], patch_bbox_latlon, nz, nx)
            _ux, _uz = _nxg - _px, _nzg - _pz            # 内向き
            _un = (_ux * _ux + _uz * _uz) ** 0.5
            if _un < 1e-6:                               # 退化(重複点) → 円板
                _ax, _az, _bx, _bz = _px, _pz, _px, _pz
            else:
                _ux, _uz = _ux / _un, _uz / _un
                _ax, _az = _px - _ux * _rout, _pz - _uz * _rout   # 外(進入路)端
                _bx, _bz = _px + _ux * _rin, _pz + _uz * _rin     # 内(トンネル)端
            _dx, _dz = _bx - _ax, _bz - _az
            _l2 = _dx * _dx + _dz * _dz
            if _l2 < 1e-6:
                _d2 = (_pii - _ax) ** 2 + (_pjj - _az) ** 2
            else:
                _t = np.clip(((_pii - _ax) * _dx + (_pjj - _az) * _dz) / _l2, 0.0, 1.0)
                _d2 = (_pii - (_ax + _t * _dx)) ** 2 + (_pjj - (_az + _t * _dz)) ** 2
            return _d2 <= _pr2

        for _tb in tunnels:
            _co = _tb.get("coords") or []
            if len(_co) < 2:
                continue
            _pmask |= _portal_capsule(_co[0], _co[1])
            _pmask |= _portal_capsule(_co[-1], _co[-2])
        _pmask &= land_mask                          # 海/水柱は対象外
        if _pmask.any():
            _under_depth = np.where(
                _pmask, np.maximum(_under_depth, _pdep), _under_depth).astype(np.int32)
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
        def _roof_solid(_id):      # その棟が「屋根を型単色・オルソ非焼込」指定か
            return bool(building_roof_solid is not None
                        and 0 <= _id < len(building_roof_solid) and building_roof_solid[_id])
        roof_by_id = None          # 各建物の代表屋根キー
        roof_dom_rgb = None        # 代表屋根キーの RGB（同系統判定用）
        if building_id is not None and building_roof_keys is not None:
            roof_by_id = list(building_roof_keys)
            if color_building_roofs:
                from collections import Counter
                from block_palette import BLOCKS as _BP
                bsel = (building_id >= 0) & building_mask & land_mask
                if _ortho_ds is not None and _ortho_ds.shape[:2] == surf_block.shape:
                    # 屋根色: footprint の生オルソ平均RGBを増彩度マッチ(水色/黄緑の色落ち対策)。
                    # surf_block の最頻ブロック(=通常彩度マッチ)だと淡い青タイル/黄緑屋根がグレーに落ちる。
                    from ortho_surface import classify_rgb_to_palette_saturated as _csat
                    ids = building_id[bsel]
                    cols = _ortho_ds[bsel].astype(np.float64)
                    nb = len(roof_by_id)
                    rgb_sum = np.zeros((nb, 3)); cnt = np.zeros(nb)
                    np.add.at(rgb_sum, ids, cols); np.add.at(cnt, ids, 1)
                    have = cnt > 0
                    mean_rgb = np.zeros((nb, 3), np.uint8)
                    mean_rgb[have] = (rgb_sum[have] / cnt[have, None]).round().astype(np.uint8)
                    matched = _csat(mean_rgb.reshape(1, nb, 3)).reshape(nb)
                    for _id in range(nb):
                        if have[_id] and not _roof_solid(_id):   # 新設建物はオルソを焼かず型単色を保持
                            roof_by_id[_id] = str(matched[_id])
                else:
                    acc: dict[int, Counter] = {}
                    for _id, _c in zip(building_id[bsel].tolist(),
                                       np.asarray(surf_block)[bsel].tolist()):
                        acc.setdefault(_id, Counter())[_c] += 1
                    for _id, c in acc.items():
                        if 0 <= _id < len(roof_by_id) and not _roof_solid(_id):
                            roof_by_id[_id] = c.most_common(1)[0][0]
                roof_dom_rgb = [(_BP[k][1] if k in _BP else (128, 128, 128))
                                for k in roof_by_id]
        _tol2 = float(roof_color_tol) * float(roof_color_tol)
        from block_palette import BLOCKS as _BP2

        # ── 連絡通路(渡り廊下)の検出と 1F 抜き ──
        #   FGD道路バッファは広場/駐車場まで含み広大で、「道路がfootprintに重なる」だけでは普通の道路脇
        #   建物まで誤検出する(9基準で検証済・分離不可)。実際の連絡通路は『2棟の建物を繋ぐ細長い渡り廊下を
        #   道路が貫く』形状。そこで次の4条件で検出する:
        #     (1) 細長い         : footprint座標のPCA主軸/副軸比 >= 2.2
        #     (2) 橋渡し         : 膨張2セルで隣接する別建物が 2 種以上(行き止まり/単独棟を排除)
        #     (3) 道路が貫く     : footprint ∩ 塗り潰し道路 >= 10 セル
        #     (4) 道路被覆率高い : (footprint∩道路)/footprint >= 0.37
        #         ＝渡り廊下は床の大半を道路が占める。端をかすめるだけの大建物(rib/nfp≤0.22)を排除。
        #         南部bboxで真の3棟は 0.41/0.69/0.85, 過剰反応の大建物は ≤0.22 と明確に分離。0.37は
        #         「3棟がギリギリ(0.41)認識できるより少しだけ余裕」を持たせた値(ユーザ指定)。
        #   検出した渡り廊下の footprint∩道路 の 1F を抜いて下を通れるようにする(上階は通路天井に残る)。
        tunnel_cells = None
        if (road_under_building and building_id is not None and _road_filled is not None
                and _road_filled.shape == building_mask.shape):
            from scipy.ndimage import binary_dilation as _tdil
            tunnel_cells = np.zeros((nz, nx), bool)
            road_in_bld = _road_filled & building_mask
            _n_tunnel = 0
            RIB_FRAC = 0.37          # (4) 道路被覆率の下限(端かすめ大建物の過剰検出を排除)
            _dbg = bool(__import__("os").environ.get("TUNNEL_DEBUG"))
            _dbg_rows = []
            _bids = np.unique(building_id[(building_id >= 0) & building_mask])
            for _b in _bids.tolist():
                fp = (building_id == _b)
                nfp = int(fp.sum())
                if nfp < 12:
                    continue
                rib = road_in_bld & fp
                ribn = int(rib.sum())
                if _dbg and ribn >= 1:
                    ys2, xs2 = np.where(fp)
                    pts = np.column_stack((ys2 - ys2.mean(), xs2 - xs2.mean())).astype(float)
                    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
                    pca_d = (ev[0] / max(float(ev[1]), 1e-6)) ** 0.5
                    nb_d = _tdil(fp, iterations=2) & ~fp
                    neigh_d = int(np.unique(building_id[nb_d & (building_id >= 0)]).size)
                    cy, cx = float(ys2.mean()), float(xs2.mean())
                    _dbg_rows.append((nfp, ribn, round(pca_d, 2), neigh_d,
                                      round(ribn / max(nfp, 1), 2), int(cy), int(cx)))
                if ribn < 10 or ribn < nfp * RIB_FRAC:
                    continue              # 道路が貫いていない/被覆率低い(端かすめ大建物) → 対象外
                ys2, xs2 = np.where(fp)                        # (1) 細長さ = PCA主軸/副軸
                pts = np.column_stack((ys2 - ys2.mean(), xs2 - xs2.mean())).astype(float)
                ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
                pca = (ev[0] / max(float(ev[1]), 1e-6)) ** 0.5
                if pca < 2.2:
                    continue                                  # 細長くない(本体/道路脇の普通建物) → 対象外
                nb = _tdil(fp, iterations=2) & ~fp            # (2) 2棟以上の別建物を橋渡しするか
                neigh = np.unique(building_id[nb & (building_id >= 0)])
                if int(neigh.size) < 2:
                    continue                                  # 行き止まり/単独棟 → 対象外
                tunnel_cells |= rib
                _n_tunnel += 1
            if _dbg:
                print(f"  [連絡通路-DBG] 候補(nfp>=12 & road接触) {len(_dbg_rows)}件 "
                      f"[nfp, rib, pca, neigh, rib/nfp, cz, cx] (rib降順):")
                for r in sorted(_dbg_rows, key=lambda t: -t[1]):
                    _pass = (r[1] >= 10 and r[1] >= r[0] * RIB_FRAC
                             and r[2] >= 2.2 and r[3] >= 2)
                    print(f"      {'PASS' if _pass else 'rej '} nfp={r[0]:4d} rib={r[1]:3d} "
                          f"pca={r[2]:5.2f} neigh={r[3]} rib/nfp={r[4]:.2f} @z{r[5]},x{r[6]}")
            if _n_tunnel:
                print(f"  [連絡通路] 渡り廊下(細長×2棟橋渡し×道路貫通)を {_n_tunnel}棟検出し1Fを抜いて通路化 "
                      f"({int(tunnel_cells.sum())} cells)")

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
            if roof_dom_rgb is not None and 0 <= bid_c < len(roof_by_id) and not _roof_solid(bid_c):
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
            # アーキタイプ由来の外壁装飾スペック(trim=角柱/帯, win=窓材, parapet, 店頭, 床帯)
            fac = (building_facade_by_id[bid_c]
                   if (building_facade_by_id is not None and 0 <= bid_c < len(building_facade_by_id))
                   else None)
            trim_kind = fac["trim"] if fac else wall_kind
            win_kind = fac["window"] if fac else window_block
            shopfront = bool(fac["shopfront"]) if fac else False
            floor_band = bool(fac["floor_band"]) if fac else False
            parapet_n = int(fac["parapet"]) if fac else 0
            # 屋根色(オルソ)を壁に反映: 暖色屋根の戸建は暖色壁へ(色も種別の代理特徴=ユーザ方針)
            if (fac and style == "wood_house" and roof_dom_rgb is not None
                    and 0 <= bid_c < len(roof_dom_rgb) and roof_dom_rgb[bid_c] is not None):
                _rr = roof_dom_rgb[bid_c]
                if _rr[0] > _rr[2] + 18:                      # R≫B = 暖色屋根
                    wall_kind = "sandstone"
            is_wall = bool(wall_cell[j, i_])
            is_corner = bool(corner_cell[j, i_])
            is_door = bool(door_cell[j, i_])
            light_here = (bx_v % 5 == 2) and (bz_v % 5 == 2)   # 疎な内側格子に照明
            attic = hollow_buildings and (ceil_y < top_y)        # 寄棟で軒より上に屋根裏がある棟
            # 連絡通路セル: この棟の下を道路がくぐる位置。1F(y_base+1..y_base+fh)を抜いて通路化する。
            tunnel_here = bool(tunnel_cells[j, i_]) if tunnel_cells is not None else False
            # 基礎: 地盤(y_top)から基準面の1つ下(y_base-1)まで壁材で充填（傾斜地の段差を塞ぐ）。
            # y_base = 1F の床面: 室内=床材を敷き(地表オルソが室内に透けるのを防ぐ)、壁直下=壁材。
            # 通路下(連絡通路)は塞がず道路を露出させ通れるようにする。
            if not tunnel_here:
                for fy in range(y_top + 1, y_base):
                    blocks.append(nbtlib.Compound({
                        "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(fy), nbtlib.Int(bz_v)]),
                        "state": block_id(wall_kind),
                    }))
                blocks.append(nbtlib.Compound({
                    "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v), nbtlib.Int(y_base), nbtlib.Int(bz_v)]),
                    "state": block_id(wall_kind if bool(wall_cell[j, i_]) else floor_block),
                }))
            for fy in range(y_base + 1, top_y + 1):
                # 連絡通路: 1F分(y_base+fh まで)はブロックを置かず空ける。2Fの床スラブ以上は残し通路の天井に。
                if tunnel_here and fy <= y_base + fh:
                    continue
                rel = fy - (y_base + 1)
                r = rel % fh                       # 階内位置 0..fh-1（0=床スラブ位置）
                is_slab = (r == 0 and fy != y_base + 1)
                if fy == top_y:
                    kind = roof_kind                                      # 屋根面（全 footprint）
                elif attic and fy > ceil_y:
                    kind = roof_kind                                      # 軒より上＝屋根裏を中実化(筒抜け防止)
                elif is_slab or (attic and fy == ceil_y):
                    # 各階の床 / 軒の天井（全 footprint）。外壁の床ラインは帯/角柱(arnis風)、内側は床/灯
                    if is_wall and (is_corner or floor_band):
                        kind = trim_kind                                 # 角柱(quoin) or 床ライン帯
                    else:
                        kind = interior_light if (not is_wall and light_here) else floor_block
                elif not hollow_buildings:
                    kind = win_kind if (r == 2 and fy < top_y - 1) else wall_kind  # 旧ソリッド
                elif is_wall:
                    if is_door and fy <= y_base + 2:
                        continue                                         # 出入口（接地2マス開口）
                    if is_corner:
                        kind = trim_kind                                 # 角柱(quoin)で縦の陰影
                    else:
                        _run = bz_v if wall_x[j, i_] else bx_v            # 壁の走る向きに沿って窓を間引く
                        _ground = (fy <= y_base + fh)                    # 1F(地上階)
                        if shopfront and _ground and r != 0:
                            kind = win_kind if (_run % 2 == 0) else wall_kind   # 店頭=広いガラス面
                        elif _is_window(style, r, fh, _run):
                            kind = win_kind                              # 窓(アーキタイプ別パターン)
                        else:
                            kind = wall_kind
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
            # parapet(陸屋根の外周を屋根上に1〜2段立ち上げる。勾配屋根=attic の棟は対象外)
            if parapet_n > 0 and is_wall and not attic and not tunnel_here:
                for _pz in range(1, parapet_n + 1):
                    _pk = trim_kind if _pz == parapet_n else wall_kind   # 最上段=トリムで笠木風
                    blocks.append(nbtlib.Compound({
                        "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(bx_v),
                                                          nbtlib.Int(top_y + _pz), nbtlib.Int(bz_v)]),
                        "state": block_id(_pk),
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
            road_mask=(road_mask if (road_mask is not None
                       and road_mask.shape == (nz, nx)) else None),
        )
        max_y = max(max_y, bridge_ymax + 2)

    # --- トンネル（OSM tunnel=yes）。地形を刳り貫く。地表構築後・橋の後に処理 ---
    if tunnels and patch_bbox_latlon is not None:
        tunnel_ymax = add_tunnel_blocks(
            blocks, tunnels, patch_bbox_latlon, nz, nx,
            y_surf_land=y_surf_land, h_res_block_m=h_res_block,
            surf_block=surf_block, sea_mask=sea_mask,
            road_mask=(road_mask if (road_mask is not None
                       and road_mask.shape == (nz, nx)) else None),
            core_always_covered=tunnel_core_always_covered,
            core_cover_slack=tunnel_core_cover_slack,
            cover_close_blocks=tunnel_cover_close_blocks,
        )
        max_y = max(max_y, tunnel_ymax + 2)

    # --- 送電線・鉄塔（OSM power=line/tower）。橋の後に立体化 ---
    if (powerlines or power_towers) and patch_bbox_latlon is not None:
        power_ymax = add_power_blocks(
            blocks, powerlines, power_towers, patch_bbox_latlon, nz, nx,
            y_surf_land=y_surf_land, sea_mask=sea_mask, scale_land=scale_land,
            clip_spans_to_grid=power_clip_spans_to_grid,
        )
        max_y = max(max_y, power_ymax + 2)

    # --- 鉄道（FG-GML RailCL）。道床＋枕木＋レールを地表敷設 ---
    if rails and patch_bbox_latlon is not None:
        rail_ymax = add_rail_blocks(
            blocks, rails, patch_bbox_latlon, nz, nx,
            y_surf_land=y_surf_land, sea_mask=sea_mask,
        )
        max_y = max(max_y, rail_ymax + 2)

    # --- 駐車場の停車車両（オルソ検出）を 1 段持ち上げて配置 ---
    for (ix, iy, iz, key) in _parking_cars:
        if 0 <= ix < nx and 0 <= iz < nz and 0 <= iy <= 500:
            blocks.append(nbtlib.Compound({
                "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(ix), nbtlib.Int(iy), nbtlib.Int(iz)]),
                "state": block_id(key)}))
            max_y = max(max_y, iy + 1)

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
