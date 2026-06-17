"""
anvil_loader.py
Tellus Fabric mod (mc1211 / mc261) が生成した Minecraft Anvil world を読み、
flood_pso の `dem_patch` (np.ndarray, m) と `surface_block_grid` (str) を返す。

【設計方針】（ユーザー指定: 2026-05-13）
  - 地形は Tellus の高さをそのまま使う（v_exag=1, h_res=1m が基本）
  - Tellus の block 名は flood_pso の 8-パレットに集約
    （deepslate→stone, podzol→grass, oak_log→grass, ice→blue_ice 等）
  - inundation を後から overlay する用に numpy grid を返す

【座標系】
  - Tellus は Web Mercator 投影。`EarthProjection.java` を Python 移植：
      blockX = lon * METERS_PER_DEGREE / world_scale
      blockZ = -EARTH_R * ln(tan(π/4 + lat_rad/2)) / world_scale
  - world_scale=1（既定）で 1 block = 1 m （実スケール Earth）
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from nbt import region as _nbt_region

# ─────────────────────────────────────────────────────────────
# Tellus EarthProjection (Mercator) — Python port
# ─────────────────────────────────────────────────────────────

METERS_PER_DEGREE = 111319.49166666667
EARTH_RADIUS_M = METERS_PER_DEGREE * 180.0 / math.pi    # 6378137.0
MAX_MERCATOR_LAT = 85.05112878


def lat_to_blockZ(lat: float, world_scale: float = 1.0) -> float:
    if world_scale <= 0.0:
        return 0.0
    lat = max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, lat))
    return -EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) / world_scale


def lon_to_blockX(lon: float, world_scale: float = 1.0) -> float:
    if world_scale <= 0.0:
        return 0.0
    return lon * METERS_PER_DEGREE / world_scale


def blockZ_to_lat(z: float, world_scale: float = 1.0) -> float:
    if world_scale <= 0.0:
        return 0.0
    mY = -z * world_scale
    return math.degrees(math.atan(math.sinh(mY / EARTH_RADIUS_M)))


def blockX_to_lon(x: float, world_scale: float = 1.0) -> float:
    if world_scale <= 0.0:
        return 0.0
    return x * world_scale / METERS_PER_DEGREE


def latlon_bbox_to_blockbox(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    world_scale: float = 1.0,
) -> tuple[int, int, int, int]:
    """(lat,lon) bbox → (bx_min, bx_max, bz_min, bz_max) inclusive integer block range.

    +Z は南向き（lat 小）。lat_max → bz_min, lat_min → bz_max。
    """
    bx_min = int(math.floor(lon_to_blockX(lon_min, world_scale)))
    bx_max = int(math.ceil(lon_to_blockX(lon_max, world_scale)))
    bz_min = int(math.floor(lat_to_blockZ(lat_max, world_scale)))   # 北端 = 小さな z
    bz_max = int(math.ceil(lat_to_blockZ(lat_min, world_scale)))    # 南端 = 大きな z
    return bx_min, bx_max, bz_min, bz_max


# ─────────────────────────────────────────────────────────────
# Tellus block 名 → flood_pso 8-palette マッピング
# ─────────────────────────────────────────────────────────────

PALETTE_FALLBACK = "stone"

TELLUS_TO_PALETTE: dict[str, str] = {
    # 岩盤・深層
    "minecraft:bedrock":   "bedrock",
    "minecraft:deepslate":  "stone",
    "minecraft:cobbled_deepslate": "stone",
    "minecraft:polished_deepslate": "stone",
    "minecraft:tuff":       "stone",
    "minecraft:calcite":    "stone",
    # 石（火成岩・変成岩・堆積岩を一括）
    "minecraft:stone":      "stone",
    "minecraft:cobblestone": "stone",
    "minecraft:mossy_cobblestone": "stone",
    "minecraft:smooth_stone": "stone",
    "minecraft:smooth_stone_slab": "stone",
    "minecraft:stone_slab":   "stone",
    "minecraft:stone_bricks": "stone",
    "minecraft:mossy_stone_bricks": "stone",
    "minecraft:cracked_stone_bricks": "stone",
    "minecraft:chiseled_stone_bricks": "stone",
    "minecraft:stone_brick_wall": "stone",
    "minecraft:stone_brick_slab": "stone",
    "minecraft:stone_brick_stairs": "stone",
    "minecraft:granite":    "stone",
    "minecraft:polished_granite": "stone",
    "minecraft:diorite":    "stone",
    "minecraft:polished_diorite": "stone",
    "minecraft:andesite":   "stone",
    "minecraft:polished_andesite": "stone",
    "minecraft:basalt":     "stone",
    "minecraft:smooth_basalt": "stone",
    "minecraft:blackstone": "stone",
    "minecraft:netherrack": "stone",
    # コンクリート・テラコッタ系（建物外壁）
    "minecraft:white_concrete": "stone",
    "minecraft:light_gray_concrete": "stone",
    "minecraft:gray_concrete": "stone",
    "minecraft:black_concrete": "stone",
    "minecraft:cyan_concrete": "stone",
    "minecraft:blue_concrete": "stone",
    "minecraft:green_concrete": "stone",
    "minecraft:brown_concrete": "stone",
    "minecraft:red_concrete": "stone",
    "minecraft:orange_concrete": "stone",
    "minecraft:yellow_concrete": "stone",
    "minecraft:white_concrete_powder": "sand",
    "minecraft:gray_concrete_powder": "sand",
    "minecraft:cyan_terracotta": "gravel",
    "minecraft:glazed_terracotta": "gravel",
    # 装飾石材
    "minecraft:bricks":           "stone",
    "minecraft:brick_slab":       "stone",
    "minecraft:brick_wall":       "stone",
    "minecraft:deepslate_bricks": "stone",
    "minecraft:deepslate_tiles":  "stone",
    "minecraft:smooth_quartz":    "stone",
    "minecraft:quartz_block":     "stone",
    "minecraft:quartz_pillar":    "stone",
    "minecraft:quartz_bricks":    "stone",
    "minecraft:chiseled_quartz_block": "stone",
    "minecraft:magma_block":      "stone",
    "minecraft:smooth_sandstone": "sand",
    "minecraft:cut_sandstone":    "sand",
    "minecraft:chiseled_sandstone": "sand",
    # 羊毛・カーペット（村の建材として使われる）
    "minecraft:white_wool":       "grass",
    "minecraft:light_gray_wool":  "grass",
    "minecraft:gray_wool":        "grass",
    "minecraft:black_wool":       "grass",
    "minecraft:brown_wool":       "grass",
    "minecraft:red_wool":         "grass",
    "minecraft:orange_wool":      "grass",
    "minecraft:yellow_wool":      "grass",
    "minecraft:green_wool":       "grass",
    "minecraft:lime_wool":        "grass",
    "minecraft:cyan_wool":        "grass",
    "minecraft:blue_wool":        "grass",
    "minecraft:purple_wool":      "grass",
    "minecraft:pink_wool":        "grass",
    "minecraft:magenta_wool":     "grass",
    # 砂・砂岩
    "minecraft:sand":       "sand",
    "minecraft:red_sand":   "sand",
    "minecraft:sandstone":  "sand",
    "minecraft:red_sandstone": "sand",
    # 砂利・土系
    "minecraft:gravel":     "gravel",
    "minecraft:dirt":       "gravel",
    "minecraft:coarse_dirt": "gravel",
    "minecraft:rooted_dirt": "gravel",
    "minecraft:clay":       "gravel",
    "minecraft:mud":        "gravel",
    "minecraft:packed_mud": "gravel",
    "minecraft:terracotta": "gravel",
    "minecraft:white_terracotta": "gravel",
    "minecraft:orange_terracotta": "gravel",
    "minecraft:yellow_terracotta": "gravel",
    "minecraft:brown_terracotta": "gravel",
    "minecraft:red_terracotta": "gravel",
    "minecraft:light_gray_terracotta": "gravel",
    "minecraft:gray_terracotta": "gravel",
    # 草地・植生（地表）
    "minecraft:grass_block": "grass",
    "minecraft:dirt_path":   "grass",
    "minecraft:moss_block":  "grass",
    "minecraft:moss_carpet": "grass",
    "minecraft:podzol":      "grass",
    "minecraft:mycelium":    "grass",
    # 雪は grass（白色は表現できないので近似）
    "minecraft:snow_block":  "grass",
    "minecraft:powder_snow": "grass",
    "minecraft:snow":        "grass",
    # 氷は blue_ice
    "minecraft:ice":         "blue_ice",
    "minecraft:packed_ice":  "blue_ice",
    "minecraft:blue_ice":    "blue_ice",
    "minecraft:frosted_ice": "blue_ice",
    # 水
    "minecraft:water":         "water",
    "minecraft:flowing_water": "water",
    "minecraft:bubble_column": "water",
    "minecraft:kelp":          "water",
    "minecraft:kelp_plant":    "water",
    "minecraft:seagrass":      "water",
    "minecraft:tall_seagrass": "water",
    # 木材・建材（村の家屋・橋など。色がない 8-palette では grass で代用）
    "minecraft:oak_planks":     "grass",
    "minecraft:birch_planks":   "grass",
    "minecraft:spruce_planks":  "grass",
    "minecraft:jungle_planks":  "grass",
    "minecraft:acacia_planks":  "grass",
    "minecraft:dark_oak_planks": "grass",
    "minecraft:cherry_planks":  "grass",
    "minecraft:mangrove_planks": "grass",
    "minecraft:bamboo_planks":  "grass",
    "minecraft:oak_fence":      "grass",
    "minecraft:spruce_fence":   "grass",
    "minecraft:birch_fence":    "grass",
    "minecraft:dark_oak_fence": "grass",
    "minecraft:oak_door":       "grass",
    "minecraft:spruce_door":    "grass",
    "minecraft:birch_door":     "grass",
    "minecraft:oak_trapdoor":   "grass",
    "minecraft:spruce_trapdoor": "grass",
    "minecraft:birch_trapdoor": "grass",
    "minecraft:oak_stairs":     "grass",
    "minecraft:spruce_stairs":  "grass",
    "minecraft:birch_stairs":   "grass",
    "minecraft:oak_slab":       "grass",
    "minecraft:spruce_slab":    "grass",
    "minecraft:birch_slab":     "grass",
    "minecraft:barrel":         "grass",
    "minecraft:bookshelf":      "grass",
    "minecraft:crafting_table": "grass",
    "minecraft:loom":           "grass",
    "minecraft:cartography_table": "grass",
    "minecraft:smithing_table": "stone",
    "minecraft:stonecutter":    "stone",
    "minecraft:furnace":        "stone",
    "minecraft:blast_furnace":  "stone",
    "minecraft:smoker":         "stone",
    "minecraft:composter":      "grass",
    # 光源・装飾
    "minecraft:sea_lantern":    "blue_ice",
    "minecraft:glowstone":      "sand",
    "minecraft:lantern":        "sand",
    "minecraft:torch":          "sand",
    "minecraft:wall_torch":     "sand",
    # ステンドグラス（透過扱い → blue_ice 近似）
    "minecraft:white_stained_glass":      "blue_ice",
    "minecraft:light_gray_stained_glass": "blue_ice",
    "minecraft:gray_stained_glass":       "blue_ice",
    "minecraft:light_blue_stained_glass": "blue_ice",
    "minecraft:blue_stained_glass":       "blue_ice",
    "minecraft:cyan_stained_glass":       "blue_ice",
    "minecraft:glass":                    "blue_ice",
    "minecraft:glass_pane":               "blue_ice",
    # キノコブロック
    "minecraft:red_mushroom_block":       "grass",
    "minecraft:brown_mushroom_block":     "grass",
    "minecraft:mushroom_stem":            "grass",
    # 樹木（地表に立つ → grass 扱い）
    "minecraft:oak_log":       "grass",
    "minecraft:birch_log":     "grass",
    "minecraft:spruce_log":    "grass",
    "minecraft:jungle_log":    "grass",
    "minecraft:acacia_log":    "grass",
    "minecraft:dark_oak_log":  "grass",
    "minecraft:cherry_log":    "grass",
    "minecraft:mangrove_log":  "grass",
    "minecraft:oak_leaves":    "grass",
    "minecraft:birch_leaves":  "grass",
    "minecraft:spruce_leaves": "grass",
    "minecraft:jungle_leaves": "grass",
    "minecraft:acacia_leaves": "grass",
    "minecraft:dark_oak_leaves": "grass",
    "minecraft:cherry_leaves": "grass",
    "minecraft:mangrove_leaves": "grass",
    "minecraft:azalea_leaves": "grass",
    "minecraft:flowering_azalea_leaves": "grass",
    # 草・花・装飾植物（純粋に植生、出力時も grass 扱い）
    "minecraft:short_grass":   "grass",
    "minecraft:tall_grass":    "grass",
    "minecraft:grass":         "grass",
    "minecraft:fern":          "grass",
    "minecraft:large_fern":    "grass",
    "minecraft:dead_bush":     "grass",
    "minecraft:vine":          "grass",
    "minecraft:dandelion":     "grass",
    "minecraft:poppy":         "grass",
    "minecraft:blue_orchid":   "grass",
    "minecraft:allium":        "grass",
    "minecraft:azure_bluet":   "grass",
    "minecraft:sunflower":     "grass",
    "minecraft:peony":         "grass",
    "minecraft:rose_bush":     "grass",
    "minecraft:lilac":         "grass",
    "minecraft:cornflower":    "grass",
    "minecraft:wither_rose":   "grass",
    "minecraft:lily_of_the_valley": "grass",
    "minecraft:sweet_berry_bush": "grass",
    "minecraft:sugar_cane":    "grass",
    "minecraft:bamboo":        "grass",
    "minecraft:bamboo_sapling": "grass",
    "minecraft:azalea":        "grass",
    "minecraft:flowering_azalea": "grass",
    "minecraft:leaf_litter":   "grass",
    "minecraft:moss_carpet":   "grass",
    # ぶら下がりや小型装飾
    "minecraft:cobweb":        "grass",
    "minecraft:hanging_roots": "grass",
    "minecraft:cave_vines":    "grass",
    "minecraft:cave_vines_plant": "grass",
    "minecraft:bush":          "grass",
    "minecraft:oxeye_daisy":   "grass",
    "minecraft:red_mushroom":  "grass",
    "minecraft:brown_mushroom": "grass",
    "minecraft:bee_nest":      "grass",
    "minecraft:beehive":       "grass",
    "minecraft:hay_block":     "grass",
    "minecraft:firefly_bush":  "grass",
    "minecraft:pumpkin":       "grass",
    "minecraft:carved_pumpkin": "grass",
    "minecraft:jack_o_lantern": "sand",
    "minecraft:melon":         "grass",
}

# 「surface とみなさない」透過/装飾ブロック群（surface 探索でスキップする）
SURFACE_TRANSPARENT: set[str] = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air",
    # 短草・花・ベリーなど
    "minecraft:short_grass", "minecraft:tall_grass", "minecraft:grass",
    "minecraft:fern", "minecraft:large_fern",
    "minecraft:dead_bush", "minecraft:vine",
    "minecraft:dandelion", "minecraft:poppy", "minecraft:sweet_berry_bush",
    "minecraft:sugar_cane", "minecraft:bamboo", "minecraft:bamboo_sapling",
    "minecraft:azalea", "minecraft:flowering_azalea",
    # 葉ブロック・原木は surface としては使わず、地表の grass を残す
    "minecraft:oak_leaves", "minecraft:birch_leaves", "minecraft:spruce_leaves",
    "minecraft:jungle_leaves", "minecraft:acacia_leaves", "minecraft:dark_oak_leaves",
    "minecraft:cherry_leaves", "minecraft:mangrove_leaves",
    "minecraft:azalea_leaves", "minecraft:flowering_azalea_leaves",
    "minecraft:oak_log", "minecraft:birch_log", "minecraft:spruce_log",
    "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log",
    "minecraft:cherry_log", "minecraft:mangrove_log",
    # 雪レイヤ（薄く積もったやつ）も装飾扱い
    "minecraft:snow",
    # 流体
    "minecraft:water", "minecraft:flowing_water", "minecraft:bubble_column",
    "minecraft:lava", "minecraft:flowing_lava",
    "minecraft:kelp", "minecraft:kelp_plant",
    "minecraft:seagrass", "minecraft:tall_seagrass",
}

AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}


def map_tellus_to_palette(name: str) -> str:
    return TELLUS_TO_PALETTE.get(name, PALETTE_FALLBACK)


# ─────────────────────────────────────────────────────────────
# Section block_states デコーダ（MC 1.16+ 仕様: long を跨がない）
# ─────────────────────────────────────────────────────────────

_MASK_U64 = 0xFFFFFFFFFFFFFFFF


def _decode_section(palette_names: list[str], data_long_array) -> np.ndarray:
    """1 セクション (16x16x16) を YZX 順の (Y, Z, X) ndarray にデコード。

    ・1.16 以降は packed-bit indices が long を跨がないパッキング方式。
    ・bits/block = max(4, ceil(log2(palette_size)))。palette が 1 つなら data 不要。
    """
    n_pal = len(palette_names)
    if n_pal <= 1 or data_long_array is None:
        return np.zeros((16, 16, 16), dtype=np.int16)

    bpb = max(4, (n_pal - 1).bit_length())
    indices_per_long = 64 // bpb
    mask = (1 << bpb) - 1

    out = np.zeros(4096, dtype=np.int16)
    pos = 0
    for L in data_long_array:
        u = int(L) & _MASK_U64
        for k in range(indices_per_long):
            if pos >= 4096:
                break
            out[pos] = (u >> (k * bpb)) & mask
            pos += 1
        if pos >= 4096:
            break
    # NBT 仕様: index = y*256 + z*16 + x （Y, Z, X 順）
    return out.reshape(16, 16, 16)


# ─────────────────────────────────────────────────────────────
# TellusWorld: region/chunk LRU + grid 抽出
# ─────────────────────────────────────────────────────────────

class TellusWorld:
    """Tellus 生成の Anvil world を読むラッパ。

    Parameters
    ----------
    world_dir : path
      level.dat のあるディレクトリ。中に dimensions/<ns>/<key>/region/r.X.Z.mca が必要。
    world_scale : float
      Tellus 世界生成時の `world_scale`（1.0 で 1 block=1 m, real-scale Earth）
    dimension : str
      "minecraft:overworld" 等
    """

    def __init__(self, world_dir: str | Path, world_scale: float = 1.0,
                 dimension: str = "minecraft:overworld"):
        self.world_dir = Path(world_dir)
        self.world_scale = float(world_scale)
        ns, key = dimension.split(":", 1)
        self.region_dir = self.world_dir / "dimensions" / ns / key / "region"
        if not self.region_dir.is_dir():
            raise FileNotFoundError(f"region directory not found: {self.region_dir}")
        self._regions: dict[tuple[int, int], object] = {}     # (rx,rz) -> RegionFile|None
        self._chunks: dict[tuple[int, int], object] = {}      # (cx,cz) -> root NBT|None

    # ── region / chunk アクセス ──
    def _region(self, rx: int, rz: int):
        if (rx, rz) in self._regions:
            return self._regions[(rx, rz)]
        f = self.region_dir / f"r.{rx}.{rz}.mca"
        if not f.exists():
            self._regions[(rx, rz)] = None
            return None
        try:
            rf = _nbt_region.RegionFile(str(f))
        except Exception as e:
            print(f"[anvil] failed to open {f.name}: {e}")
            rf = None
        self._regions[(rx, rz)] = rf
        return rf

    def _chunk(self, cx: int, cz: int):
        key = (cx, cz)
        if key in self._chunks:
            return self._chunks[key]
        rf = self._region(cx >> 5, cz >> 5)
        if rf is None:
            self._chunks[key] = None
            return None
        try:
            ch = rf.get_chunk(cx & 31, cz & 31)
        except Exception:
            ch = None
        self._chunks[key] = ch
        return ch

    # ── 単一カラム ──
    def column_blocks(self, blockX: int, blockZ: int) -> list[tuple[int, str]]:
        """指定 (blockX, blockZ) の柱を [(y, name), ...] で返す（air は除外）。"""
        ch = self._chunk(blockX >> 4, blockZ >> 4)
        if ch is None:
            return []
        sections = ch.get('sections')
        if sections is None:
            return []
        lx, lz = blockX & 15, blockZ & 15
        out: list[tuple[int, str]] = []
        for s in sections:
            bs = s.get('block_states')
            if bs is None:
                continue
            pal = bs.get('palette')
            if pal is None or len(pal) == 0:
                continue
            pal_names = [p['Name'].value for p in pal]
            if all(n in AIR_NAMES for n in pal_names):
                continue
            data = bs.get('data')
            arr = _decode_section(pal_names, list(data) if data is not None else None)
            sy_base = int(s['Y'].value) * 16
            for ly in range(16):
                idx = int(arr[ly, lz, lx])
                name = pal_names[idx]
                if name in AIR_NAMES:
                    continue
                out.append((sy_base + ly, name))
        return out

    def surface(self, blockX: int, blockZ: int) -> tuple[int | None, str | None]:
        """そのカラムの surface (y, block_name) — 透過物・水・葉はスキップして地表の固体まで降りる。"""
        col = self.column_blocks(blockX, blockZ)
        if not col:
            return None, None
        col.sort(key=lambda t: -t[0])
        topmost_transp = None
        for y, name in col:
            if name in SURFACE_TRANSPARENT:
                if topmost_transp is None:
                    topmost_transp = (y, name)
                continue
            return y, name
        return topmost_transp if topmost_transp is not None else (None, None)

    # ── grid 取得（chunk 単位デコードを再利用） ──
    def fetch_grid(
        self,
        bx_min: int, bx_max: int, bz_min: int, bz_max: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """指定 block 範囲の DEM (m) と surface block (palette key) grid を返す。

        Grid 並び:
          - axis 0 (rows) = +Z 方向（北→南、bz_min が row=0）
          - axis 1 (cols) = +X 方向（西→東、bx_min が col=0）

        Returns
        -------
        dem_m : (H, W) float32
            surface y (m, world_scale=1 想定)。chunk 未生成 / 全 air は NaN
        surf  : (H, W) object (str)
            flood_pso 8-palette キー（'stone','grass',... or 'air'）
        stats : dict
            {'n_cells', 'n_loaded', 'n_chunks_loaded', 'n_chunks_missing', ...}
        """
        if bx_max < bx_min or bz_max < bz_min:
            raise ValueError("invalid bbox")
        H = bz_max - bz_min + 1
        W = bx_max - bx_min + 1
        dem = np.full((H, W), np.nan, dtype=np.float32)
        surf = np.full((H, W), 'air', dtype=object)

        cx_min, cx_max = bx_min >> 4, bx_max >> 4
        cz_min, cz_max = bz_min >> 4, bz_max >> 4
        n_chunks_loaded = 0
        n_chunks_missing = 0

        for cz in range(cz_min, cz_max + 1):
            for cx in range(cx_min, cx_max + 1):
                ch = self._chunk(cx, cz)
                if ch is None:
                    n_chunks_missing += 1
                    continue
                sections = ch.get('sections')
                if sections is None:
                    n_chunks_missing += 1
                    continue
                # decode 全 section（高い Y から並べて surface 探索を最短化）
                decoded: list[tuple[int, list[str], np.ndarray]] = []
                for s in sections:
                    bs = s.get('block_states')
                    if bs is None:
                        continue
                    pal = bs.get('palette')
                    if pal is None or len(pal) == 0:
                        continue
                    pal_names = [p['Name'].value for p in pal]
                    if all(n in AIR_NAMES for n in pal_names):
                        continue
                    data = bs.get('data')
                    arr = _decode_section(pal_names, list(data) if data is not None else None)
                    decoded.append((int(s['Y'].value), pal_names, arr))
                if not decoded:
                    continue
                decoded.sort(key=lambda t: -t[0])
                n_chunks_loaded += 1

                # bbox とこの chunk の交差範囲
                lz0 = max(0, bz_min - (cz << 4))
                lz1 = min(15, bz_max - (cz << 4))
                lx0 = max(0, bx_min - (cx << 4))
                lx1 = min(15, bx_max - (cx << 4))

                for lz in range(lz0, lz1 + 1):
                    row = (cz << 4) + lz - bz_min
                    for lx in range(lx0, lx1 + 1):
                        col = (cx << 4) + lx - bx_min
                        topmost_transp: tuple[int, str] | None = None
                        for sy, pal_names, arr in decoded:
                            base = sy * 16
                            found = False
                            for ly in range(15, -1, -1):
                                idx = int(arr[ly, lz, lx])
                                name = pal_names[idx]
                                if name in AIR_NAMES:
                                    continue
                                if name in SURFACE_TRANSPARENT:
                                    if topmost_transp is None:
                                        topmost_transp = (base + ly, name)
                                    continue
                                dem[row, col] = float(base + ly)
                                surf[row, col] = map_tellus_to_palette(name)
                                found = True
                                break
                            if found:
                                break
                        else:
                            if topmost_transp is not None:
                                y, name = topmost_transp
                                dem[row, col] = float(y)
                                surf[row, col] = map_tellus_to_palette(name)

        stats = {
            'n_cells': int(H * W),
            'n_loaded_cells': int(np.isfinite(dem).sum()),
            'n_chunks_total': (cx_max - cx_min + 1) * (cz_max - cz_min + 1),
            'n_chunks_loaded': n_chunks_loaded,
            'n_chunks_missing': n_chunks_missing,
            'bbox_blocks': (bx_min, bx_max, bz_min, bz_max),
            'shape': (H, W),
        }
        return dem, surf, stats

    # ── lat,lon 中心 + width/depth で fetch ──
    def fetch_grid_around(
        self,
        center_lat: float, center_lon: float,
        width_m: float, depth_m: float,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """(center, width_m × depth_m) を Tellus 投影で block 範囲に変換し fetch_grid。

        block 中心は center_lat/lon が原点。h_res は world_scale=1 で 1 m/block 固定。
        """
        bx_c = lon_to_blockX(center_lon, self.world_scale)
        bz_c = lat_to_blockZ(center_lat, self.world_scale)
        half_w = width_m / 2.0 / self.world_scale
        half_d = depth_m / 2.0 / self.world_scale
        bx_min = int(math.floor(bx_c - half_w))
        bx_max = int(math.ceil(bx_c + half_w)) - 1
        bz_min = int(math.floor(bz_c - half_d))
        bz_max = int(math.ceil(bz_c + half_d)) - 1
        return self.fetch_grid(bx_min, bx_max, bz_min, bz_max)


# ─────────────────────────────────────────────────────────────
# CLI: ``python anvil_loader.py <world_dir> <lat> <lon> <w_m> <d_m>``
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("usage: anvil_loader.py <world_dir> <lat> <lon> <width_m> <depth_m> [world_scale=1]")
        sys.exit(1)
    world_dir = sys.argv[1]
    lat = float(sys.argv[2]); lon = float(sys.argv[3])
    width_m = float(sys.argv[4]); depth_m = float(sys.argv[5])
    ws = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

    w = TellusWorld(world_dir, world_scale=ws)
    bx_c = lon_to_blockX(lon, ws); bz_c = lat_to_blockZ(lat, ws)
    print(f"center block: ({bx_c:.1f}, {bz_c:.1f})  region: ({int(bx_c)>>9}, {int(bz_c)>>9})")

    dem, surf, stats = w.fetch_grid_around(lat, lon, width_m, depth_m)
    print(f"shape: {dem.shape}  loaded cells: {stats['n_loaded_cells']}/{stats['n_cells']}  "
          f"chunks: {stats['n_chunks_loaded']}/{stats['n_chunks_total']}")
    valid = dem[np.isfinite(dem)]
    if valid.size:
        print(f"y range (m): min={valid.min():.1f}  median={np.median(valid):.1f}  max={valid.max():.1f}")
        from collections import Counter
        cnt = Counter(surf[np.isfinite(dem)].tolist())
        print("surface palette mix:", cnt.most_common())
    else:
        print("(no cells loaded — bbox may be outside generated chunks)")
