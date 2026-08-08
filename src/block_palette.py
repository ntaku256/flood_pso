"""
block_palette.py
flood_pso 全体で使う **バニラブロック色テーブルの単一真実源**。

- NBT 出力パレット（nbt_export.PALETTE）
- 空中写真の最近傍カラーマッチ（ortho_surface）
- トップダウンプレビュー（nbt_preview）
- ネイティブビューア（flood_pso_viewer/src/block_colors.json を本ファイルから生成）

各エントリ: key(パレットキー) → (minecraft_name, (r,g,b), role)
  role: "air"=空白 / "opaque"=不透明地表 / "water","ice"=半透明（マッチ対象外）

既存コードが使うキー（stone/grass/sand/gravel/water/blue_ice/bedrock/air）は互換維持。
それ以外は concrete16 / terracotta16 / wool16 / 自然・木・葉 を追加した ~80 色。
"""
from __future__ import annotations

import json
from pathlib import Path

# key: (minecraft_name, (r, g, b), role)
BLOCKS: dict[str, tuple[str, tuple[int, int, int], str]] = {
    "air":      ("minecraft:air", (0, 0, 0), "air"),
    # 半透明（flood/氷レイヤ用。カラーマッチ対象外）
    "water":    ("minecraft:blue_stained_glass", (40, 100, 200), "water"),
    "blue_ice": ("minecraft:cyan_stained_glass", (150, 220, 240), "ice"),

    # ── 自然・地形（既存キー stone/grass/sand/gravel/bedrock を含む） ──
    "stone":             ("minecraft:stone", (125, 125, 125), "opaque"),
    "grass":             ("minecraft:grass_block", (91, 153, 66), "opaque"),
    "dirt":              ("minecraft:dirt", (134, 96, 66), "opaque"),
    "coarse_dirt":       ("minecraft:coarse_dirt", (122, 86, 57), "opaque"),
    "podzol":            ("minecraft:podzol", (94, 67, 31), "opaque"),
    "rooted_dirt":       ("minecraft:rooted_dirt", (144, 103, 76), "opaque"),
    "sand":              ("minecraft:sand", (219, 207, 162), "opaque"),
    "red_sand":          ("minecraft:red_sand", (190, 103, 33), "opaque"),
    "gravel":            ("minecraft:gravel", (131, 124, 121), "opaque"),
    "clay":              ("minecraft:clay", (160, 166, 179), "opaque"),
    "terracotta":        ("minecraft:terracotta", (152, 94, 67), "opaque"),
    "mud":               ("minecraft:mud", (60, 55, 53), "opaque"),
    "packed_mud":        ("minecraft:packed_mud", (143, 110, 79), "opaque"),
    "moss_block":        ("minecraft:moss_block", (89, 109, 45), "opaque"),
    "mossy_cobblestone": ("minecraft:mossy_cobblestone", (110, 118, 92), "opaque"),
    "cobblestone":       ("minecraft:cobblestone", (122, 122, 122), "opaque"),
    "andesite":          ("minecraft:andesite", (136, 136, 137), "opaque"),
    "diorite":           ("minecraft:diorite", (189, 189, 191), "opaque"),
    "granite":           ("minecraft:granite", (149, 103, 84), "opaque"),
    "deepslate":         ("minecraft:deepslate", (77, 77, 80), "opaque"),
    "tuff":              ("minecraft:tuff", (108, 109, 103), "opaque"),
    "calcite":           ("minecraft:calcite", (224, 225, 220), "opaque"),
    "sandstone":         ("minecraft:sandstone", (216, 203, 156), "opaque"),
    "snow_block":        ("minecraft:snow_block", (243, 247, 247), "opaque"),
    "bedrock":           ("minecraft:bedrock", (85, 85, 85), "opaque"),
    "prismarine":        ("minecraft:prismarine", (99, 156, 151), "opaque"),

    # ── 建物内部（窓・照明）。地表オルソ色マッチには使わない（_NON_MATCH で除外） ──
    "glass":             ("minecraft:glass", (200, 224, 233), "opaque"),
    "glowstone":         ("minecraft:glowstone", (255, 226, 142), "opaque"),
    "iron_bars":         ("minecraft:iron_bars", (130, 130, 135), "opaque"),

    # ── 鉄道レール（同一 minecraft 名で shape 別の 6 キー。Properties は KEY で分岐）──
    "rail_ns":           ("minecraft:rail", (84, 82, 80), "opaque"),
    "rail_ew":           ("minecraft:rail", (84, 82, 80), "opaque"),
    "rail_ne":           ("minecraft:rail", (84, 82, 80), "opaque"),
    "rail_nw":           ("minecraft:rail", (84, 82, 80), "opaque"),
    "rail_se":           ("minecraft:rail", (84, 82, 80), "opaque"),
    "rail_sw":           ("minecraft:rail", (84, 82, 80), "opaque"),

    # ── 凡例(地下データ層)用の色付きガラス。地表オルソ色マッチには使わない（_NON_MATCH 除外） ──
    "white_stained_glass":      ("minecraft:white_stained_glass", (236, 240, 240), "opaque"),
    "red_stained_glass":        ("minecraft:red_stained_glass", (165, 46, 38), "opaque"),
    "orange_stained_glass":     ("minecraft:orange_stained_glass", (216, 118, 33), "opaque"),
    "light_blue_stained_glass": ("minecraft:light_blue_stained_glass", (88, 158, 210), "opaque"),
    "blue_stained_glass":       ("minecraft:blue_stained_glass", (44, 60, 140), "opaque"),
    "green_stained_glass":      ("minecraft:green_stained_glass", (84, 109, 28), "opaque"),
    "black_stained_glass":      ("minecraft:black_stained_glass", (25, 22, 22), "opaque"),

    # ── 木材・原木・葉 ──
    "oak_planks":        ("minecraft:oak_planks", (162, 131, 79), "opaque"),
    "spruce_planks":     ("minecraft:spruce_planks", (114, 84, 48), "opaque"),
    "birch_planks":      ("minecraft:birch_planks", (196, 179, 123), "opaque"),
    "jungle_planks":     ("minecraft:jungle_planks", (160, 115, 80), "opaque"),
    "acacia_planks":     ("minecraft:acacia_planks", (168, 90, 50), "opaque"),
    "dark_oak_planks":   ("minecraft:dark_oak_planks", (66, 43, 20), "opaque"),
    "oak_log":           ("minecraft:oak_log", (109, 85, 51), "opaque"),
    "spruce_log":        ("minecraft:spruce_log", (58, 38, 20), "opaque"),
    "stripped_oak_log":  ("minecraft:stripped_oak_log", (181, 143, 84), "opaque"),
    "oak_leaves":        ("minecraft:oak_leaves", (59, 87, 42), "opaque"),
    "spruce_leaves":     ("minecraft:spruce_leaves", (48, 64, 42), "opaque"),
    "birch_leaves":      ("minecraft:birch_leaves", (110, 131, 66), "opaque"),

    # ── Concrete 16 ──
    "white_concrete":      ("minecraft:white_concrete", (207, 213, 214), "opaque"),
    "orange_concrete":     ("minecraft:orange_concrete", (224, 97, 1), "opaque"),
    "magenta_concrete":    ("minecraft:magenta_concrete", (169, 48, 159), "opaque"),
    "light_blue_concrete": ("minecraft:light_blue_concrete", (35, 137, 198), "opaque"),
    "yellow_concrete":     ("minecraft:yellow_concrete", (240, 175, 21), "opaque"),
    "lime_concrete":       ("minecraft:lime_concrete", (94, 168, 24), "opaque"),
    "pink_concrete":       ("minecraft:pink_concrete", (213, 101, 142), "opaque"),
    "gray_concrete":       ("minecraft:gray_concrete", (54, 57, 61), "opaque"),
    "gray_concrete_powder": ("minecraft:gray_concrete_powder", (77, 80, 84), "opaque"),  # 道路舗装（※重力：直下に固体必須）
    "light_gray_concrete": ("minecraft:light_gray_concrete", (125, 125, 115), "opaque"),
    "cyan_concrete":       ("minecraft:cyan_concrete", (21, 118, 136), "opaque"),
    "purple_concrete":     ("minecraft:purple_concrete", (100, 31, 156), "opaque"),
    "blue_concrete":       ("minecraft:blue_concrete", (44, 46, 143), "opaque"),
    "brown_concrete":      ("minecraft:brown_concrete", (96, 59, 31), "opaque"),
    "green_concrete":      ("minecraft:green_concrete", (73, 91, 36), "opaque"),
    "red_concrete":        ("minecraft:red_concrete", (142, 32, 32), "opaque"),
    "sea_lantern":         ("minecraft:sea_lantern", (211, 227, 207), "opaque"),  # 避難所マーカー発光
    "black_concrete":      ("minecraft:black_concrete", (8, 10, 15), "opaque"),

    # ── Terracotta 16（dyed） ──
    "white_terracotta":      ("minecraft:white_terracotta", (209, 178, 161), "opaque"),
    "orange_terracotta":     ("minecraft:orange_terracotta", (161, 83, 37), "opaque"),
    "magenta_terracotta":    ("minecraft:magenta_terracotta", (149, 88, 108), "opaque"),
    "light_blue_terracotta": ("minecraft:light_blue_terracotta", (113, 108, 137), "opaque"),
    "yellow_terracotta":     ("minecraft:yellow_terracotta", (186, 133, 35), "opaque"),
    "lime_terracotta":       ("minecraft:lime_terracotta", (103, 117, 52), "opaque"),
    "pink_terracotta":       ("minecraft:pink_terracotta", (161, 78, 78), "opaque"),
    "gray_terracotta":       ("minecraft:gray_terracotta", (57, 42, 35), "opaque"),
    "light_gray_terracotta": ("minecraft:light_gray_terracotta", (135, 106, 97), "opaque"),
    "cyan_terracotta":       ("minecraft:cyan_terracotta", (86, 91, 91), "opaque"),
    "purple_terracotta":     ("minecraft:purple_terracotta", (118, 70, 86), "opaque"),
    "blue_terracotta":       ("minecraft:blue_terracotta", (74, 59, 91), "opaque"),
    "brown_terracotta":      ("minecraft:brown_terracotta", (77, 51, 35), "opaque"),
    "green_terracotta":      ("minecraft:green_terracotta", (76, 82, 42), "opaque"),
    "red_terracotta":        ("minecraft:red_terracotta", (143, 61, 46), "opaque"),
    "black_terracotta":      ("minecraft:black_terracotta", (37, 22, 16), "opaque"),

    # ── Wool 16 ──
    "white_wool":      ("minecraft:white_wool", (233, 236, 236), "opaque"),
    "orange_wool":     ("minecraft:orange_wool", (240, 118, 19), "opaque"),
    "magenta_wool":    ("minecraft:magenta_wool", (189, 68, 179), "opaque"),
    "light_blue_wool": ("minecraft:light_blue_wool", (58, 175, 217), "opaque"),
    "yellow_wool":     ("minecraft:yellow_wool", (248, 197, 39), "opaque"),
    "lime_wool":       ("minecraft:lime_wool", (112, 185, 25), "opaque"),
    "pink_wool":       ("minecraft:pink_wool", (237, 141, 172), "opaque"),
    "gray_wool":       ("minecraft:gray_wool", (62, 68, 71), "opaque"),
    "light_gray_wool": ("minecraft:light_gray_wool", (142, 142, 134), "opaque"),
    "cyan_wool":       ("minecraft:cyan_wool", (21, 137, 145), "opaque"),
    "purple_wool":     ("minecraft:purple_wool", (121, 42, 172), "opaque"),
    "blue_wool":       ("minecraft:blue_wool", (53, 57, 157), "opaque"),
    "brown_wool":      ("minecraft:brown_wool", (114, 71, 40), "opaque"),
    "green_wool":      ("minecraft:green_wool", (84, 109, 27), "opaque"),
    "red_wool":        ("minecraft:red_wool", (160, 39, 34), "opaque"),
    "black_wool":      ("minecraft:black_wool", (20, 21, 25), "opaque"),
}

# パレットキーの確定順（NBT パレット index になる。air=0 を保証）
PALETTE_KEYS: list[str] = list(BLOCKS.keys())

# カラーマッチに使うアンカー（opaque のみ。air/water/ice は除外）
# 発光等のマーカー専用ブロックは地表オルソ色マッチのアンカーから除外
_NON_MATCH = {"sea_lantern", "glass", "glowstone", "iron_bars",
              # 重力ブロック: オルソ色マッチで地表(斜面/縁=直下が空)に湧くと落下するため候補外
              "gray_concrete_powder",
              "white_stained_glass", "red_stained_glass", "orange_stained_glass",
              "light_blue_stained_glass", "blue_stained_glass",
              "green_stained_glass", "black_stained_glass",
              # 鉄道レールは色マッチで地表/屋根に湧くと「レール屋根」になるため候補から除外
              "rail_ns", "rail_ew", "rail_ne", "rail_nw", "rail_se", "rail_sw"}
MATCH_KEYS: list[str] = [k for k, (_, _, role) in BLOCKS.items()
                         if role == "opaque" and k not in _NON_MATCH]


def minecraft_name(key: str) -> str:
    return BLOCKS[key][0]


def rgb(key: str) -> tuple[int, int, int]:
    return BLOCKS[key][1]


def role(key: str) -> str:
    return BLOCKS[key][2]


def block_state_properties(name: str) -> dict[str, str] | None:
    """minecraft ブロック名 → BlockState の Properties（単一真実源, None=Properties 無し）。

    NBT (nbt_export) と litematic (nbt_to_litematic) のパレット生成が共通で使う。
    - grass_block : snowy=false（従来挙動）
    - *_leaves    : persistent=true → ワールド配置後に葉が時間で崩壊して消えるのを防ぐ
      （構造物/litematic に Properties 無しで置くと worldgen 既定 persistent=false で
       幹から離れた葉が decay して消える）。distance=1 も明示。
    """
    if name == "minecraft:grass_block":
        return {"snowy": "false"}
    if name.endswith("_leaves"):
        return {"persistent": "true", "distance": "1"}
    return None


# 鉄道レール: palette KEY → minecraft:rail の shape。同一名で複数 state を持つため
# 名前ベースの block_state_properties では分岐できず、KEY で解決する。
RAIL_SHAPES: dict[str, str] = {
    "rail_ns": "north_south", "rail_ew": "east_west",
    "rail_ne": "north_east",  "rail_nw": "north_west",
    "rail_se": "south_east",  "rail_sw": "south_west",
}


def block_state_properties_for_key(key: str) -> dict[str, str] | None:
    """palette KEY → BlockState Properties（NBT/litematic/anvil 共通の単一真実源）。
    rail のように同一 minecraft 名で複数 state を持つキーはここで分岐し、
    それ以外は名前ベースの block_state_properties() に委譲する。"""
    if key in RAIL_SHAPES:
        return {"shape": RAIL_SHAPES[key]}
    return block_state_properties(BLOCKS[key][0])


def name_to_rgb() -> dict[str, tuple[int, int, int]]:
    """minecraft:name → (r,g,b)。プレビュー/ビューア用。"""
    return {mc: c for (mc, c, _) in BLOCKS.values()}


def export_viewer_json(path: str) -> None:
    """flood_pso_viewer 用の {minecraft:name: [r,g,b,role]} を書き出す。"""
    out = {mc: {"rgb": list(c), "role": rl} for (mc, c, rl) in BLOCKS.values()}
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


_ROLE_CODE = {"opaque": 0, "water": 1, "ice": 2, "air": 3}


def export_viewer_rust(path: str) -> None:
    """flood_pso_viewer 用の block_colors.rs（自動生成）を書き出す。
    `block_rgb_role(name) -> Option<([f32;3], u8)>`（srgb 0..1, role 0=opaque/1=water/2=ice）。
    air/未知は None。serde_json 不要のコンパイル時テーブル。"""
    lines = [
        "// AUTO-GENERATED from flood_pso/src/block_palette.py — do not edit by hand.",
        "// role: 0=opaque, 1=water(translucent), 2=ice(translucent)",
        "/// minecraft ブロック名 → (srgb [r,g,b] 0..1, role)。air/未知は None。",
        "pub fn block_rgb_role(name: &str) -> Option<([f32; 3], u8)> {",
        "    let v = match name {",
    ]
    _seen_mc: set[str] = set()
    for mc, (r, g, b), rl in ((mc, c, rl) for (mc, c, rl) in BLOCKS.values()):
        if rl == "air" or mc in _seen_mc:   # 同一 minecraft 名（例: rail の shape 別キー）は1度だけ
            continue
        _seen_mc.add(mc)
        code = _ROLE_CODE[rl]
        lines.append(f'        "{mc}" => ([{r/255:.4f}, {g/255:.4f}, {b/255:.4f}], {code}u8),')
    lines += [
        "        _ => return None,",
        "    };",
        "    Some(v)",
        "}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "export-viewer":
        dst = sys.argv[2] if len(sys.argv) > 2 else "../flood_pso_viewer/src/block_colors.rs"
        export_viewer_rust(dst)
        print(f"wrote {dst}  ({len(BLOCKS)} blocks)")
    else:
        print(f"{len(BLOCKS)} blocks, {len(MATCH_KEYS)} match anchors")
