"""
make_nbt_tellus_preview.py

Tellus mod が生成した Anvil world を、地形そのまま (8-palette に集約) で
Structure NBT に書き出す「プレビュー専用」エクスポータ。

- 洪水シミュレーション無し（benchmark JSON 不要）
- 全カラムの non-air ブロックを出力（地表だけでなく木・地下も含む）
- 8-palette に無い block 名はマッピング表でフォールバック（既定 stone）。
  未知ブロックがあれば末尾サマリで列挙。

【典型的な使い方】
    # spawn 周辺 500m × 500m を h_res=1m で
    .venv/bin/python src/make_nbt_tellus_preview.py \
        --world-dir "../flood_pso_viewer/新規ワールド (3)" \
        --center-lat 33.833 --center-lon 135.177 \
        --width 500 --depth 500

    # 生成済領域全体を自動 bbox で（デカいので注意）
    .venv/bin/python src/make_nbt_tellus_preview.py \
        --world-dir "../flood_pso_viewer/新規ワールド (3)" --auto-bbox

    # 地下も全部入れたい（重い）
    ... --include-underground
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import nbtlib

sys.path.insert(0, str(Path(__file__).parent))
from anvil_loader import (  # noqa: E402
    TellusWorld,
    lon_to_blockX, lat_to_blockZ, blockX_to_lon, blockZ_to_lat,
    AIR_NAMES, TELLUS_TO_PALETTE, PALETTE_FALLBACK, _decode_section,
)
from nbt_export import (  # noqa: E402
    PALETTE_LIST_KEYS, PALETTE_INDEX, write_nbt_structure,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "nbt" / "tellus_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 自動 bbox: world フォルダの region ファイル名から境界を割り出す
# ─────────────────────────────────────────────────────────────

_R_RE = re.compile(r"^r\.(-?\d+)\.(-?\d+)\.mca$")


def detect_region_bbox(world_dir: Path) -> tuple[int, int, int, int] | None:
    """region 名から (rx_min, rx_max, rz_min, rz_max) を返す（origin 周辺の捨て region は除外）。"""
    rd = world_dir / "dimensions" / "minecraft" / "overworld" / "region"
    if not rd.is_dir():
        return None
    rxs, rzs = [], []
    for p in rd.iterdir():
        m = _R_RE.match(p.name)
        if not m:
            continue
        rx, rz = int(m.group(1)), int(m.group(2))
        # origin 近傍 (|rx|<=2 かつ |rz|<=2) は spawn 自動生成の捨てチャンクと推定して除外
        if abs(rx) <= 2 and abs(rz) <= 2:
            continue
        rxs.append(rx); rzs.append(rz)
    if not rxs:
        return None
    return min(rxs), max(rxs), min(rzs), max(rzs)


# ─────────────────────────────────────────────────────────────
# 全カラムスキャン → NBT block list
# ─────────────────────────────────────────────────────────────

def fetch_columns_full(
    tw: TellusWorld,
    bx_min: int, bx_max: int, bz_min: int, bz_max: int,
    *, include_underground: bool, deep_ground: int, head_room: int,
) -> tuple[list, tuple[int, int], dict]:
    """指定 block 範囲を全カラム scan。

    Returns
    -------
    block_entries : [(rx, ry, rz, palette_key), ...]   ry は y_min を 0 とした相対値
    (y_min, y_max) : 出力に使った絶対 y 範囲
    stats          : 統計 dict（unknown_blocks など）
    """
    H = bz_max - bz_min + 1
    W = bx_max - bx_min + 1
    cx_min, cx_max = bx_min >> 4, bx_max >> 4
    cz_min, cz_max = bz_min >> 4, bz_max >> 4

    # ── 1st pass: 各 (rx, rz) の surface y を取り、y 範囲決定 ──
    print(f"  [scan] pass 1 / 2: 表面 y 範囲を測定中 ({W}×{H} cells)...")
    surface_y = np.full((H, W), np.nan, dtype=np.float32)
    for cz in range(cz_min, cz_max + 1):
        for cx in range(cx_min, cx_max + 1):
            ch = tw._chunk(cx, cz)
            if ch is None:
                continue
            sections = ch.get('sections')
            if sections is None:
                continue
            decoded = []
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
            lz0 = max(0, bz_min - (cz << 4))
            lz1 = min(15, bz_max - (cz << 4))
            lx0 = max(0, bx_min - (cx << 4))
            lx1 = min(15, bx_max - (cx << 4))
            for lz in range(lz0, lz1 + 1):
                row = (cz << 4) + lz - bz_min
                for lx in range(lx0, lx1 + 1):
                    col = (cx << 4) + lx - bx_min
                    for sy, pal_names, arr in decoded:
                        base = sy * 16
                        found = False
                        for ly in range(15, -1, -1):
                            idx = int(arr[ly, lz, lx])
                            name = pal_names[idx]
                            if name in AIR_NAMES:
                                continue
                            surface_y[row, col] = base + ly
                            found = True
                            break
                        if found:
                            break
    valid = surface_y[np.isfinite(surface_y)]
    if valid.size == 0:
        return [], (0, 0), {'n_loaded': 0, 'unknown_blocks': Counter()}
    y_surf_min = int(valid.min())
    y_surf_max = int(valid.max())

    # 出力 y 範囲：地下 deep_ground、地上 head_room（木・建物用）
    if include_underground:
        y_out_min = int(min(y_surf_min, valid.min()) - deep_ground)
        # 完全地下まで含めると重いのでとりあえず -64 を下限
        y_out_min = max(y_out_min, -64)
    else:
        y_out_min = y_surf_min - deep_ground
    y_out_max = y_surf_max + head_room
    print(f"  [scan] surface y range: {y_surf_min}..{y_surf_max}  "
          f"→ output y: {y_out_min}..{y_out_max} ({y_out_max - y_out_min + 1} layers)")

    # ── 2nd pass: 範囲内の全 non-air を出力 ──
    print(f"  [scan] pass 2 / 2: ブロック書き出し中...")
    block_entries: list = []
    unknown = Counter()
    n_chunks_loaded = 0
    n_chunks_total = (cx_max - cx_min + 1) * (cz_max - cz_min + 1)

    for cz in range(cz_min, cz_max + 1):
        for cx in range(cx_min, cx_max + 1):
            ch = tw._chunk(cx, cz)
            if ch is None:
                continue
            sections = ch.get('sections')
            if sections is None:
                continue
            decoded = []
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
                sy = int(s['Y'].value)
                base = sy * 16
                if base + 15 < y_out_min or base > y_out_max:
                    continue
                data = bs.get('data')
                arr = _decode_section(pal_names, list(data) if data is not None else None)
                decoded.append((sy, pal_names, arr))
            if not decoded:
                continue
            n_chunks_loaded += 1

            lz0 = max(0, bz_min - (cz << 4))
            lz1 = min(15, bz_max - (cz << 4))
            lx0 = max(0, bx_min - (cx << 4))
            lx1 = min(15, bx_max - (cx << 4))
            for sy, pal_names, arr in decoded:
                base = sy * 16
                # palette key を section ごとに事前変換
                pal_key = []
                for n in pal_names:
                    if n in AIR_NAMES:
                        pal_key.append(None)
                    elif n in TELLUS_TO_PALETTE:
                        pal_key.append(TELLUS_TO_PALETTE[n])
                    else:
                        unknown[n] += 1
                        pal_key.append(PALETTE_FALLBACK)
                for ly in range(16):
                    y = base + ly
                    if y < y_out_min or y > y_out_max:
                        continue
                    ry = y - y_out_min
                    for lz in range(lz0, lz1 + 1):
                        rz = (cz << 4) + lz - bz_min
                        for lx in range(lx0, lx1 + 1):
                            rx = (cx << 4) + lx - bx_min
                            idx = int(arr[ly, lz, lx])
                            k = pal_key[idx]
                            if k is None:
                                continue
                            block_entries.append((rx, ry, rz, k))
    stats = {
        'n_loaded_chunks': n_chunks_loaded,
        'n_total_chunks': n_chunks_total,
        'n_loaded_cells': int(np.isfinite(surface_y).sum()),
        'n_total_cells': int(H * W),
        'unknown_blocks': unknown,
    }
    return block_entries, (y_out_min, y_out_max), stats


def to_nbt_blocks(entries: list[tuple[int, int, int, str]]) -> list:
    """(rx, ry, rz, palette_key) → nbtlib.Compound の Structure block entry list."""
    out = []
    for rx, ry, rz, k in entries:
        out.append(nbtlib.Compound({
            "pos":   nbtlib.List[nbtlib.Int]([nbtlib.Int(rx), nbtlib.Int(ry), nbtlib.Int(rz)]),
            "state": nbtlib.Int(PALETTE_INDEX[k]),
        }))
    return out


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-dir", required=True, help="Tellus 生成 world フォルダ (level.dat があるパス)")
    ap.add_argument("--world-scale", type=float, default=1.0)
    ap.add_argument("--center-lat", type=float, default=None,
                    help="中心緯度。--auto-bbox の場合は不要")
    ap.add_argument("--center-lon", type=float, default=None)
    ap.add_argument("--width", type=float, default=500.0,
                    help="東西幅 [m]（既定 500）")
    ap.add_argument("--depth", type=float, default=500.0,
                    help="南北幅 [m]（既定 500）")
    ap.add_argument("--auto-bbox", action="store_true",
                    help="region フォルダから自動的に bbox を決定（origin 周辺は除外）。"
                         "--width/--depth を無視し全領域を出力。重い場合あり。")
    ap.add_argument("--include-underground", action="store_true",
                    help="地表より下 deep_ground 以上も全部出力する（既定 OFF, deep_ground=8 のみ）")
    ap.add_argument("--deep-ground", type=int, default=8,
                    help="地表からの deep ground 厚 [block]（既定 8）")
    ap.add_argument("--head-room", type=int, default=30,
                    help="地表より上に取る厚さ [block]（木や建物のため、既定 30）")
    ap.add_argument("--out", type=str, default=None,
                    help="出力ファイルパス（省略時 results/nbt/tellus_preview/<name>.nbt）")
    ap.add_argument("--tag", type=str, default="preview",
                    help="出力ファイル名のタグ（既定 preview）")
    args = ap.parse_args()

    world_dir = Path(args.world_dir).expanduser()
    if not world_dir.is_dir():
        sys.exit(f"world dir not found: {world_dir}")
    print(f"world dir: {world_dir}")

    tw = TellusWorld(world_dir, world_scale=args.world_scale)

    if args.auto_bbox:
        rb = detect_region_bbox(world_dir)
        if rb is None:
            sys.exit("auto-bbox failed: region 名から bbox を決められませんでした")
        rx0, rx1, rz0, rz1 = rb
        bx_min = rx0 * 512
        bx_max = (rx1 + 1) * 512 - 1
        bz_min = rz0 * 512
        bz_max = (rz1 + 1) * 512 - 1
        print(f"[auto-bbox] region rx={rx0}..{rx1}, rz={rz0}..{rz1}  "
              f"→ block X={bx_min}..{bx_max} ({bx_max-bx_min+1}b)  "
              f"Z={bz_min}..{bz_max} ({bz_max-bz_min+1}b)")
        bx_c = (bx_min + bx_max) / 2
        bz_c = (bz_min + bz_max) / 2
        c_lat = blockZ_to_lat(bz_c, args.world_scale)
        c_lon = blockX_to_lon(bx_c, args.world_scale)
        print(f"[auto-bbox] center lat,lon = ({c_lat:.6f}, {c_lon:.6f})")
    else:
        if args.center_lat is None or args.center_lon is None:
            sys.exit("--center-lat / --center-lon を指定するか --auto-bbox を使ってください")
        c_lat, c_lon = args.center_lat, args.center_lon
        bx_c = lon_to_blockX(c_lon, args.world_scale)
        bz_c = lat_to_blockZ(c_lat, args.world_scale)
        half_w = args.width / 2.0 / args.world_scale
        half_d = args.depth / 2.0 / args.world_scale
        bx_min = int(math.floor(bx_c - half_w))
        bx_max = bx_min + int(math.ceil(2 * half_w)) - 1
        bz_min = int(math.floor(bz_c - half_d))
        bz_max = bz_min + int(math.ceil(2 * half_d)) - 1
        print(f"center (lat,lon)=({c_lat:.6f},{c_lon:.6f}) → block ({bx_c:.1f},{bz_c:.1f})  "
              f"region ({int(bx_c)>>9},{int(bz_c)>>9})")
        print(f"bbox blockX {bx_min}..{bx_max} ({bx_max-bx_min+1}b)  "
              f"blockZ {bz_min}..{bz_max} ({bz_max-bz_min+1}b)")

    entries, (y_min, y_max), stats = fetch_columns_full(
        tw, bx_min, bx_max, bz_min, bz_max,
        include_underground=args.include_underground,
        deep_ground=args.deep_ground,
        head_room=args.head_room,
    )
    H = bz_max - bz_min + 1
    W = bx_max - bx_min + 1
    Y = y_max - y_min + 1

    print(f"\nstats: chunks {stats['n_loaded_chunks']}/{stats['n_total_chunks']} loaded  "
          f"surface cells {stats['n_loaded_cells']}/{stats['n_total_cells']}")
    if stats['unknown_blocks']:
        print(f"\nunknown blocks ({len(stats['unknown_blocks'])} types) → '{PALETTE_FALLBACK}' fallback:")
        for n, c in stats['unknown_blocks'].most_common(20):
            print(f"  {n}: {c}")
        print("  (anvil_loader.TELLUS_TO_PALETTE に追加してマッピングを上書きできます)")

    if not entries:
        sys.exit("(no blocks emitted — bbox が generated chunks の外かも)")

    print(f"\nemitting Structure NBT: size W={W} × Y={Y} × D={H} = {W*Y*H:,} cells, "
          f"{len(entries):,} non-air blocks ({len(entries)/max(1,W*Y*H)*100:.1f}% fill)")

    blocks = to_nbt_blocks(entries)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = OUT_DIR / f"tellus_{args.tag}_lat{c_lat:.4f}_lon{c_lon:.4f}_{W}x{H}.nbt"

    meta = {
        "experiment": "tellus_world_preview",
        "method": "preview",
        "method_long": "Tellus mod world direct dump (no flood overlay)",
        "tellus_world_dir": str(world_dir),
        "tellus_world_scale": float(args.world_scale),
        "center_lat": float(c_lat),
        "center_lon": float(c_lon),
        "bbox_blockX": [int(bx_min), int(bx_max)],
        "bbox_blockZ": [int(bz_min), int(bz_max)],
        "bbox_y": [int(y_min), int(y_max)],
        "structure_size_xyz": [int(W), int(Y), int(H)],
        "n_block_entries": int(len(entries)),
        "deep_ground": int(args.deep_ground),
        "head_room": int(args.head_room),
        "include_underground": bool(args.include_underground),
        "n_unknown_block_types": int(len(stats['unknown_blocks'])),
        "ref_doc": "flood_pso/docs/07_地形レンダ改善.md",
    }
    write_nbt_structure(blocks, [W, Y, H], str(out_path), meta=meta)


if __name__ == "__main__":
    main()
