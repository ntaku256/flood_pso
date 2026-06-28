"""
nbt_to_litematic.py
flood_pso が出力する Java Structure NBT (.nbt, gzip) を
**Litematica 形式 (.litematic)** に変換する。

redtact のローダ `@taku128/java-schematic` が読む litematic 規約に厳密一致：
  - region.Size / Position（Compound {x,y,z}）
  - BlockStatePalette（List[Compound] {Name, Properties?}）
  - BlockStates（Long_Array, contiguous bit packing。long を跨ぐ）
  - bitsPerBlock = max(2, bitLength(paletteCount - 1))
  - index = y*absSizeX*absSizeZ + z*absSizeX + x   （YZX 順, X 最速）
  - air はパレット内で Name=="minecraft:air" を検索（既定 0 でも可）
  - root は MinecraftDataVersion と Regions のみ必須（Version はゲートされない）

使い方:
  .venv/bin/python src/nbt_to_litematic.py results/nbt/hd/<file>.nbt [out.litematic]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import nbtlib
from nbtlib import Compound, Int, Long, String
from nbtlib import List as NbtList

LITEMATICA_VERSION = 6      # 現行スキーマ（読み手はゲートしないが Litematica mod 互換のため）
LITEMATICA_SUBVERSION = 1


def _xyz(x, y, z) -> Compound:
    return Compound({"x": Int(int(x)), "y": Int(int(y)), "z": Int(int(z))})


def _find_air_index(palette) -> int:
    for i, p in enumerate(palette):
        if str(p.get("Name", "")) == "minecraft:air":
            return i
    return 0


def pack_litematica_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """
    YZX 順に並んだ palette index 配列を Litematica の contiguous bit packing で
    signed int64 long 配列へ。値は long を跨いで連続詰めされる。
    チャンク処理でメモリを抑える（境界の long は bitwise_or で正しく合成される）。
    """
    n = values.shape[0]
    n_longs = (n * bits + 63) // 64
    packed = np.zeros(n_longs, dtype=np.uint64)
    chunk = 8_000_000
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        v = values[s0:s1].astype(np.uint64)
        start = np.arange(s0, s1, dtype=np.int64) * bits
        long_idx = start >> 6
        bit_off = (start & 63)
        np.bitwise_or.at(packed, long_idx, v << bit_off.astype(np.uint64))
        over = (bit_off + bits) > 64
        if over.any():
            sh = (64 - bit_off[over]).astype(np.uint64)
            np.bitwise_or.at(packed, long_idx[over] + 1, v[over] >> sh)
    return packed.view(np.int64)


def _entry_to_compound(name: str, props: dict | None) -> Compound:
    """パレット名＋Properties → BlockStatePalette 用 Compound。Properties は構造NBT側
    （nbt_export の _palette_compound）が単一真実源で、_parse_fast が読み取った値を
    そのまま使う。これにより同一 minecraft 名で複数 state（例: rail の shape 別）を保持できる。"""
    if props:
        return Compound({"Name": String(name),
                         "Properties": Compound({k: String(v) for k, v in props.items()})})
    return Compound({"Name": String(name)})


def structure_nbt_to_litematic(in_path: str, out_path: str | None = None,
                               name: str | None = None, verbose: bool = True) -> str:
    in_path = str(in_path)
    t0 = time.time()
    # nbt_preview の自前バイナリパーサで pos/state を numpy 高速抽出（nbtlib ループ回避）
    from nbt_preview import _parse_fast
    size, names, palprops, (xs, ys, zs, sts) = _parse_fast(in_path)
    sx, sy, sz = (int(v) for v in size)
    air_idx = names.index("minecraft:air") if "minecraft:air" in names else 0
    if verbose:
        print(f"[litematic] in={Path(in_path).name}  size={sx}×{sy}×{sz}  "
              f"palette={len(names)}  blocks={len(xs):,}  air_idx={air_idx}")

    # ── dense index 配列（YZX 順, ベクトル化） ──
    L = sx * sy * sz
    dense = np.full(L, air_idx, dtype=np.int32)
    valid = (xs >= 0) & (xs < sx) & (ys >= 0) & (ys < sy) & (zs >= 0) & (zs < sz)
    flat = (ys.astype(np.int64) * sz + zs.astype(np.int64)) * sx + xs.astype(np.int64)
    dense[flat[valid]] = sts[valid].astype(np.int32)   # 重なりセルは後勝ち
    nonair = int(np.count_nonzero(dense != air_idx))

    palette = names
    data_version = 4671
    bits = max(2, (len(palette) - 1).bit_length())
    longs = pack_litematica_bits(dense, bits)
    if verbose:
        print(f"[litematic] bits/block={bits}  longs={len(longs):,}  "
              f"non-air blocks={nonair:,}  ({time.time()-t0:.1f}s)")

    # ── パレット（名前→Compound, air を同じ index に保つ） ──
    palette_list = NbtList[Compound](
        [_entry_to_compound(n, (palprops[i] if palprops else None))
         for i, n in enumerate(palette)])
    region_name = name or Path(in_path).stem
    now_ms = int(time.time() * 1000)

    region = Compound({
        "Position": _xyz(0, 0, 0),
        "Size": _xyz(sx, sy, sz),
        "BlockStatePalette": palette_list,
        "BlockStates": nbtlib.LongArray(longs),
        "TileEntities": NbtList[Compound]([]),
        "Entities": NbtList[Compound]([]),
        "PendingBlockTicks": NbtList[Compound]([]),
        "PendingFluidTicks": NbtList[Compound]([]),
    })

    litematic = Compound({
        "MinecraftDataVersion": Int(data_version),
        "Version": Int(LITEMATICA_VERSION),
        "SubVersion": Int(LITEMATICA_SUBVERSION),
        "Metadata": Compound({
            "Name": String(region_name),
            "Author": String("flood_pso"),
            "Description": String("Converted from Java Structure NBT by flood_pso/nbt_to_litematic.py"),
            "RegionCount": Int(1),
            "TotalVolume": Int(L),
            "TotalBlocks": Int(nonair),
            "TimeCreated": Long(now_ms),
            "TimeModified": Long(now_ms),
            "EnclosingSize": _xyz(sx, sy, sz),
        }),
        "Regions": Compound({region_name: region}),
    })

    if out_path is None:
        out_path = str(Path(in_path).with_suffix(".litematic"))
    out_path = str(out_path)
    nbtlib.File(litematic).save(out_path, gzipped=True)
    mb = Path(out_path).stat().st_size / 1e6
    if verbose:
        print(f"[litematic] saved {out_path}  ({mb:.1f} MB)  total {time.time()-t0:.1f}s")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: nbt_to_litematic.py <in.nbt> [out.litematic]")
    structure_nbt_to_litematic(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
