"""
nbt_preview.py
Minecraft を使わずに Structure NBT を俯瞰プレビュー画像化する。

各 (x, z) 列について最上位の非空気ブロックを取り、材質色＋高さ陰影（hillshade）で
2D マップとして描画し、洪水（水ブロック）を半透明の青で重畳する。

高速化:
  - nbtlib を使わず自前のバイナリパーサで blocks を numpy 配列へ直接抽出
  - 最上ブロック集計を numpy でベクトル化（lexsort）
  - 抽出結果を results/nbt_preview/.cache/*.npz にキャッシュ（再実行は即時）

材質色は flood_pso_viewer のパレットに準拠。

使い方:
  .venv/bin/python src/nbt_preview.py results/nbt/hd/gobo_hd_K16_seed0_md_5m_ccpso2.nbt
  .venv/bin/python src/nbt_preview.py "results/nbt/**/*.nbt"   # 一括 + コンタクトシート
"""
import os
import sys
import gzip
import time
import glob
import struct
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# 材質名 -> RGB（flood_pso_viewer 準拠）
MAT_COLOR = {
    "minecraft:stone": (130, 130, 130),
    "minecraft:grass_block": (95, 150, 70),
    "minecraft:sand": (222, 210, 160),
    "minecraft:gravel": (128, 118, 108),
    "minecraft:dirt": (134, 96, 67),
    "minecraft:coarse_dirt": (120, 86, 60),
    "minecraft:bedrock": (25, 25, 25),
}
# 水・氷系（旧形式の生 water/ice と stained_glass 版の両対応）
WATER_NAMES = {"minecraft:blue_stained_glass", "minecraft:cyan_stained_glass",
               "minecraft:water", "minecraft:blue_ice", "minecraft:ice", "minecraft:packed_ice"}
WATER_COLOR = {"minecraft:blue_stained_glass": (40, 95, 225),
               "minecraft:cyan_stained_glass": (95, 200, 225),
               "minecraft:water": (40, 95, 225),
               "minecraft:blue_ice": (120, 190, 225),
               "minecraft:ice": (150, 200, 230),
               "minecraft:packed_ice": (150, 200, 230)}
UNKNOWN_COLOR = (220, 40, 220)
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "nbt_preview"
CACHE_DIR = OUT_DIR / ".cache"


# ─────────────────────────────────────────────────────────────
# 自前 NBT バイナリパーサ（blocks を numpy へ高速抽出）
# ─────────────────────────────────────────────────────────────
class _Reader:
    __slots__ = ("b", "p")

    def __init__(self, b):
        self.b = b
        self.p = 0

    def u8(self):
        v = self.b[self.p]; self.p += 1; return v

    def u16(self):
        v = struct.unpack_from(">H", self.b, self.p)[0]; self.p += 2; return v

    def i32(self):
        v = struct.unpack_from(">i", self.b, self.p)[0]; self.p += 4; return v

    def name(self):
        l = self.u16(); s = self.b[self.p:self.p + l]; self.p += l
        return s.decode("utf-8", "replace")

    def skip(self, tid):
        b = self.b
        if tid == 1: self.p += 1
        elif tid == 2: self.p += 2
        elif tid in (3, 5): self.p += 4
        elif tid in (4, 6): self.p += 8
        elif tid == 7: self.p += 4 + struct.unpack_from(">i", b, self.p)[0]   # byte array (len は含めて飛ばす)
        elif tid == 8: self.p += 2 + struct.unpack_from(">H", b, self.p)[0]   # string
        elif tid == 9:                                                        # list
            etid = self.u8(); n = self.i32()
            for _ in range(n): self.skip(etid)
        elif tid == 10:                                                       # compound
            while True:
                t = self.u8()
                if t == 0: break
                self.name(); self.skip(t)
        elif tid == 11: self.p += 4 + 4 * struct.unpack_from(">i", b, self.p)[0]  # int array
        elif tid == 12: self.p += 4 + 8 * struct.unpack_from(">i", b, self.p)[0]  # long array
        else:
            raise ValueError(f"unknown tag id {tid} @ {self.p}")

    def parse_blocks(self, n):
        """blocks リスト（compound 要素 n 個）を numpy へ。各要素から pos[3], state を抽出。"""
        b = self.b
        xs = np.empty(n, np.int32); ys = np.empty(n, np.int32)
        zs = np.empty(n, np.int32); st = np.empty(n, np.int32)
        for k in range(n):
            while True:
                tid = b[self.p]; self.p += 1
                if tid == 0:
                    break
                l = struct.unpack_from(">H", b, self.p)[0]; self.p += 2
                nm = b[self.p:self.p + l]; self.p += l
                if tid == 3 and nm == b"state":
                    st[k] = struct.unpack_from(">i", b, self.p)[0]; self.p += 4
                elif tid == 9 and nm == b"pos":
                    self.p += 1                                    # element tag id (=3)
                    ln = struct.unpack_from(">i", b, self.p)[0]; self.p += 4
                    xs[k] = struct.unpack_from(">i", b, self.p)[0]
                    ys[k] = struct.unpack_from(">i", b, self.p + 4)[0]
                    zs[k] = struct.unpack_from(">i", b, self.p + 8)[0]
                    self.p += 4 * ln
                else:
                    self.skip(tid)
        return xs, ys, zs, st


def _scan(r):
    """compound を1つ読み size/palette/blocks を返す。子 compound にあれば再帰探索
    （入れ子ラッパー root→''/'root'→{size,palette,blocks} 形式に対応）。"""
    size = names = arrays = None
    while True:
        tid = r.u8()
        if tid == 0:
            break
        nm = r.name()
        if tid == 9 and nm == "size":
            r.u8(); n = r.i32(); size = [r.i32() for _ in range(n)]
        elif tid == 9 and nm == "palette":
            r.u8(); n = r.i32(); pal = []
            for _ in range(n):
                nm2 = None
                while True:
                    t = r.u8()
                    if t == 0:
                        break
                    pn = r.name()
                    if t == 8 and pn == "Name":
                        nm2 = r.name()
                    else:
                        r.skip(t)
                pal.append(nm2 or "minecraft:air")
            names = pal
        elif tid == 9 and nm == "blocks":
            r.u8(); n = r.i32(); arrays = r.parse_blocks(n)
        elif tid == 10:
            s2, n2, a2 = _scan(r)
            if a2 is not None:
                arrays = a2
                if s2 is not None: size = s2
                if n2 is not None: names = n2
        else:
            r.skip(tid)
    return size, names, arrays


def _parse_fast(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    r = _Reader(raw)
    root_tid = r.u8()
    if root_tid != 10:
        raise ValueError("root is not a compound")
    r.name()  # root name
    size, names, arrays = _scan(r)
    if size is None or names is None or arrays is None:
        raise ValueError("size/palette/blocks not found")
    return size, names, arrays


def _aggregate(path, verbose=True):
    """最上ブロック集計（top_h/top_st/wat_h/wat_st, names, size）。npz キャッシュ付き。"""
    st = os.stat(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{Path(path).stem}_{int(st.st_mtime)}_{st.st_size}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return (d["top_h"], d["top_st"], d["wat_h"], d["wat_st"],
                list(d["names"]), tuple(int(v) for v in d["size"]))

    t0 = time.time()
    if verbose:
        print(f"    parsing {Path(path).name} ({st.st_size/1e6:.1f}MB)...", flush=True)
    size, names, (xs, ys, zs, sts) = _parse_fast(path)
    sx, sy, sz = size
    is_water = np.array([n in WATER_NAMES for n in names], bool)
    is_air = np.array([n == "minecraft:air" for n in names], bool)
    inb = (xs >= 0) & (xs < sx) & (zs >= 0) & (zs < sz)
    cell = zs.astype(np.int64) * sx + xs

    def top_per_cell(selmask):
        c = cell[selmask]; y = ys[selmask]; s = sts[selmask]
        h = np.full(sx * sz, -1, np.int32); stt = np.full(sx * sz, -1, np.int32)
        if len(c):
            order = np.lexsort((y, c))           # cell 昇順→y 昇順
            c2 = c[order]; y2 = y[order]; s2 = s[order]
            last = np.empty(len(c2), bool); last[-1] = True; last[:-1] = c2[1:] != c2[:-1]
            h[c2[last]] = y2[last]; stt[c2[last]] = s2[last]
        return h.reshape(sz, sx), stt.reshape(sz, sx)

    top_h, top_st = top_per_cell((~is_water[sts]) & (~is_air[sts]) & inb)
    wat_h, wat_st = top_per_cell(is_water[sts] & inb)
    if verbose:
        print(f"      done {len(xs)} blocks in {time.time()-t0:.1f}s", flush=True)

    np.savez_compressed(cache, top_h=top_h, top_st=top_st, wat_h=wat_h, wat_st=wat_st,
                        names=np.array(names, dtype=object), size=np.array(size))
    return top_h, top_st, wat_h, wat_st, names, (sx, sy, sz)


def render_topdown(path, verbose=True):
    top_h, top_st, wat_h, wat_st, names, (sx, sy, sz) = _aggregate(path, verbose=verbose)
    valid = top_h >= 0

    # hillshade
    if valid.any():
        hv = top_h.astype(float)
        hv[~valid] = np.nanmin(hv[valid]) if valid.any() else 0.0
        gy, gx = np.gradient(hv)
        slope = gx - gy
        shade = np.clip(0.85 + 0.10 * (slope / (np.abs(slope).max() + 1e-6)), 0.6, 1.15)
    else:
        shade = np.ones((sz, sx))

    mat_rgb = np.array([MAT_COLOR.get(n, UNKNOWN_COLOR) for n in names] + [UNKNOWN_COLOR], float)
    flat_st = np.where(valid, top_st, len(names))
    img = np.full((sz, sx, 3), 245, dtype=np.uint8)
    terr = mat_rgb[flat_st] * shade[:, :, None]
    img[valid] = np.clip(terr[valid], 0, 255).astype(np.uint8)

    wmask = wat_h >= 0
    if wmask.any():
        # 浸水深 = 水面 y - 地表 y（地表が無ければ水面 y）。浅い=明るい青 / 深い=濃い青。
        depth = np.where(top_h >= 0, wat_h - top_h, wat_h).astype(float)
        depth = np.clip(depth, 0, None)
        t = np.clip(depth / 12.0, 0, 1)[:, :, None]
        light = np.array([150, 205, 238], float)   # 浅
        dark = np.array([12, 45, 150], float)       # 深
        wcol = light * (1 - t) + dark * t
        a = 0.72
        blended = (1 - a) * img.astype(float) + a * wcol
        img[wmask] = np.clip(blended[wmask], 0, 255).astype(np.uint8)

    pim = Image.fromarray(img, "RGB")
    m = max(sx, sz)
    if m > 1400:
        pim = pim.resize((sx * 1400 // m, sz * 1400 // m), Image.NEAREST)
    elif m < 720:
        sc = max(1, round(720 / m))
        pim = pim.resize((sx * sc, sz * sc), Image.NEAREST)

    info = dict(size=(sx, sy, sz), terrain_cells=int(valid.sum()), water_cells=int(wmask.sum()))
    return pim, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="NBT ファイル or グロブ")
    ap.add_argument("--no-sheet", action="store_true", help="コンタクトシートを作らない")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files += [Path(f) for f in glob.glob(p, recursive=True)]
    files = sorted(set(files))
    if not files:
        print("NBT が見つかりません"); return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} NBT をプレビュー化 → {OUT_DIR}", flush=True)

    thumbs = []
    for f in files:
        t0 = time.time()
        try:
            pim, info = render_topdown(f)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}"); continue
        out = OUT_DIR / (f.stem + ".png")
        pim.save(out)
        sx, sy, sz = info["size"]
        print(f"  {f.name}: {sx}x{sy}x{sz}  terr={info['terrain_cells']} "
              f"water={info['water_cells']}  ({time.time()-t0:.1f}s)", flush=True)
        thumbs.append((f.stem, pim))

    if not args.no_sheet and thumbs:
        _contact_sheet(thumbs, OUT_DIR / "_contact_sheet.png")
        print(f"コンタクトシート: {OUT_DIR/'_contact_sheet.png'}")


def _contact_sheet(thumbs, path, cols=4, cell=320, pad=8):
    from PIL import ImageDraw
    n = len(thumbs); rows = (n + cols - 1) // cols
    label_h = 22
    W = cols * (cell + pad) + pad
    H = rows * (cell + label_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad); y = pad + r * (cell + label_h + pad)
        t = im.copy(); t.thumbnail((cell, cell), Image.NEAREST)
        sheet.paste(t, (x + (cell - t.width) // 2, y + label_h + (cell - t.height) // 2))
        short = name if len(name) <= 46 else name[:43] + "..."
        draw.text((x, y + 4), short, fill=(0, 0, 0))
    sheet.save(path)


if __name__ == "__main__":
    main()
