#!/usr/bin/env python3
"""nbt_section.py — 生成ワールドの **鉛直断面** を PNG 化する検証ツール。

src/nbt_preview.py は俯瞰(トップダウン)投影しか出せないため、

  * 橋のデッキが地形に埋没していないか / 水没していないか / 途中で段差がないか
  * トンネルが実際に貫通しているか / 天井が閉じているか / 平坦地で地表に箱が生えていないか
  * 崖の側面に穴(air)が空いていないか

といった **高さ方向の不具合が一切判定できない**。本ツールは指定した線に沿って
ワールドを縦に切り、1 ブロック = 1 ピクセルで色分け描画する。

────────────────────────────────────────────────────────────────────────
入力
────────────────────────────────────────────────────────────────────────
  * Structure NBT     … results/nbt/hd/*.nbt （flood_pso_meta から緯度経度を自動取得）
  * Anvil ワールド dir … results/anvil/gobo_crop 等（緯度経度は --center/--size か
                          --bbox で明示。y はワールド絶対 y をそのまま使う）

断面の線は次のいずれかで指定する:
  --from LAT,LON --to LAT,LON     2 点を結ぶ直線
  --way OSM_WAY_ID                OSM way の折れ線に沿う（--geom-json / 既定の橋 JSON から検索）
  --from-block X,Z --to-block X,Z 緯度経度不要（メタが無い Anvil でも使える）

────────────────────────────────────────────────────────────────────────
使い方（橋・トンネルを直すときの実例）
────────────────────────────────────────────────────────────────────────
  # 0) 対象範囲にどの橋/トンネルがあるか一覧する
  .venv/bin/python tools/nbt_section.py --list-ways \
      --geom-json data_cache/osm/gobo_bridges_south4_geom.json \
      --bbox 33.8320,33.8390,135.1820,135.1910

  # 1) 橋 1 本の縦断面（way に沿って切る）
  .venv/bin/python tools/nbt_section.py results/nbt/hd/xxx.nbt \
      --way 385194099 --geom-json data_cache/osm/gobo_bridges_011_geom.json \
      --margin 40 -o /tmp/bridge.png

  # 2) トンネルの貫通確認（坑口から坑口まで way に沿って縦断）
  .venv/bin/python tools/nbt_section.py results/anvil/gobo_crop \
      --center 33.8337,135.1789 --size 400x400 --scale 1.5 \
      --way <TUNNEL_WAY_ID> --geom-json data_cache/osm/gobo_tunnels_geom.json \
      --margin 30 --cavity-max 12 -o /tmp/tunnel.png

  # 3) 崖の穴チェック（崖を横切る短い線 + 拡大 + 厚み3）
  #    ※ thick>1 は柱を OR 合成するので穴(空洞)の数は過小評価になる。
  #      穴を数えるときは --thick 1 でも切って [warn] の thick=1 相当値と見比べること。
  .venv/bin/python tools/nbt_section.py results/nbt/hd/xxx.nbt \
      --from 33.8360,135.1850 --to 33.8360,135.1870 --thick 3 --px 6 -o /tmp/cliff.png

────────────────────────────────────────────────────────────────────────
読み方
────────────────────────────────────────────────────────────────────────
  * 各ブロックは block_palette.py の色で描く（地表・道路・建物・橋が材質で見分く）
  * 水(blue/cyan stained glass, water, ice) は青系
  * **空 sky**（白に近い灰）  = その柱の最上位固体より上の air
  * **空洞 cavity**（黄）     = 上下を固体に挟まれた高さ --cavity-max(既定16)以下の air。
    トンネル内空(5-9block)・建物内部・桁下がここに出る。**検査対象はこの黄色**。
    トンネルが埋まれば黄色が途切れ、平坦地に箱が生えれば黄色が地表より上に浮く。
  * **大空洞 big gap**（ベージュ） = 閉じているが背が高い air。地表の下 deep_ground(=8)
    しか地盤を書かないので山の内部はここになる（**仕様であってバグではない**）。
  * **VOID**（青灰）          = 下が抜けている air（最下スラブより下＝モデル外の地下）
  * **NO DATA**（赤の斜線ハッチ） = その柱の chunk/列そのものが取れなかった station。
    **空（白）と必ず区別する**こと。ここは「貫通していない」のではなく「見えていない」。
  * `--mark minecraft:andesite` 等で特定ブロックをマゼンタで強調（橋デッキ/橋脚=andesite）

標準出力には最上位固体 y とその最大段差、空洞セル数と y 範囲、水面 y、
--mark 指定時はデッキ上の被り厚 (cover) と埋没/水没の station 数を出す。
cover は **デッキに接している固体の連続段数** なので、上空の樹冠や架線では増えない。

注意: `--thick>1` は法線方向の柱を **OR 合成**（どれか1本でも固体なら固体）するため、
幅が thick 未満の空洞（トンネル内空など）は埋まって見えなくなる。貫通確認は
`--thick 1`（既定）で行うこと。thick>1 を指定すると thick=1 相当の空洞数を警告に出す。
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from block_palette import BLOCKS, name_to_rgb  # noqa: E402

MAT_COLOR = dict(name_to_rgb())
# 名前 → role（air / opaque / water / ice）。BLOCKS は key→(name, rgb, role)。
NAME_ROLE = {v[0]: v[2] for v in BLOCKS.values()}
# 旧形式 / バニラ生水など、パレット外でも水として扱う名前
EXTRA_WATER = {"minecraft:water", "minecraft:flowing_water", "minecraft:ice",
               "minecraft:packed_ice", "minecraft:blue_ice",
               "minecraft:blue_stained_glass", "minecraft:cyan_stained_glass",
               "minecraft:light_blue_stained_glass"}
AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air", "air"}

# block_palette に無いが意味の分かる名前（バニラ生水/氷など）。
# これを入れないと _is_water が水と判定するのに描画は UNKNOWN(マゼンタ)になり、
# --mark のハイライト(255,0,255)と見分けが付かない。
EXTRA_COLOR = {
    "minecraft:water": (40, 100, 200),
    "minecraft:flowing_water": (40, 100, 200),
    "minecraft:bubble_column": (40, 100, 200),
    "minecraft:ice": (150, 220, 240),
    "minecraft:packed_ice": (150, 220, 240),
    "minecraft:blue_ice": (150, 220, 240),
}
for _n, _c in EXTRA_COLOR.items():
    MAT_COLOR.setdefault(_n, _c)

SKY_RGB = (250, 250, 252)      # 最上位固体より上の air（＝空。水と紛れないよう白に近い灰）
CAVITY_RGB = (255, 224, 80)    # 閉じた小空洞（トンネル/建物内部/桁下）＝ここが検査対象
BIGGAP_RGB = (236, 228, 196)   # 閉じているが cavity_max 超（地形シェル内部など）
VOID_RGB = (178, 186, 204)     # 下が抜けている air（＝モデル化されていない地下）。
#                                stone(125,125,125)/deepslate と紛れないよう青寄りにする
UNKNOWN_RGB = (220, 40, 220)
MARK_RGB = (255, 0, 255)
# データ欠損（chunk/列そのものが無い）。空(白)と絶対に混同しないよう赤の斜線ハッチで描く。
NODATA_RGB = (198, 74, 74)
NODATA_BG_RGB = (247, 224, 224)

LAT_M = 111320.0
CACHE_DIR = REPO_ROOT / "data_cache" / "section_cache"


def _is_water(name: str) -> bool:
    return NAME_ROLE.get(name) in ("water", "ice") or name in EXTRA_WATER


def _is_air(name: str) -> bool:
    return name in AIR_NAMES or NAME_ROLE.get(name) == "air"


# ══════════════════════════════════════════════════════════════════════
# Structure NBT ストリーミングリーダ
#   nbt_export._write_nbt_dense は 1 ブロック = 固定 36 byte で書くので、
#   その並びを numpy でベクトル復号する（60M ブロックでも Python ループ無し）。
# ══════════════════════════════════════════════════════════════════════
_VOXEL_TMPL = (b"\x09\x00\x03pos\x03\x00\x00\x00\x03" + b"\x00" * 12
               + b"\x03\x00\x05state" + b"\x00" * 4 + b"\x00")
assert len(_VOXEL_TMPL) == 36


class _Stream:
    """gzip/生ファイルを前方向にだけ読む簡易バッファ。巨大 blocks を貯めずに流す。"""

    def __init__(self, fobj):
        self.f = fobj
        self.buf = b""
        self.pos = 0

    def _need(self, n: int):
        while len(self.buf) - self.pos < n:
            chunk = self.f.read(max(1 << 20, n))
            if not chunk:
                raise EOFError("unexpected end of NBT stream")
            self.buf = self.buf[self.pos:] + chunk
            self.pos = 0

    def read(self, n: int) -> bytes:
        self._need(n)
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v

    def discard(self, n: int):
        have = len(self.buf) - self.pos
        if n <= have:
            self.pos += n
            return
        n -= have
        self.buf = b""
        self.pos = 0
        while n > 0:
            chunk = self.f.read(min(n, 1 << 22))
            if not chunk:
                raise EOFError("unexpected end of NBT stream (discard)")
            n -= len(chunk)

    def u8(self) -> int:
        return self.read(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.read(2))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack(">q", self.read(8))[0]

    def name(self) -> str:
        return self.read(self.u16()).decode("utf-8", "replace")


def _read_payload(s: _Stream, tid: int):
    """TAG payload を Python 値として読む（blocks 以外は小さいので素直に読む）。"""
    if tid == 1:
        return struct.unpack(">b", s.read(1))[0]
    if tid == 2:
        return struct.unpack(">h", s.read(2))[0]
    if tid == 3:
        return s.i32()
    if tid == 4:
        return s.i64()
    if tid == 5:
        return struct.unpack(">f", s.read(4))[0]
    if tid == 6:
        return struct.unpack(">d", s.read(8))[0]
    if tid == 7:
        return s.read(s.i32())
    if tid == 8:
        return s.name()
    if tid == 9:
        et = s.u8()
        n = s.i32()
        return [_read_payload(s, et) for _ in range(max(0, n))]
    if tid == 10:
        out = {}
        while True:
            t = s.u8()
            if t == 0:
                return out
            nm = s.name()
            out[nm] = _read_payload(s, t)
    if tid == 11:
        return [s.i32() for _ in range(s.i32())]
    if tid == 12:
        return [s.i64() for _ in range(s.i32())]
    raise ValueError(f"unknown tag id {tid}")


def _skip_payload(s: _Stream, tid: int):
    if tid == 1:
        s.discard(1)
    elif tid == 2:
        s.discard(2)
    elif tid in (3, 5):
        s.discard(4)
    elif tid in (4, 6):
        s.discard(8)
    elif tid == 7:
        s.discard(s.i32())
    elif tid == 8:
        s.discard(s.u16())
    elif tid == 9:
        et = s.u8()
        n = s.i32()
        for _ in range(max(0, n)):
            _skip_payload(s, et)
    elif tid == 10:
        while True:
            t = s.u8()
            if t == 0:
                return
            s.discard(s.u16())
            _skip_payload(s, t)
    elif tid == 11:
        s.discard(4 * s.i32())
    elif tid == 12:
        s.discard(8 * s.i32())
    else:
        raise ValueError(f"unknown tag id {tid}")


def _open_nbt(path: Path) -> _Stream:
    with open(path, "rb") as f:
        magic = f.read(2)
    fobj = gzip.open(str(path), "rb") if magic == b"\x1f\x8b" else open(path, "rb")
    s = _Stream(fobj)
    if s.u8() != 10:
        raise ValueError(f"{path}: root is not a TAG_Compound")
    s.name()
    return s


def _scan_nbt(path: Path, want_cols: np.ndarray | None, nx_hint: int | None,
              verbose: bool = True):
    """Structure NBT を1パスで流し読みする。

    want_cols : bool 配列 (nz*nx) — True の (x,z) 柱のブロックだけ拾う。None なら
                ブロックを一切読まない（メタだけ欲しいとき）。
    戻り値: dict(size, names, meta, cells) — cells は (x, y, z, state) の tuple of arrays
    """
    t0 = time.time()
    s = _open_nbt(path)
    size = names = meta = None
    xs_o = ys_o = zs_o = st_o = None
    while True:
        tid = s.u8()
        if tid == 0:
            break
        nm = s.name()
        if tid == 9 and nm == "size":
            s.u8(); n = s.i32()
            size = [s.i32() for _ in range(n)]
        elif tid == 9 and nm == "palette":
            s.u8(); n = s.i32()
            pal = []
            for _ in range(n):
                d = _read_payload(s, 10)
                pal.append(str(d.get("Name", "minecraft:air")))
            names = pal
        elif tid == 9 and nm == "blocks":
            et = s.u8(); n = s.i32()
            if et != 10:
                raise ValueError("blocks list element is not a compound")
            xs_o, ys_o, zs_o, st_o = _stream_blocks(
                s, n, want_cols, nx_hint if nx_hint else (size[0] if size else 1), verbose)
        elif tid == 10 and nm == "flood_pso_meta":
            meta = _read_payload(s, 10)
        else:
            _skip_payload(s, tid)
    if verbose:
        print(f"    [nbt] scanned {path.name} in {time.time()-t0:.1f}s", flush=True)
    return dict(size=size, names=names, meta=meta,
                cells=(xs_o, ys_o, zs_o, st_o) if xs_o is not None else None)


def _stream_blocks(s: _Stream, n: int, want_cols, nx: int, verbose: bool):
    """blocks リスト本体を読む。want_cols=None なら丸ごと読み飛ばす。"""
    if want_cols is None:
        # 固定 36 byte 形式なら一気に飛ばせる。そうでなくても要素単位で skip。
        if _peek_is_template(s):
            s.discard(36 * n)
        else:
            for _ in range(n):
                _skip_payload(s, 10)
        return None, None, None, None

    if not _peek_is_template(s):
        return _stream_blocks_generic(s, n, want_cols, nx)

    CH = 500_000
    XS, YS, ZS, ST = [], [], [], []
    done = 0
    while done < n:
        m = min(CH, n - done)
        raw = s.read(36 * m)
        a = np.frombuffer(raw, dtype=np.uint8).reshape(m, 36)
        x = np.ascontiguousarray(a[:, 11:15]).view(">i4").ravel().astype(np.int32)
        y = np.ascontiguousarray(a[:, 15:19]).view(">i4").ravel().astype(np.int32)
        z = np.ascontiguousarray(a[:, 19:23]).view(">i4").ravel().astype(np.int32)
        st = np.ascontiguousarray(a[:, 31:35]).view(">i4").ravel().astype(np.int32)
        cell = z.astype(np.int64) * nx + x
        ok = (cell >= 0) & (cell < want_cols.size)
        sel = np.zeros(m, bool)
        sel[ok] = want_cols[cell[ok]]
        if sel.any():
            XS.append(x[sel]); YS.append(y[sel]); ZS.append(z[sel]); ST.append(st[sel])
        done += m
    cat = (np.concatenate(XS) if XS else np.empty(0, np.int32),
           np.concatenate(YS) if YS else np.empty(0, np.int32),
           np.concatenate(ZS) if ZS else np.empty(0, np.int32),
           np.concatenate(ST) if ST else np.empty(0, np.int32))
    if verbose:
        print(f"    [nbt] blocks={n:,} → corridor {len(cat[0]):,}", flush=True)
    return cat


def _peek_is_template(s: _Stream) -> bool:
    """次の 36 byte が nbt_export の固定ボクセルテンプレートか（消費しない）。"""
    try:
        s._need(36)
    except EOFError:
        return False
    b = s.buf[s.pos:s.pos + 36]
    return (b[0:11] == _VOXEL_TMPL[0:11] and b[23:31] == _VOXEL_TMPL[23:31]
            and b[35] == 0)


def _stream_blocks_generic(s: _Stream, n: int, want_cols, nx: int):
    """テンプレート外（nbtlib 経路で書かれた .nbt）の遅いフォールバック。"""
    XS, YS, ZS, ST = [], [], [], []
    for _ in range(n):
        d = _read_payload(s, 10)
        p = d.get("pos") or [0, 0, 0]
        x, y, z = int(p[0]), int(p[1]), int(p[2])
        cell = z * nx + x
        if 0 <= cell < want_cols.size and want_cols[cell]:
            XS.append(x); YS.append(y); ZS.append(z); ST.append(int(d.get("state", 0)))
    return (np.array(XS, np.int32), np.array(YS, np.int32),
            np.array(ZS, np.int32), np.array(ST, np.int32))


def read_nbt_meta(path: Path, verbose: bool = True) -> dict:
    """flood_pso_meta（+ size / palette）だけを取る。結果は npz/json にキャッシュ。"""
    st = os.stat(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{path.stem}_{int(st.st_mtime)}_{st.st_size}.meta.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if verbose:
        print(f"    [nbt] reading meta of {path.name} ({st.st_size/1e6:.1f}MB, "
              f"初回のみ全体を流し読み)...", flush=True)
    r = _scan_nbt(path, None, None, verbose=verbose)
    meta = r["meta"] or {}
    out = {"size": r["size"], "meta": {k: v for k, v in meta.items()
                                       if isinstance(v, (int, float, str, list))}}
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ══════════════════════════════════════════════════════════════════════
# Anvil ワールドリーダ（chunk 単位で1回だけデコードして柱を取る）
# ══════════════════════════════════════════════════════════════════════
class AnvilReader:
    """chunk / region file のキャッシュは **LRU で上限を持つ**。長い線（数千 station）を
    切ると素の dict では復号済み chunk が際限なく積み上がってメモリが伸び続けるため。
    1 chunk は (y範囲, 16, 16) int32 ＝ y400 で約 0.4MB。既定 192 枚で上限 ~75MB。"""

    def __init__(self, world_dir: str | Path, dimension: str = "minecraft:overworld",
                 max_chunks: int = 192, max_regions: int = 8):
        from nbt import region as _region
        from anvil_loader import _decode_section
        self._region_mod = _region
        self._decode = _decode_section
        self.dir = Path(world_dir)
        ns, key = dimension.split(":", 1)
        cands = [self.dir / "dimensions" / ns / key / "region", self.dir / "region"]
        self.region_dir = next((c for c in cands if c.is_dir()), None)
        if self.region_dir is None:
            raise FileNotFoundError(
                f"region dir not found under {self.dir} (tried: "
                + ", ".join(str(c) for c in cands) + ")")
        self._rf: OrderedDict = OrderedDict()
        self._ch: OrderedDict = OrderedDict()
        self.max_chunks = max(1, int(max_chunks))
        self.max_regions = max(1, int(max_regions))
        self.chunks_read = 0        # 復号した延べ chunk 数（キャッシュ内数とは別）

    def _put_chunk(self, key, val):
        self._ch[key] = val
        self._ch.move_to_end(key)
        while len(self._ch) > self.max_chunks:
            self._ch.popitem(last=False)
        return val

    def _region_file(self, rx, rz):
        if (rx, rz) in self._rf:
            self._rf.move_to_end((rx, rz))
            return self._rf[(rx, rz)]
        f = self.region_dir / f"r.{rx}.{rz}.mca"
        try:
            rf = self._region_mod.RegionFile(str(f)) if f.exists() else None
        except Exception:
            rf = None
        self._rf[(rx, rz)] = rf
        while len(self._rf) > self.max_regions:
            _, old = self._rf.popitem(last=False)
            try:
                if old is not None:
                    old.close()
            except Exception:
                pass
        return rf

    def chunk(self, cx, cz):
        """(y0, arr[Y,Z,X] int16 palette-local, names) or None"""
        if (cx, cz) in self._ch:
            self._ch.move_to_end((cx, cz))
            return self._ch[(cx, cz)]
        rf = self._region_file(cx >> 5, cz >> 5)
        root = None
        if rf is not None:
            try:
                root = rf.get_chunk(cx & 31, cz & 31)
            except Exception:
                root = None
        if root is None or root.get("sections") is None:
            return self._put_chunk((cx, cz), None)
        secs = []
        for s in root["sections"]:
            bs = s.get("block_states")
            if bs is None:
                continue
            pal = bs.get("palette")
            if pal is None or len(pal) == 0:
                continue
            pal_names = [p["Name"].value for p in pal]
            data = bs.get("data")
            arr = self._decode(pal_names, list(data) if data is not None else None)
            secs.append((int(s["Y"].value), pal_names, arr))
        if not secs:
            return self._put_chunk((cx, cz), None)
        secs.sort(key=lambda t: t[0])
        y0 = secs[0][0] * 16
        y1 = secs[-1][0] * 16 + 16
        names = ["minecraft:air"]
        idx = {"minecraft:air": 0}
        out = np.zeros((y1 - y0, 16, 16), np.int32)
        for sy, pal_names, arr in secs:
            remap = np.empty(len(pal_names), np.int32)
            for i, nm in enumerate(pal_names):
                if nm not in idx:
                    idx[nm] = len(names)
                    names.append(nm)
                remap[i] = idx[nm]
            base = sy * 16 - y0
            out[base:base + 16] = remap[arr]
        self.chunks_read += 1
        return self._put_chunk((cx, cz), (y0, out, names))

    def column(self, x: int, z: int):
        """(y0, names_idx array over y, names) — 無ければ None"""
        c = self.chunk(x >> 4, z >> 4)
        if c is None:
            return None
        y0, arr, names = c
        return y0, arr[:, z & 15, x & 15], names

    def world_block_bounds(self):
        """region ファイル名から world の chunk 範囲 → ブロック範囲を推定。"""
        xs, zs = [], []
        for f in self.region_dir.glob("r.*.*.mca"):
            try:
                _, rx, rz, _ = f.name.split(".")
                xs.append(int(rx)); zs.append(int(rz))
            except Exception:
                continue
        if not xs:
            return None
        return (min(xs) * 512, max(xs) * 512 + 511, min(zs) * 512, max(zs) * 512 + 511)


# ══════════════════════════════════════════════════════════════════════
# ジオリファレンス
# ══════════════════════════════════════════════════════════════════════
class Geo:
    """ブロック格子 ↔ 緯度経度。ワールド左上(北西)ブロック (x0,z0) を基準にする。"""

    def __init__(self, lat_max, lon_min, mpp, x0=0, z0=0, dlat=None, dlon=None):
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.mpp = float(mpp)
        self.x0 = int(x0)
        self.z0 = int(z0)
        # bbox から作るときは deg/block を厳密に渡す（cos(lat) 近似を挟まない）
        self.dlat = float(dlat) if dlat else self.mpp / LAT_M
        self.dlon = (float(dlon) if dlon else
                     self.mpp / (LAT_M * math.cos(math.radians(self.lat_max))))

    def to_block(self, lat, lon):
        x = self.x0 + (lon - self.lon_min) / self.dlon
        z = self.z0 + (self.lat_max - lat) / self.dlat
        return x, z

    def to_latlon(self, x, z):
        lon = self.lon_min + (x - self.x0) * self.dlon
        lat = self.lat_max - (z - self.z0) * self.dlat
        return lat, lon

    def __repr__(self):
        return (f"Geo(lat_max={self.lat_max:.7f}, lon_min={self.lon_min:.7f}, "
                f"mpp={self.mpp:.4f}, origin=({self.x0},{self.z0}))")


def geo_from_bbox(bbox, nx, nz, x0=0, z0=0):
    la0, la1, lo0, lo1 = bbox
    dlat = (la1 - la0) / max(1, nz)
    dlon = (lo1 - lo0) / max(1, nx)
    return Geo(la1, lo0, dlat * LAT_M, x0, z0, dlat=dlat, dlon=dlon)


def _ncells(half_m, mpp):
    """make_nbt_hd/nbt_export と同じ切り捨て。0.6666667 のような丸め誤差入り mpp でも
    480 が 479 に落ちないよう相対 1e-6 の許容を足す。"""
    v = half_m / mpp
    return int(v + 1e-6 * max(1.0, v))


def geo_from_center(clat, clon, width_m, depth_m, mpp, x0=0, z0=0):
    nx = 2 * _ncells(width_m / 2.0, mpp)
    nz = 2 * _ncells(depth_m / 2.0, mpp)
    lat_max = clat + (nz / 2.0) * mpp / LAT_M
    lon_min = clon - (nx / 2.0) * mpp / (LAT_M * math.cos(math.radians(clat)))
    return Geo(lat_max, lon_min, mpp, x0, z0), nx, nz


# ══════════════════════════════════════════════════════════════════════
# OSM way
# ══════════════════════════════════════════════════════════════════════
DEFAULT_GEOM_GLOBS = ["data_cache/osm/gobo_bridges_*_geom.json",
                      "data_cache/osm/gobo_bridges_geom.json",
                      "data_cache/osm/gobo_tunnels*_geom.json"]


def iter_ways(geom_jsons):
    for p in geom_jsons:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] {p}: {e}")
            continue
        for e in d.get("elements", []):
            g = e.get("geometry") or []
            if len(g) < 2:
                continue
            yield str(p), e, [(float(q["lat"]), float(q["lon"])) for q in g]


def resolve_geom_jsons(args):
    if args.geom_json:
        return [s.strip() for s in args.geom_json.split(",") if s.strip()]
    out = []
    for g in DEFAULT_GEOM_GLOBS:
        out += sorted(str(p) for p in REPO_ROOT.glob(g))
    return out


# ══════════════════════════════════════════════════════════════════════
# 断面サンプリング
# ══════════════════════════════════════════════════════════════════════
def polyline_stations(pts_block, step=1.0):
    """[(x,z), ...] を step ブロック間隔でサンプル → (xs, zs, dist)"""
    xs, zs, ds = [], [], []
    acc = 0.0
    for (x0, z0), (x1, z1) in zip(pts_block, pts_block[1:]):
        seg = math.hypot(x1 - x0, z1 - z0)
        n = max(1, int(round(seg / step)))
        for k in range(n):
            t = k / n
            xs.append(x0 + (x1 - x0) * t)
            zs.append(z0 + (z1 - z0) * t)
            ds.append(acc + seg * t)
        acc += seg
    xs.append(pts_block[-1][0]); zs.append(pts_block[-1][1]); ds.append(acc)
    return np.array(xs), np.array(zs), np.array(ds)


def extend_polyline(pts_block, margin):
    """両端を margin ブロックだけ外側へ延長（橋のアプローチ/取り付けを見るため）。"""
    if margin <= 0 or len(pts_block) < 2:
        return pts_block
    p = [list(map(float, q)) for q in pts_block]
    def unit(a, b):
        dx, dz = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dz) or 1.0
        return dx / n, dz / n
    ux, uz = unit(p[1], p[0])
    head = [p[0][0] + ux * margin, p[0][1] + uz * margin]
    vx, vz = unit(p[-2], p[-1])
    tail = [p[-1][0] + vx * margin, p[-1][1] + vz * margin]
    return [head] + p + [tail]


def perp_offsets(xs, zs, thick):
    """各 station の法線方向単位ベクトル（thick>1 のとき横に少しずらして探すため）。"""
    dx = np.gradient(xs); dz = np.gradient(zs)
    n = np.hypot(dx, dz)
    n[n == 0] = 1.0
    return -dz / n, dx / n     # 法線 = 進行方向を +90deg 回転


# ══════════════════════════════════════════════════════════════════════
# 断面データ構築
# ══════════════════════════════════════════════════════════════════════
def build_section_nbt(path, xs, zs, thick, verbose=True):
    meta = read_nbt_meta(Path(path), verbose=verbose)
    size = meta["size"]
    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])
    cols = _sample_columns(xs, zs, thick, nx, nz)
    want = np.zeros(nz * nx, bool)
    for (x, z) in cols:
        want[z * nx + x] = True
    r = _scan_nbt(Path(path), want, nx, verbose=verbose)
    names = r["names"]
    bx, by, bz, bs = r["cells"]
    colmap: dict[tuple[int, int], dict[int, int]] = {}
    for x, y, z, s in zip(bx.tolist(), by.tolist(), bz.tolist(), bs.tolist()):
        colmap.setdefault((x, z), {})[y] = s
    return colmap, names, (nx, ny, nz), meta.get("meta", {}), 0


def build_section_anvil(path, xs, zs, thick, verbose=True):
    rd = AnvilReader(path)
    cols = _sample_columns(xs, zs, thick, None, None)
    colmap: dict[tuple[int, int], dict[int, int]] = {}
    names = ["minecraft:air"]
    idx = {"minecraft:air": 0}
    for (x, z) in cols:
        c = rd.column(x, z)
        if c is None:
            continue
        y0, arr, cnames = c
        remap = []
        for nm in cnames:
            if nm not in idx:
                idx[nm] = len(names)
                names.append(nm)
            remap.append(idx[nm])
        remap = np.array(remap, np.int32)
        loc = remap[arr]
        nz_idx = np.nonzero(loc != 0)[0]
        if len(nz_idx) == 0:
            colmap[(x, z)] = {}
            continue
        colmap[(x, z)] = {int(y0 + i): int(loc[i]) for i in nz_idx.tolist()}
    if verbose:
        print(f"    [anvil] columns={len(colmap)}/{len(cols)} "
              f"chunks decoded={rd.chunks_read} (cache {len(rd._ch)}/{rd.max_chunks})",
              flush=True)
    if not colmap:
        bb = rd.world_block_bounds()
        raise SystemExit(
            "Anvil から1柱も読めませんでした。ジオリファレンス(--center/--size/--scale か "
            "--bbox, --origin)を確認して下さい。\n"
            f"  要求した block 範囲: x[{int(min(xs))},{int(max(xs))}] "
            f"z[{int(min(zs))},{int(max(zs))}]\n"
            + (f"  region ファイルが覆う範囲: x[{bb[0]},{bb[1]}] z[{bb[2]},{bb[3]}]"
               if bb else "  region ファイルが1つもありません"))
    return colmap, names, None, {}, 0


def _sample_columns(xs, zs, thick, nx, nz):
    px, pz = perp_offsets(xs, zs, thick)
    half = (thick - 1) // 2
    cols = []
    seen = set()
    for i in range(len(xs)):
        for k in range(-half, half + 1):
            x = int(round(xs[i] + px[i] * k))
            z = int(round(zs[i] + pz[i] * k))
            if nx is not None and not (0 <= x < nx and 0 <= z < nz):
                continue
            if (x, z) not in seen:
                seen.add((x, z))
                cols.append((x, z))
    return cols


def assemble_grid(colmap, names, xs, zs, thick, nx=None, nz=None):
    """station × y の材質 index グリッドを作る（0=air/未取得）。

    戻り値の hit[i] は「その station の柱にブロックが 1 つでもあったか」。
    flood_pso の出力は必ずベッドロック板 + 地盤シェルを書くので、1 ブロックも無い柱は
    構造外 / 未生成 chunk / タイル外＝**データ欠損**。False の列は描画で NO DATA
    ハッチにし（空 sky の白と区別する）、統計からも除く。
    """
    px, pz = perp_offsets(xs, zs, thick)
    half = (thick - 1) // 2
    n = len(xs)
    # y 範囲
    ymin, ymax = None, None
    for d in colmap.values():
        if not d:
            continue
        a, b = min(d), max(d)
        ymin = a if ymin is None else min(ymin, a)
        ymax = b if ymax is None else max(ymax, b)
    if ymin is None:
        raise SystemExit("断面上にブロックが1つもありません（線の位置 or ジオリファレンスを確認）")
    H = ymax - ymin + 1
    grid = np.zeros((H, n), np.int32)
    hit = np.zeros(n, bool)
    for i in range(n):
        for k in sorted(range(-half, half + 1), key=abs):
            x = int(round(xs[i] + px[i] * k))
            z = int(round(zs[i] + pz[i] * k))
            d = colmap.get((x, z))
            if not d:                         # 柱が無い/空 = データ欠損（描画は NO DATA）
                continue
            for y, s in d.items():
                if grid[y - ymin, i] == 0:
                    grid[y - ymin, i] = s
            hit[i] = True
            if thick == 1:
                break
    return grid, ymin, hit


# ══════════════════════════════════════════════════════════════════════
# 描画
# ══════════════════════════════════════════════════════════════════════
def classify_air(solid, fill, cavity_max):
    """air セルを sky / cavity / biggap / void に分類する。

    列ごとに air の連続区間(run)を取り、
      * 上に固体が無い run              → sky（空）
      * 下に固体が無い run              → void（モデル外）
      * 最下段スラブの直上から始まる run → void（＝地盤の下の未モデル化空間）
      * 上下とも固体で高さ<=cavity_max   → cavity（トンネル/建物内部/桁下＝**検査対象の空洞**）
      * 上下とも固体だが高さ>cavity_max  → biggap（地形シェル内部など）
    後半2つを分ける理由: 出力は地表の下 deep_ground(既定8)ブロックしか地盤を書かないので
    山の内部は巨大な空洞になる。同じ色で塗ると本物のトンネル(内空5-9block)が埋もれる。
    """
    H, W = solid.shape
    air = ~fill
    sky = np.zeros_like(air)
    cavity = np.zeros_like(air)
    biggap = np.zeros_like(air)
    void = np.zeros_like(air)
    for j in range(W):
        acol = air[:, j]
        if not acol.any():
            continue
        scol = solid[:, j]
        s_below = np.cumsum(scol) - scol > 0
        s_above = np.cumsum(scol[::-1])[::-1] - scol > 0
        # 最下段スラブ（bedrock 等）の上端。この直上から始まる air は未モデル化の地下。
        srows = np.nonzero(scol)[0]
        slab_top = -2
        if len(srows):
            k = 0
            while k + 1 < len(srows) and srows[k + 1] == srows[k] + 1:
                k += 1
            slab_top = int(srows[k])
        idx = np.nonzero(acol)[0]
        brk = np.nonzero(np.diff(idx) > 1)[0]
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk, [len(idx) - 1]))
        for a_, b_ in zip(idx[starts], idx[ends]):
            if not s_above[a_]:
                sky[a_:b_ + 1, j] = True
            elif not s_below[b_] or a_ == slab_top + 1:
                void[a_:b_ + 1, j] = True
            elif (b_ - a_ + 1) <= cavity_max:
                cavity[a_:b_ + 1, j] = True
            else:
                biggap[a_:b_ + 1, j] = True
    return sky, cavity, biggap, void


def grid_masks(grid, names):
    """材質 index グリッド → (solid, water, fill) bool マスク。"""
    is_air = np.array([_is_air(n) for n in names] + [False])
    is_wat = np.array([_is_water(n) for n in names] + [False])
    solid = (~is_air[grid]) & (~is_wat[grid])
    water = is_wat[grid]
    return solid, water, solid | water


def cavity_stats(grid, names, cavity_max):
    """(空洞セル数, 空洞のある station 数) — thick 違いの比較用に描画抜きで数える。"""
    solid, _water, fill = grid_masks(grid, names)
    _sky, cav, _big, _void = classify_air(solid, fill, cavity_max)
    return int(cav.sum()), int((cav.sum(axis=0) > 0).sum())


def cover_above(mtop, anym, solid):
    """marked の最上段に **接している** 固体の連続段数。

    旧実装は「その柱の最上位固体 − marked 最上段」だったため、間に空気があっても
    上空の樹冠(--trees)や送電線があるだけで大きな値になり、埋没の偽陽性を出していた。
    ここでは marked の直上から連続する固体だけを数える（空気に当たった時点で 0 で止まる）。
    """
    H, W = solid.shape
    rows = np.arange(H, dtype=np.int64)[:, None]
    above = rows > mtop[None, :]
    first_air = np.where(above & (~solid), rows, H).min(axis=0)
    return np.where(anym, np.maximum(first_air - mtop - 1, 0), 0).astype(np.int64)


def render(grid, ymin, names, ds_m, title, subtitle, mark_names=(), px=None,
           cavity_max=16, ylim=None, max_w=1700, max_h=900, hit=None):
    """ylim=(ymin,ymax) は **分類の後で** 切る（切ってから分類すると天井/床が消えて
    空洞判定が壊れるため）。"""
    H, W = grid.shape
    rgb_tab = np.array([MAT_COLOR.get(n, UNKNOWN_RGB) for n in names] + [UNKNOWN_RGB], float)
    marked = np.array([n in mark_names for n in names] + [False])

    solid, water, fill = grid_masks(grid, names)
    sky, cavity, biggap, void = classify_air(solid, fill, cavity_max)
    miss = (np.zeros(W, bool) if hit is None else ~np.asarray(hit, bool))

    # 各 station の最上位「固体」行（画像は上が高 y なので行 index は反転させて扱う）
    top_y = np.full(W, -1, np.int64)
    any_solid = solid.any(axis=0)
    if any_solid.any():
        top_y[any_solid] = (H - 1) - np.argmax(solid[::-1, :], axis=0)[any_solid]

    img = np.zeros((H, W, 3), float)
    img[:] = SKY_RGB
    img[sky] = SKY_RGB
    img[void] = VOID_RGB
    img[biggap] = BIGGAP_RGB
    img[cavity] = CAVITY_RGB
    img[fill] = rgb_tab[grid][fill]
    mk = marked[grid]
    if mk.any():
        img[mk] = 0.35 * img[mk] + 0.65 * np.array(MARK_RGB, float)

    # データ欠損の列は赤の斜線ハッチ（空 sky の白と絶対に混同させない）
    if miss.any():
        rr, cc = np.mgrid[0:H, 0:W]
        mcol = np.broadcast_to(miss[None, :], (H, W))
        hatch = ((rr + cc) % 6) < 2
        img[mcol & hatch] = NODATA_RGB
        img[mcol & ~hatch] = NODATA_BG_RGB

    # 表示だけを ylim で切る（統計・空洞分類は全高のまま返す）
    stats = dict(top_y=np.where(any_solid, top_y + ymin, np.nan),
                 cavity=cavity, biggap=biggap, void=void, water=water,
                 solid=solid, marked=mk, ymin=ymin, H=H, W=W, any_solid=any_solid,
                 missing=miss)
    ymin_d, H_d, img_d, fill_d, grid_d = ymin, H, img, fill, grid
    stats["disp"] = None
    stats["ylim_applied"] = None
    if ylim is not None and (ylim[0] is not None or ylim[1] is not None):
        lo = max(ymin, ylim[0] if ylim[0] is not None else ymin)
        hi = min(ymin + H - 1, ylim[1] if ylim[1] is not None else ymin + H - 1)
        a, b = lo - ymin, hi - ymin + 1
        stats["ylim_applied"] = b > a
        if b > a:
            img_d, fill_d, grid_d = img[a:b], fill[a:b], grid[a:b]
            ymin_d, H_d = lo, b - a

    # 上下反転（y が大きいほど上）
    img, fill, grid, ymin, H = img_d, fill_d, grid_d, ymin_d, H_d
    stats["disp"] = (ymin_d, H_d)
    img = np.flipud(img).astype(np.uint8)
    sec = Image.fromarray(img, "RGB")

    if px is None:
        px = max(1, min(8, max_w // max(1, W), max_h // max(1, H)))
    sec = sec.resize((W * px, H * px), Image.NEAREST)

    # 余白付きキャンバス
    L, R, T, B = 62, 210, 46, 40
    cw, chh = sec.size
    canvas = Image.new("RGB", (L + cw + R, T + chh + B), (255, 255, 255))
    canvas.paste(sec, (L, T))
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([L - 1, T - 1, L + cw, T + chh], outline=(90, 90, 90))

    # y 軸目盛
    ystep = 10 if H <= 120 else (20 if H <= 260 else 50)
    for y in range(int(math.ceil(ymin / ystep) * ystep), ymin + H, ystep):
        row = (ymin + H - 1 - y)
        py = T + row * px
        dr.line([L - 5, py, L, py], fill=(90, 90, 90))
        dr.text((4, py - 5), f"y={y:>4d}", fill=(60, 60, 60))
    # x 軸目盛（m）
    tot = ds_m[-1] if len(ds_m) else 0
    xstep = 50 if tot <= 400 else (100 if tot <= 1200 else 250)
    for d in range(0, int(tot) + 1, xstep):
        i = int(np.searchsorted(ds_m, d))
        i = min(i, W - 1)
        pxx = L + i * px
        dr.line([pxx, T + chh, pxx, T + chh + 5], fill=(90, 90, 90))
        dr.text((pxx - 10, T + chh + 8), f"{d}m", fill=(60, 60, 60))

    dr.text((6, 6), title[:210], fill=(0, 0, 0))
    dr.text((6, 20), subtitle[:210], fill=(70, 70, 70))
    dr.text((6, 32), f"sky=white   CAVITY(enclosed air <={cavity_max} blocks tall)=yellow   "
                     "big enclosed gap=beige   VOID(open below)=gray   water=blue   "
                     + ("NO DATA=red hatch   " if miss.any() else "")
                     + "(block colors: src/block_palette.py)",
            fill=(110, 110, 110))

    # 凡例（出現数の多い材質）
    vals, cnt = np.unique(grid[fill], return_counts=True)
    order = np.argsort(-cnt)
    y = T + 2
    dr.text((L + cw + 8, y), "materials", fill=(0, 0, 0)); y += 14
    for k in order[:22]:
        v = int(vals[k])
        nm = names[v] if v < len(names) else "?"
        c = tuple(int(q) for q in rgb_tab[v])
        dr.rectangle([L + cw + 8, y, L + cw + 18, y + 9], fill=c, outline=(60, 60, 60))
        dr.text((L + cw + 22, y - 1), f"{nm.replace('minecraft:','')[:20]} {int(cnt[k])}",
                fill=(40, 40, 40))
        y += 12
        if y > T + chh - 24:
            break
    legend = [(CAVITY_RGB, f"CAVITY (<={cavity_max}b tall)"),
              (BIGGAP_RGB, "big enclosed gap"),
              (VOID_RGB, "VOID (unmodelled)")]
    if miss.any():
        legend.append((NODATA_RGB, f"NO DATA x{int(miss.sum())} (not sky!)"))
    for c, lab in legend:
        dr.rectangle([L + cw + 8, y, L + cw + 18, y + 9], fill=c, outline=(60, 60, 60))
        dr.text((L + cw + 22, y - 1), lab, fill=(40, 40, 40))
        y += 12
    return canvas, stats


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════
def parse_pair(s, what):
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        raise SystemExit(f"--{what} は 'A,B' 形式で指定して下さい: {s!r}")


def parse_bbox(s):
    parts = [p for p in str(s).split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise SystemExit("--bbox は 'lat_min,lat_max,lon_min,lon_max' の 4 要素で指定して"
                         f"下さい（{len(parts)} 要素でした: {s!r}）")
    try:
        la0, la1, lo0, lo1 = (float(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--bbox の要素が数値ではありません: {s!r}")
    if la0 >= la1 or lo0 >= lo1:
        raise SystemExit(f"--bbox は lat_min<lat_max, lon_min<lon_max の順です: {s!r}")
    return la0, la1, lo0, lo1


def parse_origin(s):
    parts = [p for p in str(s).split(",") if p.strip() != ""]
    if len(parts) != 2:
        raise SystemExit(f"--origin は 'X,Z' の 2 要素で指定して下さい: {s!r}")
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        raise SystemExit(f"--origin の要素が数値ではありません: {s!r}")


def parse_size(s):
    try:
        w, d = (float(v) for v in str(s).lower().split("x"))
    except Exception:
        raise SystemExit(f"--size は 'WIDTHxDEPTH'（例 400x400）で指定して下さい: {s!r}")
    if w <= 0 or d <= 0:
        raise SystemExit(f"--size は正の値で指定して下さい: {s!r}")
    return w, d


def main():
    ap = argparse.ArgumentParser(
        description="生成ワールドの鉛直断面を PNG 化（橋デッキ/トンネル貫通/崖の穴の検証用）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world", nargs="?", help=".nbt ファイル または Anvil world ディレクトリ")
    ap.add_argument("--from", dest="p_from", help="始点 LAT,LON")
    ap.add_argument("--to", dest="p_to", help="終点 LAT,LON")
    ap.add_argument("--from-block", help="始点 X,Z（ブロック座標。ジオリファレンス不要）")
    ap.add_argument("--to-block", help="終点 X,Z")
    ap.add_argument("--way", type=int, help="OSM way id（その折れ線に沿って切る）")
    ap.add_argument("--geom-json", help="way を探す geom JSON（カンマ区切り可）")
    ap.add_argument("--list-ways", action="store_true", help="geom JSON の way を一覧して終了")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="way/線の両端を延長する長さ[m]（橋の取り付け部を見るのに使う）")
    ap.add_argument("--thick", type=int, default=1,
                    help="断面の厚み[block, **奇数のみ**]。既定 1。>1 は法線方向の柱を OR 合成"
                         "する（どれか1本でも固体なら固体）ので、線が1ブロックずれても構造を"
                         "捉えられる代わりに **幅が thick 未満の空洞は埋まって消える**。"
                         "トンネルの貫通確認は必ず --thick 1 で行うこと")
    ap.add_argument("--bbox", help="lat_min,lat_max,lon_min,lon_max（生成ログの [patch_bbox]）")
    ap.add_argument("--center", help="LAT,LON（--width/--depth と同じ値。Anvil の georef 用）")
    ap.add_argument("--size", help="WIDTHxDEPTH [m]（例 400x400）")
    ap.add_argument("--h-res", type=float, default=None,
                    help="m/block（既定: NBT は meta から、Anvil は --scale から）")
    ap.add_argument("--scale", type=float, default=None,
                    help="生成時の --scale。m/block = 1/scale（未指定なら 1.5 = 0.6667m/block "
                         "を仮定し警告する）。--h-res を明示したらそちらが優先")
    ap.add_argument("--origin", default="0,0", help="Anvil world 上の構造原点 X,Z（既定 0,0）")
    ap.add_argument("--ymin", type=int, default=None)
    ap.add_argument("--ymax", type=int, default=None)
    ap.add_argument("--px", type=int, default=None, help="1 ブロックのピクセル数（既定 自動）")
    ap.add_argument("--cavity-max", type=int, default=16,
                    help="空洞(黄)として強調する air 連続区間の最大高さ[block]（既定 16）。"
                         "トンネル内空は 5-9block, 建物階高は 3-4block。これを超える上下閉塞"
                         "air は『地形シェル内部』としてベージュにする")
    ap.add_argument("--mark", default="", help="強調するブロック名(カンマ区切り)。例 minecraft:andesite")
    ap.add_argument("-o", "--out", default=None, help="出力 PNG")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    # ── 引数検証（黙って別の値に化けるのを防ぐ） ──
    if args.ymin is not None and args.ymax is not None and args.ymin > args.ymax:
        raise SystemExit(f"--ymin({args.ymin}) > --ymax({args.ymax}) です。逆に指定して下さい")
    if args.cavity_max < 1:
        raise SystemExit(f"--cavity-max は 1 以上で指定して下さい: {args.cavity_max}")
    if args.px is not None and args.px < 1:
        raise SystemExit(f"--px は 1 以上で指定して下さい: {args.px}")
    if args.thick < 1:
        raise SystemExit(f"--thick は 1 以上の奇数で指定して下さい: {args.thick}")
    if args.thick % 2 == 0:
        print(f"  [warn] --thick は奇数のみ。{args.thick} → {args.thick | 1} として扱います")

    geom_jsons = resolve_geom_jsons(args)

    # ── way 一覧 ──
    if args.list_ways:
        bbox = parse_bbox(args.bbox) if args.bbox else None
        n = 0
        seen: dict[int, list[str]] = {}
        rows: list[tuple[int, str]] = []
        for src, e, g in iter_ways(geom_jsons):
            if bbox and not any(bbox[0] <= la <= bbox[1] and bbox[2] <= lo <= bbox[3]
                                for la, lo in g):
                continue
            if int(e["id"]) in seen:                    # 図郭別 JSON は同じ way を重複して持つ
                seen[int(e["id"])].append(Path(src).name)
                continue
            seen[int(e["id"])] = [Path(src).name]
            t = e.get("tags", {})
            L = sum(math.hypot((g[i + 1][0] - g[i][0]) * LAT_M,
                               (g[i + 1][1] - g[i][1]) * LAT_M
                               * math.cos(math.radians(g[i][0])))
                    for i in range(len(g) - 1))
            cla = sum(p[0] for p in g) / len(g); clo = sum(p[1] for p in g) / len(g)
            rows.append((int(e["id"]),
                         f"way {e['id']:>12}  L={L:6.0f}m n={len(g):3d}  "
                         f"c=({cla:.5f},{clo:.5f})  "
                         f"hw={t.get('highway')} layer={t.get('layer')} "
                         f"tunnel={t.get('tunnel')} bridge={t.get('bridge')} "
                         f"name={t.get('name')}"))
            n += 1
        for wid, line in rows:
            print(f"{line}  [{','.join(seen[wid])}]")
        print(f"-- {n} ways ({len(geom_jsons)} JSON) --")
        return

    if not args.world:
        raise SystemExit("world (.nbt か Anvil ディレクトリ) を指定して下さい")
    wpath = Path(args.world)
    if not wpath.exists():
        raise SystemExit(f"not found: {wpath}")
    is_nbt = wpath.is_file()

    # ── ジオリファレンス ──
    geo = None
    nx = nz = None
    meta = {}
    if is_nbt:
        m = read_nbt_meta(wpath, verbose=verbose)
        meta = m.get("meta", {})
        sz = m.get("size") or []
        if len(sz) == 3:
            nx, nz = int(sz[0]), int(sz[2])
    DEFAULT_SCALE = 1.5
    mpp = args.h_res
    mpp_src = "--h-res"
    if mpp is None:
        mpp = float(meta.get("h_res_m_per_block", 0.0)) or None
        mpp_src = "flood_pso_meta"
    if mpp is None:
        sc = float(args.scale) if args.scale else DEFAULT_SCALE
        if sc <= 0:
            raise SystemExit(f"--scale は正の値で指定して下さい: {args.scale}")
        mpp = 1.0 / sc
        mpp_src = f"--scale {sc}"
        if args.scale is None:
            mpp_src = f"既定 --scale {DEFAULT_SCALE}(仮定)"
            print(f"  [warn] {'flood_pso_meta が無い' if is_nbt else 'Anvil はメタを持たない'}"
                  f"ので m/block を {mpp:.4f} (--scale {DEFAULT_SCALE}) と仮定しました。"
                  "生成時の --scale が違うと距離[m]・緯度経度が食い違います "
                  "(--scale か --h-res を明示して下さい)")

    ox, oz = parse_origin(args.origin)
    if args.bbox:
        bb = parse_bbox(args.bbox)
        if nx is None or nz is None:
            if not args.size:
                raise SystemExit("--bbox を Anvil に使うときは --size も指定して下さい")
            w, d = parse_size(args.size)
            nx, nz = 2 * _ncells(w / 2, mpp), 2 * _ncells(d / 2, mpp)
        geo = geo_from_bbox(bb, nx, nz, ox, oz)
    elif args.center:
        clat, clon = parse_pair(args.center, "center")
        if not args.size:
            raise SystemExit("--center には --size WxD が要ります")
        w, d = parse_size(args.size)
        geo, nx, nz = geo_from_center(clat, clon, w, d, mpp, ox, oz)
    elif meta.get("center_lat") is not None and nx:
        geo, _nx, _nz = geo_from_center(float(meta["center_lat"]), float(meta["center_lon"]),
                                        float(meta["width_m"]), float(meta["depth_m"]),
                                        mpp, ox, oz)
        # meta の width/depth と実サイズがズレたら実サイズを優先して中心合わせ
        if _nx != nx or _nz != nz:
            lat_max = float(meta["center_lat"]) + (nz / 2.0) * mpp / LAT_M
            lon_min = (float(meta["center_lon"])
                       - (nx / 2.0) * mpp / (LAT_M * math.cos(math.radians(meta["center_lat"]))))
            geo = Geo(lat_max, lon_min, mpp, ox, oz)
    if verbose and geo:
        print(f"  georef: {geo}  size=({nx},{nz}) blocks  m/block from {mpp_src}")

    # ── 断面の線 ──
    label = ""
    if args.from_block and args.to_block:
        a = parse_pair(args.from_block, "from-block")
        b = parse_pair(args.to_block, "to-block")
        pts = [a, b]
        label = f"block {a} -> {b}"
    elif args.way is not None:
        if geo is None:
            raise SystemExit("--way は緯度経度が要ります（--bbox か --center/--size を指定）")
        found = None
        for src, e, g in iter_ways(geom_jsons):
            if int(e["id"]) == args.way:
                found = (src, e, g)
                break
        if not found:
            raise SystemExit(f"way {args.way} が {geom_jsons} に見つかりません "
                             f"(--list-ways で確認)")
        src, e, g = found
        pts = [geo.to_block(la, lo) for la, lo in g]
        t = e.get("tags", {})
        label = (f"way {args.way} {t.get('name') or ''} hw={t.get('highway')} "
                 f"layer={t.get('layer')} bridge={t.get('bridge')} tunnel={t.get('tunnel')}")
    elif args.p_from and args.p_to:
        if geo is None:
            raise SystemExit("--from/--to は緯度経度が要ります（--bbox か --center/--size を指定）")
        la0, lo0 = parse_pair(args.p_from, "from")
        la1, lo1 = parse_pair(args.p_to, "to")
        pts = [geo.to_block(la0, lo0), geo.to_block(la1, lo1)]
        label = f"({la0:.5f},{lo0:.5f}) -> ({la1:.5f},{lo1:.5f})"
    else:
        raise SystemExit("--from/--to か --from-block/--to-block か --way を指定して下さい")

    mpp_eff = geo.mpp if geo else (mpp or 1.0)
    if args.margin:
        pts = extend_polyline(pts, args.margin / mpp_eff)
    xs, zs, ds = polyline_stations(pts, step=1.0)
    ds_m = ds * mpp_eff
    thick = max(1, args.thick | 1)
    if verbose:
        print(f"  section: {len(xs)} stations, {ds_m[-1]:.0f}m, thick={thick}, "
              f"x[{xs.min():.0f},{xs.max():.0f}] z[{zs.min():.0f},{zs.max():.0f}]")

    if is_nbt:
        colmap, names, size, meta2, _ = build_section_nbt(wpath, xs, zs, thick, verbose)
    else:
        colmap, names, size, meta2, _ = build_section_anvil(wpath, xs, zs, thick, verbose)

    grid, ymin, hit = assemble_grid(colmap, names, xs, zs, thick)

    marks = {m.strip() if ":" in m else "minecraft:" + m.strip()
             for m in args.mark.split(",") if m.strip()}
    title = f"SECTION  {wpath.name}"
    sub = f"{label}   len={ds_m[-1]:.0f}m  {mpp_eff:.3f} m/block  thick={thick}"
    img, st = render(grid, ymin, names, ds_m, title, sub, marks, args.px,
                     cavity_max=args.cavity_max, ylim=(args.ymin, args.ymax), hit=hit)

    out = Path(args.out) if args.out else (REPO_ROOT / "results" / "inspect"
                                           / f"section_{wpath.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    dy0, dH = st["disp"]
    print(f"saved {out}  ({img.size[0]}x{img.size[1]}px, drawn {grid.shape[1]} stations "
          f"x y[{dy0},{dy0+dH-1}])")
    if st.get("ylim_applied") is False:
        print(f"  [warn] --ymin/--ymax ({args.ymin},{args.ymax}) が断面の y 範囲 "
              f"[{ymin},{ymin+grid.shape[0]-1}] と重ならないので無視しました（全高を描画）")

    # ── 統計 ──
    H, W = grid.shape
    cav, wat, sol, mk = st["cavity"], st["water"], st["solid"], st["marked"]
    per_col = cav.sum(axis=0)
    miss = int((~hit).sum())
    ok = st["any_solid"]
    print(f"  y range      : {ymin} .. {ymin+H-1}   stations={W} "
          f"(with blocks: {int(ok.sum())}, no data: {miss})")

    def _adj_maxstep(vals, valid):
        """隣接する **有効 station 同士** の段差だけを取る。

        旧実装は有効値だけを詰めた配列に diff をかけていたため、欠損区間を跨いだ
        『幻の段差』が出て、位置 @i/N の N も詰めた後の数になっていた。
        戻り値: (maxstep, 左側 station index, 欠損でスキップした隣接対の数)
        """
        v = np.asarray(vals, float)
        if len(v) < 2:
            return 0, -1, 0
        pair = valid[:-1] & valid[1:]
        d = np.where(pair, np.abs(np.diff(np.nan_to_num(v, nan=0.0))), -1.0)
        skipped = int((~pair).sum())
        if not pair.any():
            return 0, -1, skipped
        i = int(np.argmax(d))
        return int(d[i]), i, skipped

    if ok.any():
        ty = st["top_y"][ok]
        mstep, mi, skipped = _adj_maxstep(st["top_y"], ok)
        print(f"  top solid y  : min={int(ty.min())} max={int(ty.max())} "
              f"maxstep={mstep} block"
              + (f"  @{mi}->{mi+1}/{W}" if mstep > 0 else "")
              + (f"  (隣接有効 station 間のみ。欠損で {skipped} 対をスキップ)"
                 if skipped else ""))
    else:
        print("  top solid y  : (固体が 1 つもありません)")
    print(f"  CAVITY cells : {int(cav.sum())}  "
          f"(stations with cavity: {int((per_col>0).sum())}/{W}, "
          f"max per station={int(per_col.max()) if W else 0})")
    if cav.any():
        rows = np.nonzero(cav.any(axis=1))[0]
        print(f"    cavity y   : {ymin+int(rows.min())} .. {ymin+int(rows.max())}")
    if st["biggap"].any():
        print(f"  big gap cells: {int(st['biggap'].sum())} "
              f"(地形シェル内部など。--cavity-max で境界を変えられる)")
    if thick > 1:
        # thick>1 は法線方向の柱を OR 合成する＝幅の細い空洞が埋まる。どれだけ失ったか出す。
        g1, _y1, _h1 = assemble_grid(colmap, names, xs, zs, 1)
        c1, s1 = cavity_stats(g1, names, args.cavity_max)
        if c1 > int(cav.sum()):
            print(f"  [warn] --thick {thick} は法線方向 {thick} 柱を OR 合成するため "
                  f"幅<{thick} の空洞が埋まって消えています: "
                  f"thick=1 なら CAVITY {c1} cells / {s1}/{W} stations "
                  f"(現在 {int(cav.sum())} cells / {int((per_col>0).sum())} stations)。"
                  " トンネルの貫通確認は --thick 1 で行うこと")
        else:
            print(f"  [info] thick=1 相当の CAVITY は {c1} cells / {s1}/{W} stations "
                  "(この thick で失われた空洞は無し)")

    def _top_row(mask):
        top = np.full(W, -1, np.int64)
        a = mask.any(axis=0)
        if a.any():
            top[a] = ((H - 1) - np.argmax(mask[::-1, :], axis=0))[a]
        return top, a

    wtop, anyw = _top_row(wat)
    if anyw.any():
        print(f"  water surf y : {int((wtop[anyw]+ymin).min())} .. {int((wtop[anyw]+ymin).max())} "
              f"on {int(anyw.sum())}/{W} stations")
        drowned = int((sol & (np.arange(H)[:, None] < wtop[None, :]) & anyw[None, :]).sum())
        print(f"    solid under water surface: {drowned} cells")
    if mk.any():
        mtop, anym = _top_row(mk)
        cover = cover_above(mtop, anym, sol)          # デッキに**接した**固体の連続段数
        clear = np.where(anym,                        # 最上位固体までの距離（非接触を含む）
                         np.nan_to_num(st["top_y"], nan=float(ymin)).astype(np.int64)
                         - ymin - mtop, 0)
        mstep2, mi2, skip2 = _adj_maxstep(np.where(anym, mtop, np.nan), anym)
        print(f"  MARKED (--mark) : {int(mk.sum())} cells on {int(anym.sum())}/{W} stations")
        print(f"    marked top y  : {int((mtop[anym]+ymin).min())} .. {int((mtop[anym]+ymin).max())}  "
              f"maxstep={mstep2} block" + (f" @{mi2}->{mi2+1}/{W}" if mstep2 > 0 else "")
              + (f" (欠損で {skip2} 対をスキップ)" if skip2 else ""))
        print(f"    cover above   : min={int(cover[anym].min())} "
              f"median={int(np.median(cover[anym]))} max={int(cover[anym].max())} block  "
              f"(marked の直上に**接している**固体の連続段数。橋デッキなら 0-2)")
        buried = anym & (cover > 2)
        print(f"    BURIED?(接している固体 >2 blocks): {int(buried.sum())}/{int(anym.sum())} stations")
        if int(clear[anym].max()) > int(cover[anym].max()):
            print(f"    (参考) 最上位固体まで max={int(clear[anym].max())} block — "
                  "樹冠(--trees)や送電線など **接していない** 上空の固体を含む。"
                  "埋没判定には使わない")
        if anyw.any():
            sub = anym & anyw & (wtop > mtop)
            print(f"    SUBMERGED (water above marked top) : {int(sub.sum())}/{int(anym.sum())} stations")

    used = np.unique(grid).tolist()
    unk = sorted({names[v] for v in used if 0 <= v < len(names)
                  and names[v] not in MAT_COLOR and not _is_air(names[v])})
    if unk:
        print(f"  [warn] block_palette に無い {len(unk)} 種を UNKNOWN(マゼンタ)で描きました: "
              + ", ".join(n.replace("minecraft:", "") for n in unk[:8])
              + (" ..." if len(unk) > 8 else ""))
    if miss:
        print(f"  [warn] {miss}/{W} stations had no chunk/column data (範囲外か未生成 chunk)。"
              "PNG では **赤の斜線ハッチ**（空の白ではない）。"
              "その区間は『貫通していない』ではなく『見えていない』")


if __name__ == "__main__":
    main()
