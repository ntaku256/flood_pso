"""
anvil_export.py  ── 施策⑤: native Anvil (.mca) 直書き

施策③で得た密ブロック配列（uint16 グローバルパレット index, 軸順 (Y, Z, X)）を、
Minecraft 1.18+ の Anvil world（region/r.X.Z.mca + level.dat）へ直接書き出す。
これにより Tellus mod 非依存で「歩ける御坊ワールド」を生成できる。

【フォーマット】(src/anvil_loader._decode_section の逆変換に厳密一致)
  chunk root (1.18+, Level ラッパ無し):
    DataVersion / xPos,yPos,zPos / Status="minecraft:full" / sections[] / block_entities[]
  section:
    Y(byte) / block_states{ palette[{Name,Properties?}], data(LongArray) } / biomes{palette,data?}
  block_states.data: bits=max(4,(n_pal-1).bit_length()), 各 long に floor(64/bits) 個を
    LSB 詰めで非 straddling、index = y*256 + z*16 + x（YZX）。palette が1つなら data 省略。

region コンテナは NBT(twoolie) パッケージ `nbt.region.RegionFile.write_chunk` に委譲。
"""
from __future__ import annotations

from pathlib import Path
import struct
import numpy as np
from nbt import nbt as N
from nbt import region as R

from block_palette import PALETTE_KEYS, minecraft_name, block_state_properties
from anvil_loader import _decode_section   # merge 読み戻し（read 側と同一デコーダで対称性保証）

# 施策③/nbt_export と同じ DataVersion（1.21.x。古い MC で開くなら下げる）
DATA_VERSION = 4671
DEFAULT_BIOME = "minecraft:plains"
SECTION = 16

# グローバルパレット index → (Minecraft 名, Properties dict|None) を事前計算
_NAMES = [minecraft_name(k) for k in PALETTE_KEYS]
_PROPS = [block_state_properties(_NAMES[i]) for i in range(len(PALETTE_KEYS))]
_AIR_IDX = PALETTE_KEYS.index("air") if "air" in PALETTE_KEYS else 0
assert _NAMES[_AIR_IDX] in ("minecraft:air", "air"), f"air index想定外: {_NAMES[_AIR_IDX]}"
# Minecraft 名 → グローバル index（merge 時の読み戻しデコード用。最初の出現を採用）
_NAME_TO_IDX: dict[str, int] = {}
for _i, _nm in enumerate(_NAMES):
    _NAME_TO_IDX.setdefault(_nm, _i)


# ─────────────────────────────────────────────────────────────
# NBT 構築ヘルパ（twoolie nbt.nbt）
# ─────────────────────────────────────────────────────────────

def _named(tag, name):
    tag.name = name
    return tag


def _block_compound(global_idx: int):
    """グローバルパレット index → block_states.palette の 1 要素 {Name, Properties?}。"""
    c = N.TAG_Compound()
    c.tags.append(_named(N.TAG_String(_NAMES[global_idx]), "Name"))
    props = _PROPS[global_idx]
    if props:
        pc = N.TAG_Compound()
        pc.name = "Properties"
        for k, v in props.items():
            pc.tags.append(_named(N.TAG_String(str(v)), str(k)))
        c.tags.append(pc)
    return c


def _pack_indices(local: np.ndarray, bpb: int) -> np.ndarray:
    """local(len 4096, 各 0..n_pal-1) を Anvil 非 straddling LongArray(int64) へ。
    各 long に ipl=64//bpb 個を LSB 詰め。_decode_section の (u>>(k*bpb))&mask の逆。"""
    ipl = 64 // bpb
    n_longs = (4096 + ipl - 1) // ipl
    pad = np.zeros(n_longs * ipl, dtype=np.uint64)
    pad[:4096] = local.astype(np.uint64)
    pad = pad.reshape(n_longs, ipl)
    longs = np.zeros(n_longs, dtype=np.uint64)
    b = np.uint64(bpb)
    for k in range(ipl):
        longs |= pad[:, k] << (np.uint64(k) * b)
    return longs.view(np.int64)


def _section_compound(sub: np.ndarray, sy: int):
    """16×16×16 の (Y,Z,X) グローバル index 配列 → section Compound。
    戻り値 (compound, all_air:bool)。all_air なら呼び出し側で省略してよい。"""
    flat = np.ascontiguousarray(sub).reshape(-1)          # 順序 y*256+z*16+x
    uniq = np.unique(flat)
    n = int(uniq.size)
    sec = N.TAG_Compound()
    sec.tags.append(_named(N.TAG_Byte(sy), "Y"))
    bs = N.TAG_Compound(); bs.name = "block_states"
    pal = N.TAG_List(type=N.TAG_Compound); pal.name = "palette"
    for gi in uniq.tolist():
        pal.tags.append(_block_compound(int(gi)))
    bs.tags.append(pal)
    if n > 1:
        local = np.searchsorted(uniq, flat)               # uniq 昇順=ローカル index
        bpb = max(4, (n - 1).bit_length())
        longs = _pack_indices(local, bpb)
        la = N.TAG_Long_Array(); la.name = "data"
        la.value = longs.tolist()
        bs.tags.append(la)
    sec.tags.append(bs)
    # biomes（1.18+ 必須）: 単一 plains
    bio = N.TAG_Compound(); bio.name = "biomes"
    bpal = N.TAG_List(type=N.TAG_String); bpal.name = "palette"
    bpal.tags.append(N.TAG_String(DEFAULT_BIOME))
    bio.tags.append(bpal)
    sec.tags.append(bio)
    all_air = (n == 1 and int(uniq[0]) == _AIR_IDX)
    return sec, all_air


def _gather_chunk_slab(dense, x_off, z_off, gcx, gcz, n_sec_y):
    """グローバル chunk(gcx,gcz) を覆う (n_sec_y*16, 16, 16) のグローバル index slab を、
    dense(Y,Z,X, ローカル原点が world(x_off,z_off)) から収集（範囲外は air）。"""
    ny, nz, nx = dense.shape
    slab = np.full((n_sec_y * SECTION, SECTION, SECTION), _AIR_IDX, dtype=dense.dtype)
    # この chunk の world ブロック範囲 [gx0,gx0+16) を dense ローカルへ写す
    gx0, gz0 = gcx * SECTION, gcz * SECTION
    lx0, lz0 = gx0 - x_off, gz0 - z_off          # dense 内のローカル開始
    # dense と chunk の重なり（ローカル座標で交差）
    sx0 = max(0, lx0); sx1 = min(nx, lx0 + SECTION)
    sz0 = max(0, lz0); sz1 = min(nz, lz0 + SECTION)
    if sx1 <= sx0 or sz1 <= sz0:
        return slab, True                         # この chunk は dense と重ならない
    dx0, dz0 = sx0 - lx0, sz0 - lz0               # slab 内の書込開始
    yh = min(n_sec_y * SECTION, ny)
    slab[:yh, dz0:dz0 + (sz1 - sz0), dx0:dx0 + (sx1 - sx0)] = dense[:yh, sz0:sz1, sx0:sx1]
    return slab, False


def _read_chunk_slab(ch, n_sec_y):
    """既存 chunk root → (n_sec_y*16,16,16) グローバル index slab（merge 用読み戻し）。"""
    slab = np.full((n_sec_y * SECTION, SECTION, SECTION), _AIR_IDX, dtype=np.uint16)
    secs = ch.get("sections")
    if secs is None:
        return slab
    for s in secs:
        bs = s.get("block_states")
        if bs is None:
            continue
        pal = bs.get("palette")
        if pal is None or len(pal) == 0:
            continue
        pal_names = [p["Name"].value for p in pal]
        data = bs.get("data")
        local = _decode_section(pal_names, list(data) if data is not None else None)  # (16,16,16)
        gmap = np.array([_NAME_TO_IDX.get(nm, _AIR_IDX) for nm in pal_names], dtype=np.uint16)
        sy = int(s["Y"].value)
        if 0 <= sy < n_sec_y:
            slab[sy * SECTION:(sy + 1) * SECTION] = gmap[local]
    return slab


def _chunk_nbt_from_slab(slab: np.ndarray, gcx: int, gcz: int, data_version: int) -> N.NBTFile:
    """(n_sec_y*16,16,16) グローバル index slab → chunk root NBT。"""
    root = N.NBTFile()
    root.name = ""
    root.tags.append(_named(N.TAG_Int(data_version), "DataVersion"))
    root.tags.append(_named(N.TAG_Int(gcx), "xPos"))
    root.tags.append(_named(N.TAG_Int(0), "yPos"))      # 最下 section Y index = 0（world Y0 始まり）
    root.tags.append(_named(N.TAG_Int(gcz), "zPos"))
    root.tags.append(_named(N.TAG_String("minecraft:full"), "Status"))
    n_sec_y = slab.shape[0] // SECTION
    sections = N.TAG_List(type=N.TAG_Compound); sections.name = "sections"
    for sy in range(n_sec_y):
        sub = slab[sy * SECTION:(sy + 1) * SECTION]
        sec, all_air = _section_compound(sub, sy)
        if all_air:
            continue                                     # 全 air section は省略（MC が air 補完）
        sections.tags.append(sec)
    root.tags.append(sections)
    root.tags.append(_named(N.TAG_List(type=N.TAG_Compound), "block_entities"))
    return root


# ─────────────────────────────────────────────────────────────
# region ファイル（空ヘッダ作成 → write_chunk）
# ─────────────────────────────────────────────────────────────

def _open_region(path: Path) -> R.RegionFile:
    """region ファイルを開く。無ければ 8KB ゼロヘッダ（locations+timestamps）で作成。"""
    if not path.exists():
        with open(path, "wb") as f:
            f.write(b"\x00" * 8192)
    return R.RegionFile(str(path))


# ─────────────────────────────────────────────────────────────
# level.dat（最小・void フラットで MC が地形を上書きしないように）
# ─────────────────────────────────────────────────────────────

def _clone_level_from_template(template: Path, out_path: Path, level_name: str, spawn_xyz):
    """既存の正規ワールドの level.dat を雛形にして LevelName/spawn を差し替え、Player を除去。
    実機が作った level.dat（Version/DataPacks/game_rules 等が全て揃う）を流用するので、
    『サポートされていないバージョン』警告が出ない。雛形の generator(地表生成)は維持。"""
    lv = N.NBTFile(str(template))
    D = lv["Data"]
    # LevelName
    if "LevelName" in D:
        D["LevelName"].value = level_name
    else:
        D.tags.append(_named(N.TAG_String(level_name), "LevelName"))
    sx, sy, sz = (int(spawn_xyz[0]), int(spawn_xyz[1]), int(spawn_xyz[2]))
    # 1.21 系: Data.spawn{pos:[x,y,z], dimension,...}
    if "spawn" in D:
        pos = D["spawn"]["pos"]
        try:
            if hasattr(pos, "tags") and len(pos.tags) >= 3:        # TAG_List
                pos[0].value, pos[1].value, pos[2].value = sx, sy, sz
            else:                                                  # TAG_Int_Array 等
                pos.value = [sx, sy, sz]
        except Exception:
            pass
    # 旧式 SpawnX/Y/Z も併記（読む実装向け）
    for nm, v in (("SpawnX", sx), ("SpawnY", sy), ("SpawnZ", sz)):
        if nm in D:
            D[nm].value = v
        else:
            D.tags.append(_named(N.TAG_Int(v), nm))
    # Player を除去（実機が新規プレイヤーを world spawn に作る＝生成地形にスポーン）
    D.tags = [t for t in D.tags if t.name != "Player"]
    lv.write_file(str(out_path))


def _write_level_dat(path: Path, level_name: str, spawn_xyz, data_version: int):
    root = N.NBTFile()
    root.name = ""
    data = N.TAG_Compound(); data.name = "Data"
    def D(tag, name): data.tags.append(_named(tag, name))
    D(N.TAG_Int(data_version), "DataVersion")
    ver = N.TAG_Compound(); ver.name = "Version"
    ver.tags.append(_named(N.TAG_Int(data_version), "Id"))
    ver.tags.append(_named(N.TAG_String("1.21"), "Name"))
    ver.tags.append(_named(N.TAG_Byte(0), "Snapshot"))
    data.tags.append(ver)
    D(N.TAG_String(level_name), "LevelName")
    D(N.TAG_Int(1), "GameType")            # creative
    D(N.TAG_Byte(1), "allowCommands")
    D(N.TAG_Byte(0), "Difficulty")
    D(N.TAG_Byte(1), "initialized")
    D(N.TAG_Long(0), "Time"); D(N.TAG_Long(6000), "DayTime")
    D(N.TAG_Int(int(spawn_xyz[0])), "SpawnX")
    D(N.TAG_Int(int(spawn_xyz[1])), "SpawnY")
    D(N.TAG_Int(int(spawn_xyz[2])), "SpawnZ")
    D(N.TAG_Int(19133), "version")
    # WorldGenSettings: void フラット（layers 空）→ 未生成 chunk は何も無し
    wgs = N.TAG_Compound(); wgs.name = "WorldGenSettings"
    wgs.tags.append(_named(N.TAG_Long(0), "seed"))
    wgs.tags.append(_named(N.TAG_Byte(0), "generate_features"))
    dims = N.TAG_Compound(); dims.name = "dimensions"
    ov = N.TAG_Compound(); ov.name = "minecraft:overworld"
    ov.tags.append(_named(N.TAG_String("minecraft:overworld"), "type"))
    gen = N.TAG_Compound(); gen.name = "generator"
    gen.tags.append(_named(N.TAG_String("minecraft:flat"), "type"))
    settings = N.TAG_Compound(); settings.name = "settings"
    settings.tags.append(_named(N.TAG_List(type=N.TAG_Compound), "layers"))  # 空=void
    settings.tags.append(_named(N.TAG_String("minecraft:plains"), "biome"))
    gen.tags.append(settings)
    ov.tags.append(gen)
    dims.tags.append(ov)
    wgs.tags.append(dims)
    data.tags.append(wgs)
    root.tags.append(data)
    root.write_file(str(path))   # gzip


# ─────────────────────────────────────────────────────────────
# メイン: 密配列 → Anvil world ディレクトリ
# ─────────────────────────────────────────────────────────────

def write_anvil_world(dense: np.ndarray, size, out_dir: str | Path, *,
                      x_offset: int = 0, z_offset: int = 0,
                      merge: bool = False,
                      level_name: str = "flood_pso",
                      data_version: int = DATA_VERSION,
                      write_level: bool = True,
                      level_template: str | None = None,
                      verbose: bool = True) -> dict:
    """密ブロック配列を Anvil world として out_dir/ に書き出す。

    dense    : (Y, Z, X) uint16 グローバルパレット index（施策③ の出力そのまま）
    size     : [nx, ny, nz]（参考。dense.shape から導出）
    x_offset / z_offset : この dense の world ブロック原点（タイルを1ワールドへ配置するため）。
    merge    : True なら既存 chunk を読み戻し、新 dense の非 air だけを上書き（タイル境界を
               跨ぐ chunk で前タイルの内容を残す。施策④ halo でコアは密着するので overlay で十分）。
    返り値: {"world_dir", "n_chunks", "n_regions", "size"}
    """
    dense = np.ascontiguousarray(dense)
    ny, nz, nx = dense.shape
    n_sec_y = (ny + SECTION - 1) // SECTION

    out_dir = Path(out_dir)
    region_dir = out_dir / "region"          # vanilla overworld レイアウト
    region_dir.mkdir(parents=True, exist_ok=True)

    # この dense が触れるグローバル chunk 範囲
    gcx0, gcx1 = x_offset >> 4, (x_offset + nx - 1) >> 4
    gcz0, gcz1 = z_offset >> 4, (z_offset + nz - 1) >> 4
    regions: dict[tuple[int, int], R.RegionFile] = {}

    def _region(rx, rz):
        rf = regions.get((rx, rz))
        if rf is None:
            rf = _open_region(region_dir / f"r.{rx}.{rz}.mca")
            regions[(rx, rz)] = rf
        return rf

    n_chunks = 0
    for gcz in range(gcz0, gcz1 + 1):
        for gcx in range(gcx0, gcx1 + 1):
            slab, empty = _gather_chunk_slab(dense, x_offset, z_offset, gcx, gcz, n_sec_y)
            if empty:
                continue
            rx, rz = gcx >> 5, gcz >> 5
            rf = _region(rx, rz)
            if merge:
                try:
                    ex = rf.get_chunk(gcx & 31, gcz & 31)
                except Exception:
                    ex = None
                if ex is not None:
                    base = _read_chunk_slab(ex, n_sec_y)
                    fill = slab != _AIR_IDX            # 新 dense の非 air だけ上書き
                    base[fill] = slab[fill]
                    slab = base
            root = _chunk_nbt_from_slab(slab, gcx, gcz, data_version)
            if len(root["sections"]) == 0:
                continue                      # 全 air chunk は書かない
            rf.write_chunk(gcx & 31, gcz & 31, root)
            n_chunks += 1
        if verbose and ((gcz - gcz0) % 16 == 0 or gcz == gcz1):
            print(f"  [anvil] chunk row {gcz-gcz0+1}/{gcz1-gcz0+1}  ({n_chunks} chunks, "
                  f"{len(regions)} regions)")

    for rf in regions.values():               # flush（次タイルの merge 読み戻しのため確実に閉じる）
        try:
            rf.close()
        except Exception:
            pass

    if write_level:
        spawn = (x_offset + nx // 2, int(ny + 8), z_offset + nz // 2)
        if level_template and Path(level_template).exists():
            _clone_level_from_template(Path(level_template), out_dir / "level.dat",
                                       level_name, spawn)
            if verbose:
                print(f"  [anvil] level.dat は雛形 {Path(level_template).name} から複製"
                      f"（バージョン警告回避, spawn={spawn}）")
        else:
            _write_level_dat(out_dir / "level.dat", level_name, spawn, data_version)

    if verbose:
        print(f"[anvil] world '{out_dir.name}': +{n_chunks} chunks / {len(regions)} regions  "
              f"size(blocks) {nx}×{ny}×{nz} @off({x_offset},{z_offset}) merge={merge}")
    return {"world_dir": str(out_dir), "n_chunks": n_chunks,
            "n_regions": len(regions), "size": [nx, ny, nz]}
