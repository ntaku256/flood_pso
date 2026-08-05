#!/usr/bin/env python3
"""make_crop_dem.py — 既存の結合 mosaic DEM npz から小さな bbox を切り出して
`<name>_grd.grid<res>m_pp.npz` として保存する（＝ローダのキャッシュに化けさせる）。

src/wakayama_pcd.load_wakayama_dem は
    data_cache/wakayama_lidar/<stem>.grid{res:g}m_pp.npz
というキャッシュがあれば **txt を読まずに即返す**。そこで結合 mosaic を bbox で切って
同じ命名で置いておくと、

    --wakayama-grd data_cache/wakayama_lidar/<name>_grd.txt

と渡すだけで「その範囲だけの小さな DEM」で生成が回る。全域 DEM のロードと洪水 sim を
避けられるので、橋・トンネルのレンダリングを直す反復が数十秒で回る。
（リポジトリ root の未追跡 scratch_make_crop.py を tools/ の正式版にしたもの。）

使い方:
  .venv/bin/python tools/make_crop_dem.py secbridge 33.8320 33.8390 135.1820 135.1910
  .venv/bin/python tools/make_crop_dem.py secbridge 33.832 33.839 135.182 135.191 --scale 1.5
  .venv/bin/python tools/make_crop_dem.py --list          # 使える mosaic npz を一覧
  .venv/bin/python tools/make_crop_dem.py --clean secbridge   # 作った切り出しを消す

注意: 生成した npz は data_cache/wakayama_lidar/ に置かれる（数 MB）。使い終わったら
      --clean で消すこと。

既存データの保護（実データの DEM キャッシュは 1 図郭 860MB の生 txt からしか作り直せない）:
  * **出力先が既に存在したら無条件で中止する**。図郭別キャッシュ名
    （例 `06RC904`）を誤って渡しても実データは壊れない。本当に置き換えるときだけ
    `--force` を付ける。ただし読み込み元 mosaic 自身への上書きは `--force` でも許可しない。
  * `--clean` も同様で、**本ツールが作った切り出し以外は削除しない**（本ツール製の npz は
    `crop_row0` 等の目印キーを持つ）。目印の無い npz を消すには `--force` が要る。

ジオリファレンス:
  * 元 npz の `res_lat`/`res_lon` を **そのまま引き継ぐ**（切り出しは元グリッドの部分格子）。
    元 npz に無い場合だけ `(lat_max-lat_min)/(H-1)` で導出する。キャッシュは
    `H = round(span/res_lat)+1`（src/wakayama_pcd.py）で作られており、除数は H ではなく H-1。
    `span/H` で再計算すると 0.02% 低い解像度になり、切り出し世界が全域世界に対して
    約 0.4 ブロックずれる（4図郭 mosaic での実測）。
  * 宣言する lat_min/lon_max は **最終行・最終列の画素の緯度経度**（ローダの
    `H = round(span/res)+1` 規約と一致させるため。span/res+1 == 実 shape になる）。

既知の差（揃えられないもの）:
  * make_nbt_hd は経度方向のメートル換算 `lon_per_m` を **全域 DEM の中心緯度**の cos で
    作る。切り出し DEM を渡すとその中心緯度（＝タイル中心緯度）の cos になるため、
    全域から `--tiles` で切ったタイルと東西の幅がわずかに食い違う。実測で 12 タイル列の
    lon_min ばらつきが 0.516m ≒ 0.77 ブロック（南北方向は 0.000m で厳密）。
    単体タイルなら ±0.4 ブロック程度。**目視反復には十分だが、座標を突き合わせる検証
    （全域生成との bit 比較など）には使わないこと**。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LID = REPO_ROOT / "data_cache" / "wakayama_lidar"

# 本ツールが作った切り出しであることの目印（npz のキー）。実データのキャッシュには無い。
CROP_MARK = "crop_row0"


def list_mosaics(res_tag: str | None = None):
    """(path, (lat_min, lat_max, lon_min, lon_max), shape, is_crop) のリストを返す。"""
    pats = sorted(LID.glob("*grid*m_pp.npz"))
    out = []
    for p in pats:
        if res_tag and f".grid{res_tag}m_pp" not in p.name:
            continue
        try:
            z = np.load(p, mmap_mode="r")
            out.append((p, tuple(float(z[k]) for k in
                                 ("lat_min", "lat_max", "lon_min", "lon_max")),
                        tuple(z["dem"].shape), CROP_MARK in z.files))
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
    return out


def is_crop_npz(path: Path) -> bool:
    """本ツールが作った切り出しなら True（読めない/目印が無いものは False = 実データ扱い）。"""
    try:
        with np.load(path, mmap_mode="r") as z:
            return CROP_MARK in z.files
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="出力名（<name>_grd.grid<res>m_pp.npz になる）")
    ap.add_argument("bbox", nargs="*", type=float,
                    help="lat_min lat_max lon_min lon_max")
    ap.add_argument("--scale", type=float, default=1.5,
                    help="生成時に使う --scale。res = 1/scale（既定 1.5 → 0.666667m）")
    ap.add_argument("--src", default=None, help="元 mosaic npz（既定: 最も広いものを自動選択）")
    ap.add_argument("--list", action="store_true", help="使える mosaic npz を一覧して終了")
    ap.add_argument("--clean", action="store_true", help="<name> の切り出しを削除して終了")
    ap.add_argument("--force", action="store_true",
                    help="既存ファイルへの上書き / 本ツール製でない npz の削除を明示的に許可"
                         "（読み込み元 mosaic への上書きだけは常に拒否）")
    args = ap.parse_args()

    res_m = 1.0 / float(args.scale)
    res_tag = f"{res_m:g}"

    if args.list:
        for p, bb, sh, is_crop in list_mosaics():
            print(f"{p.name}{'  [crop]' if is_crop else ''}\n"
                  f"    lat[{bb[0]:.6f},{bb[1]:.6f}] lon[{bb[2]:.6f},{bb[3]:.6f}] "
                  f"shape={sh}")
        return

    if not args.name:
        ap.error("name が必要です（--list / --clean 以外）")
    dst = LID / f"{args.name}_grd.grid{res_tag}m_pp.npz"

    if args.clean:
        if not dst.exists():
            print(f"not found: {dst}")
            return
        if not is_crop_npz(dst) and not args.force:
            sys.exit(f"本ツールが作った切り出しではありません（削除拒否）: {dst}\n"
                     f"  実データのキャッシュを消すと生 txt からの再構築が要ります。\n"
                     f"  本当に消すなら --force を付けてください。")
        if not is_crop_npz(dst):
            print(f"  [warn] --force: 本ツール製でない npz を削除します: {dst.name}")
        dst.unlink()
        print(f"removed {dst}")
        return

    if len(args.bbox) != 4:
        ap.error("bbox は lat_min lat_max lon_min lon_max の4値")
    la0, la1, lo0, lo1 = args.bbox

    # ── 上書き防止: 出力先が既にあるなら --force が無い限り何もせず中止する。
    #    （図郭別キャッシュ名を誤って渡したときに実データを黙って壊さないため）
    if dst.exists() and not args.force:
        kind = "本ツール製の切り出し" if is_crop_npz(dst) else "**実データの可能性がある npz**"
        sys.exit(f"出力先が既に存在します（上書き防止で中止）: {dst}\n"
                 f"  種別: {kind}  size={dst.stat().st_size/1e6:.1f}MB\n"
                 f"  別の名前を使うか、本当に置き換えるなら --force を付けてください"
                 f"（切り出しなら --clean {args.name} で消してから作り直すのが安全）。")

    cands = list_mosaics(res_tag)
    if args.src:
        src = Path(args.src)
        if not src.exists():
            sys.exit(f"--src が見つかりません: {src}")
    else:
        # 図郭を最も多く結合した(=名前に '+' が多い)ものを広域 mosaic とみなす。
        # 本ツール製の切り出しは元データではないので候補から除く。
        pool = [t for t in cands if not t[3]]
        if not pool:
            sys.exit(f"res={res_tag}m の mosaic npz が {LID} にありません")
        src = max(pool, key=lambda t: (t[0].name.count("+"), t[2][0] * t[2][1]))[0]
    if dst.resolve() == src.resolve():
        sys.exit(f"出力名が元 mosaic と同じです（--force でも上書き禁止）: {dst}")
    if dst.exists():   # ここに来るのは --force のときだけ
        print(f"  [warn] --force: 既存ファイルを上書きします: {dst.name} "
              f"({'crop' if is_crop_npz(dst) else '実データの可能性あり'})")

    z = np.load(src)
    dem = z["dem"]
    LA0, LA1 = float(z["lat_min"]), float(z["lat_max"])
    LO0, LO1 = float(z["lon_min"]), float(z["lon_max"])
    H, W = dem.shape
    # 解像度は元 npz の値をそのまま使う（切り出しは元グリッドの部分格子でなければならない）。
    # 無い古い npz だけ span/(N-1) で導出する（N = round(span/res)+1 の逆算）。
    if "res_lat" in z.files and "res_lon" in z.files:
        rlat, rlon = float(z["res_lat"]), float(z["res_lon"])
        res_from = "npz"
    else:
        rlat = (LA1 - LA0) / max(H - 1, 1)
        rlon = (LO1 - LO0) / max(W - 1, 1)
        res_from = "derived(span/(N-1))"
    # 元 npz の res と shape が食い違っていたら（別規約で作られた npz）気付けるように警告
    h_chk = int(round((LA1 - LA0) / rlat)) + 1
    w_chk = int(round((LO1 - LO0) / rlon)) + 1
    if abs(h_chk - H) > 1 or abs(w_chk - W) > 1:
        print(f"  [warn] 元 npz の res と shape が不整合: res から算出 {h_chk}x{w_chk} "
              f"vs 実 shape {H}x{W}（切り出し位置がずれる可能性）")

    if not (LA0 <= la0 < la1 <= LA1 and LO0 <= lo0 < lo1 <= LO1):
        print(f"  [warn] 要求 bbox が mosaic 範囲 lat[{LA0:.6f},{LA1:.6f}] "
              f"lon[{LO0:.6f},{LO1:.6f}] をはみ出しています（クリップします）")
    # row0 = lat_max（北）。要求 bbox を必ず覆うように外側へ丸める。
    r0 = max(0, math.floor((LA1 - la1) / rlat))
    r1 = min(H, math.ceil((LA1 - la0) / rlat) + 1)
    c0 = max(0, math.floor((lo0 - LO0) / rlon))
    c1 = min(W, math.ceil((lo1 - LO0) / rlon) + 1)
    if r1 <= r0 or c1 <= c0:
        sys.exit("切り出し範囲が空です")
    crop = dem[r0:r1, c0:c1].copy()
    # lat_min/lon_max は「最終行・最終列の画素」の座標。ローダの H=round(span/res)+1 規約と
    # 一致させる（r1/c1 は排他端なので -1 する。旧版は 1 セル分広く宣言していた）。
    out_lat_max = LA1 - r0 * rlat
    out_lat_min = LA1 - (r1 - 1) * rlat
    out_lon_min = LO0 + c0 * rlon
    out_lon_max = LO0 + (c1 - 1) * rlon
    np.savez_compressed(dst, dem=crop, lat_min=out_lat_min, lat_max=out_lat_max,
                        lon_min=out_lon_min, lon_max=out_lon_max,
                        res_lat=rlat, res_lon=rlon,
                        # 本ツール製である目印 兼 元グリッド上の位置（数値のみ。ローダは
                        # 0次元配列を float() するので文字列キーは入れられない）
                        crop_row0=float(r0), crop_col0=float(c0),
                        crop_src_rows=float(H), crop_src_cols=float(W))
    fin = int(np.isfinite(crop).sum())
    print(f"src {src.name}  res={res_from} "
          f"res_lat={rlat:.9g} res_lon={rlon:.9g}")
    print(f"saved {dst}")
    print(f"  shape={crop.shape}  lat[{out_lat_min:.6f},{out_lat_max:.6f}] "
          f"lon[{out_lon_min:.6f},{out_lon_max:.6f}]  "
          f"valid={fin}/{crop.size} ({100*fin/crop.size:.1f}%)  "
          f"{dst.stat().st_size/1e6:.1f}MB")
    print(f"  元グリッド上の位置: rows[{r0}:{r1}] cols[{c0}:{c1}] of {H}x{W}"
          f"（部分格子なので全域 DEM と画素が一致）")
    print(f"  → 生成時は --wakayama-grd {LID}/{args.name}_grd.txt --scale {args.scale:g} "
          f"（txt は存在しなくてよい。上の npz がキャッシュとして拾われる）")


if __name__ == "__main__":
    main()
