#!/usr/bin/env python3
"""和歌山県 3次元点群データ 自動ダウンローダ

data_cache/wakayama_lidar/{code}_grd.txt / {code}_org.txt を図郭コード指定で取得する。
従来は GeoCloud の Web GIS を手動操作するしかなく、消失した図郭を取り直せなかった。

--------------------------------------------------------------------------
配信機構（2026-08-05 実測で解明。実データ取得まで検証済み）
--------------------------------------------------------------------------
1. https://wakayamaken.geocloud.jp/webgis/Item?...&itemId=N
     -> 図郭コード + Box 共有URL（オリジナル=org / グラウンド=grd）
     ※ **ブラウザ相当の User-Agent が必須**。無いと WAF が 403 を返す。
        `python-requests/2.31.0` や短い `Mozilla/5.0` でも 403 になるので
        完全な Chrome UA 文字列を使うこと。Cookie / Referer / 規約同意は不要。
     ※ itemId は 1..1798 を総当りすれば全図郭を列挙できる（クリック座標不要）。
        静岡県と違い S3 の ListObjectsV2 相当は無いので、これが正攻法。
2. https://wakayamakendo.box.com/s/<shared_name>
     -> HTML 中の "typedID":"f_<file_id>"（1ページに必ず1個）
3. https://wakayamakendo.app.box.com/index.php
     ?rm=box_download_shared_file&shared_name=<sn>&file_id=f_<id>
     -> 302 -> public.boxcloud.com の署名付きURL -> zip 本体
     ※ Box 側は完全匿名。認証も UA も不要。Range 対応。
     ※ 署名付きURLは期限付きなので**保存・再利用しないこと**（毎回 302 を辿る）。

zip の中身は 1 ファイルのみ（{code}_grd.txt / {code}_org.txt、CRLF 区切り）。
  grd: ID,x,y,z          org: ID,x,y,z,class
座標系は JGD2011 平面直角座標系 第VI系 EPSG:6674。図郭は 2000m(東西) x 1500m(南北)。

--------------------------------------------------------------------------
利用条件（マップ属性 civilTermsOfUse より。2026-08-05 確認）
--------------------------------------------------------------------------
- オリジナルデータ(org) / グラウンドデータ(grd) は **申請不要**。出典の記載のみ求められる
- グリッドデータ(1mメッシュ)だけが測量法第43・44条の申請対象（本スクリプトでは扱わない）
- 出典表記例: 「和歌山県 3次元点群データ（航空レーザ測量）を加工して作成」
- 測量時期は令和元年度（北山村付近のみ平成25年度）
- 担当: 和歌山県 県土整備部 河川･下水道局 砂防課

サーバに負荷をかけないこと。索引は tools/wakayama_mesh_index.json に同梱してあるので、
通常は `index` を実行し直す必要は無い（Box URL が変わったときだけ）。

--------------------------------------------------------------------------
使い方
--------------------------------------------------------------------------
  # 御坊で欠けている図郭を grd/org まとめて取得
  python3 tools/wakayama_lidar_dl.py get --kind both --out data_cache/wakayama_lidar \\
      06RC701 06RC702 06RC711 06RC713 06RC803 06RC811 06RC813 06RC911

  # 緯度経度から図郭コードを引く
  python3 tools/wakayama_lidar_dl.py locate 33.8837 135.1676

  # コード接頭辞で一覧（取得可否つき）
  python3 tools/wakayama_lidar_dl.py list --prefix 06RC8

  # 索引を作り直す（約22秒 / 約2000リクエスト。通常は不要）
  python3 tools/wakayama_lidar_dl.py index
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
ITEM = ("https://wakayamaken.geocloud.jp/webgis/Item"
        "?srs=EPSG:4326&mapId=22-0&layerId=0&itemId={}")
SCAN = ("https://wakayamaken.geocloud.jp/webgis/Scan"
        "?srs=EPSG:4326&x={lon}&y={lat}&mapId=22-0&params=-1&level=14&buffer=0.002")
BOXDL = ("https://wakayamakendo.app.box.com/index.php?rm=box_download_shared_file"
         "&shared_name={sn}&file_id={fid}")

INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "wakayama_mesh_index.json")
MAX_ITEM_ID = 2000  # 実測: 有効 itemId は 1..1798


def _open(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=120)


# ---------- 1. 索引 ----------
def fetch_item(i):
    d = None
    for _ in range(3):
        try:
            d = json.loads(_open(ITEM.format(i)).read().decode())
            break
        except Exception:
            d = None
    if not d or d.get("status") != "OK":
        return None
    m = d["ret"]["main"]
    a = {k: v["value"] for k, v in m["attributes"].items()}
    return {"itemId": i, "code": m["name"], "lon": m["x"], "lat": m["y"],
            "org": a.get("オリジナルデータURL", ""),
            "grd": a.get("グラウンドデータURL", ""),
            "grid_viewer": a.get("３Dビューア起動URL（グリッド データ）", "")}


def build_index(path=INDEX, workers=8):
    print(f"[index] itemId 1..{MAX_ITEM_ID - 1} を走査中（約22秒）...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(fetch_item, range(1, MAX_ITEM_ID)) if r]
    rows.sort(key=lambda r: r["itemId"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    dl = sum(1 for r in rows if r["grd"])
    print(f"[index] {len(rows)} メッシュ -> {path}（うち取得可能 {dl}）")
    return rows


def load_index(path=INDEX):
    if not os.path.exists(path):
        return build_index(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 2. Box 共有URL -> 直URL ----------
def resolve_box(shared_url):
    sn = shared_url.rstrip("/").rsplit("/s/", 1)[1]
    html = _open(shared_url).read().decode("utf-8", "replace")
    m = re.search(r'"typedID"\s*:\s*"(f_\d+)"', html)
    if not m:
        raise RuntimeError(f"Box ページに file_id が見つかりません: {sn}")
    return BOXDL.format(sn=sn, fid=m.group(1))


# ---------- 3. 取得 ----------
def download(code, kind, outdir, extract=True, force=False):
    idx = {r["code"]: r for r in load_index()}
    if code not in idx:
        raise SystemExit(f"{code}: 図郭が索引にありません（list で確認してください）")
    share = idx[code][kind]
    if not share:
        raise SystemExit(f"{code}/{kind}: 県のデータ取得範囲外（URL 未設定）")

    txt = os.path.join(outdir, f"{code}_{kind}.txt")
    if extract and os.path.exists(txt) and not force:
        print(f"  [skip] {code}_{kind}.txt 既存 "
              f"({os.path.getsize(txt):,} B) — 取り直すなら --force")
        return

    url = resolve_box(share)
    os.makedirs(outdir, exist_ok=True)
    zpath = os.path.join(outdir, f"{code}_{kind}.zip")
    with _open(url) as r, open(zpath, "wb") as f:
        total = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            total += len(chunk)
    print(f"  {code}_{kind}.zip  {total:,} B")
    if extract:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(outdir)
            names = z.namelist()
        os.remove(zpath)
        for n in names:
            print(f"  -> {n}  {os.path.getsize(os.path.join(outdir, n)):,} B")


def locate(lat, lon):
    """緯度経度からその地点の図郭を引く（Scan API）。

    レスポンスは ret.polygon[] に {itemId, name, x, y} が入る（ret.items ではない）。
    """
    d = json.loads(_open(SCAN.format(lat=lat, lon=lon)).read().decode())
    hits = d.get("ret", {}).get("polygon", []) if d.get("status") == "OK" else []
    if not hits:
        print(f"  ({lat}, {lon}) に対応する図郭はありません（県の取得範囲外の可能性）")
        return
    idx = {r["itemId"]: r for r in load_index()}
    for h in hits:
        r = idx.get(h.get("itemId"), {})
        ok = "取得可" if r.get("grd") else "範囲外"
        print(f"  {h.get('name')}  itemId={h.get('itemId')}  "
              f"中心({h.get('y'):.5f},{h.get('x'):.5f})  {ok}")


def main():
    p = argparse.ArgumentParser(
        description="和歌山県 3次元点群データ ダウンローダ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="出典表記が必要: 和歌山県 3次元点群データ（航空レーザ測量）を加工して作成")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="図郭索引を作り直す（通常は不要）")

    g = sub.add_parser("get", help="図郭を取得して展開する")
    g.add_argument("codes", nargs="+", help="図郭コード（例 06RC701）")
    g.add_argument("--kind", choices=["grd", "org", "both"], default="grd",
                   help="grd=地表 / org=全点(DSM) / both（既定 grd）")
    g.add_argument("--out", default="data_cache/wakayama_lidar")
    g.add_argument("--keep-zip", action="store_true", help="展開せず zip のまま残す")
    g.add_argument("--force", action="store_true", help="既存 txt があっても取り直す")

    ls = sub.add_parser("list", help="索引を一覧する")
    ls.add_argument("--prefix", default="", help="コード接頭辞で絞る（例 06RC8）")
    ls.add_argument("--available", action="store_true", help="取得可能なものだけ")

    lo = sub.add_parser("locate", help="緯度経度から図郭コードを引く")
    lo.add_argument("lat", type=float)
    lo.add_argument("lon", type=float)

    a = p.parse_args()
    if a.cmd == "index":
        build_index()
    elif a.cmd == "list":
        rows = [r for r in load_index() if r["code"].startswith(a.prefix)]
        if a.available:
            rows = [r for r in rows if r["grd"]]
        for r in sorted(rows, key=lambda r: r["code"]):
            ok = "取得可" if r["grd"] else "範囲外"
            print(f"  {r['code']}  itemId={r['itemId']:4d}  "
                  f"中心({r['lat']:.5f},{r['lon']:.5f})  {ok}")
        print(f"  --- {len(rows)} 件 ---")
    elif a.cmd == "locate":
        locate(a.lat, a.lon)
    else:
        kinds = ["grd", "org"] if a.kind == "both" else [a.kind]
        for c in a.codes:
            for k in kinds:
                download(c, k, a.out, extract=not a.keep_zip, force=a.force)


if __name__ == "__main__":
    main()
