"""
osm_cache.py
Overpass API 取得とディスクキャッシュの共通基盤（tellus_data / bridge_osm / power_osm 共用）。

解決する問題
------------
1. **bbox のわずかなズレでキャッシュミス**
   従来のキャッシュキーは要求 bbox の浮動小数点をそのまま埋め込んでいたため、
   crop を 1 m ずらすだけで別ファイルになり Overpass へ再問い合わせしていた
   （実例: `osm_33.83140_33.83499_...` と `osm_33.83140_33.83500_...` が別物として並存）。
   → bbox を `BBOX_QUANT_DEG`（既定 0.001 度 ≒ 111 m）刻みで **外側へ量子化**して
     キーにする。量子化 bbox は要求 bbox を必ず内包するので、取得結果を
     要求範囲でフィルタして返せば意味的に等価。

   注意（実態）: 効くのは **これから作られるキャッシュ** だけで、既存の旧形式
   `osm_{要求bbox}.json` が消えたり 1 ファイルへまとまったりはしない。旧完全一致キーの
   探索を先に走らせる（後方互換）ので、既存 9 ファイルは今後もそれぞれ個別に配信される。
   ディスク上でのファイル統合は起きず、新規 bbox が `osmq_` に集約されるだけ。

2. **オフライン再現性**
   環境変数 `FLOOD_PSO_OFFLINE=1` でネットワーク取得を完全に禁止する
   （tizucraft の `TIZU_OFFLINE` と同じ思想）。キャッシュが無ければ黙って
   空データを返さず `OfflineError` を送出する。

3. 破損キャッシュ対策として書き込みは tmp → `os.replace` の atomic write。

4. **部分結果（Overpass の runtime error）をキャッシュしない**
   Overpass はクエリがタイムアウトしても HTTP 200 のまま
   `{"remark": "runtime error: Query timed out ...", "elements": [...部分...]}`
   を返すことがある。これを検査せずキャッシュすると「橋 0 本」等の汚染が永続し、
   しかも bbox 量子化により 1 件の汚染が ~111 m セル全域へ波及する。
   → `overpass_post` は `remark` 付きレスポンスを **失敗** として扱い、
     別ミラーへ切り替える。全ミラーで駄目なら `OverpassError`（＝キャッシュ書込なし）。
   → 既に汚染済みのキャッシュも `read_cache` が `remark` を見て破棄・再取得する。
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data_cache"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass は混雑時に 429/504 を返すので複数ミラーを順に叩く。
OVERPASS_MIRRORS = [
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
USER_AGENT = "flood_pso/tellus_data"

#: オフライン強制の環境変数名（tizucraft の TIZU_OFFLINE 相当）
OFFLINE_ENV = "FLOOD_PSO_OFFLINE"

#: キャッシュキー用 bbox 量子化幅 [度]。0.001 度 ≒ 緯度 111 m / 経度 93 m（御坊）。
BBOX_QUANT_DEG = 0.001


class OfflineError(BaseException):
    """FLOOD_PSO_OFFLINE=1 の状態でキャッシュミス＝ネットワークが必要になった。

    **Exception ではなく BaseException を継承している**のは意図的。
    呼び出し側（nbt_export の道路 curb 回廊 / gap_fill の建物高さ補完など）には
    「取得に失敗したら劣化させて続行する」ための広域 `except Exception` が点在していて、
    OfflineError が Exception だと *オフライン指定が無言で握り潰される*。
    それではこの仕組みの目的（欠損を黙って通さない）を果たせないので、
    KeyboardInterrupt / SystemExit と同じく「握り潰さない」側に置く。

    ネットワーク由来の通常の失敗（Overpass 落ち・404 等）は従来どおり
    RuntimeError / URLError（＝ Exception）なので、劣化継続の挙動は変わらない。
    """


class OverpassError(RuntimeError):
    """Overpass の応答が使えない（全ミラー失敗 / `remark` 付きの部分結果 / 壊れた JSON）。

    Exception 派生なので、呼び出し側の「取れなければ劣化継続」は従来どおり働く。
    重要なのは **この例外が出た場合キャッシュへは何も書かれない** こと。
    """


# ─────────────────────────────────────────────────────────────
# オフラインガード
# ─────────────────────────────────────────────────────────────

def is_offline() -> bool:
    """FLOOD_PSO_OFFLINE が真値（空 / 0 / false / no 以外）なら True。"""
    v = os.environ.get(OFFLINE_ENV, "")
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def offline_guard(what: str) -> None:
    """オフライン時にネットワーク取得へ進もうとしたら明示エラー。

    「無言で空データを返す」ことを禁じるのが目的なので、呼び出し側は
    OfflineError を握り潰さないこと（tellus_data のタイルループは再送出する）。
    """
    if is_offline():
        raise OfflineError(
            f"{OFFLINE_ENV}=1 のためネットワーク取得は禁止されています: {what}\n"
            f"  キャッシュ（{DEFAULT_CACHE_DIR}）に必要なファイルがありません。"
            f" オンラインで一度生成するか、{OFFLINE_ENV} を外してください。"
        )


# ─────────────────────────────────────────────────────────────
# bbox 量子化とキャッシュキー
# ─────────────────────────────────────────────────────────────

def require_bbox(lat_min, lat_max, lon_min, lon_max, *, what: str = "bbox") -> bool:
    """bbox の指定状態を判定する。

    Returns
    -------
    True  : 4 要素すべて指定済み（ネットワーク取得へ進んでよい）
    False : 4 要素すべて None（bbox 未指定＝呼び出し側の「取得しない」経路）

    Raises
    ------
    ValueError : 一部だけ指定されている場合。以前はこの状態で `quantize_bbox` の
                 内部まで進み `TypeError: '>' not supported between ... and 'NoneType'`
                 という原因の分からない例外になっていた。
    """
    vals = {"lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max}
    missing = [k for k, v in vals.items() if v is None]
    if not missing:
        return True
    if len(missing) == 4:
        return False
    given = ", ".join(f"{k}={v}" for k, v in vals.items() if v is not None)
    raise ValueError(
        f"{what}: bbox が不完全です（{', '.join(missing)} が None / 指定済み: {given}）。"
        " lat_min / lat_max / lon_min / lon_max の 4 つすべてを指定するか、"
        " 4 つとも省略してください。"
    )


def quantize_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                  step: float = BBOX_QUANT_DEG) -> tuple[float, float, float, float]:
    """bbox を step 刻みの格子へ **外側** に丸める（要求 bbox を必ず内包）。

    近接する crop 同士が同じキーへ落ちるのでキャッシュヒット率が上がる。
    浮動小数の誤差で内包が壊れないよう、下限は floor / 上限は ceil のみ使う。
    None / 数値化できない値が混ざっていたら、どの要素が駄目かを示す ValueError。
    """
    bad = []
    nums = []
    for k, v in (("lat_min", lat_min), ("lat_max", lat_max),
                 ("lon_min", lon_min), ("lon_max", lon_max)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            bad.append(f"{k}={v!r}")
            f = 0.0
        else:
            if math.isnan(f):
                bad.append(f"{k}=nan")
        nums.append(f)
    if bad:
        raise ValueError(
            "quantize_bbox: bbox の要素は数値でなければなりません（不正: "
            + ", ".join(bad) + "）。lat_min / lat_max / lon_min / lon_max を"
            " すべて数値で渡してください。")
    lat_min, lat_max, lon_min, lon_max = nums
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    i0 = math.floor(lat_min / step)
    i1 = math.ceil(lat_max / step)
    j0 = math.floor(lon_min / step)
    j1 = math.ceil(lon_max / step)
    if i1 <= i0:
        i1 = i0 + 1          # 退化 bbox でも 1 セル分の幅を持たせる
    if j1 <= j0:
        j1 = j0 + 1
    return (round(i0 * step, 9), round(i1 * step, 9),
            round(j0 * step, 9), round(j1 * step, 9))


def bbox_key(prefix: str, bbox: tuple[float, float, float, float]) -> str:
    """bbox（lat_min, lat_max, lon_min, lon_max）→ キャッシュファイル名。"""
    return (f"{prefix}_{bbox[0]:.5f}_{bbox[1]:.5f}_"
            f"{bbox[2]:.5f}_{bbox[3]:.5f}.json")


def bbox_intersects(las, los, bbox) -> bool:
    """要素の緯度列・経度列が bbox と重なるか（bridge_osm._bbox_hit と同規約）。"""
    if bbox is None:
        return True
    lat_min, lat_max, lon_min, lon_max = bbox
    return not (max(las) < lat_min or min(las) > lat_max
                or max(los) < lon_min or min(los) > lon_max)


# ─────────────────────────────────────────────────────────────
# 取得（オフラインガード・ミラー・リトライ・atomic 書込）
# ─────────────────────────────────────────────────────────────

def atomic_write_text(path: Path, text: str) -> None:
    """同じキーを同時に書いても壊れた JSON が残らないよう tmp→replace。

    tmp 名は **プロセス内並列でも衝突しない**よう pid + thread id + uuid で作る
    （pid だけだと同一プロセスの複数スレッドが同じ tmp を掴み、先に replace した側の
    あとで `os.replace` が FileNotFoundError を投げる）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}-"
                         f"{threading.get_ident():x}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)      # 中途半端な tmp を残さない
        raise


def _response_remark(data) -> str:
    """Overpass レスポンスの `remark`（＝ runtime error / タイムアウト等）を返す。無ければ ""。

    Overpass はクエリがタイムアウトしても HTTP 200 で
    `{"remark": "runtime error: Query timed out in \"query\" at line ...", "elements": [...]}`
    を返す。elements は空か部分結果なので、そのままキャッシュすると汚染が永続する。
    """
    if not isinstance(data, dict):
        return ""
    rm = data.get("remark")
    return str(rm).strip() if rm else ""


def overpass_post(query: str, *, timeout: float = 180.0, tries: int = 6,
                  verbose: bool = True, what: str = "overpass") -> dict:
    """Overpass へ POST してレスポンス JSON を dict で返す（ミラー巡回 + リトライ）。

    HTTP エラーだけでなく、**HTTP 200 でも `remark` 付き（＝部分結果 / runtime error）**
    なら失敗として次のミラーへ回す。全試行が駄目なら OverpassError を送出するので、
    呼び出し側がキャッシュへ書き込むことは無い。
    """
    offline_guard(f"Overpass {what}")
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_err = None
    for attempt in range(tries):
        url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        host = url.split("//")[1].split("/")[0]
        req = urllib.request.Request(
            url, data=body,
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            data = json.loads(raw.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if verbose:
                print(f"  [osm] {host} failed ({e}); retry...")
        except (ValueError, UnicodeDecodeError) as e:     # JSONDecodeError も ValueError
            last_err = OverpassError(f"invalid JSON from {host}: {e}")
            if verbose:
                print(f"  [osm] {host} returned invalid JSON ({e}); retry...")
        else:
            remark = _response_remark(data)
            if not remark:
                return data
            n_el = len(data.get("elements", []) or [])
            last_err = OverpassError(f"{host} remark: {remark} (elements={n_el})")
            if verbose:
                print(f"  [osm] {host} returned a partial/failed result "
                      f"(elements={n_el}) — remark: {remark[:160]}; "
                      f"キャッシュせず別ミラーへ retry...")
        if attempt < tries - 1:
            time.sleep(2.0 * (attempt + 1))
    raise OverpassError(f"Overpass API failed after {tries} tries ({what}): {last_err}")


def read_cache(cache_path: Path) -> dict | None:
    """キャッシュ JSON を読む。無い / 壊れている場合は None（＝再取得させる）。

    過去に書かれてしまった `remark` 入り（部分結果）キャッシュも None 扱いにして
    捨てる。オフライン時はここが None になることで OfflineError となり、
    汚染データが黙って配信され続けることは無い。
    """
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    remark = _response_remark(data)
    if remark:
        print(f"  [osm] 破棄: {cache_path.name} は Overpass の部分結果でした "
              f"(remark: {remark[:120]}) → 再取得します")
        return None
    return data


# ─────────────────────────────────────────────────────────────
# bbox 量子化キャッシュ付き Overpass geom 取得（橋/トンネル/電力/駐車場）
# ─────────────────────────────────────────────────────────────

#: kind → Overpass のクエリ本体（{S},{W},{N},{E} を埋める）。
#: 出力は `out geom tags;` で、bridge_osm / power_osm / parking_osm が
#: そのままパースできる生 Overpass レスポンス形式を保つ。
GEOM_QUERIES: dict[str, str] = {
    "bridge":  'way["bridge"]["highway"]({S},{W},{N},{E});',
    "tunnel":  'way["tunnel"]["highway"]({S},{W},{N},{E});',
    "power":   ('way["power"~"^(line|minor_line|cable)$"]({S},{W},{N},{E});'
                'node["power"~"^(tower|pole)$"]({S},{W},{N},{E});'),
    "parking": ('way["amenity"="parking"]({S},{W},{N},{E});'
                'relation["amenity"="parking"]({S},{W},{N},{E});'),
    # 用水路・排水路。river も引いておき、採否は load_waterways 側の kinds で決める
    # （FG-GML の面がある範囲では river を落とすため。クエリを絞ると面の無い所を
    #   後から補えなくなり、量子化キャッシュを引き直す羽目になる）。
    "waterway": ('way["waterway"~"^(stream|drain|ditch|canal|river)$"]'
                 '({S},{W},{N},{E});'),
    # 交通信号（highway=traffic_signals ノード）。灯器位置ではなく停止線ノードだが、
    # 1m/block では見た目の差は出ない。信号柱＋3灯で立体化する。
    "signal":  'node["highway"="traffic_signals"]({S},{W},{N},{E});',
}


def fetch_overpass_geom(kind: str,
                        lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float,
                        cache_dir: Path | str = DEFAULT_CACHE_DIR,
                        verbose: bool = True,
                        step: float = BBOX_QUANT_DEG) -> dict:
    """`out geom tags` 形式の生 Overpass レスポンスを量子化 bbox キャッシュ付きで取得。

    Returns: {"elements": [...], ...}（Overpass の生 JSON）。
    要求 bbox でのフィルタは各 loader（bridge_osm 等）の bbox 引数が行う。
    `remark` 付き（部分結果）が返った場合は OverpassError で、キャッシュには書かない。
    """
    if kind not in GEOM_QUERIES:
        raise ValueError(f"unknown overpass geom kind: {kind} (available: {sorted(GEOM_QUERIES)})")
    if not require_bbox(lat_min, lat_max, lon_min, lon_max,
                        what=f"fetch_overpass_geom(kind={kind!r})"):
        raise ValueError(f"fetch_overpass_geom(kind={kind!r}): bbox が未指定です"
                         "（lat_min / lat_max / lon_min / lon_max が必要）。")
    qb = quantize_bbox(lat_min, lat_max, lon_min, lon_max, step=step)
    cache_path = Path(cache_dir) / "osm" / bbox_key(f"osmgeom_{kind}", qb)
    cached = read_cache(cache_path)
    if cached is not None:
        if verbose:
            print(f"[osm:{kind}] cache hit {cache_path.name} "
                  f"({len(cached.get('elements', []))} elements)")
        return cached

    # 「クエリを投げます」と出した直後に OfflineError で落ちて読み手が混乱しないよう、
    # ログより先にオフライン判定する（overpass_post 側でも再度ガードされる）。
    offline_guard(f"Overpass {kind} geom qbbox lat[{qb[0]:.4f},{qb[1]:.4f}] "
                  f"lon[{qb[2]:.4f},{qb[3]:.4f}] (cache: {cache_path})")
    body = GEOM_QUERIES[kind].format(S=f"{qb[0]:.6f}", W=f"{qb[2]:.6f}",
                                     N=f"{qb[1]:.6f}", E=f"{qb[3]:.6f}")
    query = f"[out:json][timeout:120];({body});out geom tags;"
    if verbose:
        print(f"[osm:{kind}] Overpass query qbbox lat[{qb[0]:.4f},{qb[1]:.4f}] "
              f"lon[{qb[2]:.4f},{qb[3]:.4f}]")
    data = overpass_post(query, verbose=verbose, what=f"{kind} geom")
    atomic_write_text(cache_path, json.dumps(data, ensure_ascii=False))
    if verbose:
        print(f"[osm:{kind}] cached {cache_path.name} "
              f"({len(data.get('elements', []))} elements)")
    return data


# ─────────────────────────────────────────────────────────────
# 建物 + 道路（tellus_data.fetch_osm_buildings_roads の実体）
# ─────────────────────────────────────────────────────────────

def clip_buildings_roads(osm: dict,
                         lat_min: float, lat_max: float,
                         lon_min: float, lon_max: float) -> dict:
    """量子化 bbox で取った建物/道路を、要求 bbox に交差するものだけへ絞る。

    Overpass の `way[...](bbox)` は「bbox に掛かる way を全ジオメトリ付きで返す」
    ので、bbox 交差判定で絞れば直接問い合わせた場合と同じ集合（外接矩形判定なので
    厳密には superset）になる。ラスタ化側は grid 外を捨てるため superset で無害。
    """
    bbox = (lat_min, lat_max, lon_min, lon_max)

    def _hit(coords) -> bool:
        if not coords:
            return False
        return bbox_intersects([c[0] for c in coords], [c[1] for c in coords], bbox)

    buildings = [b for b in osm.get("buildings", []) if _hit(b.get("coords"))]
    roads = [r for r in osm.get("roads", []) if _hit(r.get("coords"))]
    return {
        "buildings": buildings, "roads": roads,
        "bbox": [lat_min, lat_max, lon_min, lon_max],
        "n_buildings": len(buildings), "n_roads": len(roads),
    }


def parse_buildings_roads(response: dict, highway_width_m: dict) -> dict:
    """Overpass レスポンス → {"buildings": [...], "roads": [...]}（bbox 情報は付けない）。"""
    buildings: list = []
    roads: list = []
    for el in response.get("elements", []):
        tags = el.get("tags", {}) or {}
        # way: geometry に [{lat, lon}, ...] が入る
        if el.get("type") == "way" and "geometry" in el:
            coords = [[g["lat"], g["lon"]] for g in el["geometry"]]
            if "building" in tags:
                buildings.append({"coords": coords, "tags": tags})
            elif "highway" in tags:
                ht = tags.get("highway", "")
                roads.append({
                    "coords": coords, "tags": tags,
                    "width_m": float(highway_width_m.get(ht, 4)),
                })
        # relation (multipolygon building) は outer ring を抽出
        elif el.get("type") == "relation" and "members" in el and "building" in tags:
            for m in el["members"]:
                if m.get("type") == "way" and m.get("role") == "outer" and "geometry" in m:
                    coords = [[g["lat"], g["lon"]] for g in m["geometry"]]
                    buildings.append({"coords": coords, "tags": tags})
    return {"buildings": buildings, "roads": roads}


def fetch_buildings_roads(lat_min: float, lat_max: float,
                          lon_min: float, lon_max: float,
                          highway_width_m: dict,
                          cache_dir: Path | str = DEFAULT_CACHE_DIR,
                          verbose: bool = True,
                          step: float = BBOX_QUANT_DEG) -> dict:
    """建物 + 道路を量子化 bbox キャッシュ付きで取得し、要求 bbox でフィルタして返す。

    キャッシュ探索順（後方互換）:
      1. 旧形式の完全一致キー `osm_{要求bbox}.json` → **そのまま返す**（従来と同一挙動）
      2. 量子化キー `osmq_{量子化bbox}.json` → 要求 bbox でフィルタして返す
      3. オフラインなら OfflineError、そうでなければ量子化 bbox で Overpass 取得

    1 を先に見るので、既存の `osm_*.json` はディスク上でまとめられることはなく、
    今後もそれぞれが個別に使われ続ける（量子化による集約は新規取得ぶんだけ）。
    """
    if not require_bbox(lat_min, lat_max, lon_min, lon_max,
                        what="fetch_buildings_roads"):
        raise ValueError("fetch_buildings_roads: bbox が未指定です"
                         "（lat_min / lat_max / lon_min / lon_max が必要）。")
    cache_root = Path(cache_dir) / "osm"
    legacy_path = cache_root / bbox_key("osm", (lat_min, lat_max, lon_min, lon_max))
    legacy = read_cache(legacy_path)
    if legacy is not None:
        return legacy

    qb = quantize_bbox(lat_min, lat_max, lon_min, lon_max, step=step)
    cache_path = cache_root / bbox_key("osmq", qb)
    cached = read_cache(cache_path)
    if cached is None:
        query = (
            "[out:json][timeout:60];("
            f'way["building"]({qb[0]:.6f},{qb[2]:.6f},{qb[1]:.6f},{qb[3]:.6f});'
            f'way["highway"]({qb[0]:.6f},{qb[2]:.6f},{qb[1]:.6f},{qb[3]:.6f});'
            f'relation["building"]({qb[0]:.6f},{qb[2]:.6f},{qb[1]:.6f},{qb[3]:.6f});'
            ");out geom;"
        )
        # ログより先にオフライン判定（「query 投げます」の直後に落ちるのを避ける）
        offline_guard(f"Overpass buildings+roads qbbox lat[{qb[0]:.4f},{qb[1]:.4f}] "
                      f"lon[{qb[2]:.4f},{qb[3]:.4f}] (cache: {cache_path})")
        if verbose:
            print(f"[osm] Overpass query qbbox lat[{qb[0]:.4f},{qb[1]:.4f}] "
                  f"lon[{qb[2]:.4f},{qb[3]:.4f}]  "
                  f"(要求 lat[{lat_min:.4f},{lat_max:.4f}] lon[{lon_min:.4f},{lon_max:.4f}])")
        response = overpass_post(query, verbose=verbose, what="buildings+roads")
        parsed = parse_buildings_roads(response, highway_width_m)
        cached = {
            "buildings": parsed["buildings"], "roads": parsed["roads"],
            "bbox": [qb[0], qb[1], qb[2], qb[3]],
            "n_buildings": len(parsed["buildings"]), "n_roads": len(parsed["roads"]),
            "quantized": True, "quant_step_deg": step,
        }
        atomic_write_text(cache_path, json.dumps(cached, ensure_ascii=False))
        if verbose:
            print(f"[osm] cached {cache_path.name}  "
                  f"buildings={cached['n_buildings']}  roads={cached['n_roads']}")

    return clip_buildings_roads(cached, lat_min, lat_max, lon_min, lon_max)


if __name__ == "__main__":
    # 量子化・bbox ガード・remark 判定の自己確認（ネットワーク不要）
    reqs = [(33.83140, 33.83499, 135.17524, 135.17956),
            (33.83140, 33.83500, 135.17524, 135.17955),
            (33.83141, 33.83499, 135.17524, 135.17956)]
    for r in reqs:
        q = quantize_bbox(*r)
        assert q[0] <= r[0] and q[1] >= r[1] and q[2] <= r[2] and q[3] >= r[3], (r, q)
        print(f"{r} -> {bbox_key('osmq', q)}")

    assert require_bbox(33.83, 33.84, 135.17, 135.18) is True
    assert require_bbox(None, None, None, None) is False
    try:
        require_bbox(33.83, None, None, None, what="selftest")
    except ValueError as e:
        print(f"partial bbox -> ValueError ✓  ({e})")
    else:
        raise AssertionError("partial bbox must raise ValueError")

    assert _response_remark({"elements": []}) == ""
    assert _response_remark({"remark": "runtime error: Query timed out",
                             "elements": []}).startswith("runtime error")
    print("remark detection ✓")
    print(f"offline={is_offline()}  ({OFFLINE_ENV}={os.environ.get(OFFLINE_ENV, '')!r})")
