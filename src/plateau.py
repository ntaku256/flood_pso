"""
plateau.py
PLATEAU CityGML（国土交通省 Project PLATEAU）の建物（bldg）ローダー。

LOD0/LOD1 の建物フットプリント（lod0RoofEdge）と実測高さ（measuredHeight）を抽出し、
fgd_vector.load_buildings と同じ形式
  [{"coords":[[lat,lon],...], "holes":[], "tags":{"fgd_type","height_m"}}]
で返す。これにより terrain_render.build_building_maps にそのまま渡せる。

CityGML 建物は EPSG:6697（緯度経度・標高）。posList は "lat lon height lat lon height ..."。
"""
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np


def _ln(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def mesh_bbox(code: str):
    """標準地域メッシュ8桁コード → (lat_min, lat_max, lon_min, lon_max)[deg]。"""
    p = int(code[0:2]); q = int(code[2:4])
    r = int(code[4]);   s = int(code[5])
    t = int(code[6]);   u = int(code[7])
    lat0 = p * (2 / 3) + r * (2 / 3 / 8) + t * (2 / 3 / 80)
    lon0 = (q + 100) + s * (1 / 8) + u * (1 / 80)
    return lat0, lat0 + (2 / 3 / 80), lon0, lon0 + (1 / 80)


def _first_poslist(elem):
    """要素配下の最初の gml:posList を [[lat,lon],...] にして返す（z は無視）。"""
    for pl in elem.iter():
        if _ln(pl.tag) == "posList" and pl.text:
            v = list(map(float, pl.text.split()))
            if len(v) >= 9:  # 3点(=closed triangle)以上
                return [[v[i], v[i + 1]] for i in range(0, len(v) - 2, 3)]
    return None


def _footprint(bl):
    """建物のフットプリント外周。lod0RoofEdge → lod0FootPrint → lod1Solid の順に探す。"""
    for tag in ("lod0RoofEdge", "lod0FootPrint"):
        for e in bl.iter():
            if _ln(e.tag) == tag:
                f = _first_poslist(e)
                if f:
                    return f
    for e in bl.iter():
        if _ln(e.tag) == "lod1Solid":
            f = _first_poslist(e)
            if f:
                return f
    return None


def _lod2_surfaces(bl):
    """建物の LOD2 ジオメトリ（lod2Solid/lod2MultiSurface）の全ポリゴンを
    [[(lat,lon,z),...], ...] で返す（無ければ空）。屋根形状の描画に使う。"""
    polys = []
    for g in bl.iter():
        if _ln(g.tag) in ("lod2Solid", "lod2MultiSurface"):
            for pl in g.iter():
                if _ln(pl.tag) == "posList" and pl.text:
                    v = list(map(float, pl.text.split()))
                    if len(v) >= 9:
                        polys.append([(v[i], v[i + 1], v[i + 2]) for i in range(0, len(v) - 2, 3)])
    return polys


# PLATEAU 建物用途コード(Building_usage) → 壁材カテゴリ（terrain_render.BUILDING_WALL_BY_TYPE のキー）
USAGE_CAT = {
    "401": "商業ビル", "403": "商業ビル",
    "402": "宿泊",
    "404": "住宅", "412": "住宅", "414": "住宅",
    "411": "マンション", "413": "マンション",
    "415": "公共", "421": "公共",
    "422": "倉庫", "451": "倉庫", "452": "倉庫", "454": "倉庫",
    "431": "工場",
    "441": "農業施設",
}


def _category(usage, struct, h):
    """用途コード＋構造(0=木造)＋高さ から壁材カテゴリを決める。用途不明は高さで概略。"""
    cat = USAGE_CAT.get(usage)
    if cat is None:
        return "堅ろう建物" if (h or 0) >= 12 else "普通建物"
    if cat in ("住宅", "マンション") and struct == "0":   # 木造住宅
        return "木造住宅"
    return cat


def load_plateau_buildings(bldg_dir: str, *, lat_min, lat_max, lon_min, lon_max,
                           lod2: bool = False, verbose: bool = True) -> list:
    """
    PLATEAU bldg ディレクトリ内の *.gml から、指定 bbox に重なる建物を抽出。

    Returns
    -------
    [{"coords":[[lat,lon],...], "holes":[], "tags":{"fgd_type","height_m"}}]
    fgd_type は高さで概略マッピング（>=12m を堅ろう建物、無壁は判定せず普通建物）。
    height_m は measuredHeight（無い場合 None → build_building_maps が LiDAR 高さで補完）。
    """
    paths = sorted(glob.glob(os.path.join(bldg_dir, "*.gml")))
    out = []
    n_files = 0
    for p in paths:
        base = os.path.basename(p)
        code = base.split("_")[0]
        if not (code.isdigit() and len(code) >= 8):
            continue
        a, b, c, d = mesh_bbox(code[:8])
        if b < lat_min or a > lat_max or d < lon_min or c > lon_max:
            continue  # メッシュが bbox 外
        try:
            root = ET.parse(p).getroot()
        except Exception as e:
            if verbose:
                print(f"  [plateau] parse fail {base}: {e}")
            continue
        n_files += 1
        for bl in root.iter():
            if _ln(bl.tag) != "Building":
                continue
            h = None; usage = None; struct = None
            for e in bl.iter():
                lt = _ln(e.tag)
                if lt == "measuredHeight" and e.text and h is None:
                    try:
                        h = float(e.text)
                    except ValueError:
                        pass
                elif lt == "usage" and e.text and usage is None:
                    usage = e.text.strip()
                elif lt == "buildingStructureOrgType" and e.text and struct is None:
                    struct = e.text.strip()
            foot = _footprint(bl)
            if not foot or len(foot) < 4:
                continue
            arr = np.asarray(foot)
            if (arr[:, 0].max() < lat_min or arr[:, 0].min() > lat_max
                    or arr[:, 1].max() < lon_min or arr[:, 1].min() > lon_max):
                continue
            tp = _category(usage, struct, h)
            d = {"coords": foot, "holes": [],
                 "tags": {"fgd_type": tp, "height_m": h, "usage": usage}}
            if lod2:
                surfs = _lod2_surfaces(bl)
                if surfs:
                    d["roof3d"] = surfs
                    zs = [z for p in surfs for (_, _, z) in p]
                    if zs and (max(zs) - min(zs)) >= 20:   # 城などの高いLOD2は白壁ランドマーク扱い
                        d["tags"]["fgd_type"] = "ランドマーク"
            out.append(d)
    if verbose:
        print(f"  [plateau] {n_files} メッシュ → 建物 {len(out)} 棟（bbox内）")
    return out


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    bs = load_plateau_buildings(d, lat_min=34.225, lat_max=34.239, lon_min=135.153, lon_max=135.175)
    hs = [b["tags"]["height_m"] for b in bs if b["tags"]["height_m"]]
    print(f"buildings={len(bs)}  height median={np.median(hs):.1f}m max={np.max(hs):.1f}m" if hs else f"buildings={len(bs)}")
