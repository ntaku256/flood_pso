"""
parking_render.py
OSM 駐車場(amenity=parking)を「衛星写真の色に寄せて」立体化する画像処理レンダラ。

terrain_render.dem_to_blocks_enhanced から呼ばれ、surf_block(地表ブロックキー grid)を
その場で上書きし、検出した停車車両を 3D ブロック (ix,iy,iz,key) のリストで返す。

オルソ RGB がある時の方針（駐車「枠」は描かず衛星写真に任せ、行の「ベースライン」だけ引く）:
  - アスファルト = 写真色をそのまま残す
  - 車          = アスファルト中央色から外れた色塊を検出 → 1 段持ち上げて停車車両ブロックに
  - ベースライン = 車の「背面」側に、行と平行な1本線:
      * 検出した車から行の並ぶ向き(スタッキング軸)を自動判定
      * 行が背中合わせに近接 → その中間に1本
      * 片側が空いている行 → 駐車場境界に近い側に1本
      * 車が途中で切れたら、それまでの行間隔を外挿して延長
      * 車が無い領域 → 衛星写真の白線を検出してその位置に
      * どちらも無ければデフォルト間隔
オルソ RGB が無い時: アスファルト=black_concrete、ベースライン=デフォルト間隔。
"""
from __future__ import annotations

import math
import numpy as np


def _grid_xy(lat, lon, bbox, gh, gw):
    lat_min, lat_max, lon_min, lon_max = bbox
    x = (lon - lon_min) / (lon_max - lon_min) * gw
    z = (lat_max - lat) / (lat_max - lat_min) * gh
    return x, z


def _poly_mask(coords, bbox, gh, gw):
    import matplotlib.path as mpath
    if len(coords) < 3:
        return np.zeros((gh, gw), bool), np.zeros((0, 2))
    pts = np.array([_grid_xy(la, lo, bbox, gh, gw) for la, lo in coords])
    path = mpath.Path(pts)
    xs, zs = np.meshgrid(np.arange(gw) + 0.5, np.arange(gh) + 0.5)
    inside = path.contains_points(np.column_stack([xs.ravel(), zs.ravel()]))
    return inside.reshape(gh, gw), pts


def _long_axis(pts):
    """ポリゴン頂点列 (Nx2 grid xy) の最長辺方向の単位ベクトル。"""
    best, vec = -1.0, (1.0, 0.0)
    for (x0, z0), (x1, z1) in zip(pts, np.roll(pts, -1, axis=0)):
        d = math.hypot(x1 - x0, z1 - z0)
        if d > best:
            best, vec = d, ((x1 - x0) / (d + 1e-9), (z1 - z0) / (d + 1e-9))
    return vec


def _bandedness(p):
    """1D 射影値 p のヒストグラムが「帯状(行)」にどれだけ集中しているか（std/mean）。"""
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 2:
        return 0.0
    h = np.histogram(p, bins=int(hi - lo) + 1)[0].astype(float)
    return float(h.std() / (h.mean() + 1e-9))


def _find_rows(p, smin, smax, min_dist):
    """射影値 p のヒストグラム極大 = 行(車列)の中心位置リスト（射影単位）。"""
    from scipy.signal import find_peaks
    nb = int(smax - smin) + 1
    if nb < 3 or len(p) < 3:
        return []
    h = np.histogram(p, bins=nb, range=(smin, smax))[0].astype(float)
    hs = np.convolve(h, np.ones(3) / 3.0, mode="same")
    pk, _ = find_peaks(hs, distance=max(1, int(min_dist)),
                       prominence=max(1.0, 0.2 * hs.max()))
    return [smin + int(i) + 0.5 for i in pk]


def _med_spacing(p, smin, smax):
    """射影 p のピーク間隔の中央値（行/ストール判別用）。ピーク<2 なら None。"""
    rows = _find_rows(p, smin, smax, min_dist=2)
    if len(rows) < 2:
        return None
    return float(np.median(np.diff(sorted(rows))))


def _baselines_from_rows(rows, smin, smax, back_thr, cd):
    """行中心列 → ベースライン射影値リスト（背中合わせ=中間/開き=境界側/外挿）。"""
    rows = sorted(rows)
    n = len(rows)
    half = cd / 2.0
    bl = []
    # 連続行の間: 近接(背中合わせ)→中間に1本 / 離れ(アイスル)→開けて引かない
    for i in range(n - 1):
        if rows[i + 1] - rows[i] < back_thr:
            bl.append((rows[i] + rows[i + 1]) / 2.0)
    # 端の行: 内側がアイスル(離れ)なら外=境界側が背面 → 境界寄りに1本
    if n == 1:
        # 単行は「境界に近い側」へ1本だけ（両側に引くと [ ] 化するため）
        if rows[0] - smin <= smax - rows[0]:
            bl.append(max(smin, rows[0] - half))
        else:
            bl.append(min(smax, rows[0] + half))
    elif n >= 2:
        if (rows[1] - rows[0]) >= back_thr:
            bl.append(max(smin, rows[0] - half))
        if (rows[-1] - rows[-2]) >= back_thr:
            bl.append(min(smax, rows[-1] + half))
    # 外挿: 車が途中で切れた領域を、確立した行間隔で境界まで延長
    if n >= 2:
        sp = float(np.median(np.diff(rows)))
        if sp > 1.5:
            s = rows[0] - sp
            while s > smin + half:
                bl.append(s); s -= sp
            s = rows[-1] + sp
            while s < smax - half:
                bl.append(s); s += sp
    return bl


def _white_line_cells(rgb, mask):
    """オルソの白線セル（高輝度・低彩度）。車の無い領域のベースライン推定に使う。"""
    r, g, b = (rgb[..., c].astype(int) for c in range(3))
    bright = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return mask & (bright > 185) & (sat < 35)


def render_parking(parkings, patch_bbox_latlon, nz, nx, *,
                   surf_block, y_surf_land, land_mask,
                   ortho_rgb=None, h_res_block_m: float = 0.667,
                   car_depth_m: float = 5.0, default_sp_m: float = 5.0,
                   line_key: str = "white_concrete",
                   asphalt_key: str = "black_concrete",
                   boundary_key: str = "gray_concrete") -> list:
    """surf_block を駐車場で上書きし、(検出車両の 3D ブロックリスト, 駐車場領域 union mask) を返す。
    駐車場領域 mask は呼び出し側で「道路境界線を駐車場内に引かない」抑制に使う。"""
    from scipy import ndimage

    car_blocks: list = []
    area_mask = np.zeros((nz, nx), bool)                  # 駐車場ポリゴン union(陸セル) — curb抑制用
    boundary_mask = np.zeros((nz, nx), bool)              # 駐車場境界線(外周1セル) — 道路描画後に再描画して保護
    has_rgb = ortho_rgb is not None and ortho_rgb.shape[:2] == (nz, nx)
    cd = car_depth_m / max(h_res_block_m, 0.1)            # 車奥行(セル)
    back_thr = 1.5 * cd                                   # 行中心間 < これ = 背中合わせ
    default_sp = max(3.0, default_sp_m / max(h_res_block_m, 0.1))

    for pk in (parkings or []):
        mask, pts = _poly_mask(pk["coords"], patch_bbox_latlon, nz, nx)
        mask &= land_mask
        if not mask.any() or len(pts) < 3:
            continue
        area_mask |= mask
        ys, xs = np.where(mask)
        ux, uz = _long_axis(pts)
        vx, vz = -uz, ux

        if not has_rgb:
            surf_block[mask] = asphalt_key

        # ── 車検出（行判定用）。白線(明・低彩度)は車から除外し、暗い/彩度のある塊だけ車に ──
        car_mask = np.zeros((nz, nx), bool)
        if has_rgb:
            rgbm = ortho_rgb[ys, xs].astype(np.float32)
            base_b = float(np.median(rgbm.mean(1)))          # アスファルト代表輝度
            bright = rgbm.mean(1)
            sat = rgbm.max(1) - rgbm.min(1)
            is_dark = bright < base_b - 35                   # アスファルトより暗い=車体/影
            is_color = sat > 35                              # 彩度のある車
            is_white = (bright > base_b + 15) & (sat < 25)   # 白線/明部 → 車でない(除外)
            sel = (is_dark | is_color) & ~is_white
            car_mask[ys[sel], xs[sel]] = True
            lab, n = ndimage.label(car_mask)
            keep = np.zeros((nz, nx), bool)
            for k in range(1, n + 1):
                if 3 <= int((lab == k).sum()) <= 40:         # ~2–12m の塊=車
                    keep |= (lab == k)
            car_mask = keep & mask

        # ── スタッキング軸(行が積み重なる向き)を「ピーク間隔の大きい方」で判定 ──
        #   積み重なる軸=行ピッチ(5-8m)で離散、行方向=ストールピッチ(2.5m)か連続。
        #   よってピーク間隔が大きい軸が積み重なる軸。車がまばらでも頑健。
        pv_lot = xs * vx + ys * vz
        pu_lot = xs * ux + ys * uz
        s_hat, s_lot, base_s = (vx, vz), pv_lot, None
        if car_mask.any():
            cys, cxs = np.where(car_mask)
            pu_c, pv_c = cxs * ux + cys * uz, cxs * vx + cys * vz
            su = _med_spacing(pu_c, float(pu_lot.min()), float(pu_lot.max()))
            sv = _med_spacing(pv_c, float(pv_lot.min()), float(pv_lot.max()))
            use_u = (sv is None) if su is not None else False
            if su is not None and sv is not None:
                use_u = su >= sv
            if use_u:
                s_hat, s_lot, s_c = (ux, uz), pu_lot, pu_c
            else:
                s_hat, s_lot, s_c = (vx, vz), pv_lot, pv_c
            smin, smax = float(s_lot.min()), float(s_lot.max())
            rows = _find_rows(s_c, smin, smax, min_dist=cd * 0.8)
            if rows:
                base_s = _baselines_from_rows(rows, smin, smax, back_thr, cd)
        if base_s is None:
            # 車から決まらない → 衛星白線 → デフォルト間隔（行方向=長辺と仮定, スタッキング=v）
            smin, smax = float(pv_lot.min()), float(pv_lot.max())
            s_hat, s_lot = (vx, vz), pv_lot
            if has_rgb:
                wl = _white_line_cells(ortho_rgb, mask)
                if wl.any():
                    wy, wx = np.where(wl)
                    base_s = _find_rows(wx * vx + wy * vz, smin, smax, min_dist=cd * 0.8)
            if not base_s:
                k = max(1, int((smax - smin) / default_sp))
                base_s = [smin + (i + 0.5) * (smax - smin) / k for i in range(k)]

        # 最小間隔(≈車奥行)でデデュープ：物理的に1行1本、密な縞化を防ぐ
        if base_s:
            base_s = sorted(base_s)
            ded = [base_s[0]]
            for b in base_s[1:]:
                if b - ded[-1] >= 0.9 * cd:
                    ded.append(b)
            base_s = ded

        # ── ベースライン描画（スタッキング軸の射影が base 値に近い行=行と平行な線）──
        sxh, szh = s_hat
        proj = xs * sxh + ys * szh
        lm = np.zeros((nz, nx), bool)
        for b in base_s:
            sel = np.abs(proj - b) <= 0.7
            lm[ys[sel], xs[sel]] = True
        lm &= mask & ~car_mask
        surf_block[lm] = line_key

        # ── 駐車場の境界線（外周1セル）= 控えめだが明確な縁取り(curb) ──
        if boundary_key:
            edge = mask & ~ndimage.binary_erosion(mask)
            surf_block[edge] = boundary_key
            boundary_mask |= edge
        # 車は立体化しない（オルソ写真色のまま）。car_mask は行/ベースライン判定にのみ使用。

    return car_blocks, area_mask, boundary_mask
