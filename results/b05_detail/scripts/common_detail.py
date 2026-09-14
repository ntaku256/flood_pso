"""b05_detail 共通: パス・格子の前処理 (npz キャッシュ)・S5 指標の定義。

洪水パートの位置づけ:
  「浸水想定図 (25 m・5 段ランク) を地形に整合した連続水面へ変換する高次元逆問題」
  降雨・時間は入力に取らない = シミュレーションではなく **想定図の変換器**。
  指標は IoU (想定図との整合の目安) と水面品質 (1 m 階段率 / 連結成分数 /
  水面<地形の違反 / 取り得る水深値の数)。

元スクリプト (読み取り専用の退避ディレクトリ /home/ntaku/research-rescue-20260728) からの移植:
  s5_vizquality.py / src_reg.py / src_reg_final.py / hand_optimize20.py / fast_objective.py
差分は scratchpad 依存パスの除去と、S5 指標の定義を s5_vizquality.py に一本化した点のみ。
"""
from __future__ import annotations
import os
import sys
import numpy as np

FP = "/home/ntaku/laravel-project/flood_pso"
OUT = f"{FP}/results/b05_detail"
RESCUE = "/home/ntaku/research-rescue-20260728"
DEM_DIR = "/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620"
BBOX = {"lat_min": 33.855, "lat_max": 33.905, "lon_min": 135.145, "lon_max": 135.215}

CACHE = os.environ.get(
    "B05D_CACHE",
    "/tmp/claude-1000/-home-ntaku-laravel-project/"
    "2ad68a62-bd0b-4106-b379-6430fbc56344/scratchpad/b05cache",
)
os.makedirs(CACHE, exist_ok=True)

sys.path.insert(0, f"{FP}/src")
sys.path.insert(0, f"{FP}/results/b05_slides/scripts")

ST8 = np.ones((3, 3), dtype=int)
DMAX = 20.0          # HAND 水深場の上限 (d0∈[0,16] + dd∈[-4,4])
K_OPT = 16           # 最適化のブロック分割 (D = 1 + K*K = 257)
BUDGET = 5000        # 評価回数
SEEDS = (0, 1, 2)


def load_grids(verbose: bool = True) -> dict:
    """DEM / 想定図 GT / HAND / 水源マスクを作って npz にキャッシュする。

    land     : 標高 (NaN は 9999 に置換)
    valid    : DEM が有効なセル
    gt       : 公式浸水想定図 (L2) の浸水セル
    gt_depth : 同 代表水深 (ランク中央値)
    LS       : gaussian_filter(land, 0.5)  = シミュレーション側の実効地形
    zd       : 最近傍排水路セルの標高
    HS/HR    : HAND (平滑版 / 生)
    src_bbox : 現行の水源マスク (矩形 bbox ∩ 標高<=5m)
    src_drain: 実河道の水源マスク (谷底 ∩ bbox ∩ 標高<=5m)
    """
    p = f"{CACHE}/grids.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["info"] = d["info"].item()
        return d

    from scipy.ndimage import gaussian_filter, minimum_filter, distance_transform_edt
    from dem_parser import mosaic_tiles, downsample
    from hazard_gt import load_hazard_gt
    from flood_sim import make_river_source

    info = downsample(mosaic_tiles(DEM_DIR), 1)
    dem = info["dem"]
    gt_depth, gt = load_hazard_gt(info, zoom=16)
    land = np.where(np.isnan(dem), 9999.0, dem).astype(np.float64)
    valid = ~np.isnan(dem)
    gt = gt & valid
    H, W = land.shape
    LS = gaussian_filter(land, sigma=0.5)
    land_eff = np.maximum(LS, land + 0.05)

    src_bbox = make_river_source(
        dem, lat_max=info["lat_max"], res_lat=info["res_lat"],
        lon_min=info["lon_min"], res_lon=info["res_lon"],
        river_bbox=BBOX, elev_max=5.0)

    lm = minimum_filter(np.where(valid, land, 9999.0), size=201, mode="nearest")
    drain = valid & ((land - lm) < 0.5)
    rows = np.arange(H)
    cols = np.arange(W)
    lats = info["lat_max"] - rows * info["res_lat"]
    lons = info["lon_min"] + cols * info["res_lon"]
    inb = np.zeros((H, W), bool)
    inb[np.ix_((lats >= BBOX["lat_min"]) & (lats <= BBOX["lat_max"]),
               (lons >= BBOX["lon_min"]) & (lons <= BBOX["lon_max"]))] = True
    src_drain = drain & inb & (land <= 5.0)

    _, (iy, ix) = distance_transform_edt(~drain, return_indices=True)
    zd = land[iy, ix]

    out = dict(land=land, valid=valid, gt=gt, gt_depth=np.nan_to_num(gt_depth),
               LS=LS, land_eff=land_eff, zd=zd, drain=drain,
               src_bbox=src_bbox, src_drain=src_drain,
               HS=LS - zd, HR=land - zd,
               info=np.array(info, dtype=object))
    np.savez_compressed(p, **out)
    if verbose:
        print(f"[grids] cached -> {p}", flush=True)
    out["info"] = info
    return out


# ────────────────────────────────────────────────────────────────
# S5 水面品質指標 (定義は s5_vizquality.py に一致させる)
# ⚠ 「隣接段差 非ゼロ率」は「区分定数か連続か」を測っていただけで
#    品質指標として誤りだったため出さない (memory 2026-07-25 の訂正)。
# ────────────────────────────────────────────────────────────────
def viz_metrics(name: str, mask, wse, land, valid, gt, verbose: bool = True) -> dict:
    """mask=浸水セル, wse=水面標高場 (浸水セルのみ有効)。"""
    from scipy.ndimage import label as nd_label
    m = mask & valid
    n = int(m.sum())
    tp = np.count_nonzero(m & gt)
    fp = np.count_nonzero(m & ~gt & valid)
    fn = np.count_nonzero(~m & gt)
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    py = m[:-1] & m[1:]
    px = m[:, :-1] & m[:, 1:]
    # 2. 水面が地形より低いのに浸水扱い
    bad = int(np.count_nonzero(m & (wse < land)))
    # 3. 水面の連結成分数
    _, ncomp = nd_label(m, structure=ST8)
    # 4. 水深の取り得る値の数
    d = np.where(m, np.maximum(wse - land, 0.0), np.nan)
    nvals = int(len(np.unique(np.round(d[m], 3)))) if n else 0
    # 5. 1m 丸めで階段になる隣接ペア (Minecraft の 1 ブロック段差)
    Wb = np.where(m, np.floor(wse), np.nan)
    by = (np.abs(np.diff(Wb, axis=0)) > 0.5) & py
    bx = (np.abs(np.diff(Wb, axis=1)) > 0.5) & px
    steps = int(np.count_nonzero(by) + np.count_nonzero(bx))
    pairs = int(np.count_nonzero(py) + np.count_nonzero(px))
    r = dict(name=name, cells=n, iou=float(iou), bad=bad, ncomp=int(ncomp),
             nvals=nvals, steps=steps, pairs=pairs,
             stair_rate=float(steps / pairs) if pairs else 0.0)
    if verbose:
        print(f"■ {name}: 浸水 {n:,} / IoU {iou:.4f} / 1m階段 {steps:,}/{pairs:,} "
              f"= {100*r['stair_rate']:.2f}% / 連結成分 {ncomp:,} / 水深値 {nvals:,} / 違反 {bad:,}",
              flush=True)
    return r


class HandObj:
    """HAND 基準 + depth∈[-4,20] の目的関数 (src_reg.py の Obj と同じ。
    高速化 = 到達可能域クロップ + 分離型 bilinear + float32 + LUT 索引)。"""

    def __init__(self, g, K, srcmask, lam=0.0, dmax=DMAX):
        HS, HR, gt = g["HS"], g["HR"], g["gt"]
        H, W = HS.shape
        pot = HS < dmax
        rs = np.where(pot.any(axis=1))[0]
        cs = np.where(pot.any(axis=0))[0]
        self.r0, self.r1 = int(rs[0]), int(rs[-1]) + 1
        self.c0, self.c1 = int(cs[0]), int(cs[-1]) + 1
        Hc, Wc = self.r1 - self.r0, self.c1 - self.c0
        sl = (slice(self.r0, self.r1), slice(self.c0, self.c1))
        self.F = np.ascontiguousarray(HS[sl]).astype(np.float32)
        self.R05 = np.ascontiguousarray(HR[sl]).astype(np.float32) + np.float32(0.05)
        self.src = np.ascontiguousarray(srcmask[sl])
        self.GT = np.ascontiguousarray(gt[sl])
        self.ngt = int(np.count_nonzero(gt))
        self.K, self.lam, self.n_evals = K, lam, 0
        self.shape = (H, W)

        def co(n_in, nf, lo, no):
            x = (np.arange(lo, lo + no)) * (n_in - 1) / (nf - 1)
            i0 = np.clip(np.floor(x).astype(np.intp), 0, n_in - 2)
            return i0, (x - i0).astype(np.float32)
        self.ry, ty = co(K, H, self.r0, Hc)
        self.rx, self.tx = co(K, W, self.c0, Wc)
        self.ty = ty[:, None]

    def _up(self, dd):
        d = dd.astype(np.float32, copy=False)
        a = d[:, self.rx]
        b = d[:, self.rx + 1]
        col = a + (b - a) * self.tx
        t = col[self.ry]
        u = col[self.ry + 1]
        return t + (u - t) * self.ty

    def mask(self, x):
        from scipy.ndimage import label as nd_label
        K = self.K
        tf = self._up(np.asarray(x[1:1 + K * K]).reshape(K, K))
        tf += np.float32(x[0])
        cand = self.F < tf
        sv = self.src & cand
        if not sv.any():
            return None, tf
        lab, nl = nd_label(cand, structure=ST8)
        u = np.unique(lab[sv])
        u = u[u > 0]
        lut = np.zeros(nl + 1, bool)
        lut[u] = True
        m = lut[lab]
        m &= tf > self.R05
        return m, tf

    def iou(self, x):
        m, _ = self.mask(x)
        if m is None:
            return 0.0
        tp = np.count_nonzero(m & self.GT)
        nm = np.count_nonzero(m)
        return tp / (nm + self.ngt - tp) if (nm + self.ngt - tp) else 0.0

    def __call__(self, x):
        self.n_evals += 1
        v = 1.0 - self.iou(x)
        if self.lam:
            from flood_sim import dh_roughness
            v += self.lam * dh_roughness(np.asarray(x[1:1 + self.K * self.K]).reshape(self.K, self.K))
        return v

    def expand(self, x, g):
        """解 x から全域の (浸水マスク, 水面標高場) を作る。
        水面標高 WSE = HAND 水深場 tf + 最近傍排水路標高 zd。"""
        m, tf = self.mask(x)
        H, W = self.shape
        mf = np.zeros((H, W), bool)
        wf = np.full((H, W), -9999.0)
        sl = (slice(self.r0, self.r1), slice(self.c0, self.c1))
        if m is not None:
            mf[sl] = m
        wf[sl] = tf.astype(np.float64) + g["zd"][sl]
        return mf, wf


def bounds_hand(K=K_OPT):
    D = 1 + K * K
    lb = np.empty(D)
    ub = np.empty(D)
    lb[0], ub[0] = 0.0, 16.0
    lb[1:], ub[1:] = -4.0, 4.0
    return lb, ub, D


def run_ccpso2(obj, K=K_OPT, seed=0, budget=BUDGET):
    from ccpso2 import CCPSO2
    lb, ub, D = bounds_hand(K)
    cc = CCPSO2(obj, dim=D, n_particles=20, group_size=16, bounds=(lb, ub),
                p_cauchy=0.5, seed=seed)
    cc.run(n_cycles=max(1, budget // (20 * ((D + 15) // 16))))
    return cc.b
