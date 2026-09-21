"""この定式化 (バスタブ + 連結成分 + K×K 水位補正場) の理論上限 IoU を実データで見積もる。

オラクル階層:
  O0  最良の単一平面水位 (dh=0)                       … 現状の 1 次元版
  O1  ブロック単位オラクル (±2m 制約あり, 連結性なし)   … K×K パラメータ化の上限
  O2  ブロック単位オラクル (制約なし, 連結性なし)       … ±2m 制約のコスト
  O3  O1 に連結性を適用                               … 連結性のコスト
  O4  セル単位オラクル (= 1.0)                        … 参照

構造的に不可能な分:
  水位場の値域 water_field ∈ [lb0-2, ub0+2] = [1, 10] を超える標高の GT セル
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, '/home/ntaku/laravel-project/flood_pso/src')
from scipy.ndimage import gaussian_filter, label as nd_label

from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source
from hazard_gt import load_hazard_gt

DEM_DIR = '/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX = {"lat_min": 33.855, "lat_max": 33.905, "lon_min": 135.145, "lon_max": 135.215}
SIGMA, DS, ZOOM = 0.5, 1, 16
LB0, UB0, DHB = 3.0, 8.0, 2.0      # calibrate_hidaka.py の範囲

t0 = time.time()
info = downsample(mosaic_tiles(DEM_DIR), DS)
dem = info['dem']
src = make_river_source(dem, lat_max=info['lat_max'], res_lat=info['res_lat'],
                        lon_min=info['lon_min'], res_lon=info['res_lon'],
                        river_bbox=BBOX, elev_max=5.0)
gt_depth, gt_mask = load_hazard_gt(info, zoom=ZOOM)
print(f'[load] {time.time()-t0:.0f}s  DEM {dem.shape}  GT {int(gt_mask.sum()):,} cells '
      f'({100*gt_mask.mean():.2f}%)  src {int(src.sum()):,} cells')

land = np.where(np.isnan(dem), 9999.0, dem).astype(np.float64)
land_s = gaussian_filter(land, sigma=SIGMA)
# sim_mask = (land_s < wf) かつ (wf - land > 0.05)  ⇔  land_eff < wf
land_eff = np.maximum(land_s, land + 0.05)
H, W = land.shape
valid = ~np.isnan(dem)

def iou(mask):
    i = np.count_nonzero(mask & gt_mask); u = np.count_nonzero(mask | gt_mask)
    return i/u if u else 0.0

def stats(mask, name):
    tp = np.count_nonzero(mask & gt_mask); fp = np.count_nonzero(mask & ~gt_mask)
    fn = np.count_nonzero(~mask & gt_mask)
    csi = tp/(tp+fp+fn) if (tp+fp+fn) else 0
    print(f'  {name:52s} IoU={csi:.4f}  TP={tp:>9,} FP={fp:>9,} FN={fn:>9,} '
          f'HR={tp/(tp+fn) if tp+fn else 0:.3f} FAR={fp/(tp+fp) if tp+fp else 0:.3f}')
    return csi

print('\n=== GT の標高分布 ===')
g = land[gt_mask & valid]
for q in (50, 90, 95, 99, 99.9, 100):
    print(f'  GT セル標高 {q:5.1f}%ile = {np.percentile(g, q):7.2f} m')
WF_MAX = UB0 + DHB   # 10.0
WF_MIN = LB0 - DHB   # 1.0
over = np.count_nonzero(gt_mask & valid & (land_eff >= WF_MAX))
print(f'\n  ★ 水位場の上限 {WF_MAX} m を超える標高の GT セル = {over:,} '
      f'({100*over/gt_mask.sum():.2f}% of GT) → 構造的に到達不能 (確定 FN)')
print(f'     この分だけで IoU 上限は {1 - over/np.count_nonzero(gt_mask):.4f} 以下')

print('\n=== O0: 最良の単一平面水位 (dh=0, 連結性なし) ===')
best = (0, None)
for w in np.arange(1.0, 14.01, 0.05):
    s = iou(land_eff < w)
    if s > best[0]: best = (s, w)
print(f'  範囲制約なしで最良: w={best[1]:.2f} m → IoU={best[0]:.4f}')
best_in = (0, None)
for w in np.arange(LB0, UB0+1e-9, 0.05):
    s = iou(land_eff < w)
    if s > best_in[0]: best_in = (s, w)
print(f'  範囲 [{LB0},{UB0}] 内で最良: w={best_in[1]:.2f} m → IoU={best_in[0]:.4f}')

def block_oracle(K, wf_lo=None, wf_hi=None, tag=''):
    """各 K×K ブロックで誤分類を最小化する水位を選ぶ (ブロック定数 = 上限側)。"""
    ys = np.linspace(0, H, K+1).astype(int); xs = np.linspace(0, W, K+1).astype(int)
    wf = np.zeros((H, W))
    for i in range(K):
        for j in range(K):
            sl = (slice(ys[i], ys[i+1]), slice(xs[j], xs[j+1]))
            le = land_eff[sl].ravel(); gm = gt_mask[sl].ravel()
            if le.size == 0: continue
            o = np.argsort(le, kind='stable'); le_s = le[o]; gm_s = gm[o]
            # 閾値を le_s[k] の直後に置く = 先頭 k+1 セルを浸水とする
            # 誤り = (浸水側の非GT) + (非浸水側のGT)
            cs_non = np.cumsum(~gm_s)          # 先頭 k+1 の非GT数 = FP
            tot_gt = gm_s.sum()
            cs_gt = np.cumsum(gm_s)
            err = cs_non + (tot_gt - cs_gt)    # FP + FN
            err = np.concatenate(([tot_gt], err))   # k=-1 (全非浸水)
            k = int(np.argmin(err))
            thr = (le_s[0] - 1e-6) if k == 0 else (le_s[k-1] + 1e-9)
            if wf_lo is not None: thr = min(max(thr, wf_lo), wf_hi)
            wf[sl] = thr
    m = land_eff < wf
    s = stats(m, f'O: K={K} ブロックオラクル {tag}')
    return wf, m, s

print('\n=== O1/O2: ブロック単位オラクル (連結性なし) ===')
res = {}
for K in (16, 24, 32, 64):
    wf_u, m_u, s_u = block_oracle(K, tag='(値域制約なし)')
    wf_b, m_b, s_b = block_oracle(K, WF_MIN, WF_MAX, tag=f'(値域 [{WF_MIN},{WF_MAX}] 制約)')
    res[K] = {'unbounded': s_u, 'bounded': s_b}

print('\n=== O3: K=32 ブロックオラクル + 連結性 ===')
wf32, m32, _ = block_oracle(32, WF_MIN, WF_MAX, tag='(再計算)')
cand = land_s < wf32
lab, _ = nd_label(cand, structure=np.ones((3,3), int))
sv = src & cand
if sv.any():
    labs = set(lab[sv].tolist()); labs.discard(0)
    fm = np.isin(lab, list(labs))
    m_conn = fm & (wf32 - land > 0.05)
    stats(m_conn, 'O3: K=32 オラクル + 連結性(水源接続のみ)')
    lost = np.count_nonzero(m32 & ~m_conn)
    print(f'     連結性で落ちたセル {lost:,} ({100*lost/max(1,np.count_nonzero(m32)):.1f}% of 浸水域)')
    # GT のうち水源に連結不能な分
    gt_unreach = np.count_nonzero(gt_mask & ~fm)
    print(f'     GT のうち水源成分に含まれない = {gt_unreach:,} ({100*gt_unreach/gt_mask.sum():.1f}% of GT)')
else:
    print('     水源が浸水候補に含まれず')

print('\n=== 実測値との比較 ===')
print(f'  実測 K=32 CCPSO2 IoU = 0.6196 / PSO = 0.5627')
print(f'  K=32 ブロックオラクル (値域制約, 連結性なし) = {res[32]["bounded"]:.4f}')
print(f'  → 最適化の余地 = {res[32]["bounded"] - 0.6196:.4f}')

json.dump({'gt_cells': int(gt_mask.sum()), 'dem_shape': list(dem.shape),
           'over_wfmax': int(over), 'O0_free': best, 'O0_bounded': best_in,
           'block': {str(k): v for k, v in res.items()}},
          open('/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/iou_ceiling.json', 'w'),
          ensure_ascii=False, indent=1, default=float)
print(f'\n総時間 {time.time()-t0:.0f}s')
