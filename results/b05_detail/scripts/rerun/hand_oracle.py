"""同じパラメータ数で「絶対標高の水面」vs「排水路基準の水深 (HAND)」を比較する。
HAND(x,y) = land(x,y) - land(最近傍の谷底セル)。浸水条件は HAND < depth_field。
これなら河川縦断の落差 (~30m) を depth_field が負担しなくて済む。"""
import sys, numpy as np, json
sys.path.insert(0,'/home/ntaku/laravel-project/flood_pso/src')
from scipy.ndimage import minimum_filter, gaussian_filter, distance_transform_edt, label as nd_label
from dem_parser import mosaic_tiles, downsample
from hazard_gt import load_hazard_gt
from flood_sim import make_river_source

DEM_DIR='/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}
info=downsample(mosaic_tiles(DEM_DIR),1); dem=info['dem']
gt_depth,gt=load_hazard_gt(info,zoom=16)
land=np.where(np.isnan(dem),9999.0,dem).astype(np.float64)
valid=~np.isnan(dem); H,W=land.shape
land_s=gaussian_filter(land,sigma=0.5)
land_eff=np.maximum(land_s,land+0.05)
gt=gt&valid

def iou(m):
    i=np.count_nonzero(m&gt); u=np.count_nonzero(m|gt); return i/u if u else 0.0
def rep(m,name):
    tp=np.count_nonzero(m&gt); fp=np.count_nonzero(m&~gt); fn=np.count_nonzero(~m&gt)
    print(f'  {name:56s} IoU={tp/(tp+fp+fn):.4f}  HR={tp/(tp+fn):.3f} FAR={fp/(tp+fp) if tp+fp else 0:.3f}')
    return tp/(tp+fp+fn)

# ── 谷底 (排水路) の抽出: 半径500m 窓の最低標高から 0.5m 以内
loc_min=minimum_filter(np.where(valid,land,9999.0),size=201,mode='nearest')
drain=valid&((land-loc_min)<0.5)
print(f'排水路セル {int(drain.sum()):,} ({100*drain.mean():.2f}% of grid)')
# ── HAND: 最近傍排水路セルの標高を引く
_,(iy,ix)=distance_transform_edt(~drain,return_indices=True)
z_drain=land[iy,ix]
hand=land_eff-z_drain
hand_raw=land-z_drain
print(f'HAND 分布: GT 中央値 {np.median(hand_raw[gt]):.2f}m 90%ile {np.percentile(hand_raw[gt],90):.2f}m 99%ile {np.percentile(hand_raw[gt],99):.2f}m')
print(f'           非GT 中央値 {np.median(hand_raw[valid&~gt]):.2f}m')

def block_oracle(field, K, lo=None, hi=None):
    ys=np.linspace(0,H,K+1).astype(int); xs=np.linspace(0,W,K+1).astype(int)
    thr_f=np.zeros((H,W))
    for i in range(K):
        for j in range(K):
            sl=(slice(ys[i],ys[i+1]),slice(xs[j],xs[j+1]))
            f=field[sl].ravel(); g=gt[sl].ravel()
            if f.size==0: continue
            o=np.argsort(f,kind='stable'); fs=f[o]; gs=g[o]
            tot=gs.sum()
            err=np.concatenate(([tot], np.cumsum(~gs)+(tot-np.cumsum(gs))))
            k=int(np.argmin(err))
            t=(fs[0]-1e-6) if k==0 else (fs[k-1]+1e-9)
            if lo is not None: t=min(max(t,lo),hi)
            thr_f[sl]=t
    return field<thr_f, thr_f

print('\n=== 同じ K でパラメータ化を比較 (連結性なし) ===')
out={}
for K in (16,32):
    m,_=block_oracle(land_eff,K,1.0,10.0);        a=rep(m,f'K={K} 絶対標高の水面  値域[1,10] (現状)')
    m,_=block_oracle(land_eff,K,None,None);       b=rep(m,f'K={K} 絶対標高の水面  値域制約なし')
    m,t=block_oracle(hand,K,0.0,10.0);            c=rep(m,f'K={K} HAND基準の水深  値域[0,10] ★')
    m,_=block_oracle(hand,K,0.0,20.0);            d=rep(m,f'K={K} HAND基準の水深  値域[0,20]')
    m,_=block_oracle(hand,K,None,None);           e=rep(m,f'K={K} HAND基準の水深  値域制約なし')
    out[K]={'abs_bounded':a,'abs_free':b,'hand_10':c,'hand_20':d,'hand_free':e}

print('\n=== 大局スカラー1個だけの比較 (D=1 相当) ===')
best=max(((iou(land_eff<w),w) for w in np.arange(1,14,0.05)))
print(f'  絶対標高 単一水面 w={best[1]:.2f} → IoU={best[0]:.4f}')
best=max(((iou(hand<d),d) for d in np.arange(0,20,0.05)))
print(f'  HAND 単一水深   d={best[1]:.2f} → IoU={best[0]:.4f}  ★★')

print('\n=== HAND オラクル + 連結性 (K=32, 値域[0,10]) ===')
m,t=block_oracle(hand,32,0.0,10.0)
src=make_river_source(dem,lat_max=info['lat_max'],res_lat=info['res_lat'],
                      lon_min=info['lon_min'],res_lon=info['res_lon'],
                      river_bbox=BBOX,elev_max=5.0)
cand=hand<t
lab,_=nd_label(cand,structure=np.ones((3,3),int))
sv=src&cand
if sv.any():
    labs=set(lab[sv].tolist()); labs.discard(0)
    fm=np.isin(lab,list(labs))
    rep(fm&cand,'K=32 HAND オラクル + 連結性')
json.dump(out,open('/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/hand_oracle.json','w'),indent=1,default=float)
