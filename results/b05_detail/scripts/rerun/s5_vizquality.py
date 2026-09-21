"""S5 可視化品質 — 「最適化する意味がない」への唯一の答えを数値化する。

比較:
  A) 公式25m図を素朴に最近傍拡大 (面 IoU 0.9195 で最強のベースライン)
  B) HAND IoU オラクル K=32 (連続水面形)
測る:
  1. 水面標高の隣接段差: 非ゼロ率・中央値・最大 (Minecraft の1ブロック単位で階段になる量)
  2. 水面が地形より低いのに浸水扱いのセル率 (物理的にありえない配置)
  3. 水面の連結成分数 (バラバラに浮いた水面の数)
  4. 水深の取り得る値の数 (離散段数)
  5. ブロック化したときの垂直段差の総本数 (1m 丸めで水面標高が隣接セル間で変わる箇所)
"""
import sys, numpy as np
sys.path.insert(0,'/home/ntaku/laravel-project/flood_pso/src')
from scipy.ndimage import gaussian_filter, label as nd_label, minimum_filter, distance_transform_edt
from dem_parser import mosaic_tiles, downsample
from hazard_gt import load_hazard_gt
from flood_sim import make_river_source
D='/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}
info=downsample(mosaic_tiles(D),1); dem=info['dem']
gt_depth,gt=load_hazard_gt(info,zoom=16)
land=np.where(np.isnan(dem),9999.0,dem).astype(np.float64); valid=~np.isnan(dem); gt=gt&valid
H,W=land.shape; LS=gaussian_filter(land,sigma=0.5); land_eff=np.maximum(LS,land+0.05)
src=make_river_source(dem,lat_max=info['lat_max'],res_lat=info['res_lat'],
                      lon_min=info['lon_min'],res_lon=info['res_lon'],river_bbox=BBOX,elev_max=5.0)
ST8=np.ones((3,3),int)

def metrics(name, mask, wse):
    """mask=浸水セル, wse=水面標高場 (浸水セルのみ有効)"""
    m=mask&valid
    n=int(m.sum())
    tp=np.count_nonzero(m&gt); fp=np.count_nonzero(m&~gt&valid); fn=np.count_nonzero(~m&gt)
    iou=tp/(tp+fp+fn) if tp+fp+fn else 0
    W_=np.where(m,wse,np.nan)
    # 1. 隣接段差 (浸水セル同士)
    dy=np.abs(np.diff(W_,axis=0)); py=m[:-1]&m[1:]
    dx=np.abs(np.diff(W_,axis=1)); px=m[:,:-1]&m[:,1:]
    j=np.concatenate([dy[py],dx[px]]); j=j[np.isfinite(j)]
    nz=j>1e-9
    # 2. 水面が地形より低いのに浸水扱い
    bad=int(np.count_nonzero(m&(wse<land)))
    # 3. 水面の連結成分数
    lab,ncomp=nd_label(m,structure=ST8)
    # 4. 水深の取り得る値の数
    d=np.where(m,np.maximum(wse-land,0.0),np.nan)
    nvals=len(np.unique(np.round(d[m],3)))
    # 5. 1m 丸めで隣接段差が生じる箇所 (Minecraft ブロック単位の階段)
    Wb=np.where(m,np.floor(wse),np.nan)
    by=(np.abs(np.diff(Wb,axis=0))>0.5)&py
    bx=(np.abs(np.diff(Wb,axis=1))>0.5)&px
    steps=int(np.count_nonzero(by)+np.count_nonzero(bx))
    tot_pairs=int(np.count_nonzero(py)+np.count_nonzero(px))
    print(f'\n■ {name}')
    print(f'   浸水セル {n:,} / 面IoU {iou:.4f}')
    print(f'   1. 水面標高の隣接段差: 非ゼロ {100*nz.mean():5.1f}%  非ゼロ時の中央値 {np.median(j[nz]) if nz.any() else 0:5.3f}m  最大 {j.max():6.2f}m')
    print(f'   2. 水面 < 地形 なのに浸水扱い: {bad:,} セル ({100*bad/max(n,1):.2f}%)')
    print(f'   3. 水面の連結成分数: {ncomp:,}')
    print(f'   4. 水深の取り得る値の数: {nvals:,}')
    print(f'   5. 1m丸めで階段になる隣接ペア: {steps:,} / {tot_pairs:,} = {100*steps/max(tot_pairs,1):5.1f}%')
    return dict(n=n,iou=iou,nz=float(nz.mean()),ncomp=int(ncomp),nvals=nvals,steps=steps,pairs=tot_pairs,bad=bad)

# ── A) 公式25m図を素朴に最近傍拡大 ──
f=5; hh,ww=H//f,W//f
blk=lambda a:a[:hh*f,:ww*f].reshape(hh,f,ww,f)
wcnt=blk(gt.astype(np.float64)).sum(axis=(1,3))
nval=blk(valid.astype(np.float64)).sum(axis=(1,3))
wet_c=(wcnt>0)&(np.divide(wcnt,np.maximum(nval,1))>=0.5)
# 粗セルの代表水深 = そのセル内の GT 水深の中央値 (公式図は階級値なので実質最頻値)
dsum=np.nan_to_num(blk(np.where(gt,np.maximum(gt_depth,0.0),0.0))).sum(axis=(1,3))
drep=np.divide(dsum,np.maximum(wcnt,1e-9))
mA=np.zeros((H,W),bool); dA=np.zeros((H,W))
mA[:hh*f,:ww*f]=np.repeat(np.repeat(wet_c,f,axis=0),f,axis=1)
dA[:hh*f,:ww*f]=np.repeat(np.repeat(drep,f,axis=0),f,axis=1)
mA&=valid
wseA=land+dA
rA=metrics('A) 公式25m図を素朴に最近傍拡大 (最強ベースライン)', mA, wseA)

# ── B) HAND IoU オラクル K=32 (連続水面形) ──
lm=minimum_filter(np.where(valid,land,9999.0),size=201,mode='nearest'); drain=valid&((land-lm)<0.5)
_,(iy,ix)=distance_transform_edt(~drain,return_indices=True); zd=land[iy,ix]
hand=land_eff-zd
K=32; ys=np.linspace(0,H,K+1).astype(int); xs=np.linspace(0,W,K+1).astype(int)
thr=np.zeros((H,W))
for i in range(K):
    for jj in range(K):
        sl=(slice(ys[i],ys[i+1]),slice(xs[jj],xs[jj+1]))
        fv=hand[sl].ravel(); g=gt[sl].ravel()
        if fv.size==0: continue
        o=np.argsort(fv,kind='stable'); fs=fv[o]; gs=g[o]; tot=gs.sum()
        err=np.concatenate(([tot],np.cumsum(~gs)+(tot-np.cumsum(gs))))
        k=int(np.argmin(err)); t=(fs[0]-1e-6) if k==0 else (fs[k-1]+1e-9)
        thr[sl]=min(max(t,0.0),10.0)
wseB=thr+zd
cand=LS<wseB; lab,nl=nd_label(cand,structure=ST8); sv=src&cand
u=np.unique(lab[sv]); u=u[u>0]; lut=np.zeros(nl+1,bool); lut[u]=True
mB=lut[lab]&((wseB-land)>0.05)
rB=metrics('B) HAND IoU オラクル K=32 (連続水面形)', mB, wseB)

print('\n=== まとめ: 連続水面形が素朴拡大に勝つ指標 ===')
for k,lbl in (('nz','隣接段差の非ゼロ率'),('steps','1m丸めの階段ペア数'),('ncomp','水面連結成分数'),('nvals','水深の段数'),('bad','水面<地形の違反')):
    a,b=rA[k],rB[k]
    better='B (連続)' if b<a else ('A (素朴)' if a<b else '同')
    print(f'  {lbl:24s} A={a:>10}  B={b:>10}  → 良いのは {better}')
print(f'  {"面IoU":24s} A={rA["iou"]:.4f}      B={rB["iou"]:.4f}      → 良いのは {"A (素朴)" if rA["iou"]>rB["iou"] else "B"}')
