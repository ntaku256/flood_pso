"""未測定だった2つの推奨を検証する。

(1) 水源マスク: 現行の「矩形bbox ∩ 標高≤5m」(198,791セル = グリッドの6%) を
    谷底抽出による実河道 (5,892セル = 0.18%) に置き換えると何が変わるか。
    → 連結性が実際に効くようになるはずだが、IoU は下がるかもしれない。
(2) dh_roughness 正則化: S5 で「水面の連結成分数が素朴拡大より多い (88 vs 47)」という弱点が出た。
    Dirichlet エネルギー項を目的関数に足すと抑えられるか。
    loss = (1 - IoU) + lam * dh_roughness(dd_map)

いずれも HAND + depth∈[0,20]、K=16、5000評価、3シード、CCPSO2 で測る。
"""
import os, sys, time, json, numpy as np
sys.path.insert(0,'/home/ntaku/laravel-project/flood_pso/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.ndimage import gaussian_filter, label as nd_label, minimum_filter, distance_transform_edt
from dem_parser import mosaic_tiles, downsample
from hazard_gt import load_hazard_gt
from flood_sim import make_river_source, dh_roughness
from ccpso2 import CCPSO2
D='/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}
info=downsample(mosaic_tiles(D),1); dem=info['dem']
gt_depth,gt=load_hazard_gt(info,zoom=16)
land=np.where(np.isnan(dem),9999.0,dem).astype(np.float64); valid=~np.isnan(dem); gt=gt&valid
H,W=land.shape; LS=gaussian_filter(land,sigma=0.5)
SRC_BBOX=make_river_source(dem,lat_max=info['lat_max'],res_lat=info['res_lat'],
                           lon_min=info['lon_min'],res_lon=info['res_lon'],river_bbox=BBOX,elev_max=5.0)
lm=minimum_filter(np.where(valid,land,9999.0),size=201,mode='nearest')
DRAIN=valid&((land-lm)<0.5)
# 実河道の水源: 谷底のうち bbox 内かつ標高<=5m (下流側に限る)
rows=np.arange(H); cols=np.arange(W)
lats=info['lat_max']-rows*info['res_lat']; lons=info['lon_min']+cols*info['res_lon']
inb=np.zeros((H,W),bool)
inb[np.ix_((lats>=BBOX['lat_min'])&(lats<=BBOX['lat_max']),(lons>=BBOX['lon_min'])&(lons<=BBOX['lon_max']))]=True
SRC_DRAIN=DRAIN&inb&(land<=5.0)
print(f'水源: bbox版 {int(SRC_BBOX.sum()):,} セル ({100*SRC_BBOX.mean():.2f}%) / 実河道版 {int(SRC_DRAIN.sum()):,} セル ({100*SRC_DRAIN.mean():.3f}%)', flush=True)
_,(iy,ix)=distance_transform_edt(~DRAIN,return_indices=True); zd=land[iy,ix]
HS=LS-zd; HR=land-zd; ST8=np.ones((3,3),int); DMAX=20.0
class Obj:
    def __init__(self,K,srcmask,lam=0.0):
        pot=HS<DMAX
        rs=np.where(pot.any(axis=1))[0]; cs=np.where(pot.any(axis=0))[0]
        self.r0,self.r1=int(rs[0]),int(rs[-1])+1; self.c0,self.c1=int(cs[0]),int(cs[-1])+1
        Hc,Wc=self.r1-self.r0,self.c1-self.c0; sl=(slice(self.r0,self.r1),slice(self.c0,self.c1))
        self.F=np.ascontiguousarray(HS[sl]).astype(np.float32)
        self.R05=np.ascontiguousarray(HR[sl]).astype(np.float32)+np.float32(0.05)
        self.src=np.ascontiguousarray(srcmask[sl]); self.GT=np.ascontiguousarray(gt[sl])
        self.ngt=int(np.count_nonzero(gt)); self.K=K; self.lam=lam; self.n_evals=0
        def co(n_in,nf,lo,no):
            x=(np.arange(lo,lo+no))*(n_in-1)/(nf-1)
            i0=np.clip(np.floor(x).astype(np.intp),0,n_in-2); return i0,(x-i0).astype(np.float32)
        self.ry,ty=co(K,H,self.r0,Hc); self.rx,self.tx=co(K,W,self.c0,Wc); self.ty=ty[:,None]
    def _up(self,dd):
        d=dd.astype(np.float32,copy=False)
        a=d[:,self.rx]; b=d[:,self.rx+1]; col=a+(b-a)*self.tx
        t=col[self.ry]; u=col[self.ry+1]; return t+(u-t)*self.ty
    def mask(self,x):
        K=self.K
        tf=self._up(np.asarray(x[1:1+K*K]).reshape(K,K)); tf+=np.float32(x[0])
        cand=self.F<tf; sv=self.src&cand
        if not sv.any(): return None,tf
        lab,nl=nd_label(cand,structure=ST8)
        u=np.unique(lab[sv]); u=u[u>0]
        lut=np.zeros(nl+1,bool); lut[u]=True
        m=lut[lab]; m&=tf>self.R05
        return m,tf
    def iou(self,x):
        m,_=self.mask(x)
        if m is None: return 0.0
        tp=np.count_nonzero(m&self.GT); nm=np.count_nonzero(m)
        return tp/(nm+self.ngt-tp) if (nm+self.ngt-tp) else 0.0
    def __call__(self,x):
        self.n_evals+=1
        v=1.0-self.iou(x)
        if self.lam:
            v+=self.lam*dh_roughness(np.asarray(x[1:1+self.K*self.K]).reshape(self.K,self.K))
        return v
def s5(obj,x):
    m,tf=obj.mask(x)
    if m is None: return dict(ncomp=0,steps=0,pairs=0,nz=0.0)
    lab,nc=nd_label(m,structure=ST8)
    Wf=np.where(m,tf,np.nan)
    py=m[:-1]&m[1:]; px=m[:,:-1]&m[:,1:]
    dy=np.abs(np.diff(Wf,axis=0)); dx=np.abs(np.diff(Wf,axis=1))
    j=np.concatenate([dy[py],dx[px]]); j=j[np.isfinite(j)]
    Wb=np.where(m,np.floor(tf),np.nan)
    by=(np.abs(np.diff(Wb,axis=0))>0.5)&py; bx=(np.abs(np.diff(Wb,axis=1))>0.5)&px
    st=int(np.count_nonzero(by)+np.count_nonzero(bx)); tp=int(np.count_nonzero(py)+np.count_nonzero(px))
    return dict(ncomp=int(nc),steps=st,pairs=tp,nz=float((j>1e-9).mean()) if j.size else 0.0)
BUDGET=5000; K=16; Dd=1+K*K; SEEDS=(0,1,2)
lb=np.empty(Dd); ub=np.empty(Dd); lb[0],ub[0]=0.0,16.0; lb[1:],ub[1:]=-4.0,4.0
CONF=[('★実河道水源・λ=0.001 (推奨最終)',SRC_DRAIN,0.001),
      ('実河道水源・λ=0.01',SRC_DRAIN,0.01)]
res={}
t0=time.time()
for name,sm,lam in CONF:
    ious=[]; m5=[]
    for sd in SEEDS:
        o=Obj(K,sm,lam)
        cc=CCPSO2(o,dim=Dd,n_particles=20,group_size=16,bounds=(lb,ub),p_cauchy=0.5,seed=sd)
        cc.run(n_cycles=max(1,BUDGET//(20*((Dd+15)//16))))
        x=cc.b; ious.append(o.iou(x)); m5.append(s5(o,x))
    a=np.array(ious)
    nc=np.mean([d['ncomp'] for d in m5]); stp=np.mean([d['steps'] for d in m5])
    pr=np.mean([d['pairs'] for d in m5]); nz=np.mean([d['nz'] for d in m5])
    res[name]={'iou_mean':a.mean(),'iou_sd':a.std(ddof=1),'iou':a.tolist(),
               'ncomp':nc,'steps':stp,'pairs':pr,'nz':nz}
    print(f'{name:26s} IoU {a.mean():.4f}±{a.std(ddof=1):.4f}  連結成分 {nc:6.1f}  '
          f'1m階段 {stp:8.0f}/{pr:8.0f} = {100*stp/max(pr,1):5.2f}%  段差非ゼロ率 {100*nz:5.2f}%  ({time.time()-t0:.0f}s)', flush=True)
print('\n=== 比較の基準 ===', flush=True)
print('  公式25m図の素朴拡大: IoU 0.9195 / 連結成分 47 / 1m階段 19.2% / 段差非ゼロ 62.0%', flush=True)
print('  HAND IoUオラクル K=32: IoU 0.7478 / 連結成分 88 / 1m階段 1.3% / 段差非ゼロ 4.3%', flush=True)
json.dump(res,open('/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/src_reg_final.json','w'),ensure_ascii=False,indent=1,default=float)
print('DONE', flush=True)
