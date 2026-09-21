"""決定的な実験: HAND 基準の正しいパラメータ化で CCPSO2 は修正PSO に勝つのか。

浸水条件を  land < water_field  から  HAND(x,y) < depth_field(x,y)  に変える。
  HAND      = land − land[最近傍排水路]
  depth_field = d_global + bilinear_upsample(dd_map)
  決定変数  x[0]=d_global ∈ [0,8] , x[1:]=dd_map ∈ [-2,2]  → depth_field ∈ [-2,10]
  (絶対標高版の w∈[3,8] + dh∈[-2,2] → wf∈[1,10] と同じ幅 10m)

値域を広げずに絶対標高の広い値域に近い表現力が得られるはず (オラクルで 0.7937 vs 0.8473)。
問題は「その正しいパラメータ化のもとで CCPSO2 の優位が残るか」。
"""
import sys, time, json, numpy as np
sys.path.insert(0,'/home/ntaku/laravel-project/flood_pso/src')
from scipy.ndimage import gaussian_filter, label as nd_label, minimum_filter, distance_transform_edt
from dem_parser import mosaic_tiles, downsample
from hazard_gt import load_hazard_gt
from flood_sim import make_river_source
from ccpso2 import CCPSO2
from pyswarms.single import GlobalBestPSO

D='/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}
info=downsample(mosaic_tiles(D),1); dem=info['dem']
gt_depth,gt=load_hazard_gt(info,zoom=16)
land=np.where(np.isnan(dem),9999.0,dem).astype(np.float64); valid=~np.isnan(dem); gt=gt&valid
H,W=land.shape
LS=gaussian_filter(land,sigma=0.5)
src=make_river_source(dem,lat_max=info['lat_max'],res_lat=info['res_lat'],
                      lon_min=info['lon_min'],res_lon=info['res_lon'],river_bbox=BBOX,elev_max=5.0)
# HAND
lm=minimum_filter(np.where(valid,land,9999.0),size=201,mode='nearest')
drain=valid&((land-lm)<0.5)
_,(iy,ix)=distance_transform_edt(~drain,return_indices=True); zd=land[iy,ix]
HAND_S=LS-zd; HAND_R=land-zd
ST8=np.ones((3,3),int)
print(f'排水路 {int(drain.sum()):,} cells / HAND: GT中央値 {np.median(HAND_R[gt]):.2f}m 90%ile {np.percentile(HAND_R[gt],90):.2f}m', flush=True)

class HandObj:
    def __init__(self,K,depth_max):
        pot=HAND_S<depth_max
        rs=np.where(pot.any(axis=1))[0]; cs=np.where(pot.any(axis=0))[0]
        self.r0,self.r1=int(rs[0]),int(rs[-1])+1; self.c0,self.c1=int(cs[0]),int(cs[-1])+1
        Hc,Wc=self.r1-self.r0,self.c1-self.c0
        sl=(slice(self.r0,self.r1),slice(self.c0,self.c1))
        self.F=np.ascontiguousarray(HAND_S[sl]).astype(np.float32)
        self.R05=np.ascontiguousarray(HAND_R[sl]).astype(np.float32)+np.float32(0.05)
        self.src=np.ascontiguousarray(src[sl]); self.GT=np.ascontiguousarray(gt[sl])
        self.ngt=int(np.count_nonzero(gt)); self.K=K; self.n_evals=0
        def co(n_in,n_out_full,lo,n_out):
            x=(np.arange(lo,lo+n_out))*(n_in-1)/(n_out_full-1)
            i0=np.clip(np.floor(x).astype(np.intp),0,n_in-2)
            return i0,(x-i0).astype(np.float32)
        self.ry,ty=co(K,H,self.r0,Hc); self.rx,self.tx=co(K,W,self.c0,Wc); self.ty=ty[:,None]
    def _up(self,dd):
        d=dd.astype(np.float32,copy=False)
        a=d[:,self.rx]; b=d[:,self.rx+1]; col=a+(b-a)*self.tx
        t=col[self.ry]; u=col[self.ry+1]
        return t+(u-t)*self.ty
    def __call__(self,x):
        self.n_evals+=1; K=self.K
        df=self._up(np.asarray(x[1:1+K*K]).reshape(K,K)); df+=np.float32(x[0])
        cand=self.F<df; sv=self.src&cand
        if not sv.any(): return 1.0
        lab,nl=nd_label(cand,structure=ST8)
        u=np.unique(lab[sv]); u=u[u>0]
        lut=np.zeros(nl+1,bool); lut[u]=True
        m=lut[lab]; m&=df>self.R05
        tp=np.count_nonzero(m&self.GT); nm=np.count_nonzero(m)
        un=nm+self.ngt-tp
        return 1.0-tp/un if un else 1.0
    def batch(self,X): return np.array([self(X[i]) for i in range(X.shape[0])])

BUDGET=5000; K=16; Dd=1+K*K; SEEDS=(0,1,2)
lb=np.empty(Dd); ub=np.empty(Dd); lb[0],ub[0]=0.0,16.0; lb[1:],ub[1:]=-4.0,4.0
DMAX=20.0
out={'PSO_periodic':[],'PSO_clip':[],'CCPSO2':[]}
t0=time.time()
for sd in SEEDS:
    o=HandObj(K,DMAX); np.random.seed(sd)
    p=GlobalBestPSO(n_particles=30,dimensions=Dd,options={"c1":1.5,"c2":1.5,"w":0.7},bounds=(lb,ub),ftol=-np.inf)
    b,_=p.optimize(o.batch,iters=BUDGET//30,verbose=False); out['PSO_periodic'].append(1-b)
    o=HandObj(K,DMAX); np.random.seed(sd); vc=0.2*(ub-lb)
    p=GlobalBestPSO(n_particles=30,dimensions=Dd,options={"c1":1.5,"c2":1.5,"w":0.7},bounds=(lb,ub),
                    bh_strategy="nearest",velocity_clamp=(-vc,vc),vh_strategy="invert",ftol=-np.inf)
    b,_=p.optimize(o.batch,iters=BUDGET//30,verbose=False); out['PSO_clip'].append(1-b)
    o=HandObj(K,DMAX)
    cc=CCPSO2(o,dim=Dd,n_particles=20,group_size=16,bounds=(lb,ub),p_cauchy=0.5,seed=sd)
    cc.run(n_cycles=max(1,BUDGET//(20*((Dd+15)//16)))); out['CCPSO2'].append(1-cc.b_cost)
    print(f'  seed={sd} done ({time.time()-t0:.0f}s)  PSO_periodic={out["PSO_periodic"][-1]:.4f} PSO_clip={out["PSO_clip"][-1]:.4f} CCPSO2={out["CCPSO2"][-1]:.4f}', flush=True)
print('\n=== HAND 基準 (d0∈[0,16], dd∈[-4,4] → depth∈[-4,20])、K=16, 5000評価, 3シード ===', flush=True)
for k,v in out.items():
    a=np.array(v); print(f'  {k:14s} IoU mean={a.mean():.4f} sd={a.std(ddof=1):.4f} seeds={np.round(a,4).tolist()}', flush=True)
cc=np.array(out['CCPSO2']); pc=np.array(out['PSO_clip'])
wins=int((cc>pc).sum())
print(f'\n  CCPSO2 − 修正PSO = {cc.mean()-pc.mean():+.4f}   シード別 {wins}勝{len(cc)-wins}敗', flush=True)
print(f'  参考: 絶対標高・狭い値域 CCPSO2 0.6204 / 修正PSO 0.6047 (+0.0158, 3勝0敗)', flush=True)
print(f'        絶対標高・広い値域 CCPSO2 0.7170 / 修正PSO 0.7145 (+0.0025, 1勝2敗)', flush=True)
print(f'        HAND オラクル K=16 = 0.7601 / K=32 = 0.7937', flush=True)
json.dump(out,open('/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/hand_optimize20.json','w'),indent=1)
print('DONE', flush=True)
