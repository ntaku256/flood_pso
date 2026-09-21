"""HAND + depth∈[0,20]、K=16、5000評価で 21 シード。修正PSO / CC現行 / CC忠実(N=20,固定s)。
sys.argv[1] = 'a'|'b'|'c' でシードを3分割して並列実行する。"""
import os, sys, time, json, numpy as np
sys.path.insert(0,'/home/ntaku/laravel-project/flood_pso/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.ndimage import gaussian_filter, label as nd_label, minimum_filter, distance_transform_edt
from dem_parser import mosaic_tiles, downsample
from hazard_gt import load_hazard_gt
from flood_sim import make_river_source
from ccpso2 import CCPSO2
from ccpso2_faithful import CCPSO2Faithful
from pyswarms.single import GlobalBestPSO
CHUNK=sys.argv[1]
ALL=list(range(21)); SEEDS={'a':ALL[0:7],'b':ALL[7:14],'c':ALL[14:21]}[CHUNK]
D='/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}
info=downsample(mosaic_tiles(D),1); dem=info['dem']
gt_depth,gt=load_hazard_gt(info,zoom=16)
land=np.where(np.isnan(dem),9999.0,dem).astype(np.float64); valid=~np.isnan(dem); gt=gt&valid
H,W=land.shape; LS=gaussian_filter(land,sigma=0.5)
src=make_river_source(dem,lat_max=info['lat_max'],res_lat=info['res_lat'],
                      lon_min=info['lon_min'],res_lon=info['res_lon'],river_bbox=BBOX,elev_max=5.0)
lm=minimum_filter(np.where(valid,land,9999.0),size=201,mode='nearest')
drain=valid&((land-lm)<0.5)
_,(iy,ix)=distance_transform_edt(~drain,return_indices=True); zd=land[iy,ix]
HS=LS-zd; HR=land-zd; ST8=np.ones((3,3),int); DMAX=20.0
class HandObj:
    def __init__(self,K):
        pot=HS<DMAX
        rs=np.where(pot.any(axis=1))[0]; cs=np.where(pot.any(axis=0))[0]
        r0,r1=int(rs[0]),int(rs[-1])+1; c0,c1=int(cs[0]),int(cs[-1])+1
        Hc,Wc=r1-r0,c1-c0; sl=(slice(r0,r1),slice(c0,c1))
        self.F=np.ascontiguousarray(HS[sl]).astype(np.float32)
        self.R05=np.ascontiguousarray(HR[sl]).astype(np.float32)+np.float32(0.05)
        self.src=np.ascontiguousarray(src[sl]); self.GT=np.ascontiguousarray(gt[sl])
        self.ngt=int(np.count_nonzero(gt)); self.K=K; self.n_evals=0
        def co(n_in,nf,lo,no):
            x=(np.arange(lo,lo+no))*(n_in-1)/(nf-1)
            i0=np.clip(np.floor(x).astype(np.intp),0,n_in-2); return i0,(x-i0).astype(np.float32)
        self.ry,ty=co(K,H,r0,Hc); self.rx,self.tx=co(K,W,c0,Wc); self.ty=ty[:,None]
    def _up(self,dd):
        d=dd.astype(np.float32,copy=False)
        a=d[:,self.rx]; b=d[:,self.rx+1]; col=a+(b-a)*self.tx
        t=col[self.ry]; u=col[self.ry+1]; return t+(u-t)*self.ty
    def __call__(self,x):
        self.n_evals+=1; K=self.K
        tf=self._up(np.asarray(x[1:1+K*K]).reshape(K,K)); tf+=np.float32(x[0])
        cand=self.F<tf; sv=self.src&cand
        if not sv.any(): return 1.0
        lab,nl=nd_label(cand,structure=ST8)
        u=np.unique(lab[sv]); u=u[u>0]
        lut=np.zeros(nl+1,bool); lut[u]=True
        m=lut[lab]; m&=tf>self.R05
        tp=np.count_nonzero(m&self.GT); nm=np.count_nonzero(m)
        return 1.0-tp/(nm+self.ngt-tp) if (nm+self.ngt-tp) else 1.0
    def batch(self,X): return np.array([self(X[i]) for i in range(X.shape[0])])
BUDGET=5000; K=16; Dd=1+K*K
lb=np.empty(Dd); ub=np.empty(Dd); lb[0],ub[0]=0.0,16.0; lb[1:],ub[1:]=-4.0,4.0
out={'PSO_clip':{},'CC_current':{},'CC_faithful':{}}
t0=time.time()
for sd in SEEDS:
    o=HandObj(K); np.random.seed(sd); vc=0.2*(ub-lb)
    p=GlobalBestPSO(n_particles=30,dimensions=Dd,options={"c1":1.5,"c2":1.5,"w":0.7},bounds=(lb,ub),
                    bh_strategy="nearest",velocity_clamp=(-vc,vc),vh_strategy="invert",ftol=-np.inf)
    b,_=p.optimize(o.batch,iters=BUDGET//30,verbose=False); out['PSO_clip'][sd]=1-b
    o=HandObj(K)
    cc=CCPSO2(o,dim=Dd,n_particles=20,group_size=16,bounds=(lb,ub),p_cauchy=0.5,seed=sd)
    cc.run(n_cycles=max(1,BUDGET//(20*((Dd+15)//16)))); out['CC_current'][sd]=1-cc.b_cost
    o=HandObj(K)
    ff=CCPSO2Faithful(o,dim=Dd,n_particles=20,group_size=16,bounds=(lb,ub),p_cauchy=0.5,seed=sd)
    r=ff.run(max_evals=BUDGET); out['CC_faithful'][sd]=1-r['best_cost']
    print(f'  seed={sd} ({time.time()-t0:.0f}s) PSO={out["PSO_clip"][sd]:.4f} CCcur={out["CC_current"][sd]:.4f} CCfai={out["CC_faithful"][sd]:.4f}', flush=True)
json.dump(out,open(f'/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/seeds21_{CHUNK}.json','w'),indent=1)
print('DONE', flush=True)
