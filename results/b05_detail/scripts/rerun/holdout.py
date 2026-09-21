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

# ═══════════════════════════════════════════════════════════════════
# 市松 holdout: K×K ブロック格子を市松に分け、train ブロックの GT だけで
# 損失を計算して最適化し、test ブロックの GT で IoU を評価する。
# bilinear アップサンプルは train ブロックから test ブロックへ内挿するので、
# 「滑らかな水面形が空間的に汎化するか」を直接測れる。
# ═══════════════════════════════════════════════════════════════════
class Split(Obj):
    """train/test マスクを持ち、損失は train 側だけ、評価は test 側だけ。"""
    def __init__(self,K,srcmask,parity=0,lam=0.0):
        super().__init__(K,srcmask,lam)
        Hc,Wc=self.F.shape
        # 各セルがどの K×K ブロックに属するか (アップサンプル座標と同じ割り当て)
        by=np.clip(((np.arange(self.r0,self.r1))*(K-1)/(H-1)+0.5).astype(int),0,K-1)
        bx=np.clip(((np.arange(self.c0,self.c1))*(K-1)/(W-1)+0.5).astype(int),0,K-1)
        chk=((by[:,None]+bx[None,:])%2==parity)
        self.TR=np.ascontiguousarray(chk); self.TE=np.ascontiguousarray(~chk)
        self.GT_TR=self.GT&self.TR; self.GT_TE=self.GT&self.TE
        self.n_tr=int(np.count_nonzero(self.GT_TR)); self.n_te=int(np.count_nonzero(self.GT_TE))
    def _iou_on(self,x,REG,ngt):
        m,_=self.mask(x)
        if m is None or ngt==0: return 0.0
        mr=m&REG
        tp=np.count_nonzero(mr&self.GT); nm=np.count_nonzero(mr)
        u=nm+ngt-tp
        return tp/u if u else 0.0
    def iou_train(self,x): return self._iou_on(x,self.TR,self.n_tr)
    def iou_test(self,x):  return self._iou_on(x,self.TE,self.n_te)
    def __call__(self,x):
        self.n_evals+=1
        v=1.0-self.iou_train(x)
        if self.lam: v+=self.lam*dh_roughness(np.asarray(x[1:1+self.K*self.K]).reshape(self.K,self.K))
        return v

BUDGET=5000; K=16; Dd=1+K*K; SEEDS=(0,1,2)
lb=np.empty(Dd); ub=np.empty(Dd); lb[0],ub[0]=0.0,16.0; lb[1:],ub[1:]=-4.0,4.0
import importlib
pso_mod=importlib.import_module('pyswarms.single')
def run_pso(o,sd):
    np.random.seed(sd)
    opt=pso_mod.GlobalBestPSO(n_particles=30,dimensions=Dd,
        options={'c1':1.5,'c2':1.5,'w':0.7},bounds=(lb,ub),
        bh_strategy='nearest',velocity_clamp=(-0.2*(ub-lb),0.2*(ub-lb)),ftol=-np.inf)
    opt.optimize(lambda P:np.array([o(p) for p in P]),iters=max(1,BUDGET//30),verbose=False)
    return opt.swarm.best_pos
def run_cc(o,sd):
    cc=CCPSO2(o,dim=Dd,n_particles=20,group_size=16,bounds=(lb,ub),p_cauchy=0.5,seed=sd)
    cc.run(n_cycles=max(1,BUDGET//(20*((Dd+15)//16))))
    return cc.b

print('\n=== 市松 holdout (実河道水源・λ=0) ===',flush=True)
o0=Split(K,SRC_DRAIN,parity=0)
print(f'  train GT {o0.n_tr:,} セル / test GT {o0.n_te:,} セル',flush=True)
res={}
t0=time.time()
for nm,fn in (('CCPSO2',run_cc),('修正PSO',run_pso)):
    tr=[];te=[];full=[]
    for sd in SEEDS:
        o=Split(K,SRC_DRAIN,parity=0)
        x=fn(o,sd)
        tr.append(o.iou_train(x)); te.append(o.iou_test(x))
        o2=Obj(K,SRC_DRAIN); full.append(o2.iou(x))
    tr=np.array(tr);te=np.array(te);fu=np.array(full)
    res[nm]={'train':tr.tolist(),'test':te.tolist(),'full':fu.tolist()}
    print(f'  {nm:8s} train {tr.mean():.4f}±{tr.std(ddof=1):.4f}  '
          f'test {te.mean():.4f}±{te.std(ddof=1):.4f}  '
          f'全域 {fu.mean():.4f}  汎化ギャップ {tr.mean()-te.mean():+.4f}  ({time.time()-t0:.0f}s)',flush=True)
# 参照: 全域で学習したときの test 領域 IoU (= 上限)
print('\n=== 参照: 全域学習 (holdout なし) の test 領域 IoU ===',flush=True)
for sd in SEEDS[:1]:
    of=Obj(K,SRC_DRAIN); x=run_cc(of,sd)
    osp=Split(K,SRC_DRAIN,parity=0)
    print(f'  seed{sd}: 全域IoU {of.iou(x):.4f}  train領域 {osp.iou_train(x):.4f}  test領域 {osp.iou_test(x):.4f}',flush=True)
json.dump(res,open('/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/holdout.json','w'),indent=1,default=float)
print('DONE',flush=True)
