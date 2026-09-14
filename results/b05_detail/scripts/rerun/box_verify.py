import os, sys,time,json,numpy as np
sys.path.insert(0,'/home/ntaku/laravel-project/flood_pso/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dem_parser import mosaic_tiles, downsample
from hazard_gt import load_hazard_gt
from flood_sim import make_river_source
from fast_objective import FastIoUObjective
from ccpso2 import CCPSO2
from pyswarms.single import GlobalBestPSO
D='/home/ntaku/laravel-project/kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620'
BBOX={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}
info=downsample(mosaic_tiles(D),1); dem=info['dem']
gt_depth,gt=load_hazard_gt(info,zoom=16)
src=make_river_source(dem,lat_max=info['lat_max'],res_lat=info['res_lat'],
                      lon_min=info['lon_min'],res_lon=info['res_lon'],river_bbox=BBOX,elev_max=5.0)
BUDGET=5000; K=16; Dd=1+K*K; SEEDS=(0,1,2)
res={}
for lim in (2.0,20.0):
    wf_max=8.0+lim
    lb=np.empty(Dd); ub=np.empty(Dd); lb[0],ub[0]=3.0,8.0; lb[1:],ub[1:]=-lim,lim
    out={'PSO_periodic':[],'PSO_clip':[],'CCPSO2':[]}
    for sd in SEEDS:
        o1=FastIoUObjective(dem,src,gt,K=K,sigma=0.5,wf_max=wf_max)
        np.random.seed(sd)
        p=GlobalBestPSO(n_particles=30,dimensions=Dd,options={"c1":1.5,"c2":1.5,"w":0.7},bounds=(lb,ub),ftol=-np.inf)
        b,_=p.optimize(o1.batch,iters=BUDGET//30,verbose=False); out['PSO_periodic'].append(1-b)
        o2=FastIoUObjective(dem,src,gt,K=K,sigma=0.5,wf_max=wf_max)
        np.random.seed(sd)
        vc=0.2*(ub-lb)
        p=GlobalBestPSO(n_particles=30,dimensions=Dd,options={"c1":1.5,"c2":1.5,"w":0.7},bounds=(lb,ub),
                        bh_strategy="nearest",velocity_clamp=(-vc,vc),vh_strategy="invert",ftol=-np.inf)
        b,_=p.optimize(o2.batch,iters=BUDGET//30,verbose=False); out['PSO_clip'].append(1-b)
        o3=FastIoUObjective(dem,src,gt,K=K,sigma=0.5,wf_max=wf_max)
        cc=CCPSO2(o3,dim=Dd,n_particles=20,group_size=16,bounds=(lb,ub),p_cauchy=0.5,seed=sd)
        cc.run(n_cycles=max(1,BUDGET//(20*((Dd+15)//16)))); out['CCPSO2'].append(1-cc.b_cost)
        print(f'  lim={lim:g} seed={sd} done', flush=True)
    res[str(lim)]=out
    print(f'--- dh in [-{lim:g},{lim:g}] ---', flush=True)
    for k,v in out.items():
        a=np.array(v); print(f'  {k:14s} IoU mean={a.mean():.4f} sd={a.std(ddof=1):.4f} seeds={np.round(a,4).tolist()}', flush=True)
json.dump(res,open('/home/ntaku/laravel-project/flood_pso/results/b05_detail/raw/box_verify.json','w'),indent=1)
print('DONE', flush=True)
