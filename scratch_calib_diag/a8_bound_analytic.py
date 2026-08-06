"""(E)/(B): 現行の探索区間 w∈[3,8], dh∈[-2,2] が課す解析的な IoU 上限。
・水面標高は必ず [w-2, w+2] に入る → GT のうち dem > w+2 は絶対に浸水できない (recall 上限)
・水源セル (dem<=5, bbox 内) は dem < w-2 なら必ず浸水 → GT 外なら必ず FP (precision 上限)
さらに dh の座標感度 (死んだ次元の割合) を測る。"""
import numpy as np, sys
from pathlib import Path
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import simulate_flood_hd, iou_loss
SC=Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d=np.load(SC/"data.npz"); dem,src,gtd=d["dem"],d["src"],d["gt_depth"]
gt=gtd>0; nan=np.isnan(dem); NGT=int(gt.sum())
print("### 現行パラメータ区間が課す解析的上限 (5m グリッド)")
print("  w   TP_max(dem<w+2)  FP_min(src&~gt&dem<w-2)  IoU上限")
best=(0,0)
for w in np.arange(3.0,8.01,0.25):
    tp=int(np.sum(gt&~nan&(dem<w+2)))
    fp=int(np.sum(src&~gt&(dem<w-2)))
    i=tp/(NGT+fp)
    if i>best[0]: best=(i,w)
    if abs(w*4-round(w*4))<1e-9 and (w*2)%1==0:
        print(f"  {w:4.1f}  {tp:9d}        {fp:8d}            {i:.4f}")
print(f"  → 最良 IoU上限 = {best[0]:.4f} (w={best[1]:.2f})   [dh を無限解像度にしても超えられない]")
print(f"  参考: dh 範囲を [-8,8] に広げた場合")
for dhb in (2,4,6,8,12,20):
    b=(0,0)
    for w in np.arange(0.0,30.01,0.25):
        tp=int(np.sum(gt&~nan&(dem<w+dhb))); fp=int(np.sum(src&~gt&(dem<w-dhb)))
        i=tp/(NGT+fp)
        if i>b[0]: b=(i,w)
    print(f"    dh∈[-{dhb},{dhb}] : IoU上限={b[0]:.4f} (w={b[1]:.2f})")

print("\n### 水源マスクの素性")
print(f"  src cells={int(src.sum())}  src∩GT={int((src&gt).sum())}  src∖GT={int((src&~gt).sum())}")
print(f"  src∖GT のうち dem<6 のもの={int((src&~gt&(dem<6)).sum())}  → w>=8 では必ず FP")

print("\n### dh 座標感度 (25m グリッド, K=32, 死んだ次元の割合)")
d5=np.load(SC/"data_ds5.npz"); dem5,src5,gtd5=d5["dem"],d5["src"],d5["gt_depth"]; gt5=gtd5>0
def ev(w,dh): return 1-iou_loss(simulate_flood_hd(dem5,src5,float(w),dh,sigma=0.5),gt5)
rs=np.random.RandomState(0)
for K in (16,32):
    dh=rs.uniform(-2,2,(K,K)); w=7.5
    base=ev(w,dh); dead=0; deltas=[]
    for idx in range(K*K):
        i,j=divmod(idx,K); old=dh[i,j]
        dh[i,j]=np.clip(old+0.5,-2,2); c=ev(w,dh); dh[i,j]=old
        deltas.append(abs(c-base))
        if abs(c-base)<1e-9: dead+=1
    deltas=np.array(deltas)
    print(f"  K={K}: base IoU={base:.4f}  ΔIoU=0 の座標 {dead}/{K*K} ({100*dead/(K*K):.1f}%)  "
          f"median|ΔIoU|={np.median(deltas):.2e}  p90={np.percentile(deltas,90):.2e}")
