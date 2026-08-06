import numpy as np, sys
from pathlib import Path
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from flood_sim import make_river_source
from hazard_gt import load_hazard_gt
SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
d = np.load(SC/"data.npz"); dem = d["dem"]
lat_min, lat_max, lon_min, lon_max, res_lat, res_lon = d["meta"]
f = 5
H, W = dem.shape; H2, W2 = H//f, W//f
crop = dem[:H2*f, :W2*f]
import warnings; warnings.filterwarnings("ignore")
ds = np.nanmean(crop.reshape(H2,f,W2,f), axis=(1,3)).astype(np.float32)
info = {"dem": ds, "lat_min": lat_min, "lat_max": lat_max-(H-H2*f)*res_lat,
        "lon_min": lon_min, "lon_max": lon_max-(W-W2*f)*res_lon,
        "res_lat": res_lat*f, "res_lon": res_lon*f}
src = make_river_source(ds, lat_max=info["lat_max"], res_lat=info["res_lat"],
                        lon_min=info["lon_min"], res_lon=info["res_lon"],
                        river_bbox={"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215},
                        elev_max=5.0)
gtd, gtm = load_hazard_gt(info, zoom=16)
print("ds5 dem", ds.shape, "src", int(src.sum()), "gt", int(gtm.sum()), f"({100*gtm.mean():.2f}%)")
np.savez_compressed(SC/"data_ds5.npz", dem=ds, src=src, gt_depth=gtd)
