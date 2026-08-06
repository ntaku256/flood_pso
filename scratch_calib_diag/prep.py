import os, sys, time
from pathlib import Path
import numpy as np
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source
from hazard_gt import load_hazard_gt

SC = Path("/tmp/claude-1000/-home-ntaku-laravel-project/956a6fb0-85d9-43d6-a55a-f68ab7911c84/scratchpad")
DEM_DIR = "../kennkyuu20260114/地形データ/FG-GML-503561-DEM5A-20250620"
HIDAKA = {"lat_min": 33.855, "lat_max": 33.905, "lon_min": 135.145, "lon_max": 135.215}

t = time.time()
info = downsample(mosaic_tiles(DEM_DIR), 1)
dem = info["dem"]
print("dem", dem.shape, "t", time.time()-t)
src = make_river_source(dem, lat_max=info["lat_max"], res_lat=info["res_lat"],
                        lon_min=info["lon_min"], res_lon=info["res_lon"],
                        river_bbox=HIDAKA, elev_max=5.0)
gt_depth, gt_mask = load_hazard_gt(info, zoom=16)
print("src cells", int(src.sum()), "gt cells", int(gt_mask.sum()))
np.savez_compressed(SC/"data.npz", dem=dem, src=src, gt_depth=gt_depth,
                    meta=np.array([info["lat_min"], info["lat_max"], info["lon_min"],
                                   info["lon_max"], info["res_lat"], info["res_lon"]]))
print("saved", time.time()-t)
