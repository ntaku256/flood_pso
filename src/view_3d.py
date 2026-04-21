"""
view_3d.py
DEMと洪水シミュレーション結果を Pyvista で3D可視化する。
NBTビューアなしで素早く地形を確認するのに使う。

実行:
    python view_3d.py               # デフォルト (2.5km四方, v_exag=3)
    python view_3d.py --width 5000  # 5km四方
    python view_3d.py --save        # PNG保存のみ (画面表示しない)
"""

import sys, argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from dem_parser import mosaic_tiles, downsample
from flood_sim import make_river_source, simulate_flood

DEM_DIR    = r"C:\Users\moriken\Documents\ntaku\特別実験\資料\地形データ\FG-GML-503561-DEM5A-20250620"
OUT_DIR    = Path(__file__).parent.parent / "results"
LAT_CENTER = 33.875
LON_CENTER = 135.168
WATER_LEVEL = 5.06
SIGMA       = 0.09
RIVER_BBOX  = {"lat_min":33.855,"lat_max":33.905,"lon_min":135.145,"lon_max":135.215}


def crop_dem(dem_info, lat_c, lon_c, width_m, depth_m):
    """DEMをエリアクロップ。"""
    dem = dem_info["dem"]
    lat_max = dem_info["lat_max"]
    lon_min = dem_info["lon_min"]
    res_lat = dem_info["res_lat"]
    res_lon = dem_info["res_lon"]

    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * np.cos(np.radians(lat_c)))

    row_c = round((lat_max - lat_c) / res_lat)
    col_c = round((lon_c - lon_min) / res_lon)
    hr = int(depth_m / 2 * lat_per_m / res_lat)
    hc = int(width_m / 2 * lon_per_m / res_lon)

    r0, r1 = max(0, row_c - hr), min(dem.shape[0], row_c + hr)
    c0, c1 = max(0, col_c - hc), min(dem.shape[1], col_c + hc)
    return dem[r0:r1, c0:c1]


def render_3d_surface(dem_patch, inundation_patch,
                      h_res_m=5.0, v_exag=3.0,
                      save_path=None, interactive=False):
    """
    Pyvista で地形サーフェスと洪水域を3D描画する。
    """
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True

    H, W = dem_patch.shape
    dem_clean = np.where(np.isnan(dem_patch), 0.0, dem_patch)

    # ─── 地形サーフェス (StructuredGrid 2D面) ───
    # PyVista StructuredGrid: dimensions=[W, H, 1]
    # points shape: (H*W, 3) — x=East, y=Elevation, z=North
    xs = (np.arange(W) * h_res_m).astype(np.float32)
    zs = (np.arange(H) * h_res_m).astype(np.float32)
    gx, gz = np.meshgrid(xs, zs)           # (H, W)
    gy = (dem_clean * v_exag).astype(np.float32)

    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    terrain = pv.StructuredGrid()
    terrain.dimensions = [W, H, 1]
    terrain.points = points
    terrain["elevation"] = dem_clean.ravel().astype(np.float32)

    # ─── 洪水面サーフェス ───
    flood_mask = (inundation_patch > 0.05) & ~np.isnan(dem_patch)
    flood_top  = np.where(flood_mask,
                          (dem_clean + inundation_patch) * v_exag,
                          gy)   # 非洪水は地形と同高さ
    flood_points = np.column_stack([gx.ravel(), flood_top.ravel().astype(np.float32), gz.ravel()])
    flood_surf = pv.StructuredGrid()
    flood_surf.dimensions = [W, H, 1]
    flood_surf.points = flood_points
    flood_surf["is_flooded"] = flood_mask.ravel().astype(np.float32)
    flood_surf = flood_surf.threshold(0.5, scalars="is_flooded")

    # ─── レンダリング ───
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background("lightyellow")

    plotter.add_mesh(terrain,
                     scalars="elevation", cmap="terrain",
                     clim=[0, float(np.nanpercentile(dem_clean, 95))],
                     show_edges=False, lighting=True,
                     show_scalar_bar=True,
                     scalar_bar_args={"title": "Elevation [m]",
                                      "position_x": 0.75, "position_y": 0.05})

    if flood_surf.n_cells > 0:
        plotter.add_mesh(flood_surf, color="dodgerblue",
                         opacity=0.7, show_edges=False, lighting=False)

    # カメラ: 南西から斜め俯瞰
    cx = W * h_res_m / 2
    cz = H * h_res_m / 2
    cy = float(np.nanmean(dem_clean)) * v_exag
    dist = max(W, H) * h_res_m
    plotter.camera_position = [
        (cx - dist * 0.8, cy + dist * 0.6, cz + dist * 0.9),
        (cx, cy, cz),
        (0, 1, 0),
    ]
    plotter.camera.zoom(1.2)

    if save_path:
        plotter.screenshot(str(save_path))
        print(f"3D render saved: {save_path}")
    if interactive:
        plotter.show()
    plotter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width",  type=float, default=2500)
    parser.add_argument("--depth",  type=float, default=2500)
    parser.add_argument("--v_exag", type=float, default=3.0)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()

    print("Loading DEM...")
    dem_info = mosaic_tiles(DEM_DIR)
    dem = dem_info["dem"]

    print("Running flood simulation...")
    source = make_river_source(dem, dem_info["lat_max"], dem_info["res_lat"],
        dem_info["lon_min"], dem_info["res_lon"], RIVER_BBOX, elev_max=5.0)
    inundation = simulate_flood(dem, source, water_level=WATER_LEVEL, sigma=SIGMA)

    print("Cropping to target area...")
    h_res_m = dem_info["res_lat"] / (1.0 / 111320.0)
    dem_patch = crop_dem(dem_info, LAT_CENTER, LON_CENTER, args.width, args.depth)
    row_c = round((dem_info["lat_max"] - LAT_CENTER) / dem_info["res_lat"])
    col_c = round((LON_CENTER - dem_info["lon_min"]) / dem_info["res_lon"])
    lat_per_m = 1.0 / 111320.0
    lon_per_m = 1.0 / (111320.0 * np.cos(np.radians(LAT_CENTER)))
    hr = int(args.depth / 2 * lat_per_m / dem_info["res_lat"])
    hc = int(args.width / 2 * lon_per_m / dem_info["res_lon"])
    r0, r1 = max(0, row_c - hr), min(dem.shape[0], row_c + hr)
    c0, c1 = max(0, col_c - hc), min(dem.shape[1], col_c + hc)
    idn_patch = inundation[r0:r1, c0:c1]

    print(f"Patch: {dem_patch.shape}, v_exag={args.v_exag}")

    save_path = OUT_DIR / f"3d_terrain_{int(args.width)}m_ve{args.v_exag:.0f}.png"
    render_3d_surface(dem_patch, idn_patch,
                      h_res_m=h_res_m, v_exag=args.v_exag,
                      save_path=str(save_path),
                      interactive=not args.save)


if __name__ == "__main__":
    main()
