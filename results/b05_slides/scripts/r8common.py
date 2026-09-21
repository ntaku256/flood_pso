import csv, numpy as np
from common import *
def load_r8():
    d = np.load(f"{R8}/decoded/gobo_r8_tsunami_depth5m.npz")
    g = dict(lat_max=float(d["lat_max"]), lon_min=float(d["lon_min"]), res_lat=float(d["res_lat"]), res_lon=float(d["res_lon"]))
    return d, g
def ll2rc(g, lat, lon):
    return (g["lat_max"] - lat) / g["res_lat"], (lon - g["lon_min"]) / g["res_lon"]
def load_evac():
    rows = list(csv.DictReader(open(f"{R8}/evac/30205_2.csv", encoding="utf-8-sig")))
    return {r["施設・場所名"].replace("　", " ").strip(): (float(r["緯度"]), float(r["経度"])) for r in rows}
def cell_dims(g, H):
    """格子 1 セルの寸法 [m] (dx=東西, dy=南北) を npz の res と中心緯度から求める (楕円体近似)。"""
    latc = np.radians(g["lat_max"] - H / 2 * g["res_lat"])
    dy = g["res_lat"] * (111132.954 - 559.822 * np.cos(2 * latc) + 1.175 * np.cos(4 * latc))
    dx = g["res_lon"] * (111412.84 * np.cos(latc) - 93.5 * np.cos(3 * latc))
    return float(dx), float(dy)
