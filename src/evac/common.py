"""evac/common.py — 避難解析モジュール共通のヘルパ (格子変換・和文フォント・入力ロード)。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/evac"
START_DELAY_S = 300.0        # 避難開始 = 地震発生 5 分後 (和歌山県 津波避難計画策定指針)
V_WALK = {"v10": 1.0, "v05": 0.5}   # 歩行速度 [m/s]


def setup_japanese_font():
    import matplotlib
    from matplotlib import font_manager
    for cand in ["/mnt/c/Windows/Fonts/NotoSansJP-VF.ttf", "/mnt/c/Windows/Fonts/meiryo.ttc",
                 str(Path.home() / ".fonts/YuGothL.ttc")]:
        if Path(cand).exists():
            try:
                font_manager.fontManager.addfont(cand)
                name = font_manager.FontProperties(fname=cand).get_name()
                matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue
    return None


class Grid:
    """5 m 格子 (row = (lat_max-lat)/res_lat, col = (lon-lon_min)/res_lon)。"""

    def __init__(self, lat_max, lon_min, res_lat, res_lon, H, W):
        self.lat_max, self.lon_min = float(lat_max), float(lon_min)
        self.res_lat, self.res_lon = float(res_lat), float(res_lon)
        self.H, self.W = int(H), int(W)
        lat_c = self.lat_max - self.res_lat * self.H / 2
        self.dy_m = self.res_lat * 111320.0                                   # row 方向セル長 [m]
        self.dx_m = self.res_lon * 111320.0 * math.cos(math.radians(lat_c))   # col 方向セル長 [m]
        self.lat_min = self.lat_max - self.res_lat * (self.H - 1)
        self.lon_max = self.lon_min + self.res_lon * (self.W - 1)

    @classmethod
    def from_npz(cls, g):
        return cls(g["lat_max"], g["lon_min"], g["res_lat"], g["res_lon"], *g["dem"].shape)

    def latlon_to_rc(self, lat, lon):
        return (self.lat_max - np.asarray(lat, float)) / self.res_lat, (np.asarray(lon, float) - self.lon_min) / self.res_lon

    def rc_to_latlon(self, r, c):
        return self.lat_max - np.asarray(r, float) * self.res_lat, self.lon_min + np.asarray(c, float) * self.res_lon

    def rc_to_xy(self, r, c):
        """格子 (row,col) → 局所メートル座標 (x 東, y 北)。原点 = (row 0, col 0)。"""
        return np.asarray(c, float) * self.dx_m, -np.asarray(r, float) * self.dy_m

    def xy_to_rc(self, x, y):
        return -np.asarray(y, float) / self.dy_m, np.asarray(x, float) / self.dx_m

    def latlon_to_xy(self, lat, lon):
        return self.rc_to_xy(*self.latlon_to_rc(lat, lon))

    def xy_to_latlon(self, x, y):
        return self.rc_to_latlon(*self.xy_to_rc(x, y))


def load_inputs():
    g = np.load(OUT / "inputs.npz", allow_pickle=False)
    grid = Grid.from_npz(g)
    fac = [dict(id=str(i), name=str(n), lat=float(la), lon=float(lo), row=float(r), col=float(c), inside=bool(ins), note=str(nt))
           for i, n, la, lo, r, c, ins, nt in zip(g["fac_id"], g["fac_name"], g["fac_lat"], g["fac_lon"],
                                                  g["fac_row"], g["fac_col"], g["fac_inside"], g["fac_note"])]
    return dict(grid=grid, dem=g["dem"], T=g["t_arrive_sec"], hmax=g["hmax"], road=g["road"], building=g["building"], fac=fac)


def hillshade(dem, dy_m, dx_m, az=315.0, alt=45.0):
    z = np.nan_to_num(dem, nan=0.0)
    gy, gx = np.gradient(z, dy_m, dx_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, altr = math.radians(az), math.radians(alt)
    return np.clip(np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect), 0, 1)
