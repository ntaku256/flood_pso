"""b05_slides 共通: フォント・配色・出力先。"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

OUT = "/home/ntaku/laravel-project/flood_pso/results/b05_slides"
FP = "/home/ntaku/laravel-project/flood_pso"
R8 = f"{FP}/data_cache/wakayama_r8_tsunami"
RESCUE = "/home/ntaku/research-rescue-20260728"

for p in ["/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/meiryob.ttc"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["Meiryo", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
# dataviz 参照パレット (light)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", muted="#8a8984", surface="#fcfcfb")
METHOD_COLOR = {"修正PSO": C["orange"], "CCPSO2 (現行)": C["blue"], "CCPSO2 (忠実版)": C["aqua"]}
