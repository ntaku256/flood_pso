"""
plot_hidaka_trend.py
実データ校正（日高川 想定最大規模・5m地形）の K スイープ結果から、
標準PSO と CCPSO2 のトレンド図を作る。

入力: results/hidaka/calib_K*_iou_ds1_seed0.json（calibrate_hidaka.py の出力）
出力: results/hidaka/trend_ds1_iou.png

主張: 次元 D=1+K² を上げるほど CCPSO2 の対PSO優位（loss差）が拡大し、
      標準PSOは劣化する（＝高次元で CCPSO2 が必要）。
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
HID = REPO / "results" / "hidaka"


def main():
    files = sorted(glob.glob(str(HID / "calib_K*_iou_ds1_seed0.json")))
    rows = []
    for f in files:
        d = json.load(open(f))
        rows.append((d["D"], d["K"], d["pso"]["loss"], d["pso"]["iou"],
                     d["ccpso2"]["loss"], d["ccpso2"]["iou"]))
    rows.sort()
    if not rows:
        print("no calib_K*_iou_ds1_seed0.json found"); return

    D = [r[0] for r in rows]
    Ks = [r[1] for r in rows]
    pso_loss = [r[2] for r in rows]; pso_iou = [r[3] for r in rows]
    cc_loss = [r[4] for r in rows]; cc_iou = [r[5] for r in rows]
    adv = [p - c for p, c in zip(pso_loss, cc_loss)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(D, pso_iou, "-o", color="C0", label="Standard PSO")
    ax[0].plot(D, cc_iou, "-s", color="C1", label="CCPSO2")
    for x, k, y in zip(D, Ks, cc_iou):
        ax[0].annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("dimension  D = 1 + K^2")
    ax[0].set_ylabel("IoU vs real hazard map")
    ax[0].set_title("5m Hidaka calibration: IoU vs D")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

    ax[1].plot(D, adv, "-D", color="C2")
    for x, a in zip(D, adv):
        ax[1].annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("dimension  D = 1 + K^2")
    ax[1].set_ylabel("CCPSO2 advantage  (PSO_loss − CCPSO2_loss)")
    ax[1].set_title("CCPSO2 advantage grows with dimension")
    ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = HID / "trend_ds1_iou.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}\n")
    print(f"{'K':>3} {'D':>5} | {'PSO_IoU':>8} {'CC_IoU':>8} | {'PSO_loss':>9} {'CC_loss':>9} {'adv(Δloss)':>11}")
    for D_, K_, pl, pi, cl, ci in rows:
        print(f"{K_:>3} {D_:>5} | {pi:>8.3f} {ci:>8.3f} | {pl:>9.4f} {cl:>9.4f} {pl-cl:>11.4f}")


if __name__ == "__main__":
    main()
