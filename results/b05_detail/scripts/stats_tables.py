"""a〜e の数値を退避済み生データから再計算して summary.json / summary.md を作る。

生データ (results/b05_detail/raw/ に退避ディレクトリからコピー):
  a. seeds21_all.json      — 21 シード IoU (修正PSO / CCPSO2現行 / CCPSO2忠実版)
  b. holdout.json          — 市松 holdout の train/test/full
  c. hand_oracle.json / iou_ceiling.json / box_verify.json — IoU ギャップ 5 分解
  d. box_verify.json / hand_optimize.json / hand_optimize20.json — 値域・パラメータ化の比較
  e. src_reg.json / src_reg_final.json — 水源マスク × 正則化 λ の 7 設定
  f. s5_quality.json       — 本ディレクトリで再計算した水面品質

f は s5_quality.py が先に走っている必要がある (無ければ f を省いて出力)。
"""
import json
import os
import sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_detail import OUT, RESCUE

RAW = f"{OUT}/raw"
os.makedirs(RAW, exist_ok=True)
COPY = ["seeds21_all.json", "holdout.json", "hand_oracle.json", "iou_ceiling.json",
        "box_verify.json", "hand_optimize.json", "hand_optimize20.json",
        "src_reg.json", "src_reg_final.json", "multiseed.json",
        "downscale_baseline.json"]
for fn in COPY:
    s, d = f"{RESCUE}/{fn}", f"{RAW}/{fn}"
    if os.path.exists(s) and not os.path.exists(d):
        open(d, "wb").write(open(s, "rb").read())

J = lambda fn: json.load(open(f"{RAW}/{fn}"))
S = {}


def a12(x, y):
    """Vargha-Delaney A12 = P(X>Y) + 0.5 P(X=Y)。large 閾値 0.71。"""
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    eq = sum((xi == y).sum() for xi in x)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def pair(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    w, p = wilcoxon(a, b)
    win = int((a > b).sum())
    return dict(pair=name, delta=float(a.mean() - b.mean()),
                win=win, lose=int(len(a) - win - int((a == b).sum())),
                W=float(w), p=float(p), A12=float(a12(a, b)))


# ── a. 21 シード ───────────────────────────────────────────────
s21 = J("seeds21_all.json")
LBL = {"PSO_clip": "修正PSO (clip+速度クランプ)", "CC_current": "CCPSO2 現行",
       "CC_faithful": "CCPSO2 忠実版"}
S["a_seeds21"] = {
    "n_seeds": len(s21["PSO_clip"]),
    "stats": {LBL[k]: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                           min=float(np.min(v)), max=float(np.max(v)),
                           median=float(np.median(v))) for k, v in s21.items()},
    "pairs": [pair("CCPSO2現行 vs 修正PSO", s21["CC_current"], s21["PSO_clip"]),
              pair("CCPSO2忠実版 vs 修正PSO", s21["CC_faithful"], s21["PSO_clip"]),
              pair("CCPSO2現行 vs 忠実版", s21["CC_current"], s21["CC_faithful"])],
    "note": "PSO の max がCCPSO2現行の min を上回るため完全分離ではない "
            f"(PSO max {max(s21['PSO_clip']):.4f} > CC min {min(s21['CC_current']):.4f})",
}

# ── b. 市松 holdout ─────────────────────────────────────────────
ho = J("holdout.json")
S["b_holdout"] = {}
for k, v in ho.items():
    tr, te, fu = np.array(v["train"]), np.array(v["test"]), np.array(v["full"])
    S["b_holdout"][k] = dict(
        n=len(tr),
        train_mean=float(tr.mean()), train_sd=float(tr.std(ddof=1)),
        test_mean=float(te.mean()), test_sd=float(te.std(ddof=1)),
        full_mean=float(fu.mean()), full_sd=float(fu.std(ddof=1)),
        gap=float(tr.mean() - te.mean()))
cc, ps = S["b_holdout"]["CCPSO2"], S["b_holdout"]["修正PSO"]
S["b_holdout"]["_summary"] = dict(
    gap_ratio=float(ps["gap"] / cc["gap"]),
    test_sd_ratio=float(ps["test_sd"] / cc["test_sd"]),
    out_of_sample_advantage=float(cc["test_mean"] - ps["test_mean"]),
    in_sample_advantage=float(cc["train_mean"] - ps["train_mean"]))

# ── c. IoU ギャップの 5 分解 ────────────────────────────────────
orc = J("hand_oracle.json")
bv = J("box_verify.json")
base = float(np.mean(bv["2.0"]["CCPSO2"]))          # 実測 (絶対標高[1,10]・bbox・K=16・3シード)
o16, o32 = orc["16"], orc["32"]
stages = [
    ("探索律速 (実測 → 同条件 K=16 オラクル)", o16["abs_bounded"] - base, o16["abs_bounded"]),
    ("パラメータ化律速 (絶対標高[1,10] → HAND[0,20])", o16["hand_20"] - o16["abs_bounded"], o16["hand_20"]),
    ("値域律速 (HAND[0,20] → 値域なし)", o16["hand_free"] - o16["hand_20"], o16["hand_free"]),
    ("解像度律速 (K=16 → K=32)", o32["hand_free"] - o16["hand_free"], o32["hand_free"]),
    ("到達不能 (オラクルでも届かない)", 1.0 - o32["hand_free"], 1.0),
]
gap = 1.0 - base
ms = J("multiseed.json")
abs10 = [v[0] for v in ms["cc_fixed16"]] + [v[0] for v in ms["cc_fixed16_small"]]
S["c_gap"] = {
    "base_iou": base,
    "base_desc": "CCPSO2 / 絶対標高[1,10] / bbox水源 / K=16 / 5000評価 / 3シード (box_verify.json)",
    "gap": gap,
    "stages": [dict(stage=n, width=float(w), share=float(w / gap), reach=float(r))
               for n, w, r in stages],
    "sum_check": float(sum(w for _, w, _ in stages)),
    "achievement": {
        "絶対標高[1,10]・bbox (3シード)": dict(meas=base, oracle=o16["abs_bounded"],
                                           rate=base / o16["abs_bounded"]),
        "絶対標高[1,10]・bbox (multiseed 10走)": dict(meas=float(np.mean(abs10)),
                                                  oracle=o16["abs_bounded"],
                                                  rate=float(np.mean(abs10)) / o16["abs_bounded"]),
        "HAND[-4,20]・bbox (21シード)": dict(meas=float(np.mean(s21["CC_current"])),
                                          oracle=o16["hand_20"],
                                          rate=float(np.mean(s21["CC_current"])) / o16["hand_20"]),
        "HAND[-4,20]・実河道 (3シード)": dict(
            meas=json.load(open(f"{RAW}/src_reg.json"))["実河道水源・正則化なし"]["iou_mean"],
            oracle=o16["hand_20"],
            rate=json.load(open(f"{RAW}/src_reg.json"))["実河道水源・正則化なし"]["iou_mean"] / o16["hand_20"]),
    },
    "caveat": "オラクルは全て連結性制約なし・実測は連結性あり。"
              "『探索律速』の一部は探索失敗ではなく連結性制約のコスト (連結性ありオラクルは未測定)。",
}

# ── d. 値域・パラメータ化の比較 ─────────────────────────────────
ho10 = J("hand_optimize.json")
ho20 = J("hand_optimize20.json")
rows = [("絶対標高 [1,10]  (w∈[3,8], dh∈[-2,2])", 9.0, True, bv["2.0"], o16["abs_bounded"]),
        ("HAND [-2,10]     (d0∈[0,8], dd∈[-2,2])", 12.0, True, ho10, o16["hand_10"]),
        ("HAND [-4,20]     (d0∈[0,16], dd∈[-4,4])", 24.0, True, ho20, o16["hand_20"]),
        ("絶対標高 [-17,28] (w∈[3,8], dh∈[-20,20])", 45.0, False, bv["20.0"], o16["abs_free"])]
S["d_bounds"] = []
for name, width, ok, d, oracle in rows:
    c, p = np.array(d["CCPSO2"]), np.array(d["PSO_clip"])
    S["d_bounds"].append(dict(
        cond=name, width_m=width, physically_valid=bool(ok),
        ccpso2=float(c.mean()), ccpso2_sd=float(c.std(ddof=1)),
        pso_clip=float(p.mean()), pso_clip_sd=float(p.std(ddof=1)),
        pso_periodic=float(np.mean(d["PSO_periodic"])),
        advantage=float(c.mean() - p.mean()),
        win=int((c > p).sum()), lose=int((c < p).sum()),
        separated=bool(c.min() > p.max()),
        oracle_K16=float(oracle), achievement=float(c.mean() / oracle)))

# ── e. 水源マスク × λ ───────────────────────────────────────────
sr = J("src_reg.json")
srf = J("src_reg_final.json")
ORDER = [("bbox水源・正則化なし (現行)", "bbox", 0.0, sr),
         ("bbox水源・λ=0.001", "bbox", 0.001, sr),
         ("bbox水源・λ=0.01", "bbox", 0.01, sr),
         ("bbox水源・λ=0.05", "bbox", 0.05, sr),
         ("実河道水源・正則化なし", "実河道", 0.0, sr),
         ("★実河道水源・λ=0.001 (推奨最終)", "実河道", 0.001, srf),
         ("実河道水源・λ=0.01", "実河道", 0.01, srf)]
S["e_src_reg"] = []
for key, src, lam, blob in ORDER:
    v = blob[key]
    S["e_src_reg"].append(dict(
        source=src, lam=lam, iou=v["iou_mean"], iou_sd=v["iou_sd"],
        ncomp=v["ncomp"], stair_rate=v["steps"] / v["pairs"],
        note="1m階段は HAND 水深場 tf 上で測ったもの (WSE ではない) — f 表を参照"))
S["e_src_reg_cells"] = {"bbox": 198791, "実河道": 2830, "grid": 1491 * 2241}

# ── f. 水面品質 (本ディレクトリで再計算) ────────────────────────
sq = f"{OUT}/s5_quality.json"
if os.path.exists(sq):
    S["f_surface_quality"] = json.load(open(sq))

# ── 参考: 素朴ダウンスケール基準 ────────────────────────────────
S["ref_downscale"] = J("downscale_baseline.json")

json.dump(S, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print(f"wrote {OUT}/summary.json")
