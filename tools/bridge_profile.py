#!/usr/bin/env python3
"""bridge_profile.py — BRIDGE_DUMP の npz から橋デッキ高プロファイル図と defect 判定を出す。

（リポジトリ root にあった未追跡の scratch_bridge_profile.py を tools/ の正式版にしたもの。
  引数・PASS/FAIL 判定・JSON 出力・終了コードを追加した。）

────────────────────────────────────────────────────────────────────────
BRIDGE_DUMP とは
────────────────────────────────────────────────────────────────────────
src/terrain_render.py::add_bridge_blocks は環境変数

    BRIDGE_DUMP=<out.npz>     … 橋チェーンごとの高さプロファイルを npz に落とす
    BRIDGE_DEBUG=1            … 同じ内容を標準出力に1行ずつ出す

を見ている（env-gate なので本番のオーバヘッドは 0）。npz には橋チェーンごとに
station 沿いの

    dy    … 実際に置くデッキ Y
    terr  … 直下の地形 Y
    floor … 橋脚の底 Y（川底 or 地表）
    wsurf … 水面 Y（水が無い station は -9999）

が入る。**ワールドを書き出さなくても**（＝Anvil/NBT を作る前の段階で）橋の
埋没・水没・段差が数値で分かるので、橋のアルゴリズムを直すときの最速ループになる。

────────────────────────────────────────────────────────────────────────
使い方
────────────────────────────────────────────────────────────────────────
  # 1) 生成時に dump を出す（小さい crop で十分。Makefile の crop ターゲット等）
  BRIDGE_DUMP=/tmp/bd.npz make crop
  #  もしくは直接:
  BRIDGE_DUMP=/tmp/bd.npz .venv/bin/python src/make_nbt_hd.py ... --bridges-json ...

  # 2) 図と判定を出す
  .venv/bin/python tools/bridge_profile.py /tmp/bd.npz -o /tmp/bd.png

  # 3) 修正前後を比べる
  BRIDGE_DUMP=/tmp/bd_before.npz make crop      # 修正前
  ...code fix...
  BRIDGE_DUMP=/tmp/bd_after.npz  make crop      # 修正後
  .venv/bin/python tools/bridge_profile.py /tmp/bd_after.npz --compare /tmp/bd_before.npz

  # 4) CI/自動チェック的に使う（defect があれば exit 1）
  .venv/bin/python tools/bridge_profile.py /tmp/bd.npz --fail-on-defect --no-plot

────────────────────────────────────────────────────────────────────────
判定（defect）
────────────────────────────────────────────────────────────────────────
  buried     : dy <  terr        デッキが地形に潜って見えない
  submerged  : dy <  wsurf       デッキが水没
  flush      : dy == terr        地形と面一（後勝ちで見えるが橋に見えない）
  maxstep    : |diff(dy)| の最大  隣接 station の段差（大きいと階段状の崩れ）
  low_pier   : floor > dy        橋脚の底がデッキより上（不整合）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_recs(path) -> list[dict]:
    z = np.load(str(path), allow_pickle=True)
    key = "recs" if "recs" in z.files else z.files[0]
    return [dict(r) for r in z[key]]


def analyse(r: dict) -> dict:
    dy = np.asarray(r["dy"], float)
    terr = np.asarray(r["terr"], float)
    flr = np.asarray(r["floor"], float)
    ws = np.asarray(r.get("wsurf", np.full_like(dy, -9999.0)), float)
    has_w = ws > -9000
    step = np.abs(np.diff(dy)) if len(dy) > 1 else np.array([0.0])
    return dict(
        kind=str(r.get("kind", "?")),
        n=int(len(dy)),
        total=float(r.get("total", 0.0)),
        startS=float(r.get("startS", 0.0)),
        endS=float(r.get("endS", 0.0)),
        min_deck=float(r.get("min_deck", 0.0)),
        deck_min=float(dy.min()), deck_max=float(dy.max()),
        buried=int((dy < terr).sum()),
        flush=int((dy == terr).sum()),
        submerged=int((has_w & (dy < ws)).sum()),
        low_pier=int((flr > dy).sum()),
        maxstep=float(step.max()),
        n_step_ge3=int((step >= 3).sum()),
        water_stations=int(has_w.sum()),
    )


def fmt_row(a: dict) -> str:
    bad = a["buried"] or a["submerged"] or a["maxstep"] >= 3 or a["low_pier"]
    return (f"  {'FAIL' if bad else 'ok  '} {a['kind']:4} L={a['total']:6.0f}b "
            f"n={a['n']:4d} deckY[{a['deck_min']:.0f},{a['deck_max']:.0f}] "
            f"startS={a['startS']:.0f} endS={a['endS']:.0f} "
            f"maxstep={a['maxstep']:.0f}b(>=3:{a['n_step_ge3']}) "
            f"buried={a['buried']} flush={a['flush']} "
            f"submerged={a['submerged']}/{a['water_stations']} "
            f"low_pier={a['low_pier']}")


def plot(recs, out, top=None, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted(recs, key=lambda r: -float(r.get("total", 0)))
    if top:
        recs = recs[:top]
    n = len(recs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n), squeeze=False)
    for ax, r in zip(axes[:, 0], recs):
        st = np.asarray(r["station"], float)
        dy = np.asarray(r["dy"], float)
        terr = np.asarray(r["terr"], float)
        flr = np.asarray(r["floor"], float)
        ax.plot(st, dy, "-", color="#d62728", lw=2.0, label="deck Y")
        ax.plot(st, terr, "-", color="#2ca02c", lw=1.2, label="terrain Y")
        ax.plot(st, flr, ":", color="#8c564b", lw=1.0, label="pier floor Y")
        lo, hi = dy.min() - 2, dy.max() + 2
        ws = np.asarray(r.get("wsurf", np.full_like(dy, -9999.0)), float)
        wm = ws > -9000
        if wm.any():
            ax.plot(st[wm], ws[wm], "-", color="#1f77b4", lw=1.4, label="water surf")
            sub = wm & (dy < ws)
            if sub.any():
                ax.fill_between(st, lo, hi, where=sub, color="blue", alpha=0.18,
                                label="SUBMERGED")
        buried = dy < terr
        if buried.any():
            ax.fill_between(st, lo, hi, where=buried, color="red", alpha=0.14,
                            label="BURIED (deck<terrain)")
        step = np.abs(np.diff(dy)) if len(dy) > 1 else np.array([0.0])
        for k in np.nonzero(step >= 3)[0]:
            ax.axvline(st[k], color="orange", lw=0.8, ls="--", alpha=0.7)
        a = analyse(r)
        ax.set_title(
            f"{a['kind']} L={a['total']:.0f}b startS={a['startS']:.0f} endS={a['endS']:.0f} "
            f"min_deck={a['min_deck']:.0f} maxstep={a['maxstep']:.0f}b "
            f"deckY[{a['deck_min']:.0f},{a['deck_max']:.0f}] "
            f"buried={a['buried']}/{a['n']} submerged={a['submerged']}", fontsize=9)
        ax.set_xlabel("station (block)")
        ax.set_ylabel("Y")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return n


def main():
    ap = argparse.ArgumentParser(
        description="BRIDGE_DUMP npz → デッキ高プロファイル図 + 埋没/水没/段差の判定",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("npz", help="BRIDGE_DUMP で出力した npz")
    ap.add_argument("-o", "--out", default=None, help="出力 PNG（既定 <npz>_profile.png）")
    ap.add_argument("--top", type=int, default=None, help="長い順に N チェーンだけ描く")
    ap.add_argument("--min-len", type=float, default=0.0, help="この長さ[block]未満は無視")
    ap.add_argument("--only", choices=["main", "ramp"], default=None, help="種別で絞る")
    ap.add_argument("--no-plot", action="store_true", help="図を作らず判定だけ出す")
    ap.add_argument("--json", dest="json_out", default=None, help="判定を JSON で書き出す")
    ap.add_argument("--compare", default=None, help="比較する別の npz（before）")
    ap.add_argument("--fail-on-defect", action="store_true",
                    help="buried/submerged/maxstep>=3/low_pier があれば exit 1")
    args = ap.parse_args()

    recs = load_recs(args.npz)
    if args.only:
        recs = [r for r in recs if str(r.get("kind")) == args.only]
    if args.min_len:
        recs = [r for r in recs if float(r.get("total", 0)) >= args.min_len]
    if not recs:
        sys.exit("該当するチェーンがありません（--only/--min-len を確認）")
    recs.sort(key=lambda r: -float(r.get("total", 0)))
    stats = [analyse(r) for r in recs]

    print(f"{Path(args.npz).name}: {len(recs)} chains")
    for a in stats:
        print(fmt_row(a))
    tot = dict(
        chains=len(stats),
        buried=sum(a["buried"] for a in stats),
        submerged=sum(a["submerged"] for a in stats),
        flush=sum(a["flush"] for a in stats),
        low_pier=sum(a["low_pier"] for a in stats),
        maxstep=max(a["maxstep"] for a in stats),
        chains_with_defect=sum(1 for a in stats
                               if a["buried"] or a["submerged"]
                               or a["maxstep"] >= 3 or a["low_pier"]),
    )
    print(f"TOTAL: chains={tot['chains']} buried={tot['buried']} "
          f"submerged={tot['submerged']} flush={tot['flush']} "
          f"low_pier={tot['low_pier']} maxstep={tot['maxstep']:.0f}b "
          f"→ defective chains {tot['chains_with_defect']}/{tot['chains']}")

    if args.compare:
        before = [analyse(r) for r in load_recs(args.compare)]
        b = dict(buried=sum(a["buried"] for a in before),
                 submerged=sum(a["submerged"] for a in before),
                 maxstep=max(a["maxstep"] for a in before) if before else 0,
                 chains=len(before))
        print(f"COMPARE vs {Path(args.compare).name}: "
              f"chains {b['chains']}→{tot['chains']}  "
              f"buried {b['buried']}→{tot['buried']}  "
              f"submerged {b['submerged']}→{tot['submerged']}  "
              f"maxstep {b['maxstep']:.0f}→{tot['maxstep']:.0f}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"file": str(args.npz), "total": tot, "chains": stats},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.json_out}")

    if not args.no_plot:
        out = args.out or str(args.npz).replace(".npz", "_profile.png")
        n = plot(recs, out, top=args.top)
        print(f"saved {out}  ({n} chains)")

    if args.fail_on_defect and tot["chains_with_defect"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
