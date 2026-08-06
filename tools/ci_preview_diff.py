#!/usr/bin/env python3
"""ci_preview_diff.py — CI プレビューのレンダとゴールデン画素差分（tizucraft から移植）。

生成した structure NBT を俯瞰プレビュー画像化し、コミット済みゴールデン画像との
画素差分率を出す。NBT のバイト非決定（.mca timestamp / gzip mtime）はレンダ画素に
影響しないので、画素差分は決定的。

  python tools/ci_preview_diff.py --render "results/nbt/hd/*_cicrop.nbt" --out preview.png
  python tools/ci_preview_diff.py --compare preview.png tests/golden/ci_crop.png \
      --diff-out diff.png --threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))   # flood_pso は src/ を直接 import する構成


def _resolve_nbt(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    # グロブ（Makefile から tag のワイルドカードで渡ってくる）
    cands = sorted(Path(REPO).glob(spec)) if not p.is_absolute() else sorted(Path("/").glob(spec.lstrip("/")))
    if not cands:
        sys.exit(f"[preview] NBT が無い: {spec}")
    if len(cands) > 1:
        print(f"[preview] 複数一致 → 先頭を使用: {[c.name for c in cands]}")
    return cands[0]


def cmd_render(spec: str, out: str) -> None:
    from nbt_preview import render_topdown
    p = _resolve_nbt(spec)
    im, info = render_topdown(str(p), verbose=False)
    im.save(out)
    print(f"[preview] {p.name} → {out} {im.size} "
          f"terrain={info['terrain_cells']} water={info['water_cells']}")


def cmd_compare(a: str, b: str, diff_out: str | None, threshold: float) -> None:
    import numpy as np
    from PIL import Image
    if not Path(b).exists():
        print(f"[preview] ゴールデン未作成: {b} → 今回のレンダで初期化してください "
              f"(make ci-golden もしくは cp {a} {b})")
        sys.exit(0)
    ia = np.asarray(Image.open(a).convert("RGB"), np.int16)
    ib = np.asarray(Image.open(b).convert("RGB"), np.int16)
    if ia.shape != ib.shape:
        print(f"[preview] サイズ相違 {ia.shape} vs {ib.shape} → 差分率 100.0%（閾値 {threshold}%）")
        sys.exit(1)
    mask = np.abs(ia - ib).max(axis=2) > 8      # 微小な圧縮/丸め差は無視
    rate = float(mask.mean()) * 100
    print(f"[preview] 画素差分率 {rate:.3f}%（閾値 {threshold}%）")
    if diff_out:
        vis = ia.copy()
        vis[mask] = [255, 0, 60]
        Image.fromarray(vis.astype(np.uint8)).save(diff_out)
    sys.exit(0 if rate <= threshold else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", default=None, help="レンダする NBT（グロブ可）")
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("PREVIEW", "GOLDEN"))
    ap.add_argument("--diff-out", default=None, help="差分可視化 PNG の出力先")
    ap.add_argument("--threshold", type=float, default=0.5, help="許容画素差分率[%%]")
    args = ap.parse_args()
    if args.render:
        cmd_render(args.render, args.out)
    elif args.compare:
        cmd_compare(args.compare[0], args.compare[1], args.diff_out, args.threshold)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
