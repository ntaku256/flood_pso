#!/usr/bin/env python3
"""ci_fixture.py — CI プレビュー用の *完全合成* fixture を生成する。

和歌山 LiDAR / FGD（基本測量成果）/ GSI オルソ等の外部データを一切使わず、
決定的な数式だけで極小の DEM（grd テキスト）を生成する。再配布・ライセンス上の
懸念がゼロで、ネットワーク不要・小さい・決定的なので golden 回帰テストに向く。

  python tools/ci_fixture.py stage    # 合成 grd を data_cache/ へ生成（CI/ローカルビルド前）

生成 DEM は「なだらかな斜面 + ガウス丘 + 東西の河谷（低地=水が溜まる）」で、
地形分類・水面・崖/斜面・enhanced ブロック化の主要経路を exercise する。
data_cache/ は .gitignore 対象なので、この合成 grd はリポジトリに残らない（毎回再生成）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GRD = REPO / "data_cache" / "wakayama_lidar" / "cifix_grd.txt"

# 合成 DEM 諸元。ここを変えたら Makefile の CI_CLAT/CI_CLON と tests/golden を更新すること。
N = 200
STEP = 1.0 / 1.5              # ≈0.667m（build の --scale 1.5 の res に一致）
E0, N0 = -76000.0, -239000.0  # EPSG:6674 第VI系（和歌山）。御坊付近の適当な原点。
# 生成 DEM の中心 lat/lon（Makefile の CI_CLAT/CI_CLON と一致させる。stage で検算する）
CENTER_LAT = 33.842123
CENTER_LON = 135.179515


def synth_grd_text() -> str:
    """id,E,N,Z（EPSG:6674[m]）の grd テキストを決定的に生成する。"""
    lines = []
    idx = 1
    for i in range(N):
        for j in range(N):
            E = E0 + j * STEP
            Nc = N0 - i * STEP
            x = j / (N - 1)
            y = i / (N - 1)
            Z = (5.0 + 10.0 * x
                 + 15.0 * np.exp(-(((x - 0.7) ** 2 + (y - 0.3) ** 2) / 0.02))  # 丘
                 - 8.0 * np.exp(-((y - 0.55) ** 2) / 0.004))                    # 東西の河谷
            lines.append(f"{idx},{E:.2f},{Nc:.2f},{Z:.2f}")
            idx += 1
    return "\n".join(lines) + "\n"


def cmd_stage(_args) -> None:
    GRD.parent.mkdir(parents=True, exist_ok=True)
    GRD.write_text(synth_grd_text(), encoding="utf-8", newline="")
    # 原点から中心 lat/lon を検算（Makefile 定数とずれたら気づけるように）。
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:6674", "EPSG:4326", always_xy=True)
        cE = E0 + (N - 1) / 2 * STEP
        cN = N0 - (N - 1) / 2 * STEP
        clon, clat = tr.transform(cE, cN)
        ok = abs(clat - CENTER_LAT) < 1e-5 and abs(clon - CENTER_LON) < 1e-5
        print(f"[ci-fixture] stage: {GRD.relative_to(REPO)} ({N}x{N} pts)  "
              f"center lat={clat:.6f} lon={clon:.6f}  {'ok' if ok else 'WARN 中心不一致'}")
        if not ok:
            raise SystemExit("[ci-fixture] CENTER_LAT/LON が原点と不一致 — Makefile の CI_CLAT/CI_CLON も更新すること")
    except ImportError:
        print(f"[ci-fixture] stage: {GRD.relative_to(REPO)} ({N}x{N} pts)  (pyproj 無しのため中心検算スキップ)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage", help="合成 grd を data_cache/ へ生成")
    args = ap.parse_args()
    {"stage": cmd_stage}[args.cmd](args)


if __name__ == "__main__":
    main()
