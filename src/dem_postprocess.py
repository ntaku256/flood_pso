"""
dem_postprocess.py
DEM 後処理パイプライン（arnis `src/elevation/postprocess.rs` からの移植）。

arnis(OSM→Minecraft, Rust) の標高後処理を numpy/scipy へ移植したもの。
flood_pso の DEM 取得経路（dem_parser / wakayama_pcd）には外れ値除去・継ぎ目
スパイク修復・NaN 補間が無く、docs/07 が「TerrainAnomalyRepair 未移植」と
自認していた穴を埋める。校正本体（PSO/CCPSO2/inundation）には不干渉で、
DEM をクリーンにしてから下流（NBT 生成・レンダ）へ渡す前処理層。

パイプライン順（arnis と同順）:
    filter_elevation_outliers → repair_terrain_anomalies → fill_nan_values
    （外れ値を NaN 化 → スパイク塊を中央値で修復 → 残り NaN を反復平均補間）

移植元 (arnis):
    repair_terrain_anomalies   postprocess.rs:22
    fill_nan_values            postprocess.rs:1061
    filter_elevation_outliers  postprocess.rs:1117

Rust 固有要素（rayon 並列・f32 最適化・所有権共有）は捨て、numpy ベクトル化で
等価実装している。閾値は DEM5A(5m 格子)/和歌山 1m 点群に合わせ調整可能。

無効化（A/B 比較用）:
    環境変数 FLOOD_PSO_DEM_POSTPROCESS=0  でグローバル off
    各関数/呼び出し側の引数でも個別 on/off 可
"""
from __future__ import annotations

import os
import warnings

import numpy as np


def postprocess_enabled(default: bool = True) -> bool:
    """環境変数 FLOOD_PSO_DEM_POSTPROCESS による後処理の有効/無効を返す。
    未設定なら default。"0"/"false"/"no"/"off" で無効。"""
    v = os.environ.get("FLOOD_PSO_DEM_POSTPROCESS")
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def filter_elevation_outliers(dem: np.ndarray,
                              iqr_factor: float = 3.0,
                              count_guard: float = 0.05,
                              verbose: bool = True) -> np.ndarray:
    """IQR×iqr_factor を超える極端な外れ値を NaN 化する（破損値・水面スペックル除去）。

    Q1-iqr_factor*IQR / Q3+iqr_factor*IQR の外側を外れ値とするが、超過セルが
    全有効セルの count_guard(=5%) を超えるバウンドは「実地形（谷底/山）」とみなし
    そのバウンドだけスキップ（連続した低地・海面下標高を誤消去しない）。
    NaN 化のみ行い、穴埋めは後段 fill_nan_values に任せる。

    移植元: postprocess.rs:1117 filter_elevation_outliers
    """
    h = np.asarray(dem, dtype=np.float32).copy()
    finite = np.isfinite(h)
    vals = h[finite]
    if vals.size < 4:
        return h

    q1, q3 = np.percentile(vals, [25.0, 75.0])
    iqr = float(q3 - q1)
    lo = q1 - iqr_factor * iqr
    hi = q3 + iqr_factor * iqr

    n = vals.size
    below = int(np.count_nonzero(vals < lo))
    above = int(np.count_nonzero(vals > hi))
    thr = int(n * count_guard)
    # カウントガード: 片側の超過が 5% を超えたら実地形とみなしそのバウンドは無効化
    filt_lo = 0 < below <= thr
    filt_hi = 0 < above <= thr
    if not filt_lo and not filt_hi:
        return h

    mask = finite & (((h < lo) & filt_lo) | ((h > hi) & filt_hi))
    h[mask] = np.nan
    if verbose:
        print(f"[dem_pp] outliers: filtered {int(mask.sum())} "
              f"(IQR bounds {lo:.1f}..{hi:.1f}m, lower={filt_lo}, upper={filt_hi})")
    return h


def repair_terrain_anomalies(dem: np.ndarray,
                             radius: int = 2,
                             passes: int = 10,
                             abs_threshold: float = 6.0,
                             relative_factor: float = 3.0,
                             verbose: bool = True) -> np.ndarray:
    """5x5 中央値 + MAD(中央絶対偏差) による反復的な地形異常修復。

    中心値が |center - median| > abs_threshold かつ > relative_factor × MAD の
    ときだけ近傍中央値で置換する。MAD 基準なので尾根/谷/堤防の実エッジは保存し、
    LiDAR 誤分類スパイク・DEM5A タイル継ぎ目の段差だけを除去する。
    各パスは直前パスのスナップショットを読み（走査順バイアス排除）、多画素の
    アーティファクト塊を外周から侵食する。anomaly を見つけなくなれば早期終了。

    閾値 6m/3× は arnis 既定（DEM5A 5m 格子向け）。和歌山 1m 点群の鋭い
    マルチパススパイクにも有効。中心セルは近傍統計から除外（arnis と同じ 24 近傍）。

    移植元: postprocess.rs:22 repair_terrain_anomalies
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h = np.asarray(dem, dtype=np.float32).copy()
    H, W = h.shape
    win = 2 * radius + 1
    if H < win or W < win:
        return h
    center_idx = (win * win) // 2

    total = 0
    passes_ran = 0
    for p in range(passes):
        padded = np.pad(h, radius, mode="constant", constant_values=np.nan)
        # (H, W, win, win) → (H, W, win*win)。読み取り専用 view を copy して中心を除外。
        stack = sliding_window_view(padded, (win, win)).reshape(H, W, win * win).copy()
        stack[:, :, center_idx] = np.nan  # 中心セルを近傍統計から除外
        finite_cnt = np.count_nonzero(~np.isnan(stack), axis=-1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            # 全 NaN 窓は finite_cnt<8 で後段マスク除外されるので警告は無害
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(stack, axis=-1)
            mad = np.nanmedian(np.abs(stack - median[..., None]), axis=-1)
        deviation = np.abs(h - median)
        mask = (
            (finite_cnt >= 8)
            & np.isfinite(h)
            & (deviation > abs_threshold)
            & (deviation > relative_factor * np.maximum(mad, 1.0))
        )
        n = int(mask.sum())
        if n == 0:
            break
        h[mask] = median[mask].astype(np.float32)
        total += n
        passes_ran = p + 1

    if verbose and total:
        print(f"[dem_pp] repaired {total} terrain anomalies in {passes_ran} pass(es)")
    return h


def _build_fill_mask(nan: np.ndarray,
                     skip_border: bool,
                     max_gap: int | None) -> np.ndarray | None:
    """補間を許可する NaN セルのブール格子を返す（None なら全 NaN を許可）。

    skip_border=True: 画像の縁に連結する NaN 成分（＝海・データ外の大域 void）を除外。
    max_gap: 指定時、面積が max_gap セルを超える NaN 成分も除外。
    どちらも内部の小欠損（点群取りこぼし・タイル継ぎ目）だけを埋めるための保護。
    """
    if not skip_border and max_gap is None:
        return None
    from scipy.ndimage import label
    lab, ncomp = label(nan)
    if ncomp == 0:
        return None
    sizes = np.bincount(lab.ravel())
    keep = np.ones(sizes.shape[0], dtype=bool)
    keep[0] = False  # 背景(非 NaN)
    if max_gap is not None:
        keep[1:] &= sizes[1:] <= max_gap
    if skip_border:
        border_labels = np.unique(
            np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        keep[border_labels[border_labels > 0]] = False
    return keep[lab]


def fill_nan_values(dem: np.ndarray,
                    max_iter: int = 500,
                    skip_border: bool = False,
                    max_gap: int | None = None,
                    verbose: bool = True) -> np.ndarray:
    """反復 3x3 近傍平均で NaN を補間する（最近傍コピーの方向性アーティファクト回避）。

    各反復で「現在のスナップショット」から 3x3 平均を取り、有効近傍が 1 個以上ある
    NaN セルだけを埋める。NaN が消えるか、これ以上埋められなくなるまで反復。
    エッジタイルの NaN パディング・点群取りこぼしを滑らかに拡散補間する。

    skip_border=True で、画像の縁に連結する大域 NaN 成分（沿岸 DEM の海・データ外）
    は埋めずに残す。御坊のように海域が NaN の場合、これを付けないと海を陸の標高で
    塗りつぶしてしまう（実測: 縁連結 65万セル vs 内部欠損 1万セル）。
    max_gap でも面積上限を別途指定可能。

    移植元: postprocess.rs:1061 fill_nan_values（+ 沿岸 DEM 向けに skip_border 追加）
    """
    from scipy.ndimage import uniform_filter

    h = np.asarray(dem, dtype=np.float32).copy()
    allowed = _build_fill_mask(np.isnan(h), skip_border, max_gap)

    it = 0
    while it < max_iter:
        nan = np.isnan(h)
        target = nan if allowed is None else (nan & allowed)
        if not target.any():
            break
        valid = (~nan).astype(np.float32)
        filled0 = np.where(nan, np.float32(0.0), h)
        # 3x3 ボックス和 = uniform_filter(平均) × 9。NaN は 0 寄与でカウントから除外。
        ssum = uniform_filter(filled0, size=3, mode="constant", cval=0.0) * 9.0
        cnt = uniform_filter(valid, size=3, mode="constant", cval=0.0) * 9.0
        fillable = target & (cnt > 0.5)
        if not fillable.any():
            break  # 許可セルに連結する有効近傍がもう無い
        h[fillable] = (ssum[fillable] / cnt[fillable]).astype(np.float32)
        it += 1

    if verbose:
        rem = int(np.isnan(h).sum())
        note = " (border/large voids kept as NaN)" if allowed is not None else ""
        print(f"[dem_pp] fill_nan: {it} iter(s), remaining NaN={rem}{note}")
    return h


def postprocess_dem(dem: np.ndarray,
                    outliers: bool = False,
                    anomalies: bool = True,
                    fill: bool = True,
                    abs_threshold: float = 6.0,
                    relative_factor: float = 3.0,
                    iqr_factor: float = 3.0,
                    fill_skip_border: bool = False,
                    fill_max_gap: int | None = None,
                    verbose: bool = True) -> np.ndarray:
    """DEM 後処理パイプライン本体（arnis 同順: outlier → anomaly → fill）。

    各段は引数で個別に on/off 可能。戻り値は float32。入力は破壊しない。

    outliers は既定 **False**。arnis のグローバル IQR 外れ値除去は「分布の外側
    <5% は破損データ」と仮定するが、御坊 DEM は平野(中央値≈51m)から山地(最大≈488m)
    まで標高が連続分布し、山頂は全体の 1% 弱しか占めない。このため IQR 上側バウンド
    (Q3+3*IQR≈348m) が実在の山頂 ~2.3万セルを誤って NaN 化してしまう（実測で確認）。
    DEM5A は NoData を parse 段で既に NaN 化済みで、残る突出値はローカルな
    LiDAR スパイク/継ぎ目段差なので、グローバル IQR でなくローカルな MAD 修復
    (repair_terrain_anomalies) で十分かつ安全。明確な大域破損が分かっている時だけ
    outliers=True を opt-in する。
    """
    h = np.asarray(dem, dtype=np.float32)
    if verbose:
        before_nan = int(np.isnan(h).sum())
        print(f"[dem_pp] start shape={h.shape} NaN={before_nan}")
    if outliers:
        h = filter_elevation_outliers(h, iqr_factor=iqr_factor, verbose=verbose)
    if anomalies:
        h = repair_terrain_anomalies(
            h, abs_threshold=abs_threshold, relative_factor=relative_factor,
            verbose=verbose)
    if fill:
        h = fill_nan_values(h, skip_border=fill_skip_border,
                            max_gap=fill_max_gap, verbose=verbose)
    return np.asarray(h, dtype=np.float32)


def _selftest() -> None:
    """合成 DEM（傾斜地形＋スパイク＋NaN 穴）で各段の効果を検証する。
    実 DEM データ（git-LFS）不要のサニティチェック。"""
    rng = np.random.default_rng(0)
    H, W = 80, 100
    yy, xx = np.mgrid[0:H, 0:W]
    base = 10.0 + 0.15 * xx + 0.05 * yy  # 緩傾斜の真地形
    base += rng.normal(0, 0.2, size=base.shape)  # 微小ノイズ
    dem = base.astype(np.float32)

    # 1) 大スパイク ±40m（ローカル MAD 修復で捕まる）
    sy = rng.integers(2, H - 2, size=30)
    sx = rng.integers(2, W - 2, size=30)
    dem[sy, sx] += rng.choice([+40.0, -40.0], size=30)
    big_truth = base[sy, sx]

    # 2) 中規模スパイク ±9m（継ぎ目段差相当, MAD 修復で捕まる）
    my = rng.integers(2, H - 2, size=25)
    mx = rng.integers(2, W - 2, size=25)
    dem[my, mx] += rng.choice([+9.0, -9.0], size=25)
    mid_truth = base[my, mx]

    # 3) 内部 NaN 穴（点群取りこぼし相当）
    dem[20:25, 30:36] = np.nan
    n_nan_in = int(np.isnan(dem).sum())

    # 既定パイプライン（outliers=False: MAD 修復 + NaN 補間のみ）
    out = postprocess_dem(dem, verbose=True)

    big_err = np.abs(out[sy, sx] - big_truth)
    mid_err = np.abs(out[my, mx] - mid_truth)
    n_nan_out = int(np.isnan(out).sum())
    base_err = np.nanmax(np.abs(out - base.astype(np.float32)))

    print("--- selftest: default pipeline (MAD repair + NaN fill) ---")
    print(f"big spikes(±40m) max residual: {big_err.max():.2f} m (< ~3)")
    print(f"mid spikes(±9m)  max residual: {mid_err.max():.2f} m (< ~3)")
    print(f"NaN before={n_nan_in} after={n_nan_out} (should be 0)")
    print(f"overall max |out-truth|: {base_err:.2f} m (real terrain preserved)")
    ok = big_err.max() < 3.0 and mid_err.max() < 3.0 and n_nan_out == 0

    # 4) 外れ値除去 opt-in の動作確認（大域破損: 数セルだけ +9999m）
    corrupt = base.astype(np.float32).copy()
    cy, cx = rng.integers(2, H - 2, size=5), rng.integers(2, W - 2, size=5)
    corrupt[cy, cx] = 9999.0
    fixed = filter_elevation_outliers(corrupt, verbose=False)
    n_corrupt_left = int(np.isfinite(fixed[cy, cx]).sum())
    print("--- selftest: outlier opt-in (gross corruption) ---")
    print(f"corrupt cells remaining finite: {n_corrupt_left} (should be 0)")
    ok = ok and n_corrupt_left == 0

    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    _selftest()
