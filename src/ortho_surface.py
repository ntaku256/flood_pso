"""
ortho_surface.py
GSI シームレス空中写真（オルソ RGB）から、地表ブロック種別マップを作る。

「衛星/航空写真の表面を Minecraft 地表に反映」する機能。色を **viewer が認識する
バニラブロック**（grass/sand/gravel/stone/water/bedrock）へヒューリスティック分類する
ことで、NBT は Minecraft 互換のまま & flood_pso_viewer でも正しい色で表示される。
（より豊かなパレットは viewer の voxel.rs 拡張が必要なので別軸。）

使い方:
  from ortho_surface import ortho_surface_grid
  surf = ortho_surface_grid(dst_meta, zoom=18)   # object 配列 (H,W) のパレットキー
  # dst_meta: {'lat_max','lon_min','res_lat','res_lon','shape':(H,W)}

分類対象キー（terrain_render / nbt_export のパレットに存在し、viewer も色を持つ）:
  grass(緑=植生)  sand(タン=裸地/畑)  gravel(灰茶=道路)  stone(灰=建物/コンクリ)
  water(青=水域)  bedrock(暗=濃い屋根/影)
"""
from __future__ import annotations

import numpy as np


from block_palette import MATCH_KEYS, rgb as _key_rgb

# 最近傍マッチ用アンカー（~80 単色バニラブロック）
_ANCHOR_KEYS = np.array(MATCH_KEYS, dtype=object)
_ANCHOR_RGB = np.array([_key_rgb(k) for k in MATCH_KEYS], dtype=np.float32)  # (M,3)


def enhance_rgb(rgb: np.ndarray, *, saturation: float = 1.6,
                white_balance: float = 1.0, contrast: float = 1.08) -> np.ndarray:
    """
    オルソの青み・霞かぶりと低彩度を補正（白黒っぽさ対策）。タイル間でロバストにするため、
    per-channel ストレッチ（赤の弱い海沿いタイルで赤を過増幅しピンク化した）ではなく
    **gray-world ホワイトバランス（比例的・過増幅しない）＋彩度＋緩いコントラスト**を使う。
      white_balance: gray-world 補正の強さ 0..1（1=チャンネル平均を完全中和）
      saturation   : 彩度倍率（>1 で鮮やか）
      contrast     : 128 中心のコントラスト
    """
    arr = rgb.astype(np.float32)
    if white_balance > 0:
        means = arr.reshape(-1, 3).mean(0)
        gray = float(means.mean())
        scale = 1.0 + white_balance * (gray / np.maximum(means, 1e-3) - 1.0)
        arr *= scale
    if saturation != 1.0:
        luma = arr @ np.array([0.299, 0.587, 0.114], np.float32)
        arr = luma[..., None] + (arr - luma[..., None]) * saturation
    arr = (arr - 128.0) * contrast + 128.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def classify_rgb_to_palette(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (H, W, 3) uint8 → object 配列 (H, W) のパレットキー。

    ~80 単色バニラブロックへの**最近傍カラーマッチ**（RGB ユークリッド）。
    写真の色をそのままブロックへ写像するので、地表が写真モザイクになる。
    水/氷は洪水・海レイヤが別途生成するためアンカーから除外している。
    """
    h, w, _ = rgb.shape
    px = rgb.reshape(-1, 3).astype(np.float32)         # (P,3)
    best = np.full(px.shape[0], np.inf, dtype=np.float32)
    idx = np.zeros(px.shape[0], dtype=np.int32)
    for j in range(_ANCHOR_RGB.shape[0]):
        a = _ANCHOR_RGB[j]
        d = (px[:, 0] - a[0]) ** 2 + (px[:, 1] - a[1]) ** 2 + (px[:, 2] - a[2]) ** 2
        m = d < best
        best[m] = d[m]
        idx[m] = j
    return _ANCHOR_KEYS[idx].reshape(h, w)


def colorize(surf: np.ndarray) -> np.ndarray:
    """パレットキー配列 → RGB 画像（ブロック代表色での較正プレビュー用）。"""
    h, w = surf.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for k in np.unique(surf.astype(str)):
        try:
            img[surf == k] = _key_rgb(str(k))
        except KeyError:
            pass
    return img


def ortho_surface_grid(dst_meta: dict, zoom: int = 18, saturation: float = 1.4,
                       cache_dir=None, verbose: bool = True) -> np.ndarray:
    """
    dst_meta の経緯度グリッドに揃えた地表ブロックキー配列 (H,W) を返す。
    saturation>0 のとき enhance_rgb（WB+彩度）でかぶりを補正してからマッチ。

    dst_meta: {'lat_min','lat_max','lon_min','lon_max','res_lat','res_lon','shape':(H,W)}
    """
    from tellus_data import fetch_gsi_ortho, reproject_to_grid, DEFAULT_CACHE_DIR

    H, W = dst_meta["shape"]
    ortho = fetch_gsi_ortho(
        lat_min=dst_meta["lat_min"], lat_max=dst_meta["lat_max"],
        lon_min=dst_meta["lon_min"], lon_max=dst_meta["lon_max"],
        zoom=zoom, cache_dir=cache_dir or DEFAULT_CACHE_DIR, verbose=verbose,
    )
    rgb_src = ortho["rgb"]
    dst = {**dst_meta, "dem": np.zeros((H, W), dtype=np.float32)}
    chans = [reproject_to_grid(rgb_src[..., c].astype(np.float32), ortho, dst,
                               fill_value=0.0) for c in range(3)]
    rgb_dst = np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)
    if saturation and saturation != 1.0:
        rgb_dst = enhance_rgb(rgb_dst, saturation=saturation)
    surf = classify_rgb_to_palette(rgb_dst)
    if verbose:
        keys, cnts = np.unique(surf.astype(str), return_counts=True)
        dist = ", ".join(f"{k}={c}" for k, c in sorted(zip(keys, cnts), key=lambda t: -t[1]))
        print(f"[ortho] surface classes ({H}×{W}): {dist}")
    return surf
