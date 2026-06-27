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


# --- Oklab 知覚色空間（arnis src/colors.rs:114-152 移植） ---
# sRGB→linear→LMS→Oklab。RGB ユークリッドより人間の知覚距離に一致するので、
# 「道路の灰／畑の緑」がスペクトル的に遠いブロックへ飛ぶ誤マッチを抑える。
_OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
_OKLAB_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB 0..255 → linear 0..1（arnis srgb_to_linear）。"""
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    np.power((c + 0.055) / 1.055, 2.4)).astype(np.float32)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB(0..255) → (..., 3) Oklab（arnis rgb_to_oklab のベクトル化）。"""
    lin = _srgb_to_linear(np.asarray(rgb))
    lms = np.cbrt(lin @ _OKLAB_M1.T)
    return (lms @ _OKLAB_M2.T).astype(np.float32)


_ANCHOR_OKLAB = rgb_to_oklab(_ANCHOR_RGB)  # (M,3) アンカーの Oklab（起動時前計算）


def enhance_rgb(rgb: np.ndarray, *, saturation: float = 1.35,
                low_pct: float = 2.0, high_pct: float = 98.0,
                scale_cap: float = 1.5) -> np.ndarray:
    """
    オルソの青み・霞かぶりと低彩度を補正（白黒っぽさ対策）。vivid かつタイル間でロバスト。

    **チャンネル別パーセンタイル・ストレッチ**（青みかぶり除去＋コントラスト＝発色）を使う。
    これは陸の多いタイル(御坊中心)で鮮やかになる一方、赤の弱い海沿いタイル(06SC002)では
    赤chを過増幅してピンク化する弱点があった。→ **各chの倍率を中央値比 scale_cap でクランプ**
    することで、弱いchだけ抑えてピンク化を防ぎ（均等なタイルは無クランプ＝フル vivid）。
    最後に控えめな彩度。
      scale_cap: 1ch の伸長倍率を「中央値×scale_cap」までに制限
    """
    arr = rgb.astype(np.float32)
    los = []
    scales = []
    for c in range(3):
        lo = float(np.percentile(arr[..., c], low_pct))
        hi = float(np.percentile(arr[..., c], high_pct))
        los.append(lo)
        scales.append(255.0 / max(hi - lo, 1e-3))
    med = float(np.median(scales))
    out = np.empty_like(arr)
    for c in range(3):
        s = min(scales[c], med * scale_cap)
        out[..., c] = (arr[..., c] - los[c]) * s
    arr = out
    if saturation != 1.0:
        luma = arr @ np.array([0.299, 0.587, 0.114], np.float32)
        arr = luma[..., None] + (arr - luma[..., None]) * saturation
    return np.clip(arr, 0, 255).astype(np.uint8)


def classify_rgb_to_palette(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (H, W, 3) uint8 → object 配列 (H, W) のパレットキー。

    ~80 単色バニラブロックへの**最近傍カラーマッチ**（Oklab 知覚距離, arnis 移植）。
    写真の色をそのままブロックへ写像するので、地表が写真モザイクになる。
    水/氷は洪水・海レイヤが別途生成するためアンカーから除外している。
    RGB ユークリッドより知覚一致が高く、道路/畑の微妙な色の取り違えを抑える。
    アンカーをまたぐ (P×M) 行列は作らず、アンカー毎ループで省メモリ。
    """
    h, w, _ = rgb.shape
    lab = rgb_to_oklab(rgb.reshape(-1, 3))             # (P,3) Oklab
    best = np.full(lab.shape[0], np.inf, dtype=np.float32)
    idx = np.zeros(lab.shape[0], dtype=np.int32)
    for j in range(_ANCHOR_OKLAB.shape[0]):
        a = _ANCHOR_OKLAB[j]
        d = (lab[:, 0] - a[0]) ** 2 + (lab[:, 1] - a[1]) ** 2 + (lab[:, 2] - a[2]) ** 2
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
                       cache_dir=None, verbose: bool = True,
                       layer: str = "seamlessphoto", return_rgb: bool = False):
    """
    dst_meta の経緯度グリッドに揃えた地表ブロックキー配列 (H,W) を返す。
    saturation>0 のとき enhance_rgb（WB+彩度）でかぶりを補正してからマッチ。
    layer: GSI 写真レイヤ（seamlessphoto=最新シームレス / ort=整備済オルソ＝高精細）。
    return_rgb=True なら (surf, rgb_dst) を返す（rgb_dst=補正後オルソ RGB grid (H,W,3)、駐車場の車検出等に使う）。

    dst_meta: {'lat_min','lat_max','lon_min','lon_max','res_lat','res_lon','shape':(H,W)}
    """
    from tellus_data import fetch_gsi_ortho, reproject_to_grid, DEFAULT_CACHE_DIR

    H, W = dst_meta["shape"]
    ortho = fetch_gsi_ortho(
        lat_min=dst_meta["lat_min"], lat_max=dst_meta["lat_max"],
        lon_min=dst_meta["lon_min"], lon_max=dst_meta["lon_max"],
        zoom=zoom, cache_dir=cache_dir or DEFAULT_CACHE_DIR, verbose=verbose, layer=layer,
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
        print(f"[ortho:{layer}] surface classes ({H}×{W}): {dist}")
    return (surf, rgb_dst) if return_rgb else surf
