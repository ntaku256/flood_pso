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


# ── 施策3+2: 地表をマテリアル分類化（「クラス内リッチ・クラス間明瞭」）─────────────
# オルソ最近傍マッチ(~80色)はピクセル毎に微妙に色が違う＝点群的なブラーが残る。
# 自然地表セルを ~6 の意味マテリアルへ畳み、3×3 多数決でごま塩(salt-and-pepper)を除去する。
# 道路/水/建物/樹はあとからマスクで別途上書きされるため、ここでは触れない（セグメント保持）。

# 畳み先（すべて viewer 既知・既存パレット内のキー）
GROUND_MATERIALS = ("grass", "coarse_dirt", "sand", "gravel", "stone", "light_gray_concrete")


def _ground_material_rgb(r: float, g: float, b: float) -> str:
    """RGB → 地表マテリアルキー。植生/裸地/砂/砂利/石/コンクリの6分類。"""
    mx = max(r, g, b); mn = min(r, g, b); sat = mx - mn
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # 植生（緑優勢）
    if g >= r and g >= b and (g - max(r, b)) > 10 and g > 55:
        return "grass"
    # 低彩度＝中立な人工/岩面。輝度で 石/砂利/コンクリ に分ける
    if sat <= 24:
        if lum < 95:
            return "stone"                  # 暗灰（磯・濃い舗装）
        if lum < 175:
            return "gravel"                 # 中灰（砂利・未舗装地）
        return "light_gray_concrete"        # 明灰/白（コンクリ広場）
    # 暖色・土系（赤茶〜タン）。輝度で 砂/裸地
    if r >= b:
        return "sand" if lum >= 178 else "coarse_dirt"
    # 残り（青紫系。地表では稀＝シート/コート等）→ 中立
    return "gravel"


# パレットキー → 地表マテリアル（アンカー RGB から事前計算）
_KEY_TO_GROUND = {k: _ground_material_rgb(*_key_rgb(k)) for k in MATCH_KEYS}


def collapse_to_ground_materials(surf: np.ndarray) -> np.ndarray:
    """パレットキー配列 (H,W) → 地表マテリアルキー配列（~6種）。入力は破壊しない。"""
    out = np.array(surf, dtype=object, copy=True)
    for k in np.unique(surf):
        mat = _KEY_TO_GROUND.get(str(k))
        if mat is not None and mat != k:
            out[surf == k] = mat
    return out


def majority_filter_keys(surf: np.ndarray, size: int = 3) -> np.ndarray:
    """カテゴリ(object)配列に size×size 多数決フィルタ。同数は元のクラスを維持。
    クラス数が少ない前提でベクトル化（uniform_filter＝各クラスの近傍出現率）。入力は破壊しない。"""
    from scipy.ndimage import uniform_filter
    keys = list(np.unique(surf))
    if len(keys) <= 1:
        return np.array(surf, dtype=object, copy=True)
    codes = np.full(surf.shape, -1, dtype=np.int16)
    for i, k in enumerate(keys):
        codes[surf == k] = i
    best_cnt = np.full(surf.shape, -1.0, dtype=np.float32)
    best_code = codes.copy()
    tie = 0.25 / float(size * size)   # 同数時に元クラスを優先する微小バイアス
    for i in range(len(keys)):
        cnt = uniform_filter((codes == i).astype(np.float32), size=size, mode="nearest")
        cnt = cnt + np.where(codes == i, tie, 0.0).astype(np.float32)
        upd = cnt > best_cnt
        best_cnt[upd] = cnt[upd]
        best_code[upd] = i
    out = np.empty(surf.shape, dtype=object)
    for i, k in enumerate(keys):
        out[best_code == i] = k
    return out


def cleanup_ground_surface(surf: np.ndarray, size: int = 3) -> np.ndarray:
    """施策3+2: 地表キー配列を ~6 マテリアルへ畳み、多数決でごま塩除去。入力は破壊しない。"""
    return majority_filter_keys(collapse_to_ground_materials(surf), size=size)


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
