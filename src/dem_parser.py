"""
dem_parser.py
国土地理院 基盤地図情報 DEM5A/5B (GML形式) を読み込み、
複数タイルをモザイク合成して numpy 配列を返す。
"""

import glob
import numpy as np
from lxml import etree
from pathlib import Path


NS = {
    "gml": "http://www.opengis.net/gml/3.2",
    "fgd": "http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema",
}


def parse_tile(xml_path: str) -> dict:
    """
    1タイルのGMLを解析して辞書を返す。
    Returns:
        {
          'lat_min', 'lat_max', 'lon_min', 'lon_max': float  (WGS84/JGD2024)
          'nx', 'ny': int  (グリッド列数・行数)
          'data': np.ndarray shape (ny, nx) 標高[m], NaN=NoData
        }
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # --- 座標範囲 ---
    env = root.find(".//gml:Envelope", NS)
    lower = [float(v) for v in env.find("gml:lowerCorner", NS).text.split()]
    upper = [float(v) for v in env.find("gml:upperCorner", NS).text.split()]
    lat_min, lon_min = lower[0], lower[1]
    lat_max, lon_max = upper[0], upper[1]

    # --- グリッドサイズ ---
    grid_env = root.find(".//gml:GridEnvelope", NS)
    high = [int(v) for v in grid_env.find("gml:high", NS).text.split()]
    nx = high[0] + 1  # x方向セル数
    ny = high[1] + 1  # y方向セル数

    # --- 標高データ ---
    tuple_list = root.find(".//gml:tupleList", NS).text
    values = []
    for line in tuple_list.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(",", 1)
        try:
            v = float(parts[-1])
        except ValueError:
            v = np.nan
        values.append(np.nan if v <= -9000 else v)

    arr = np.array(values, dtype=np.float32)
    expected = ny * nx
    if len(arr) < expected:
        # エッジタイルでデータが欠損している場合は NaN でパディング
        arr = np.concatenate([arr, np.full(expected - len(arr), np.nan)])
    elif len(arr) > expected:
        arr = arr[:expected]
    # GMLは行優先 (y=0が北)、x方向が列
    # データ順: row0 (北端) → row_ny-1 (南端)
    arr = arr.reshape(ny, nx)

    return {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "nx": nx, "ny": ny,
        "data": arr,
    }


def mosaic_tiles(dem_dir: str, pattern: str = "*.xml") -> dict:
    """
    指定ディレクトリ内の全DEM GMLタイルをモザイク合成する。
    Returns:
        {
          'dem': np.ndarray shape (H, W) 標高[m]
          'lat_min', 'lat_max', 'lon_min', 'lon_max': float
          'res_lat', 'res_lon': float  (セルサイズ[度])
        }
    """
    files = sorted(glob.glob(str(Path(dem_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"No XML files found in {dem_dir}")

    tiles = [parse_tile(f) for f in files]
    print(f"Loaded {len(tiles)} tiles")

    # 全体の範囲
    glob_lat_min = min(t["lat_min"] for t in tiles)
    glob_lat_max = max(t["lat_max"] for t in tiles)
    glob_lon_min = min(t["lon_min"] for t in tiles)
    glob_lon_max = max(t["lon_max"] for t in tiles)

    # 代表解像度（最初のタイルから）
    t0 = tiles[0]
    res_lat = (t0["lat_max"] - t0["lat_min"]) / (t0["ny"] - 1)
    res_lon = (t0["lon_max"] - t0["lon_min"]) / (t0["nx"] - 1)

    # モザイク配列サイズ
    H = round((glob_lat_max - glob_lat_min) / res_lat) + 1
    W = round((glob_lon_max - glob_lon_min) / res_lon) + 1
    mosaic = np.full((H, W), np.nan, dtype=np.float32)

    for t in tiles:
        # タイル左上 (北端・西端) のモザイク内インデックス
        row0 = round((glob_lat_max - t["lat_max"]) / res_lat)
        col0 = round((t["lon_min"] - glob_lon_min) / res_lon)

        row1 = row0 + t["ny"]
        col1 = col0 + t["nx"]
        # 既存データがある場所はスキップしない (上書き or 平均もできるが今回は上書き)
        mosaic[row0:row1, col0:col1] = np.where(
            np.isnan(mosaic[row0:row1, col0:col1]),
            t["data"],
            mosaic[row0:row1, col0:col1],
        )

    print(f"Mosaic shape: {mosaic.shape}, lat [{glob_lat_min:.5f}, {glob_lat_max:.5f}], lon [{glob_lon_min:.5f}, {glob_lon_max:.5f}]")
    print(f"Valid cells: {np.sum(~np.isnan(mosaic))}/{mosaic.size}")

    return {
        "dem": mosaic,
        "lat_min": glob_lat_min, "lat_max": glob_lat_max,
        "lon_min": glob_lon_min, "lon_max": glob_lon_max,
        "res_lat": res_lat, "res_lon": res_lon,
    }


def latlon_to_pixel(lat, lon, dem_info: dict):
    """緯度経度 → モザイク配列のピクセル座標 (row, col)"""
    row = round((dem_info["lat_max"] - lat) / dem_info["res_lat"])
    col = round((lon - dem_info["lon_min"]) / dem_info["res_lon"])
    return row, col


def downsample(dem_info: dict, factor: int) -> dict:
    """
    DEM を factor 倍ダウンサンプリングする（ブロック平均）。
    factor=5 で 5m → 25m 解像度。
    """
    dem = dem_info["dem"]
    H, W = dem.shape
    H2 = H // factor
    W2 = W // factor
    dem_crop = dem[:H2 * factor, :W2 * factor]
    # ブロック平均（NaN はスキップ）
    dem_ds = np.nanmean(
        dem_crop.reshape(H2, factor, W2, factor), axis=(1, 3)
    ).astype(np.float32)
    return {
        "dem": dem_ds,
        "lat_min": dem_info["lat_min"],
        "lat_max": dem_info["lat_max"] - (H - H2 * factor) * dem_info["res_lat"],
        "lon_min": dem_info["lon_min"],
        "lon_max": dem_info["lon_max"] - (W - W2 * factor) * dem_info["res_lon"],
        "res_lat": dem_info["res_lat"] * factor,
        "res_lon": dem_info["res_lon"] * factor,
    }


if __name__ == "__main__":
    import sys
    default_dir = Path(__file__).resolve().parent.parent.parent / "kennkyuu20260114" / "地形データ" / "FG-GML-503561-DEM5A-20250620"
    dem_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)
    info = mosaic_tiles(dem_dir)
    dem = info["dem"]
    valid = dem[~np.isnan(dem)]
    print(f"Elevation range: {valid.min():.1f} ~ {valid.max():.1f} m")
