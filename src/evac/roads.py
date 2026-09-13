"""
evac/roads.py — OSM highway 座標列 → 無向道路グラフ (scipy.sparse) と時間依存 Dijkstra。

グラフ
  ノード = way 座標を 1e-6 度で丸めて共有 (交差点は OSM のノード共有に依存)。長い区間は
  max_seg_m (既定 20 m) 以下に分割して中間ノードを挿入 (セル展開・浸水判定を密にするため)。
  各ノードに 5 m 格子から T(x,y)=30cm 到達秒 (DEM NaN セルは最近傍の陸セル値) と DEM を付与。

最遅出発時刻 (逆向き Dijkstra, 最大ヒープ)
  L(goal) = T(goal)  (施設が浸水しない = +inf)
  L(v) = min( T(v), max_u [ min(L(u), T(u)) − d(v,u)/v_walk ] )
  余裕 margin(v) = L(v) − 避難開始時刻 (5 分)。到達不能なら −inf。
"""
from __future__ import annotations

import heapq
import json
import math

import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from common import Grid, OUT


def fill_nan_nearest(a):
    """NaN を最近傍の有効値で埋める (海/河川セルの T を隣接陸セルから借りる)。"""
    m = np.isnan(a)
    if not m.any():
        return a
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


class RoadGraph:
    def __init__(self, ways, grid: Grid, T, dem, max_seg_m=20.0, extra_points=None, max_link_m=300.0):
        """extra_points: [(lat, lon), ...] 施設など。最寄り道路ノードへ直線エッジ (≤ max_link_m) で接続した
        仮想ノードを追加し、self.extra_ids[k] にその id (接続できなければ None) を入れる。"""
        self.grid = grid
        key2id: dict = {}
        xy_list: list = []
        latlon_list: list = []
        edges: dict = {}
        self.way_of_edge: dict = {}

        def node(lat, lon):
            k = (round(lat, 6), round(lon, 6))
            i = key2id.get(k)
            if i is None:
                i = len(xy_list)
                key2id[k] = i
                x, y = grid.latlon_to_xy(lat, lon)
                xy_list.append((float(x), float(y)))
                latlon_list.append((lat, lon))
            return i

        for wi, w in enumerate(ways):
            c = w["coords"]
            for (la0, lo0), (la1, lo1) in zip(c[:-1], c[1:]):
                x0, y0 = grid.latlon_to_xy(la0, lo0)
                x1, y1 = grid.latlon_to_xy(la1, lo1)
                L = math.hypot(x1 - x0, y1 - y0)
                if L < 1e-3:
                    continue
                n = max(1, int(math.ceil(L / max_seg_m)))
                prev = node(la0, lo0)
                for k in range(1, n + 1):
                    f = k / n
                    cur = node(la0 + (la1 - la0) * f, lo0 + (lo1 - lo0) * f) if k < n else node(la1, lo1)
                    if cur == prev:
                        continue
                    a, b = (prev, cur) if prev < cur else (cur, prev)
                    d = L / n
                    if (a, b) not in edges or edges[(a, b)] > d:
                        edges[(a, b)] = d
                        self.way_of_edge[(a, b)] = wi
                    prev = cur

        # 仮想ノード (施設) を最寄り道路ノードへ接続
        self.extra_ids = []
        self.extra_link_m = []
        if extra_points:
            base_tree = cKDTree(np.array(xy_list, float))
            for lat, lon in extra_points:
                x, y = grid.latlon_to_xy(lat, lon)
                d, j = base_tree.query([float(x), float(y)])
                if d > max_link_m:
                    self.extra_ids.append(None); self.extra_link_m.append(float(d))
                    continue
                i = len(xy_list)
                xy_list.append((float(x), float(y))); latlon_list.append((float(lat), float(lon)))
                a, b = (int(j), i)
                edges[(a, b)] = max(float(d), 0.5)
                self.extra_ids.append(i); self.extra_link_m.append(float(d))
        self.xy = np.array(xy_list, float)
        self.latlon = np.array(latlon_list, float)
        self.N = len(xy_list)
        ij = np.array(list(edges.keys()), int)
        dd = np.array(list(edges.values()), float)
        A = sparse.coo_matrix((np.r_[dd, dd], (np.r_[ij[:, 0], ij[:, 1]], np.r_[ij[:, 1], ij[:, 0]])), shape=(self.N, self.N))
        self.A = A.tocsr()
        self.n_edges = len(edges)
        # ノード属性
        r, c = grid.xy_to_rc(self.xy[:, 0], self.xy[:, 1])
        ri = np.clip(np.round(r).astype(int), 0, grid.H - 1)
        ci = np.clip(np.round(c).astype(int), 0, grid.W - 1)
        self.rc = np.c_[ri, ci]
        Tf = fill_nan_nearest(np.where(np.isnan(dem), np.nan, T.astype(np.float64)))
        self.T = Tf[ri, ci]
        self.T[np.isnan(self.T)] = np.inf
        self.dem = fill_nan_nearest(dem.astype(np.float64))[ri, ci]
        self.sea = np.isnan(dem)[ri, ci]
        self.tree = cKDTree(self.xy)
        # 連結成分
        self.n_comp, self.comp = sparse.csgraph.connected_components(self.A, directed=False)

    # ────────────────────────────────────────────────
    def snap(self, lat, lon, max_m=60.0):
        x, y = self.grid.latlon_to_xy(lat, lon)
        d, i = self.tree.query([float(x), float(y)])
        return (int(i), float(d)) if d <= max_m else (None, float(d))

    def neighbors(self, v):
        s, e = self.A.indptr[v], self.A.indptr[v + 1]
        return self.A.indices[s:e], self.A.data[s:e]

    # ────────────────────────────────────────────────
    def latest_departure(self, targets, v_walk):
        """逆向き最大ヒープ Dijkstra。返り値 L[v] (秒, 到達不能 = -inf), src[v] (到達先 target の index)。"""
        L = np.full(self.N, -np.inf)
        src = np.full(self.N, -1, int)
        heap = []
        for k, t in enumerate(targets):
            lt = float(self.T[t])
            if lt > L[t]:
                L[t] = lt
                src[t] = k
                heapq.heappush(heap, (-lt, t))
        done = np.zeros(self.N, bool)
        while heap:
            negl, u = heapq.heappop(heap)
            lu = -negl
            if done[u] or lu < L[u]:
                continue
            done[u] = True
            base = min(lu, float(self.T[u]))
            nb, dd = self.neighbors(u)
            for v, d in zip(nb, dd):
                if done[v]:
                    continue
                cand = min(base - d / v_walk, float(self.T[v]))
                if cand > L[v]:
                    L[v] = cand
                    src[v] = src[u]
                    heapq.heappush(heap, (-cand, v))
        return L, src

    def earliest_arrival(self, start, goals, t0, v_walk, respect_flood=True):
        """前向き時間依存 Dijkstra。通行条件: 出発時 t<T(v) かつ 到着時 t+d/v < T(u)。
        返り値 (goal, arrival_sec, path[list of node]) / 到達不能なら (None, inf, [])。"""
        goals = set(int(g) for g in goals)
        A = np.full(self.N, np.inf)
        prev = np.full(self.N, -1, int)
        A[start] = t0
        heap = [(t0, start)]
        done = np.zeros(self.N, bool)
        while heap:
            t, u = heapq.heappop(heap)
            if done[u] or t > A[u]:
                continue
            done[u] = True
            if u in goals:
                path = []
                v = u
                while v != -1:
                    path.append(v)
                    v = prev[v]
                return u, t, path[::-1]
            if respect_flood and t >= self.T[u]:
                continue
            nb, dd = self.neighbors(u)
            for v, d in zip(nb, dd):
                ta = t + d / v_walk
                if done[v] or ta >= A[v]:
                    continue
                if respect_flood and ta >= self.T[v]:
                    continue
                A[v] = ta
                prev[v] = u
                heapq.heappush(heap, (ta, v))
        return None, np.inf, []

    def path_length(self, path):
        return float(sum(self.A[a, b] for a, b in zip(path[:-1], path[1:])))


def load_ways():
    with open(OUT / "roads.json", encoding="utf-8") as f:
        return json.load(f)["ways"]
