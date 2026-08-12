"""Facade R&D sandbox — buildings ONLY, rendered as isometric 'photos' (no Minecraft/terrain).
Each preset = a decoration pattern. Outputs a catalog grid image so we can iterate on facades fast.
Run from repo root:  PYTHONUTF8=1 .venv/bin/python /tmp/facade_lab.py results/roofchk/catalog.png"""
import sys, os
sys.path.insert(0, '/tmp'); sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iso_render import iso_render
from block_palette import name_to_rgb

N2RGB = name_to_rgb()
GRASS = 'minecraft:grass_block'


def emit(W, D, F, spec):
    b = {}
    fh = spec.get('fh', 4); H = F * fh
    wall = spec['wall']; trim = spec.get('trim', wall); win = spec.get('window', 'minecraft:glass')
    perim = lambda x, z: x in (0, W - 1) or z in (0, D - 1)
    corner = lambda x, z: x in (0, W - 1) and z in (0, D - 1)

    def outs(x, z):
        r = []
        if x == 0: r.append((-1, 0))
        if x == W - 1: r.append((1, 0))
        if z == 0: r.append((0, -1))
        if z == D - 1: r.append((0, 1))
        return r

    bc = spec.get('base_course', 0); base_mat = spec.get('base_mat', trim)
    for y in range(H):
        r = y % fh
        for x in range(W):
            for z in range(D):
                if not perim(x, z):
                    continue
                run = x if z in (0, D - 1) else z
                name = wall
                if corner(x, z) and spec.get('quoins'):
                    name = trim
                elif bc and y < bc:
                    name = base_mat
                elif spec.get('string_course') and r == 0 and 0 < y < H - 1:
                    name = trim
                elif not corner(x, z):
                    wp = spec.get('win_pattern', 'grid')
                    if wp == 'grid' and r in (1, 2) and run % 2 == 0:
                        name = win
                    elif wp == 'bands' and r == 2:
                        name = win
                    elif wp == 'tall' and run % 3 == 1 and r >= 1:
                        name = win
                b[(x, y, z)] = name
    # flat roof + parapet
    rm = spec.get('roof_mat', 'minecraft:gray_concrete')
    for x in range(W):
        for z in range(D):
            b[(x, H, z)] = rm
    for k in range(1, spec.get('parapet', 0) + 1):
        for x in range(W):
            for z in range(D):
                if perim(x, z):
                    b[(x, H + k, z)] = trim if k == spec['parapet'] else wall
    # cornice: protruding slab band just under the roof
    if spec.get('cornice'):
        cm = spec.get('cornice_slab', 'minecraft:stone_brick_slab')
        for x in range(W):
            for z in range(D):
                if perim(x, z):
                    for ox, oz in outs(x, z):
                        b[(x + ox, H - 1, z + oz)] = cm
    # pilasters: protruding wall posts at intervals on straight walls
    ps = spec.get('pilasters', 0)
    pm = spec.get('pilaster_mat', 'minecraft:stone_brick_wall')
    if ps:
        for x in range(W):
            for z in range(D):
                if not perim(x, z) or corner(x, z):
                    continue
                run = x if z in (0, D - 1) else z
                if run % ps:
                    continue
                for ox, oz in outs(x, z):
                    for y in range(H):
                        b[(x + ox, y, z + oz)] = pm
    # corner posts (protruding quoin columns)
    if spec.get('corner_posts'):
        for cx, cz in ((0, 0), (0, D - 1), (W - 1, 0), (W - 1, D - 1)):
            ox = -1 if cx == 0 else 1; oz = -1 if cz == 0 else 1
            for y in range(H):
                b[(cx + ox, y, cz + oz)] = pm
    # ground pad
    for x in range(-2, W + 2):
        for z in range(-2, D + 2):
            b[(x, -1, z)] = GRASS
    return b


A = 'minecraft:'
PRESETS = [
    ("01 plain", dict(wall=A+'white_concrete', trim=A+'light_gray_concrete', win_pattern='none', parapet=1)),
    ("02 window grid", dict(wall=A+'white_concrete', trim=A+'light_gray_concrete', win_pattern='grid', parapet=1)),
    ("03 quoins", dict(wall=A+'sandstone', trim=A+'smooth_sandstone' if A+'smooth_sandstone' in N2RGB else A+'sandstone', win_pattern='grid', quoins=True, parapet=1)),
    ("04 pilasters", dict(wall=A+'white_concrete', trim=A+'light_gray_concrete', win_pattern='grid', pilasters=4, pilaster_mat=A+'stone_brick_wall', parapet=1)),
    ("05 cornice", dict(wall=A+'light_gray_terracotta', trim=A+'gray_terracotta', win_pattern='grid', cornice=True, cornice_slab=A+'stone_brick_slab', parapet=1)),
    ("06 string course", dict(wall=A+'brown_terracotta', trim=A+'brick', window=A+'light_blue_stained_glass', win_pattern='bands', string_course=True, parapet=1)),
    ("07 base plinth", dict(wall=A+'white_terracotta', trim=A+'andesite', base_course=2, base_mat=A+'andesite', win_pattern='grid', parapet=1)),
    ("08 corner posts", dict(wall=A+'orange_terracotta', trim=A+'brick', pilaster_mat=A+'brick_wall', corner_posts=True, win_pattern='tall', window=A+'glass', parapet=1)),
    ("09 FULL combo", dict(wall=A+'white_concrete', trim=A+'light_gray_concrete', win_pattern='grid',
                           pilasters=4, pilaster_mat=A+'stone_brick_wall', corner_posts=True,
                           cornice=True, cornice_slab=A+'stone_brick_slab', base_course=1, base_mat=A+'andesite', parapet=1)),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'results/roofchk/catalog.png'
    W, D, F = 7, 6, 3
    imgs = []
    for name, spec in PRESETS:
        b = emit(W, D, F, spec)
        imgs.append((name, np.asarray(iso_render(b, N2RGB, tile=16))))
    n = len(imgs); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    for ax, (name, im) in zip(axes.flat, imgs):
        ax.imshow(im); ax.set_title(name, fontsize=11); ax.axis('off')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('facade pattern catalog (iso, buildings only)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('saved', out, 'patterns=', n)


main()
