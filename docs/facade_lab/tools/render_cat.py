"""Render facade patterns at type-appropriate proportions.
Usage: render_cat.py <patterns.json> <out.png> [tile] [only_names_csv]"""
import sys, json
sys.path.insert(0, '/tmp'); sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iso_render import iso_render
from block_palette import name_to_rgb

N2RGB = name_to_rgb(); MATS = set(N2RGB)
patterns = json.load(open(sys.argv[1], encoding='utf-8'))
out = sys.argv[2]
tile = int(sys.argv[3]) if len(sys.argv) > 3 else 13
only = set(s for s in sys.argv[4].split('|') if s) if len(sys.argv) > 4 and sys.argv[4] else None

SIZES = {
    'jp_house': (8, 7, 2), 'jp_machiya_shop': (6, 9, 2), 'jp_apartment_danchi': (11, 6, 4),
    'jp_office_building': (7, 7, 5), 'west_classical': (9, 7, 3), 'west_brick_row': (6, 8, 3),
    'relief_modern': (8, 8, 5), 'institutional_warehouse': (12, 8, 2),
    # round 2 building types
    'jp_shrine_temple': (9, 9, 2), 'jp_koban': (5, 5, 2), 'jp_konbini_roadside': (10, 8, 1),
    'jp_station': (12, 6, 2), 'jp_sento_bath': (9, 8, 2), 'jp_kura_street': (6, 8, 2),
    'jp_school_gym': (13, 7, 3), 'jp_civic_hall': (10, 8, 2),
}
import builtins as _b
SAFE = {k: getattr(_b, k) for k in ('range', 'len', 'min', 'max', 'abs', 'int', 'float', 'round', 'bool',
        'str', 'list', 'dict', 'tuple', 'set', 'enumerate', 'zip', 'sum', 'sorted', 'any', 'all', 'map',
        'filter', 'divmod', 'pow', 'reversed') if hasattr(_b, k)}

imgs = []; ok = 0; fail = 0
for p in patterns:
    if only and p.get('name') not in only:
        continue
    W, D, F = SIZES.get(p.get('bucket'), (7, 6, 3))
    blocks = {}
    def put(x, y, z, name, _bk=blocks):
        if name in MATS:
            _bk[(int(x), int(y), int(z))] = name
    try:
        ns = {'__builtins__': SAFE}
        exec(p['code'], ns)
        ns['build'](W, D, F, put, MATS)
        for x in range(-2, W + 2):
            for z in range(-2, D + 2):
                put(x, -1, z, 'minecraft:grass_block')
        if len(blocks) < 10:
            raise ValueError('few blocks')
        imgs.append((p.get('bucket', ''), p.get('name', '?'), np.asarray(iso_render(blocks, N2RGB, tile=tile))))
        ok += 1
    except Exception as e:
        fail += 1; print('FAIL', p.get('name'), '::', repr(e)[:120])

print('ok', ok, 'fail', fail)
imgs.sort(key=lambda t: (t[0], t[1]))
n = len(imgs); cols = 3 if only else 5; rows = max(1, (n + cols - 1) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * (4.0 if only else 3.2), rows * (4.2 if only else 3.6)))
axf = list(np.atleast_1d(axes).flat)
for ax, (bk, nm, im) in zip(axf, imgs):
    ax.imshow(im); ax.set_title('%s\n[%s]' % (nm[:30], bk), fontsize=8); ax.axis('off')
for ax in axf[n:]:
    ax.axis('off')
fig.suptitle('facade catalog — %d (type proportions)' % ok, fontsize=14)
fig.tight_layout(); fig.savefig(out, dpi=100); print('saved', out)
