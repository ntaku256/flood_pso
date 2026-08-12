"""Render agent-authored facade patterns into one big catalog. Each pattern's `code` defines
build(W,D,F,put,M); we exec it safely, place blocks, iso-render, and tile. Broken ones are skipped.
Usage: render_catalog.py <patterns.json> <out.png>"""
import sys, json
sys.path.insert(0, '/tmp'); sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iso_render import iso_render
from block_palette import name_to_rgb

N2RGB = name_to_rgb(); MATS = set(N2RGB.keys())
patterns = json.load(open(sys.argv[1], encoding='utf-8'))
out = sys.argv[2]
W, D, F = 7, 6, 3

import builtins as _b
SAFE = {k: getattr(_b, k) for k in ('range', 'len', 'min', 'max', 'abs', 'int', 'float', 'round', 'bool',
        'str', 'list', 'dict', 'tuple', 'set', 'enumerate', 'zip', 'sum', 'sorted', 'any', 'all', 'map',
        'filter', 'divmod', 'pow', 'reversed', 'True', 'False', 'None') if hasattr(_b, k)}
SAFE['True'] = True; SAFE['False'] = False; SAFE['None'] = None

imgs = []; ok = 0; fail = 0
for p in patterns:
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
            raise ValueError('too few blocks (%d)' % len(blocks))
        imgs.append((p.get('bucket', ''), p.get('name', '?'), np.asarray(iso_render(blocks, N2RGB, tile=13))))
        ok += 1
    except Exception as e:
        fail += 1
        print('FAIL', p.get('name'), '::', repr(e)[:140])

print('ok=%d fail=%d' % (ok, fail))
imgs.sort(key=lambda t: (t[0], t[1]))
n = len(imgs); cols = 5; rows = max(1, (n + cols - 1) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
axf = list(np.atleast_1d(axes).flat)
for ax, (bk, nm, im) in zip(axf, imgs):
    ax.imshow(im); ax.set_title('%s\n[%s]' % (nm[:26], bk), fontsize=7); ax.axis('off')
for ax in axf[n:]:
    ax.axis('off')
fig.suptitle('facade pattern catalog — %d rendered' % ok, fontsize=14)
fig.tight_layout(); fig.savefig(out, dpi=100)
print('saved', out)
