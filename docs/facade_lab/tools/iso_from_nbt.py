"""Iso-render a window of a pipeline NBT world (to preview real decorated buildings).
Usage: iso_from_nbt.py <nbt> <out.png> <x0> <x1> <z0> <z1> [tile]"""
import sys
sys.path.insert(0, '/tmp'); sys.path.insert(0, 'src')
import nbtlib
from iso_render import iso_render
from block_palette import name_to_rgb

N2RGB = name_to_rgb()
d = nbtlib.load(sys.argv[1])
pal = [str(p['Name']) for p in d['palette']]
x0, x1, z0, z1 = (int(v) for v in sys.argv[3:7])
tile = int(sys.argv[7]) if len(sys.argv) > 7 else 10
SKIP = {'minecraft:oak_leaves', 'minecraft:spruce_leaves', 'minecraft:birch_leaves',
        'minecraft:blue_stained_glass'}  # trees + water glass declutter
blocks = {}
ymin = 999
for b in d['blocks']:
    x, y, z = (int(v) for v in b['pos'])
    if x0 <= x < x1 and z0 <= z < z1:
        name = pal[int(b['state'])]
        if name in SKIP:
            continue
        blocks[(x - x0, y, z - z0)] = name
        ymin = min(ymin, y)
# drop deep underground fill to lighten (keep from ymin.. a few below surface)
im = iso_render(blocks, N2RGB, tile=tile)
im.save(sys.argv[2])
print('saved', sys.argv[2], 'blocks', len(blocks))
