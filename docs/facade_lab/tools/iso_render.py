"""Shape-aware isometric block renderer -> PIL image. blocks: dict {(x,y,z): mcname}.
Slabs render half-height, *_wall as thin inset posts (pilasters), else full cubes. Faces are
shaded (top/left/right) so 3D relief reads clearly. No Minecraft needed."""
from PIL import Image, ImageDraw


def _box(name):
    """local bounds (x0,x1,y0,y1,z0,z1, is_full)."""
    if name.endswith('_slab'):
        return (0.0, 1.0, 0.0, 0.5, 0.0, 1.0, False)
    if name.endswith('_wall'):
        return (0.30, 0.70, 0.0, 1.0, 0.30, 0.70, False)   # thin post = pilaster
    return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, True)


def iso_render(blocks, name2rgb, tile=18, bg=(250, 250, 250)):
    if not blocks:
        return Image.new('RGB', (64, 64), bg)
    xs = [p[0] for p in blocks]; ys = [p[1] for p in blocks]; zs = [p[2] for p in blocks]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys); minz, maxz = min(zs), max(zs)
    tw = tile // 2; th = tile // 4; bh = tile

    def proj(x, y, z):
        return (x - z) * tw, (x + z) * th - y * bh

    corners = [proj(x, y, z) for x in (minx, maxx + 1) for y in (miny, maxy + 2) for z in (minz, maxz + 1)]
    xsc = [c[0] for c in corners]; ysc = [c[1] for c in corners]
    ox = -min(xsc) + 6; oy = -min(ysc) + 6
    img = Image.new('RGB', (int(max(xsc) - min(xsc) + 12), int(max(ysc) - min(ysc) + 12)), bg)
    dr = ImageDraw.Draw(img)
    edge = (45, 45, 45)

    def P(x, y, z):
        sx, sy = proj(x, y, z)
        return (sx + ox, sy + oy)

    def shade(c, f):
        return (min(255, int(c[0] * f)), min(255, int(c[1] * f)), min(255, int(c[2] * f)))

    for (x, y, z) in sorted(blocks.keys(), key=lambda p: (p[0] + p[2], p[1])):
        c = name2rgb.get(blocks[(x, y, z)], (150, 150, 150))
        x0, x1, y0, y1, z0, z1, full = _box(blocks[(x, y, z)])
        ax, bx = x + x0, x + x1; ay, by = y + y0, y + y1; az, bz = z + z0, z + z1
        if (not full) or ((x, y + 1, z) not in blocks):
            dr.polygon([P(ax, by, az), P(bx, by, az), P(bx, by, bz), P(ax, by, bz)], fill=shade(c, 1.0), outline=edge)
        if (not full) or ((x + 1, y, z) not in blocks):
            dr.polygon([P(bx, ay, az), P(bx, by, az), P(bx, by, bz), P(bx, ay, bz)], fill=shade(c, 0.72), outline=edge)
        if (not full) or ((x, y, z + 1) not in blocks):
            dr.polygon([P(ax, ay, bz), P(bx, ay, bz), P(bx, by, bz), P(ax, by, bz)], fill=shade(c, 0.55), outline=edge)
    return img
