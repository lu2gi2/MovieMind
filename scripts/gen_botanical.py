#!/usr/bin/env python3
"""
Generates the Sky Light botanical artwork as lightweight, transparent SVG layers.

    static/images/branch-main.svg    main branch entering from the upper-right edge (branch + attached leaves)
    static/images/leaves-float.svg   a few extra leaves on the same canvas (moves a little more than the branch)
    static/images/branch-corner.svg  small secondary branch for the lower-left page edge

Seeded and dependency-free, so the artwork is reproducible: python scripts/gen_botanical.py
Leaves use pale, natural greens, a lit and a shaded half, a midrib and side veins; the bark has a sunlit edge.
"""
import math
import os
import random

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'images')

DEFS = '''<defs>
  <linearGradient id="gA" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#C9D6B4"/><stop offset=".5" stop-color="#A9BD93"/><stop offset="1" stop-color="#7F9870"/></linearGradient>
  <linearGradient id="gB" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#D5DFBF"/><stop offset=".5" stop-color="#B7C7A0"/><stop offset="1" stop-color="#8CA27C"/></linearGradient>
  <linearGradient id="gC" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#BFCFAE"/><stop offset=".5" stop-color="#9DB38A"/><stop offset="1" stop-color="#75906A"/></linearGradient>
  <linearGradient id="bark" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#A89A86"/><stop offset=".55" stop-color="#8A7B68"/><stop offset="1" stop-color="#6E6252"/></linearGradient>
  <symbol id="lfA" viewBox="0 0 100 160" overflow="visible"><path d="M50 158C18 122 2 70 50 2c48 68 32 120 0 156Z" fill="url(#gA)"/><path d="M50 156C50 110 50 60 50 8" stroke="#6F8862" stroke-width="1.6" fill="none" opacity=".75"/><path d="M50 120C34 108 24 98 16 84M50 100c14-10 24-20 30-34M50 76C38 66 32 58 28 46M50 62c10-8 16-14 20-24" stroke="#7F987A" stroke-width=".9" fill="none" opacity=".55"/><path d="M50 2c48 68 32 120 0 156 10-40 14-96 0-156Z" fill="#fff" opacity=".12"/></symbol>
  <symbol id="lfB" viewBox="0 0 100 160" overflow="visible"><path d="M50 158C18 122 2 70 50 2c48 68 32 120 0 156Z" fill="url(#gB)"/><path d="M50 156C50 110 50 60 50 8" stroke="#728B66" stroke-width="1.6" fill="none" opacity=".7"/><path d="M50 120C34 108 24 98 16 84M50 100c14-10 24-20 30-34M50 76C38 66 32 58 28 46M50 62c10-8 16-14 20-24" stroke="#86A07F" stroke-width=".9" fill="none" opacity=".5"/><path d="M50 2c48 68 32 120 0 156 10-40 14-96 0-156Z" fill="#fff" opacity=".16"/></symbol>
  <symbol id="lfC" viewBox="0 0 100 160" overflow="visible"><path d="M50 158C18 122 2 70 50 2c48 68 32 120 0 156Z" fill="url(#gC)"/><path d="M50 156C50 110 50 60 50 8" stroke="#667F5B" stroke-width="1.6" fill="none" opacity=".75"/><path d="M50 120C34 108 24 98 16 84M50 100c14-10 24-20 30-34M50 76C38 66 32 58 28 46M50 62c10-8 16-14 20-24" stroke="#7A9373" stroke-width=".9" fill="none" opacity=".5"/><path d="M50 2c48 68 32 120 0 156 10-40 14-96 0-156Z" fill="#fff" opacity=".1"/></symbol>
</defs>'''


def bezier(p0, p1, p2, t):
    a = (1 - t) ** 2
    b = 2 * (1 - t) * t
    c = t * t
    return (a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1])


def tangent(p0, p1, p2, t):
    return (2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0]), 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1]))


def limb(p0, p1, p2, w0, w1, n=28):
    """Tapered woody limb as a polygon; returns (svg path d, highlight path d, sampled points with angles)."""
    left, right, pts = [], [], []
    for i in range(n + 1):
        t = i / n
        x, y = bezier(p0, p1, p2, t)
        tx, ty = tangent(p0, p1, p2, t)
        L = math.hypot(tx, ty) or 1
        nx, ny = -ty / L, tx / L
        w = (w0 + (w1 - w0) * (t ** 0.85)) / 2
        wobble = 1 + 0.06 * math.sin(t * 17)
        left.append((x + nx * w * wobble, y + ny * w * wobble))
        right.append((x - nx * w * wobble, y - ny * w * wobble))
        pts.append((x, y, math.degrees(math.atan2(ty, tx)), w))
    poly = left + right[::-1]
    d = 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in poly) + 'Z'
    hl = 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in left[2:-3]) + ''
    return d, hl, pts


def leaf(rnd, x, y, ang, size, variant=None):
    """One leaf <use>, base at (x, y), pointing along `ang` degrees (0 = right)."""
    v = variant or rnd.choice(['lfA', 'lfB', 'lfC', 'lfA', 'lfB'])
    w, h = size * 0.62, size
    # symbol points up (-y); rotate so its tip faces `ang`
    rot = ang + 90
    op = rnd.uniform(0.82, 0.96)
    return (f'<use href="#{v}" width="{w:.1f}" height="{h:.1f}" x="{-w / 2:.1f}" y="{-h:.1f}" opacity="{op:.2f}" '
            f'transform="translate({x:.1f} {y:.1f}) rotate({rot:.1f})"/>')


def twig(rnd, base, ang, length, leaf_size, depth=0, leaves=None, wood=None):
    """A thin side branch with alternating leaves and a terminal leaf."""
    bx, by = base
    r = math.radians(ang)
    ex, ey = bx + math.cos(r) * length, by + math.sin(r) * length
    bend = rnd.uniform(-0.25, 0.25) * length
    cx, cy = (bx + ex) / 2 - math.sin(r) * bend, (by + ey) / 2 + math.cos(r) * bend
    d, hl, pts = limb((bx, by), (cx, cy), (ex, ey), max(2.2, 7 - depth * 2.4), 1.2, n=14)
    wood.append(d)
    k = max(3, int(length / 34))
    for i in range(1, k + 1):
        t = i / (k + 1)
        x, y, a, w = pts[int(t * (len(pts) - 1))]
        side = 1 if i % 2 else -1
        leaves.append(leaf(rnd, x, y, a + side * rnd.uniform(38, 62), leaf_size * rnd.uniform(0.82, 1.1)))
    x, y, a, w = pts[-1]
    leaves.append(leaf(rnd, x, y, a + rnd.uniform(-12, 12), leaf_size * rnd.uniform(0.95, 1.2)))


def build_main(seed=11):
    rnd = random.Random(seed)
    W, H = 900, 760
    wood, leaves, extra = [], [], []
    # main limb: enters beyond the top-right corner, sweeps down-left and ends mid-canvas
    p0, p1, p2 = (W + 60, -30), (640, 130), (240, 330)
    d, hl, pts = limb(p0, p1, p2, 30, 4, n=44)
    wood.append(d)
    for t in (0.14, 0.24, 0.34, 0.45, 0.56, 0.66, 0.77, 0.87):
        x, y, a, w = pts[int(t * (len(pts) - 1))]
        side = 1 if int(t * 100) % 2 == 0 else -1
        twig(rnd, (x, y), a + side * rnd.uniform(38, 62), rnd.uniform(110, 190) * (1.1 - t * 0.4), rnd.uniform(46, 62), 0, leaves, wood)
        if rnd.random() < 0.7:
            twig(rnd, (x, y), a - side * rnd.uniform(40, 70), rnd.uniform(70, 120) * (1.1 - t * 0.4), rnd.uniform(40, 54), 1, leaves, wood)
    x, y, a, w = pts[-1]
    for da in (-26, 0, 24):
        leaves.append(leaf(rnd, x, y, a + da + rnd.uniform(-8, 8), rnd.uniform(52, 66)))
    # lower limb that drops down the right edge
    q0, q1, q2 = (W + 20, 120), (800, 260), (700, 470)
    d2, _, pts2 = limb(q0, q1, q2, 16, 3, n=30)
    wood.append(d2)
    for t in (0.3, 0.5, 0.7, 0.88):
        x, y, a, w = pts2[int(t * (len(pts2) - 1))]
        twig(rnd, (x, y), a + (1 if t > 0.5 else -1) * rnd.uniform(40, 65), rnd.uniform(70, 125), rnd.uniform(40, 54), 1, leaves, wood)
    x, y, a, w = pts2[-1]
    for da in (-30, 6, 34):
        leaves.append(leaf(rnd, x, y, a + da + rnd.uniform(-8, 8), rnd.uniform(44, 58)))
    # floating layer: loose leaves near twig tips
    fl = []
    for _ in range(16):
        t = rnd.uniform(0.12, 0.92)
        x, y, a, _w = pts[int(t * (len(pts) - 1))]
        ox, oy = rnd.uniform(-90, 90), rnd.uniform(40, 130) * rnd.choice([1, 1, -0.4])
        fl.append(leaf(rnd, x + ox, y + oy, rnd.uniform(-170, 20), rnd.uniform(34, 54)))
    for _ in range(5):
        fl.append(leaf(rnd, rnd.uniform(560, 860), rnd.uniform(380, 640), rnd.uniform(-150, -30), rnd.uniform(30, 44)))
    bark = ''.join(f'<path d="{p}" fill="url(#bark)"/>' for p in wood)
    lit = f'<path d="{hl}" fill="none" stroke="#E9DCC5" stroke-width="2.2" stroke-linecap="round" opacity=".55"/>'
    branch = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMaxYMin slice">{DEFS}<g>{bark}{lit}</g><g>{"".join(leaves)}</g></svg>'
    floating = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMaxYMin slice">{DEFS}<g>{"".join(fl)}</g></svg>'
    return branch, floating


def build_corner(seed=5):
    rnd = random.Random(seed)
    W, H = 520, 520
    wood, leaves = [], []
    # enters from the left edge, rises toward the upper-right
    p0, p1, p2 = (-40, 470), (150, 430), (330, 250)
    d, hl, pts = limb(p0, p1, p2, 20, 3, n=30)
    wood.append(d)
    for t in (0.25, 0.42, 0.58, 0.74, 0.9):
        x, y, a, w = pts[int(t * (len(pts) - 1))]
        twig(rnd, (x, y), a - rnd.uniform(40, 62), rnd.uniform(80, 140), rnd.uniform(40, 54), 0, leaves, wood)
        if rnd.random() < 0.6:
            twig(rnd, (x, y), a + rnd.uniform(40, 62), rnd.uniform(60, 100), rnd.uniform(36, 48), 1, leaves, wood)
    x, y, a, w = pts[-1]
    for da in (-22, 4, 26):
        leaves.append(leaf(rnd, x, y, a + da, rnd.uniform(46, 60)))
    bark = ''.join(f'<path d="{p}" fill="url(#bark)"/>' for p in wood)
    lit = f'<path d="{hl}" fill="none" stroke="#E9DCC5" stroke-width="2" stroke-linecap="round" opacity=".5"/>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMinYMax slice">{DEFS}<g>{bark}{lit}</g><g>{"".join(leaves)}</g></svg>'


def main():
    os.makedirs(OUT, exist_ok=True)
    branch, floating = build_main()
    for name, svg in (('branch-main.svg', branch), ('leaves-float.svg', floating), ('branch-corner.svg', build_corner())):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(svg)
        print(f'{name}: {len(svg) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
