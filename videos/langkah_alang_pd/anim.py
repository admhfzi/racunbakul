"""Langkah Alang ke Port Dickson - animated poster v3 (60 fps, motion blur, particles)."""
import cv2, numpy as np, os, math, subprocess, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
orig8 = cv2.imread('poster.jpg')
orig = orig8.astype(np.float32)
bg = cv2.imread('bg.png').astype(np.float32)
A = dict(np.load('alphas.npz'))
H, W = orig.shape[:2]
FPS, DUR = 60, 9.0
N = int(FPS * DUR)
PREVIEW = len(sys.argv) > 1 and sys.argv[1] == 'preview'
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rng = np.random.default_rng(7)
mn8, mx8 = orig8.min(2), orig8.max(2)
Bc, Gc, Rc = [orig[..., i] for i in range(3)]

def dil(m, k): return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

# ------------------------------------------------------------------ sprites
S = {}
def add_sprite(name, a, rgb=None):
    ys, xs = np.nonzero(a > 0.02)
    if len(xs) == 0: return
    p = 4
    x0, y0 = max(0, xs.min() - p), max(0, ys.min() - p)
    x1, y1 = min(W, xs.max() + 1 + p), min(H, ys.max() + 1 + p)
    src = orig if rgb is None else rgb
    S[name] = dict(rgb=src[y0:y1, x0:x1].copy(), a=a[y0:y1, x0:x1].astype(np.float32).copy(),
                   x0=x0, y0=y0, w=x1 - x0, h=y1 - y0, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)

# "Langkah" keeps only its navy strokes; white bits in its box belong to "Alang Ke"
keep = dil(mx8 < 140, 9)
A['alang'] = np.maximum(A['alang'], A['langkah'] * ~keep)
A['langkah'] = A['langkah'] * keep

# -- "Port Dickson": split into letters (white cores -> nearest-core assignment)
pa = A['port']
core = ((mn8 > 185) & (pa > 0.5)).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(core, 8)
alang_extra = np.zeros((H, W), bool)
comps = []
for i in range(1, n):
    x, y, w, h, area = st[i]
    if area < 40: continue
    if y <= 404 and h < 45:                 # descenders of "Alang Ke" sitting in Port's area
        alang_extra |= lab == i; continue
    comps.append(i)
big = [i for i in comps if st[i, 4] > 1500]
groups = {i: [i] for i in big}
for i in comps:
    if i in big: continue
    j = min(big, key=lambda b: abs(cen[b][0] - cen[i][0]) + 0.3 * abs(cen[b][1] - cen[i][1]))
    groups[j].append(i)
dists = []
keys = sorted(groups, key=lambda i: cen[i][0])
for k in keys:
    m = np.isin(lab, groups[k]).astype(np.uint8)
    dists.append(cv2.distanceTransform(1 - m, cv2.DIST_L2, 5))
assign = np.argmin(np.stack(dists), 0)
ae = dil(alang_extra, 9)
letters, swooshes = [], []
for idx, k in enumerate(keys):
    a = pa * (assign == idx) * ~ae
    nm = f'port{idx}'
    add_sprite(nm, a)
    (swooshes if st[k, 1] > 585 else letters).append(nm)
A['alang'] = np.maximum(A['alang'], pa * ae)

# -- date banner: clean board + icon + typed characters
da = A['date']
inside = da > 0.6
navy = (mx8 < 150) & inside & (yy > 700) & (yy < 815)
orange = (Rc > 190) & (Gc < 175) & (Bc < 130) & inside & (xx > 195) & (xx < 348) & (yy > 698) & (yy < 815)
hole = (dil(navy | orange, 9) * 255).astype(np.uint8)
board = cv2.inpaint(orig8, hole, 5, cv2.INPAINT_TELEA).astype(np.float32)
board = np.where(hole[..., None] > 0, cv2.GaussianBlur(board, (0, 0), 3), board)
add_sprite('board', da, board)
def feather(m, k=5): return cv2.GaussianBlur(dil(m, k).astype(np.float32), (0, 0), 1.2)
add_sprite('icon', feather(orange) * (da > 0.3))
n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(navy.astype(np.uint8), 8)
chars = []
order = sorted([i for i in range(1, n2) if st2[i, 4] > 60], key=lambda i: st2[i, 0])
for j, i in enumerate(order):
    add_sprite(f'ch{j}', feather(lab2 == i, 5) * (da > 0.3)); chars.append(f'ch{j}')

for k in ['langkah', 'alang', 'ticks1', 'ticks2', 'pin', 'umb', 'umbA', 'umbB', 'coco']:
    add_sprite(k, A[k])
title_alpha = np.clip(np.maximum.reduce([A['langkah'], A['alang'], A['port']]), 0, 1)
print('letters', len(letters), 'swooshes', len(swooshes), 'chars', len(chars))

# ------------------------------------------------------------------ easing
def prog(t, a, b): return min(1.0, max(0.0, (t - a) / (b - a)))
def out_cubic(t): return 1 - (1 - t) ** 3
def out_quint(t): return 1 - (1 - t) ** 5
def in_out(t): return t * t * t * (t * (6 * t - 15) + 10)
def out_back(t, s=1.7): t -= 1; return t * t * ((s + 1) * t + s) + 1
def spring(p, damp=5.5, freq=2.0):
    if p >= 1: return 1.0
    f = lambda q: 1 - math.exp(-damp * q) * math.cos(2 * math.pi * freq * q)
    return f(p) / f(1.0)
def bounce(t):
    n, d = 7.5625, 2.75
    if t < 1 / d: return n * t * t
    if t < 2 / d: t -= 1.5 / d; return n * t * t + .75
    if t < 2.5 / d: t -= 2.25 / d; return n * t * t + .9375
    t -= 2.625 / d; return n * t * t + .984375

# ------------------------------------------------------------------ drawing
def draw(fr, k, dx=0., dy=0., scale=1., rot=0., alpha=1., pivot=None, wipe=None, sx=None, slant=0.0):
    if alpha <= 0.003 or k not in S: return
    s = S[k]
    px, py = pivot if pivot else (s['cx'], s['cy'])
    sxv = scale if sx is None else sx
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    M = np.array([[c * sxv, -si * scale], [si * sxv, c * scale]])
    off = np.array([px + dx, py + dy]) - M @ np.array([px, py])
    T = np.hstack([M, (M @ np.array([s['x0'], s['y0']]) + off)[:, None]])
    corners = np.array([[0, 0], [s['w'], 0], [0, s['h']], [s['w'], s['h']]], np.float32)
    pts = corners @ T[:, :2].T + T[:, 2]
    bx0, by0 = int(max(0, math.floor(pts[:, 0].min()))), int(max(0, math.floor(pts[:, 1].min())))
    bx1, by1 = int(min(W, math.ceil(pts[:, 0].max()) + 1)), int(min(H, math.ceil(pts[:, 1].max()) + 1))
    if bx1 <= bx0 or by1 <= by0: return
    T2 = T.copy(); T2[:, 2] -= [bx0, by0]
    a = s['a']
    if wipe is not None and wipe < 1:
        gx = np.arange(s['w'], dtype=np.float32)[None, :] / s['w']
        gy = np.arange(s['h'], dtype=np.float32)[:, None] / s['h']
        edge = wipe * (1.14 + abs(slant)) - max(0, slant)
        a = a * np.clip((edge - gx + slant * gy) / 0.10, 0, 1)
    simple = abs(rot) < 1e-4 and abs(scale - 1) < 1e-5 and abs(sxv - 1) < 1e-5
    fl = cv2.INTER_LINEAR
    rgb = cv2.warpAffine(s['rgb'] * a[..., None], T2, (bx1 - bx0, by1 - by0), flags=fl)
    al = cv2.warpAffine(a, T2, (bx1 - bx0, by1 - by0), flags=fl)[..., None] * alpha
    reg = fr[by0:by1, bx0:bx1]
    reg[:] = reg * (1 - al) + rgb * alpha

def blob(fr, x, y, r, color, alpha):
    """soft round puff"""
    x0, y0, x1, y1 = int(max(0, x - 2.5 * r)), int(max(0, y - 2.5 * r)), int(min(W, x + 2.5 * r)), int(min(H, y + 2.5 * r))
    if x1 <= x0 or y1 <= y0: return
    g = np.exp(-((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * r * r))[..., None] * alpha
    reg = fr[y0:y1, x0:x1]; reg[:] = reg * (1 - g) + np.array(color, np.float32) * g

def star_kernel(sz):
    r = sz // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    corek = np.exp(-(x * x + y * y) / (2 * (sz * 0.07) ** 2))
    ray = np.exp(-np.abs(x) / (sz * 0.2)) * np.exp(-y * y / 2.2) + np.exp(-np.abs(y) / (sz * 0.2)) * np.exp(-x * x / 2.2)
    halo = 0.25 * np.exp(-(x * x + y * y) / (2 * (sz * 0.2) ** 2))
    return np.clip(corek + 0.85 * ray + halo, 0, 1)
STARS = {s: star_kernel(s) for s in (41, 61, 81, 111)}
def sparkle(fr, x, y, sz, inten, color=(215, 250, 255)):
    k = STARS[sz]; r = sz // 2
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + r + 1), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0: return
    kk = k[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)][..., None] * inten
    reg = fr[y0:y1, x0:x1]; reg[:] = reg * (1 - kk) + np.array(color, np.float32) * kk

# ------------------------------------------------------------------ wind / water displacement
def gauss(cx, cy, sx, sy): return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
leafw = [(gauss(130, 230, 230, 260) * np.clip(xx / 120, 0, 1), 0.0), (gauss(1050, 90, 160, 160), 1.3),
         (gauss(1090, 520, 120, 230), 2.1), (gauss(40, 880, 110, 110), 0.7), (gauss(10, 1150, 60, 60), 2.8)]
sea = ((Bc > Rc + 45) & (yy > 1128) & (yy < 1335)).astype(np.uint8)
sea = cv2.erode(sea, np.ones((9, 9), np.uint8))
seaw = cv2.GaussianBlur(sea.astype(np.float32), (0, 0), 5)
def world(t, wind, water):
    if wind < 0.03 and water < 0.03: return bg.copy()
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    for w, ph in leafw:
        g = math.sin(2 * math.pi * 0.8 * t + ph) + 0.35 * math.sin(2 * math.pi * 1.9 * t + ph * 2)
        dx += w * (g * wind); dy += w * (g * wind * 0.35)
    if water > 0.03:
        dx += seaw * water * (2.6 * np.sin(yy * 0.11 + 2 * math.pi * 0.55 * t) + 1.4 * np.sin(yy * 0.045 - 2 * math.pi * 0.3 * t + xx * 0.01))
        dy += seaw * water * (1.3 * np.sin(xx * 0.022 + 2 * math.pi * 0.45 * t))
    return cv2.remap(bg, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ------------------------------------------------------------------ particles
CONF_T0 = 5.32
PAL = [(60, 200, 255), (40, 150, 255), (255, 255, 255), (200, 190, 40), (140, 110, 255), (120, 230, 150)]  # BGR
conf = []
for i in range(90):
    ang = rng.uniform(-math.pi, 0) if i % 5 else rng.uniform(0, math.pi)
    spd = rng.uniform(380, 1150)
    conf.append(dict(x=565 + rng.uniform(-330, 330), y=757 + rng.uniform(-30, 30), vx=math.cos(ang) * spd, vy=math.sin(ang) * spd * 0.9 - 250,
                     rot=rng.uniform(0, 360), vr=rng.uniform(-720, 720), w=rng.uniform(9, 19), h=rng.uniform(5, 11),
                     col=PAL[i % len(PAL)], life=rng.uniform(1.2, 1.85), flip=rng.uniform(4, 9), ph=rng.uniform(0, 6)))
petals = []
for i in range(16):
    t0 = rng.uniform(0.2, 4.3)
    petals.append(dict(t0=t0, dur=rng.uniform(1.8, 2.6), y=rng.uniform(120, 1250), amp=rng.uniform(30, 90), ph=rng.uniform(0, 6),
                       sz=rng.uniform(9, 17), vr=rng.uniform(200, 520), drop=rng.uniform(-60, 220)))
twinkles = []
spots = [(300, 110), (790, 95), (240, 400), (880, 330), (120, 470), (1020, 500), (560, 430), (700, 280), (150, 700), (980, 705),
         (565, 810), (240, 960), (890, 965), (430, 240), (640, 600), (960, 610), (350, 520), (800, 520)]
for i, (x, y) in enumerate(spots):
    twinkles.append(dict(x=x + rng.uniform(-12, 12), y=y + rng.uniform(-12, 12), t0=5.75 + (i * 0.37) % 1.25, sz=[41, 61, 81][i % 3]))

def draw_overlay(fr, t):
    """confetti + petals on an anti-aliased overlay"""
    polys = []
    for p in petals:
        q = (t - p['t0']) / p['dur']
        if 0 < q < 1:
            x = -60 + (W + 120) * q
            y = p['y'] + p['amp'] * math.sin(q * 7 + p['ph']) + p['drop'] * q
            ang = p['vr'] * (t - p['t0'])
            polys.append(('e', x, y, p['sz'], p['sz'] * (0.35 + 0.3 * abs(math.sin(q * 9 + p['ph']))), ang, (245, 250, 255), min(1, q * 6, (1 - q) * 6)))
    tc = t - CONF_T0
    if tc > 0:
        for c in conf:
            if tc < c['life']:
                drag = math.exp(-1.6 * tc)
                x = c['x'] + c['vx'] * (1 - drag) / 1.6
                y = c['y'] + c['vy'] * (1 - drag) / 1.6 + 0.5 * 620 * tc * tc
                hh = c['h'] * abs(math.cos(c['flip'] * tc + c['ph'])) + 1.5
                fade = min(1.0, (c['life'] - tc) / 0.35)
                polys.append(('r', x, y, c['w'], hh, c['rot'] + c['vr'] * tc, c['col'], fade))
    if not polys: return
    xs = [p[1] for p in polys]; ys = [p[2] for p in polys]
    x0, y0 = int(max(0, min(xs) - 40)), int(max(0, min(ys) - 40))
    x1, y1 = int(min(W, max(xs) + 40)), int(min(H, max(ys) + 40))
    if x1 <= x0 or y1 <= y0: return
    col = np.zeros((y1 - y0, x1 - x0, 3), np.uint8); al = np.zeros((y1 - y0, x1 - x0), np.uint8)
    SH = 4
    for kind, x, y, w, h, ang, color, a in polys:
        cxy = (int((x - x0) * 16), int((y - y0) * 16))
        if kind == 'e':
            cv2.ellipse(col, cxy, (int(w * 16), int(h * 16)), ang, 0, 360, color, -1, cv2.LINE_AA, SH)
            cv2.ellipse(al, cxy, (int(w * 16), int(h * 16)), ang, 0, 360, int(235 * a), -1, cv2.LINE_AA, SH)
        else:
            box = cv2.boxPoints(((x - x0, y - y0), (w, h), ang))
            pts = (box * 16).astype(np.int32)
            cv2.fillConvexPoly(col, pts, color, cv2.LINE_AA, SH)
            cv2.fillConvexPoly(al, pts, int(255 * a), cv2.LINE_AA, SH)
    af = (al.astype(np.float32) / 255)[..., None]
    reg = fr[y0:y1, x0:x1]; reg[:] = reg * (1 - af) + col.astype(np.float32) * af

# ------------------------------------------------------------------ scene
T_END_MOTION = 7.25      # everything settled, start blending to the exact poster
T_EXACT = 7.7

def elements(fr, t):
    # big umbrella: tumbles in on the wind, lands, wobbles
    p = prog(t, 0.25, 1.9)
    if p > 0:
        e = out_cubic(p)
        dx = -900 * (1 - e) + 70 * math.sin(p * math.pi * 2) * (1 - e)
        dy = -1150 * (1 - e) ** 1.5
        rot = -220 * (1 - e) ** 1.2
        l = prog(t, 1.88, 2.9)
        wob = 0.0; sq = 0.0
        if 0 < l < 1:
            wob = 5.5 * math.exp(-4.5 * l) * math.sin(l * 2 * math.pi * 2.5)
            sq = 0.07 * math.exp(-5 * l) * math.cos(l * 2 * math.pi * 2.2)
        sc = 0.55 + 0.45 * e
        draw(fr, 'umb', dx, dy, scale=sc * (1 - sq), sx=sc * (1 + sq), rot=rot + wob, pivot=(568, 1396))
    # landing dust
    q = prog(t, 1.88, 2.55)
    if 0 < q < 1:
        for i in range(7):
            a = -math.pi * (0.08 + 0.84 * i / 6)
            d = 30 + 95 * out_cubic(q)
            blob(fr, 568 + math.cos(a) * d * 1.5, 1398 + math.sin(a) * d * 0.35, 12 + 20 * q, (190, 222, 240), 0.5 * (1 - q) ** 1.5)
    # small umbrellas ride the wind then bob
    for k, t0, t1, sx_, sy_, r_, ph in (('umbA', 0.9, 3.0, 420, 560, 38, 0.0), ('umbB', 1.2, 3.2, -340, -420, -42, 1.2)):
        p = prog(t, t0, t1)
        if p > 0:
            e = out_cubic(p)
            calm = 1 - in_out(prog(t, 5.6, 7.1))
            bob = 7 * math.sin(2 * math.pi * 0.55 * t + ph) * calm * e
            draw(fr, k, sx_ * (1 - e) + 0.5 * bob, sy_ * (1 - e) + 45 * math.sin(p * 7) * (1 - e) + bob,
                 rot=r_ * math.sin(p * 6) * (1 - e) + 5 * math.sin(2 * math.pi * 0.5 * t + ph) * calm * e)
    # coconut rolls in, hops, rocks to rest
    p = prog(t, 0.2, 1.35)
    if p > 0:
        e = out_cubic(p)
        dist = 660 * (1 - e)
        hop = -abs(math.sin(e * math.pi * 3)) * 46 * (1 - e)
        r2 = prog(t, 1.35, 2.3)
        rock = 9 * math.exp(-4 * r2) * math.sin(r2 * 2 * math.pi * 2.2) if 0 < r2 < 1 else 0
        draw(fr, 'coco', dist, hop, rot=math.degrees(dist / 76) + rock, pivot=(624, 1488) if dist < 1 else None)
    # "Langkah" written on, with a glint riding the pen
    p = prog(t, 2.5, 3.35)
    if p > 0:
        w = in_out(p)
        draw(fr, 'langkah', wipe=w, slant=0.25)
        if 0 < p < 1:
            s = S['langkah']
            sparkle(fr, s['x0'] + s['w'] * w, s['y0'] + s['h'] * (0.75 - 0.5 * w), 61, math.sin(p * math.pi) * 0.9)
    p = prog(t, 3.25, 3.6)
    if p > 0: draw(fr, 'ticks1', scale=out_back(p, 3.2), rot=-25 * (1 - out_cubic(p)), alpha=min(1, p * 4), pivot=(835, 255))
    # "Alang Ke": brush swipes across and pops
    p = prog(t, 3.2, 3.85)
    if p > 0:
        e = spring(p, 5, 1.6)
        draw(fr, 'alang', scale=0.82 + 0.18 * e, rot=-3 * (1 - e), wipe=out_cubic(min(1, p * 1.7)), slant=-0.2)
    # "Port Dickson": letters drop in one by one
    for i, nm in enumerate(letters):
        t0 = 3.85 + 0.06 * i
        p = prog(t, t0, t0 + 0.62)
        if p > 0:
            e = spring(p, 5.2, 1.9)
            draw(fr, nm, 0, -130 * (1 - e), scale=1 + 0.4 * (1 - e), rot=(10 if i % 2 else -10) * (1 - e), alpha=min(1, p * 5))
    for i, nm in enumerate(sorted(swooshes, key=lambda k: S[k]['x0'])):
        p = prog(t, 4.6 + 0.12 * i, 5.0 + 0.12 * i)
        if p > 0: draw(fr, nm, wipe=out_cubic(p))
    # location pin drops
    p = prog(t, 4.75, 5.3)
    if p > 0:
        e = bounce(p)
        draw(fr, 'pin', 0, -380 * (1 - e), alpha=min(1, p * 6))
    p = prog(t, 5.05, 5.4)
    if p > 0: draw(fr, 'ticks2', scale=out_back(p, 3.2), rot=25 * (1 - out_cubic(p)), alpha=min(1, p * 4), pivot=(925, 600))
    # date: board slams in, icon pops, date is typed
    p = prog(t, 5.3, 5.95)
    if p > 0:
        e = spring(p, 5.5, 1.8)
        draw(fr, 'board', scale=0.15 + 0.85 * e, rot=-7 * (1 - e) * math.cos(p * 8), alpha=min(1, p * 5))
        p2 = prog(t, 5.62, 5.95)
        if p2 > 0: draw(fr, 'icon', scale=out_back(p2, 2.5), alpha=min(1, p2 * 4), pivot=(256, 757))
        for i, nm in enumerate(chars):
            t0 = 5.8 + 0.045 * i
            pc = prog(t, t0, t0 + 0.22)
            if pc > 0: draw(fr, nm, 0, -14 * (1 - out_cubic(pc)), scale=1 + 0.7 * (1 - out_back(pc, 2.2)), alpha=min(1, pc * 3))
    draw_overlay(fr, t)

def post(fr, t):
    # twinkles
    for tw in twinkles:
        p = prog(t, tw['t0'], tw['t0'] + 0.55)
        if 0 < p < 1: sparkle(fr, tw['x'], tw['y'], tw['sz'], math.sin(p * math.pi) ** 1.5 * 0.95)
    # light sweep over the title
    p = prog(t, 6.15, 7.05)
    if 0 < p < 1:
        c = -350 + (W + 900) * in_out(p)
        band = np.exp(-((xx[60:660] + 0.6 * yy[60:660] - c) / 70) ** 2) * title_alpha[60:660] * 0.75
        reg = fr[60:660]; reg[:] = reg + (255 - reg) * band[..., None]
    # flashes
    f1 = prog(t, 5.3, 5.62)
    if 0 < f1 < 1: fr[:] = fr + (255 - fr) * (math.sin(f1 * math.pi) ** 2 * 0.42)
    f0 = prog(t, 0.0, 0.55)
    if f0 < 1: fr[:] = fr + (255 - fr) * ((1 - f0) ** 2)
    return fr

SHAKES = [(1.9, 9), (4.55, 4), (5.32, 7)]
def camera(fr, t):
    z = in_out(prog(t, 0.0, 1.9))
    sc = 1.45 - 0.45 * z
    ox = oy = 0.0
    for t0, amp in SHAKES:
        d = t - t0
        if 0 < d < 0.6:
            k = amp * math.exp(-8 * d)
            ox += k * math.sin(60 * d); oy += k * math.cos(48 * d)
    # slow push-in while the title builds, eased back out before the end
    breathe = 0.02 * math.sin(math.pi * prog(t, 2.0, T_END_MOTION)) ** 2
    sc += breathe
    if sc < 1.0005 and abs(ox) < 0.05 and abs(oy) < 0.05: return fr
    cx, cy = W / 2, H * 0.34 * (1 - z) + H / 2 * z
    M = np.array([[sc, 0, W / 2 - sc * cx + ox], [0, sc, H / 2 - sc * cy + oy]], np.float32)
    if z < 1: M[1, 2] = min(0, max(H - sc * H, M[1, 2]))
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

def render(t, sub):
    if t >= T_EXACT: return orig8
    wind = 15 * (1 - 0.75 * in_out(prog(t, 4.2, 6.0))) * (1 - in_out(prog(t, 6.0, T_END_MOTION)))
    water = 1 - in_out(prog(t, 6.0, T_END_MOTION))
    base = world(t, wind, water)
    acc = np.zeros_like(base)
    for j in range(sub):
        f = base.copy() if sub > 1 else base
        elements(f, t + (j / sub) / FPS)
        acc += f
    fr = acc / sub
    fr = post(fr, t)
    q = in_out(prog(t, T_END_MOTION, T_EXACT))
    if q > 0: fr = fr * (1 - q) + orig * q
    fr = camera(fr, t)
    return np.clip(fr, 0, 255).astype(np.uint8)

if PREVIEW:
    times = [float(x) for x in sys.argv[2:]]
    ims = [cv2.resize(render(t, 3), (377, 533), interpolation=cv2.INTER_AREA) for t in times]
    rows = [np.hstack(ims[i:i + 5]) for i in range(0, len(ims), 5)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite('sheet.jpg', np.vstack(rows))
    sys.exit()

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                       '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '4', '-pix_fmt', 'yuv444p', 'master.mkv'],
                      stdin=subprocess.PIPE)
for i in range(N):
    t = i / FPS
    ff.stdin.write(render(t, 4 if t < 7.2 else 1).tobytes())
    if i % 60 == 0: print('t', t, flush=True)
ff.stdin.close(); ff.wait()
print('done')
