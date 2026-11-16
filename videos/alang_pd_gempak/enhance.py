"""Gempak pass for the 'Alang ke Port Dickson' teaser: upscale to 1080x1920 and add a
beat-synced finishing layer (camera punch, bloom, light leaks, speed lines, confetti, glints)."""
import cv2, numpy as np, math, subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'sr3x.mkv')          # AI-upscaled intermediate (3x of the 464x832 original)
ORIG = os.path.join(HERE, '..', 'pdmap_in', 'src.mp4')
W0, H0, FPS, NF = 1392, 2496, 60, 1799
W, H = 1080, 1920
K0 = W / W0
rng = np.random.default_rng(11)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------------------------------------------------------------- timing data (from analysis)
BASS = [(0.44, .35), (0.94, .32), (1.44, .33), (1.94, .3), (3.45, .3), (5.94, .43), (6.44, .42), (6.94, .37), (7.44, .37),
        (11.34, .34), (11.56, .33), (12.45, .57), (12.95, .7), (13.44, .75), (13.94, .76), (14.44, .75), (14.94, .75),
        (15.44, .72), (15.94, 1.0), (16.44, .54), (16.82, .56), (17.32, .54), (17.58, .37), (17.82, .52), (18.08, .39),
        (18.32, .53), (18.58, .43), (18.82, .52), (19.08, .39), (19.32, .47), (19.57, .4), (19.82, .46), (20.07, .41),
        (20.32, .52), (20.57, .43), (20.82, .51), (21.07, .42), (21.32, .45), (21.57, .39), (21.83, .46), (22.07, .38),
        (22.33, .43), (22.57, .36), (22.83, .46), (23.08, .34), (23.32, .5), (23.58, .36), (23.82, .5), (24.07, .34),
        (24.32, .52), (24.57, .32), (24.82, .53), (25.08, .38), (25.32, .52), (25.58, .47), (25.82, .57), (26.08, .37),
        (26.32, .47), (26.95, .71), (27.45, .53), (27.95, .49)]
# (time, strength, flash)  -- cuts and reveals
IMPACTS = [(8.50, .9, .25), (12.03, 1.0, .35), (12.95, .5, .0), (15.94, 1.0, .45), (16.44, .45, .0),
           (19.03, .6, .14), (20.53, .6, .14), (22.03, .6, .14), (23.53, .6, .14), (24.28, .6, .14), (25.03, .6, .14),
           (25.78, .55, .16), (25.95, .5, .12), (26.17, .5, .12), (26.42, .5, .12),
           (26.95, 1.0, .5), (27.45, .45, .0), (29.0, .35, .0)]
BIG = [12.03, 15.94, 26.95]                    # speed lines + flare + confetti
GLITCH = [7.44, 7.67, 7.88, 8.07, 8.25, 8.5, 8.6]
DRIFT = [(0.0, 8.5, 1.0, 1.075), (9.2, 12.0, 1.0, 1.035), (15.94, 18.4, 1.0, 1.05), (26.95, 29.0, 1.0, 1.06), (29.0, 30.0, 1.0, 1.04)]
GOLD = [(12.1, 12.9), (15.94, 18.4), (26.95, 29.0)]   # scenes with gold titles -> glints

def env(t, t0, tau): return math.exp(-(t - t0) / tau) if t >= t0 else 0.0

def camera(t):
    s = 1.03
    for a, b, s0, s1 in DRIFT:
        if a <= t < b: s += (s0 + (s1 - s0) * (t - a) / (b - a)) - 1
    quiet = 0.4 if t < 8.5 else 1.0
    for tb, a in BASS: s += 0.02 * a * quiet * env(t, tb, 0.11)
    ox = oy = rot = 0.0
    for t0, st, _ in IMPACTS:
        d = t - t0
        if 0 <= d < 0.9:
            s += 0.085 * st * math.exp(-d / 0.13)
            k = 15 * st * math.exp(-d / 0.16)
            ox += k * math.sin(70 * d + t0); oy += k * math.cos(57 * d + 2 * t0)
            rot += 1.2 * st * math.exp(-d / 0.15) * math.sin(45 * d)
    # gentle handheld float
    ox += 3.0 * math.sin(2 * math.pi * 0.23 * t); oy += 3.0 * math.sin(2 * math.pi * 0.17 * t + 1)
    return s, ox, oy, rot

def split_amount(t):
    k = 0.0
    for t0, st, _ in IMPACTS: k += 11 * st * env(t, t0, 0.10)
    for g in GLITCH[:5]: k += 7 * env(t, g, 0.05)
    return k

def flash_amount(t): return sum(fl * env(t, t0, 0.07) for t0, st, fl in IMPACTS)

# ---------------------------------------------------------------- static layers
r2 = ((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2
VIG = (1 - 0.34 * np.clip(r2, 0, 1.6) ** 1.25).astype(np.float32)[..., None]
GRAIN = [cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7) * 7.0 for _ in range(6)]

def make_lines(seed):
    r = np.random.default_rng(seed)
    ang = np.arctan2(yy - H * 0.47, xx - W / 2)
    nb = 720
    v = r.random(nb) ** 6
    v = np.convolve(np.concatenate([v, v[:3]]), [0.25, 0.5, 0.25], 'same')[:nb]
    idx = ((ang + math.pi) / (2 * math.pi) * nb).astype(np.int32) % nb
    rad = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.47) / H) ** 2)
    fall = np.clip((rad - 0.22) / 0.33, 0, 1) ** 1.5
    return (v[idx] * fall).astype(np.float32)
LINES = [make_lines(s) for s in (1, 2, 3)]

def star_kernel(sz):
    r = sz // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    corek = np.exp(-(x * x + y * y) / (2 * (sz * 0.06) ** 2))
    ray = np.exp(-np.abs(x) / (sz * 0.2)) * np.exp(-y * y / 2.5) + np.exp(-np.abs(y) / (sz * 0.2)) * np.exp(-x * x / 2.5)
    d1, d2 = (x + y) / 1.414, (x - y) / 1.414
    ray2 = 0.45 * (np.exp(-np.abs(d1) / (sz * 0.1)) * np.exp(-d2 * d2 / 2) + np.exp(-np.abs(d2) / (sz * 0.1)) * np.exp(-d1 * d1 / 2))
    halo = 0.22 * np.exp(-(x * x + y * y) / (2 * (sz * 0.2) ** 2))
    return np.clip(corek + 0.9 * ray + ray2 + halo, 0, 1)[..., None].astype(np.float32)
STARS = {s: star_kernel(s) for s in (61, 91, 131)}
def stamp(fr, k, x, y, inten, color):
    r = k.shape[0] // 2
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + r + 1), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0: return
    kk = k[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)] * inten
    reg = fr[y0:y1, x0:x1]; reg += (np.array(color, np.float32) - reg) * kk

def disc_kernel(sz):
    r = sz // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / r
    return (np.clip((1 - d) / 0.25, 0, 1) * (0.55 + 0.45 * d ** 2))[..., None].astype(np.float32)
DISCS = {s: disc_kernel(s) for s in (15, 25, 41, 61)}
dust = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), vx=rng.uniform(-12, 12), vy=rng.uniform(-45, -12), sz=int(rng.choice([15, 15, 25, 25, 41, 61])),
             ph=rng.uniform(0, 6), fr=rng.uniform(0.3, 1.1), col=[(150, 215, 255), (255, 235, 210), (200, 160, 255)][i % 3]) for i in range(46)]

# confetti bursts (drawn at half res -> soft, reads as foreground depth)
PAL = [(60, 200, 255), (40, 150, 255), (255, 255, 255), (200, 190, 40), (140, 110, 255), (120, 230, 150), (255, 120, 200)]
conf = []
for tb in BIG:
    n = 70 if tb != 26.95 else 120
    for i in range(n):
        side = i % 2
        x0 = -20 if side == 0 else W + 20
        ang = rng.uniform(-1.25, -0.25) if side == 0 else rng.uniform(math.pi + 0.25, math.pi + 1.25)
        spd = rng.uniform(900, 2300)
        conf.append(dict(t0=tb + rng.uniform(0, 0.12), x=x0, y=H * rng.uniform(0.55, 0.95), vx=math.cos(ang) * spd, vy=math.sin(ang) * spd,
                         rot=rng.uniform(0, 360), vr=rng.uniform(-700, 700), w=rng.uniform(16, 44), h=rng.uniform(9, 22),
                         col=PAL[i % len(PAL)], life=rng.uniform(1.5, 2.4), flip=rng.uniform(4, 9), ph=rng.uniform(0, 6)))

def draw_confetti(fr, t):
    act = [c for c in conf if 0 <= t - c['t0'] < c['life']]
    if not act: return
    hw, hh = W // 2, H // 2
    col = np.zeros((hh, hw, 3), np.uint8); al = np.zeros((hh, hw), np.uint8)
    for c in act:
        tc = t - c['t0']; drag = math.exp(-1.5 * tc)
        x = c['x'] + c['vx'] * (1 - drag) / 1.5
        y = c['y'] + c['vy'] * (1 - drag) / 1.5 + 0.5 * 900 * tc * tc
        h2 = c['h'] * abs(math.cos(c['flip'] * tc + c['ph'])) + 2
        fade = min(1.0, (c['life'] - tc) / 0.4)
        box = cv2.boxPoints(((x / 2, y / 2), (c['w'] / 2, h2 / 2), c['rot'] + c['vr'] * tc))
        pts = (box * 16).astype(np.int32)
        cv2.fillConvexPoly(col, pts, c['col'], cv2.LINE_AA, 4)
        cv2.fillConvexPoly(al, pts, int(255 * fade), cv2.LINE_AA, 4)
    a = cv2.resize(al, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    c3 = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    fr *= (1 - a); fr += c3 * a

# light leaks on a tiny grid
LW, LH = 54, 96
ly, lx = np.mgrid[0:LH, 0:LW].astype(np.float32)
def leaks(t, boost):
    out = np.zeros((LH, LW, 3), np.float32)
    for i, (col, sp, ph) in enumerate([((40, 140, 255), 0.11, 0.0), ((200, 70, 255), 0.07, 2.1), ((255, 190, 90), 0.09, 4.0)]):
        cx = LW * (0.5 + 0.62 * math.sin(2 * math.pi * sp * t + ph))
        cy = LH * (0.5 + 0.55 * math.cos(2 * math.pi * sp * 0.8 * t + ph * 1.7))
        g = np.exp(-(((lx - cx) / (LW * 0.5)) ** 2 + ((ly - cy) / (LH * 0.33)) ** 2))
        out += g[..., None] * np.array(col, np.float32) / 255
    return cv2.resize(out * (0.085 + boost), (W, H), interpolation=cv2.INTER_CUBIC)

glints = []  # active gold glints: [x, y, t0, size]
def update_glints(fr_src, t, M, fi):
    if any(a <= t < b for a, b in GOLD) and fi % 5 == 0:
        fs = fr_src[::3, ::3]
        b, g, r = fs[..., 0].astype(np.int16), fs[..., 1].astype(np.int16), fs[..., 2].astype(np.int16)
        m = (r > 225) & (g > 185) & (b < 150) & (r - b > 90)
        ys, xs = np.nonzero(m)
        if len(xs) > 200:
            for _ in range(2):
                j = rng.integers(len(xs))
                p = M @ np.array([xs[j] * 3.0, ys[j] * 3.0, 1.0])
                glints.append([p[0], p[1], t, int(rng.choice([61, 91, 91, 131]))])
    glints[:] = [g for g in glints if t - g[2] < 0.42]

LUT = np.clip((lambda v: 255 * (0.5 + (v / 255 - 0.5) * 1.1 + 0.035 * np.sin((v / 255) * 2 * math.pi - math.pi)))(np.arange(256, dtype=np.float32)), 0, 255)

def process(src, t, fi):
    small_src = src[::3, ::3]
    lum_mean = float(small_src.mean())
    gate = min(1.0, lum_mean / 14.0)                         # overlays fade out on black frames
    den = src
    # glitch slices in the build-up to the cut
    gs = sum(env(t, g, 0.045) for g in GLITCH[:5])
    if gs > 0.15:
        den = den.copy(); r = np.random.default_rng(fi)
        for _ in range(int(3 + 5 * gs)):
            y0 = int(r.integers(0, H0 - 140)); hh = int(r.integers(24, 138)); dx = int(r.integers(-100, 100) * min(1.5, gs))
            den[y0:y0 + hh] = np.roll(den[y0:y0 + hh], dx, axis=1)
    s, ox, oy, rot = camera(t)
    c, si = math.cos(math.radians(rot)) * K0 * s, math.sin(math.radians(rot)) * K0 * s
    M = np.array([[c, -si, W / 2 + ox - c * W0 / 2 + si * H0 / 2], [si, c, H / 2 + oy - si * W0 / 2 - c * H0 / 2]], np.float32)
    up = cv2.warpAffine(den, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    fr = up.astype(np.float32)
    # sharpen (two radii)
    fr += 0.3 * (fr - cv2.GaussianBlur(fr, (0, 0), 1.3))
    np.clip(fr, 0, 255, out=fr)
    # chromatic split
    k = split_amount(t)
    ki = int(round(k))
    if ki >= 1:
        fr[..., 2] = np.roll(fr[..., 2], ki, axis=1); fr[..., 0] = np.roll(fr[..., 0], -ki, axis=1)
    # grade: saturation + contrast curve
    gray = fr @ np.array([0.114, 0.587, 0.299], np.float32)
    fr = gray[..., None] + (fr - gray[..., None]) * 1.14
    fr = LUT[np.clip(fr, 0, 255).astype(np.uint8)].astype(np.float32)
    # bloom
    small = cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 168, 0, None) * 1.9
    glow = cv2.GaussianBlur(br, (0, 0), 5) * 0.55 + cv2.GaussianBlur(br, (0, 0), 16) * 0.45
    bgain = 0.62 * float(np.clip(1.5 - lum_mean / 110.0, 0.22, 1.0))
    imp = sum(st * env(t, t0, 0.25) for t0, st, _ in IMPACTS)
    glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR) * (bgain * (1 + 0.9 * imp))
    fr += (255 - fr) * np.clip(glow / 255, 0, 1)
    # light leaks + dust
    fr += (255 - fr) * np.clip(leaks(t, 0.10 * imp) * gate, 0, 1)
    for d in dust:
        x = (d['x'] + d['vx'] * t + 30 * math.sin(0.7 * t + d['ph'])) % W
        y = (d['y'] + d['vy'] * t) % H
        a = (0.10 + 0.10 * math.sin(2 * math.pi * d['fr'] * t + d['ph'])) * gate * (0.25 if t < 8.5 else 1.0)
        if a > 0.01: stamp(fr, DISCS[d['sz']], x, y, a, d['col'])
    # big-moment extras
    for tb in BIG:
        d = t - tb
        if 0 <= d < 0.7:
            a = (1 - d / 0.7) ** 2 * 0.75
            fr += (255 - fr) * (LINES[fi % 3] * a)[..., None]
            # anamorphic streak
            cy = H * 0.47
            band = np.exp(-((yy[:, :1] - cy) / (10 + 60 * d)) ** 2) * math.exp(-d / 0.18) * 0.8
            fr += (np.array([255, 225, 170], np.float32) - fr) * band[..., None]
    draw_confetti(fr, t)
    for g in glints:
        p = (t - g[2]) / 0.42
        stamp(fr, STARS[g[3]], g[0], g[1], math.sin(p * math.pi) ** 1.3, (225, 250, 255))
    fl = flash_amount(t)
    if fl > 0.004: fr += (255 - fr) * min(0.85, fl)
    fr *= VIG
    fr += GRAIN[fi % 6][..., None] * (0.55 + 0.45 * gate)
    return np.clip(fr, 0, 255).astype(np.uint8), M

def frames():
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', SRC, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    n = W0 * H0 * 3; i = 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield i, np.frombuffer(b, np.uint8).reshape(H0, W0, 3)
        i += 1

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'preview':
        want = {int(round(float(x) * FPS)): x for x in sys.argv[2:]}
        ims = []
        for i, f in frames():
            t = i / FPS
            Mtmp = None
            s, ox, oy, rot = camera(t)
            c, si = math.cos(math.radians(rot)) * K0 * s, math.sin(math.radians(rot)) * K0 * s
            Mtmp = np.array([[c, -si, W / 2 + ox - c * W0 / 2 + si * H0 / 2], [si, c, H / 2 + oy - si * W0 / 2 - c * H0 / 2]], np.float32)
            update_glints(f, t, Mtmp, i)
            if i in want:
                out, _ = process(f, t, i)
                cv2.imwrite(os.path.join(HERE, f'pv_{want[i]}.png'), out)
                ims.append(cv2.resize(out, (360, 640), interpolation=cv2.INTER_AREA))
            if i > max(want): break
        cv2.imwrite(os.path.join(HERE, 'pv_sheet.jpg'), np.hstack(ims))
        sys.exit()
    out = os.path.join(HERE, 'video_only.mkv')
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '8', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for i, f in frames():
        t = i / FPS
        s, ox, oy, rot = camera(t)
        c, si = math.cos(math.radians(rot)) * K0 * s, math.sin(math.radians(rot)) * K0 * s
        M = np.array([[c, -si, W / 2 + ox - c * W0 / 2 + si * H0 / 2], [si, c, H / 2 + oy - si * W0 / 2 - c * H0 / 2]], np.float32)
        update_glints(f, t, M, i)
        o, _ = process(f, t, i)
        ff.stdin.write(o.tobytes())
        if i % 120 == 0: print('t', round(t, 1), flush=True)
    ff.stdin.close(); ff.wait(); print('done')
