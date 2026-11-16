import cv2, numpy as np, os, math, subprocess, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))
orig = cv2.imread('poster.jpg').astype(np.float32)
bg = cv2.imread('bg.png').astype(np.float32)
A = dict(np.load('alphas.npz'))
H, W = orig.shape[:2]
FPS, DUR = 30, 8.0
N = int(FPS * DUR)

def bbox(a):
    ys, xs = np.nonzero(a > 0.02)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

# sprites: premultiplied crops
S = {}
for k, a in A.items():
    x0, y0, x1, y1 = bbox(a)
    pad = 4
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    S[k] = dict(rgb=orig[y0:y1, x0:x1].copy(), a=a[y0:y1, x0:x1].copy(), x0=x0, y0=y0, w=x1 - x0, h=y1 - y0)

def ease_out_back(t, s=1.7):
    t -= 1; return t * t * ((s + 1) * t + s) + 1
def ease_out_cubic(t): return 1 - (1 - t) ** 3
def ease_in_out(t): return t * t * (3 - 2 * t)
def ease_out_bounce(t):
    n, d = 7.5625, 2.75
    if t < 1 / d: return n * t * t
    if t < 2 / d: t -= 1.5 / d; return n * t * t + .75
    if t < 2.5 / d: t -= 2.25 / d; return n * t * t + .9375
    t -= 2.625 / d; return n * t * t + .984375
def prog(t, a, b): return min(1.0, max(0.0, (t - a) / (b - a)))

def draw(frame, k, dx=0., dy=0., scale=1., rot=0., alpha=1., pivot=None, wipe=None, sx=None):
    """Draw sprite k transformed about pivot (default sprite centre, image coords)."""
    if alpha <= 0.003: return
    s = S[k]
    px, py = pivot if pivot else (s['x0'] + s['w'] / 2, s['y0'] + s['h'] / 2)
    sxv = scale if sx is None else sx
    # affine: p' = R*Sc*(p - pivot) + pivot + d
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    M = np.array([[c * sxv, -si * scale], [si * sxv, c * scale]])
    off = np.array([px + dx, py + dy]) - M @ np.array([px, py])
    # sprite local -> image: p = local + (x0,y0)
    T = np.hstack([M, (M @ np.array([s['x0'], s['y0']]) + off)[:, None]])
    corners = np.array([[0, 0], [s['w'], 0], [0, s['h']], [s['w'], s['h']]], np.float32)
    pts = corners @ T[:, :2].T + T[:, 2]
    bx0, by0 = int(max(0, math.floor(pts[:, 0].min()))), int(max(0, math.floor(pts[:, 1].min())))
    bx1, by1 = int(min(W, math.ceil(pts[:, 0].max()) + 1)), int(min(H, math.ceil(pts[:, 1].max()) + 1))
    if bx1 <= bx0 or by1 <= by0: return
    T2 = T.copy(); T2[:, 2] -= [bx0, by0]
    ww, hh = bx1 - bx0, by1 - by0
    a = s['a'].copy()
    if wipe is not None:  # left->right reveal, wipe in [0,1]
        xs = np.arange(s['w'])[None, :] / s['w']
        edge = wipe * 1.15
        a = a * np.clip((edge - xs) / 0.12, 0, 1)
    rgb = cv2.warpAffine(s['rgb'] * a[..., None], T2, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    al = cv2.warpAffine(a, T2, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None] * alpha
    region = frame[by0:by1, bx0:bx1]
    region[:] = region * (1 - al) + rgb * alpha

# palm sway: displacement field weighted around leaf areas
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def gauss(cx, cy, sx, sy): return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
leafw = [  # (weight field, anchor-ish phase)
    (gauss(130, 230, 230, 260) * np.clip(xx / 120, 0, 1), 0.0),     # top-left palm
    (gauss(1050, 90, 160, 160), 1.3),                                # top-right palm
    (gauss(1090, 520, 120, 230), 2.1),                               # right palm
    (gauss(40, 880, 110, 110), 0.7),                                 # left lower palm
    (gauss(10, 1150, 60, 60), 2.8),                                  # tiny palm by rocks
]
def sway(img, t, amp):
    if amp < 0.05: return img
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    for w, ph in leafw:
        g = math.sin(2 * math.pi * 0.9 * t + ph) + 0.35 * math.sin(2 * math.pi * 2.1 * t + ph * 2)
        dx += w * g * amp
        dy += w * g * amp * 0.35
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

frames_dir = 'frames'
shutil.rmtree(frames_dir, ignore_errors=True); os.makedirs(frames_dir)

for i in range(N):
    t = i / FPS
    # wind strength: strong early, settles to zero by 6.2s
    amp = 14 * (1 - prog(t, 4.5, 6.3)) ** 2 + 0.0
    fr = sway(bg, t, amp)

    # --- big umbrella flies in from top-left, tumbling, lands at 2.0s
    p = prog(t, 0.2, 2.0)
    if p > 0:
        e = ease_out_cubic(p)
        dx = (-900) * (1 - e) + 60 * math.sin(p * math.pi * 2) * (1 - e)
        dy = (-1100) * (1 - e) ** 1.5
        rot = -200 * (1 - e) + 10 * math.sin(p * 9) * (1 - p)
        land = prog(t, 1.8, 2.5)
        sq = 1.0
        if 0 < land < 1: sq = 1 + 0.05 * math.sin(land * math.pi * 3) * (1 - land)
        draw(fr, 'umb', dx, dy, scale=0.6 + 0.4 * e, rot=rot, sx=(0.6 + 0.4 * e) * (2 - sq),
             pivot=(568, 1395))
    # --- small umbrellas drift in on the wind
    p = prog(t, 0.9, 3.0)
    if p > 0:
        e = ease_out_cubic(p)
        draw(fr, 'umbA', 380 * (1 - e), 520 * (1 - e) + 50 * math.sin(p * 7) * (1 - e),
             rot=35 * math.sin(p * 6) * (1 - e) + 6 * math.sin(t * 3) * (1 - prog(t, 4, 6)))
    p = prog(t, 1.2, 3.2)
    if p > 0:
        e = ease_out_cubic(p)
        draw(fr, 'umbB', -320 * (1 - e), -380 * (1 - e) + 40 * math.sin(p * 8) * (1 - e),
             rot=-40 * math.sin(p * 6) * (1 - e) + 6 * math.sin(t * 3 + 1) * (1 - prog(t, 4, 6)))
    # --- coconut rolls from the right to centre with a little bounce
    p = prog(t, 0.2, 1.3)
    if p > 0:
        e = ease_out_cubic(p)
        dist = 640 * (1 - e)
        hop = -abs(math.sin(e * math.pi * 3)) * 40 * (1 - e)
        rot = math.degrees(dist / 76)
        draw(fr, 'coco', dist, hop, rot=rot)
    # --- "Langkah" written in (wipe) with slight slide
    p = prog(t, 2.8, 3.6)
    if p > 0:
        draw(fr, 'langkah', -30 * (1 - ease_out_cubic(p)), 0, wipe=ease_in_out(p))
    # --- "Alang Ke" brush swipe + pop
    p = prog(t, 3.4, 4.0)
    if p > 0:
        draw(fr, 'alang', 0, 0, scale=0.85 + 0.15 * ease_out_back(p), wipe=ease_out_cubic(min(1, p * 1.4)))
    # --- "Port Dickson" drops & bounces
    p = prog(t, 3.8, 4.6)
    if p > 0:
        e = ease_out_back(p, 1.4)
        draw(fr, 'port', 0, 140 * (1 - e), scale=0.85 + 0.15 * e, alpha=min(1, p * 2.5))
    # --- location pin drops
    p = prog(t, 4.3, 4.9)
    if p > 0:
        draw(fr, 'pin', 0, -360 * (1 - ease_out_bounce(p)), alpha=min(1, p * 5))
    # --- sparkle ticks pop
    for k, a0 in (('ticks1', 4.6), ('ticks2', 4.75)):
        p = prog(t, a0, a0 + 0.35)
        if p > 0: draw(fr, k, scale=ease_out_back(p, 3), alpha=min(1, p * 3))
    # --- date appears suddenly: flash + pop with wobble
    p = prog(t, 4.9, 5.5)
    if p > 0:
        e = ease_out_back(p, 2.6)
        draw(fr, 'date', scale=0.2 + 0.8 * e, rot=-8 * (1 - p) * math.cos(p * 10), alpha=min(1, p * 3))
        fl = prog(t, 4.9, 5.25)
        if fl < 1:
            flash = math.sin(fl * math.pi) * 0.55
            fr = fr * (1 - flash) + 255 * flash

    # final settle: blend to the exact original poster
    q = ease_in_out(prog(t, 5.9, 6.4))
    if q > 0: fr = fr * (1 - q) + orig * q
    # camera: start zoomed on the sky, pull back to the full poster
    z = ease_in_out(prog(t, 0.0, 1.8))
    sc = 1.4 + (1.0 - 1.4) * z
    if sc > 1.0001:
        cx, cy = W / 2, H * 0.36 * (1 - z) + H / 2 * z
        M = np.array([[sc, 0, W / 2 - sc * cx], [0, sc, H / 2 - sc * cy]], np.float32)
        M[1, 2] = min(0, max(H - sc * H, M[1, 2]))
        fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(f'{frames_dir}/{i:04d}.png', np.clip(fr, 0, 255).astype(np.uint8))

print('frames', N)
