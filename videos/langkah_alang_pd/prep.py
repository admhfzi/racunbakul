import cv2, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
img = cv2.imread('poster.jpg')
H, W = img.shape[:2]
f = img.astype(np.float32)
B, G, R = f[..., 0], f[..., 1], f[..., 2]
mx, mn = f.max(2), f.min(2)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

cyan = (B > R + 40) & (G > R + 30) & (mn < 215)
cloud = (mn > 190) & (mx - mn < 70)
leaf = (G > B + 10) & (G > R - 18) & (mx - mn > 60)          # yellow-green fronds
brushy = (R > 170) & (R - G > 18) & (B < 150)                  # orange-yellow paint/umbrella

def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True; return m
def ell(cx, cy, rx, ry):
    m = np.zeros((H, W), np.uint8); cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1); return m > 0
def dil(m, k): return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
def ero(m, k): return cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

def pushpull(im, known):
    k = known.astype(np.float32)
    if min(im.shape[:2]) <= 2:
        mean = (im * k[..., None]).sum((0, 1)) / max(k.sum(), 1e-6)
        return np.broadcast_to(mean, im.shape).astype(np.float32)
    h, w = im.shape[:2]; sz = ((w + 1) // 2, (h + 1) // 2)
    num = cv2.resize(im * k[..., None], sz, interpolation=cv2.INTER_AREA)
    den = cv2.resize(k, sz, interpolation=cv2.INTER_AREA)
    filled = pushpull(num / np.maximum(den, 1e-6)[..., None], den > 1e-3)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    return im * k[..., None] + up * (1 - k[..., None])

def smooth_fill(known, sigma):
    """Clean smooth estimate of the sky from known pixels (normalised blur)."""
    k = known.astype(np.float32)
    est = pushpull(f, known)
    num = cv2.GaussianBlur(est * 1, (0, 0), sigma)
    return num

# ---------- regions holding the elements
REG = {
    'langkah': rect(280, 70, 820, 312),
    'ticks1': rect(818, 158, 945, 268),
    'alang': rect(225, 250, 890, 460),
    'pin': rect(800, 280, 988, 448),
    'port': rect(85, 402, 1036, 650),
    'ticks2': rect(908, 585, 1015, 695),
    'date': rect(122, 668, 1003, 845),
    'umbA': rect(975, 825, 1105, 950),
    'umbB': rect(35, 975, 152, 1098),
}
allreg = np.zeros((H, W), bool)
for m in REG.values(): allreg |= m

# ---------- sky model, two passes
known = ero(cyan & ~leaf & ~cloud, 7) & ~allreg
sky = cv2.GaussianBlur(pushpull(f, known), (0, 0), 25)
d1 = np.abs(f - sky).sum(2)
known = ero(cyan & ~leaf & ~cloud & (d1 < 30), 5) & ~ero(allreg & (d1 >= 30), 1)
known &= ~dil((d1 > 30) & allreg, 15)
known[1110:] = False
sky = cv2.GaussianBlur(pushpull(f, known), (0, 0), 14)
diff = np.abs(f - sky).sum(2)
soft = np.clip((diff - 16) / 34, 0, 1)

# ---------- element alphas
alphas = {}
claimed = np.zeros((H, W), bool)
for k in ['langkah', 'ticks1', 'alang', 'pin', 'ticks2', 'port', 'date', 'umbA', 'umbB']:
    m = REG[k] & ~claimed
    a = soft * m
    if k in ('umbA', 'umbB'):
        a = a * ~(cloud & ~brushy)
    else:
        textlike = (mn > 165) | ((B > R + 25) & (B >= G - 12)) | (mx < 90)
        if k in ('alang', 'pin', 'ticks1', 'ticks2', 'date'):
            textlike = textlike | brushy | ((R > 150) & (B < 140))
        palmzone = (xx < 330) | (xx > 975)
        if k.startswith('ticks'): palmzone = np.zeros_like(palmzone)
        a = a * dil(textlike & ~(leaf & ~brushy & palmzone), 3)
        if k == 'alang':
            a = a * np.where(yy > 398, dil(brushy, 3), True)
        if k == 'port':
            near = dil(mn > 165, 25)
            a = a * np.where(xx > 975, near, True)
    alphas[k] = a
    claimed |= dil(a > 0.05, 3)

canopy = rect(212, 842, 915, 1072)
ua = soft * canopy * ~(cloud & ~brushy & (yy > 965))
ua[798:842, 548:590] = soft[798:842, 548:590] * (B[798:842, 548:590] < 150)
pole = rect(551, 1060, 586, 1398) & (R - B > 60)
alphas['umb'] = np.maximum(ua, pole.astype(np.float32))
cm = (ell(624, 1414, 77, 75) | (rect(636, 1288, 694, 1362) & (R > 170) & (R - G > 40))).astype(np.float32)
alphas['coco'] = cm
for k in alphas:
    alphas[k] = cv2.GaussianBlur(alphas[k].astype(np.float32), (3, 3), 0)

# ---------- clean plate
bg = f.copy()
grain = np.random.default_rng(3).normal(0, 1.3, f.shape).astype(np.float32)
grain = cv2.GaussianBlur(grain, (0, 0), 0.6) * 1.6

# (1) upper sky: anything text-ish replaced by the smooth sky model
skyhole = np.zeros((H, W), np.float32)
for k in ['date', 'pin', 'port', 'alang', 'langkah', 'ticks1', 'ticks2', 'umbA']:
    skyhole = np.maximum(skyhole, alphas[k])
skyhole = np.maximum(skyhole, (dil(alphas['umb'] > 0.02, 9) & (yy < 962)).astype(np.float32))
hs = dil(skyhole > 0.04, 7)
w = cv2.GaussianBlur(hs.astype(np.float32), (0, 0), 2.0)[..., None]
kn = ~hs & ~dil(leaf | (G > B - 4), 11) & ~cloud
res = cv2.GaussianBlur(pushpull((f - sky), kn), (0, 0), 9)
fill = sky + res
zone = (xx > 955) & (yy > 425) & (yy < 600)
fz = cv2.GaussianBlur(pushpull(f, ~hs), (0, 0), 5)
zw = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), 12)[..., None]
fill = fill * (1 - zw) + fz * zw
bg = bg * (1 - w) + (fill + grain) * w

# (2) leaf bites: rebuild frond pixels that sat under the text by mirroring the
#     nearest visible frond texture (inpaint on the leaf layer only, small radius)
bite = hs & dil(leaf & ~hs, 25) & ~cyan
bg8 = np.clip(bg, 0, 255).astype(np.uint8)
bg8 = cv2.inpaint(bg8, (bite * 255).astype(np.uint8), 4, cv2.INPAINT_TELEA)
bg = bg8.astype(np.float32)

# (3) lower areas: Poisson-clone shifted copies of the poster (clouds, sea, sand)
def clone(dst, hole, dx, dy):
    """fill hole with pixels taken from (x+dx, y+dy) of dst, Poisson-blended"""
    src = np.roll(np.roll(dst, -dy, 0), -dx, 1)
    m = (dil(hole, 9) * 255).astype(np.uint8)
    m[:2] = 0; m[-2:] = 0; m[:, :2] = 0; m[:, -2:] = 0
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return dst
    c = (int((xs.min() + xs.max() + 1) / 2), int((ys.min() + ys.max() + 1) / 2))
    return cv2.seamlessClone(src, dst, m, c, cv2.NORMAL_CLONE)

bg8 = np.clip(bg, 0, 255).astype(np.uint8)
uh = dil(alphas['umb'] > 0.02, 7) & (yy >= 955) & (yy < 1080)
# rough prefill so Poisson boundaries never see the umbrella
pre = (dil(uh | ((alphas['umb'] > 0.04) & (yy >= 1075)) | (alphas['umbB'] > 0.04), 13) * 255).astype(np.uint8)
bg8 = cv2.inpaint(bg8, pre, 6, cv2.INPAINT_TELEA)
# one continuous cloud band: mirror-tile the real clouds on the right (x 935..1130)
# so the copy joins the real clouds with no seam, then Poisson-blend the whole band
XR0, XR1 = 935, 1130
def tri(v, lo, hi):            # reflect coordinates into [lo, hi]
    span = hi - lo
    m = np.mod(v - lo, 2 * span)
    return lo + np.where(m <= span, m, 2 * span - m)
srcf = bg8.astype(np.float32)
acc = np.zeros_like(srcf); wsum = np.zeros((H, W), np.float32)
step, half = 105, 150
tiles = [(935, -1, 1.0, 935, 0)]          # first tile mirrors the join so it continues the real clouds
prm = [(1, 0.72, 1010, 10), (-1, 1.35, 1085, -8), (1, 1.1, 960, 14), (-1, 0.8, 1040, -4),
       (1, 1.5, 1070, 8), (-1, 0.95, 990, -12), (1, 0.75, 1100, 6), (-1, 1.2, 1020, 0)]
for i, (dr, k, sc_, dy_) in enumerate(prm):
    tiles.append((935 - step * (i + 1), dr, k, sc_, dy_))
for c, dr, k, sc_, dy_ in tiles:
    w = np.cos(np.clip(np.abs(xx - c) / half, 0, 1) * np.pi / 2) ** 2
    mapx = tri(sc_ + (xx - c) * k * dr, XR0 + 2, XR1 - 2).astype(np.float32)
    t_ = cv2.remap(srcf, mapx, (yy + dy_).astype(np.float32), cv2.INTER_LINEAR)
    acc += t_ * w[..., None]; wsum += w
src = bg8.copy()
blend = acc / np.maximum(wsum, 1e-4)[..., None]
src[:, :XR0] = np.clip(blend[:, :XR0], 0, 255).astype(np.uint8)
band = np.zeros((H, W), np.uint8); band[872:1084, 216:XR0] = 255
bg8 = cv2.seamlessClone(src, bg8, band, ((216 + XR0) // 2, (872 + 1084) // 2), cv2.NORMAL_CLONE)
bg8 = clone(bg8, alphas['umbB'] > 0.04, 125, -6)
# pole over clouds/sea/sand: copy the strip just beside it
bg8 = clone(bg8, (alphas['umb'] > 0.04) & (yy >= 1075), 50, 0)
# coconut spot: mat/sand texture from the right, following the stripe slope
ch = alphas['coco'] > 0.04
bg8 = cv2.inpaint(bg8, (dil(ch, 13) * 255).astype(np.uint8), 6, cv2.INPAINT_TELEA)
bg8 = clone(bg8, ch & (yy >= 1395), 260, 75)
# upper coconut spot: sand/foam bands are horizontal -> row interpolation
up = dil(ch, 11) & (yy < 1402)
fb = bg8.astype(np.float32)
for y in range(1280, 1402):
    xs = np.nonzero(up[y])[0]
    if len(xs) == 0: continue
    xl, xr = max(xs.min() - 1, 0), min(xs.max() + 1, W - 1)
    t = np.linspace(0, 1, xr - xl + 1)[:, None]
    fb[y, xl:xr + 1] = fb[y, xl] * (1 - t) + fb[y, xr] * t
fb += grain * up[..., None] * 1.5
bg8 = np.clip(fb, 0, 255).astype(np.uint8)
bg8 = clone(bg8, up & ~dil(ch & (yy >= 1395), 3), 0, 0)
cv2.imwrite('bg.png', bg8)
np.savez_compressed('alphas.npz', **alphas)
cv2.imwrite('bgtop.jpg', bg8[:1150])
cv2.imwrite('bgcoco.jpg', np.hstack([bg8[1250:1550, 400:850], img[1250:1550, 400:850]]))
