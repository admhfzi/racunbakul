import cv2, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
img = cv2.imread('poster.jpg')  # BGR
H, W = img.shape[:2]
f = img.astype(np.float32)
B, G, R = f[..., 0], f[..., 1], f[..., 2]
mx, mn = f.max(2), f.min(2)

cyan = (B > R + 40) & (G > R + 30) & (mn < 200)
cloud = (mn > 195) & (mx - mn < 60)
leaf = (G > B + 10) & (G > R - 25) & (mx - mn > 70) & ~((R > 200) & (G > 170) & (R >= G))  # yellow-green leaves
yellow = (R > 170) & (B < 150) & (R - B > 70)
brown = (R > 110) & (R - B > 70) & (B < 130)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def pushpull(img, known):
    k = known.astype(np.float32)
    if min(img.shape[:2]) <= 2:
        tot = k.sum()
        mean = (img * k[..., None]).sum((0, 1)) / max(tot, 1e-6)
        return np.broadcast_to(mean, img.shape).astype(np.float32)
    h, w = img.shape[:2]
    num = cv2.resize(img * k[..., None], ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    den = cv2.resize(k, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    small = num / np.maximum(den, 1e-6)[..., None]
    filled = pushpull(small, den > 1e-3)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    return img * k[..., None] + up * (1 - k[..., None])
RECTS = [(80,62,1050,700),(115,662,1010,850),(205,792,920,1072),(968,818,1112,955),(28,968,158,1090)]
known = np.ones((H, W), bool)
for (a0,b0,a1,b1) in RECTS: known[b0:b1,a0:a1] = False
sky = pushpull(f, known)
diff = np.abs(f - sky).sum(2)

def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True; return m
def ell(cx, cy, rx, ry):
    m = np.zeros((H, W), np.uint8); cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1); return m > 0
def dil(m, k):
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

softdiff = np.clip((diff - 28) / 40, 0, 1)
leafd = dil(leaf & (diff > 40), 3)
regions = {
    'langkah': rect(280, 70, 820, 312),
    'ticks1': rect(818, 158, 945, 268),
    'alang': rect(225, 250, 890, 460),
    'port': rect(85, 402, 1048, 650),
    'pin': rect(800, 280, 1000, 452),
    'ticks2': rect(908, 585, 1015, 695),
    'date': rect(122, 668, 1003, 845),
    'umbA': rect(975, 825, 1105, 950),
    'umbB': rect(35, 975, 152, 1098),
}
order = ['umbA', 'umbB', 'date', 'port', 'alang', 'pin', 'langkah', 'ticks1', 'ticks2']
alphas = {}
claimed = np.zeros((H, W), bool)
for k in order:
    m = regions[k] & ~claimed
    a = softdiff * m
    if k == 'port':
        a = a * ~leafd
    elif k not in ('umbA', 'umbB'):
        a = a * ~(leafd & ~yellow)
    else:
        a = a * ~cloud
    alphas[k] = a
    claimed |= regions[k]

# big umbrella: canopy (above clouds) + pole
canopy = rect(212, 798, 915, 1070)
ua = np.where(canopy, softdiff * (~cloud | yellow | (yy < 965)), 0)
ua[:842] = 0
ua[798:842, 548:590] = softdiff[798:842, 548:590] * (B[798:842, 548:590] < 150)
pole = rect(551, 1060, 586, 1398) & brown
ua = np.maximum(ua, pole.astype(np.float32))
alphas['umb'] = ua
# coconut
cm = (ell(624, 1414, 77, 75) | (rect(636, 1288, 694, 1362) & (R > 170) & (R - G > 40))).astype(np.float32)
alphas['coco'] = cv2.GaussianBlur(cm, (5, 5), 0)

for k in alphas:
    a = alphas[k].astype(np.float32)
    alphas[k] = cv2.GaussianBlur(a, (3, 3), 0)

# ---- background
hole = np.zeros((H, W), np.float32)
for k in order + ['umb']:
    hole = np.maximum(hole, (alphas[k] > 0.03).astype(np.float32))
hole[1060:] = 0
hole = cv2.dilate(hole, np.ones((11, 11), np.uint8)) > 0
bg = pushpull(f, ~hole & ~dil(leaf, 9))
for _ in range(2):  # smooth inside hole a bit more
    bl = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = np.where(hole[..., None], bl, bg)
bg = np.where(hole[..., None], bg, f)
noise = np.random.default_rng(1).normal(0, 1.2, f.shape).astype(np.float32)
bg = bg + noise * hole[..., None]
# pole: row interpolation
x0, x1 = 546, 592
for y in range(1030, 1400):
    l, r = bg[y, x0], bg[y, x1]
    t = np.linspace(0, 1, x1 - x0)[:, None]
    bg[y, x0:x1] = l * (1 - t) + r * t
bg8 = np.clip(bg, 0, 255).astype(np.uint8)
cm8 = (cv2.dilate(cm, np.ones((9, 9), np.uint8)) > 0).astype(np.uint8) * 255
hole2 = cm8 > 0
fill = pushpull(bg8.astype(np.float32), ~hole2)
fill = cv2.GaussianBlur(fill, (0, 0), 10)
m2 = cv2.GaussianBlur(hole2.astype(np.float32), (0, 0), 3)[..., None]
bg8 = (bg8 * (1 - m2) + fill * m2).astype(np.uint8)
sh = np.zeros((H, W), np.float32)
cv2.ellipse(sh, (624, 1420), (88, 80), 0, 0, 360, 1, -1)
sh = cv2.GaussianBlur(sh, (0, 0), 14)[..., None]
bg8 = (bg8 * (1 - 0.5 * sh)).astype(np.uint8)
cv2.imwrite('bg.png', bg8)
np.savez_compressed('alphas.npz', **alphas)
tot = np.zeros((H, W), np.float32)
for a in alphas.values(): tot = np.maximum(tot, a)
vis = (f * (1 - tot[..., None]) + np.array([255, 0, 255]) * tot[..., None]).astype(np.uint8)
print({k:float(v.max()) for k,v in alphas.items()}); print(alphas['langkah'][150:250,400:500].mean())
cv2.imwrite('cmp.jpg', cv2.resize(np.hstack([vis, bg8]), None, fx=0.5, fy=0.5))
