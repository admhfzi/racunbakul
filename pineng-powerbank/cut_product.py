from PIL import Image, ImageFilter
import numpy as np
from collections import deque
im=Image.open('assets/product_source.jpg').convert('RGB')
a=np.asarray(im).astype(int)
lum=a.mean(2); sat=a.max(2)-a.min(2)
b=a[:,:,2];r=a[:,:,0]
dark=(lum<150)&((b-r)<10)
# largest component
H,W=dark.shape; lab=np.zeros((H,W),int); best=(0,0); n=0
for y in range(H):
  for x in range(W):
    if dark[y,x] and not lab[y,x]:
      n+=1;q=deque([(y,x)]);lab[y,x]=n;c=0
      while q:
        cy,cx=q.popleft();c+=1
        for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
          ny,nx=cy+dy,cx+dx
          if 0<=ny<H and 0<=nx<W and dark[ny,nx] and not lab[ny,nx]:
            lab[ny,nx]=n;q.append((ny,nx))
      if c>best[0]: best=(c,n)
m=(lab==best[1])
# fill holes: flood outside
out=np.zeros_like(m);q=deque()
for y in range(H):
  for x in (0,W-1):
    if not m[y,x]: out[y,x]=1;q.append((y,x))
for x in range(W):
  for y in (0,H-1):
    if not m[y,x] and not out[y,x]: out[y,x]=1;q.append((y,x))
while q:
  cy,cx=q.popleft()
  for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
    ny,nx=cy+dy,cx+dx
    if 0<=ny<H and 0<=nx<W and not m[ny,nx] and not out[ny,nx]:
      out[ny,nx]=1;q.append((ny,nx))
mask=Image.fromarray(((~out.astype(bool))*255).astype('uint8'))
mask=mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
rgba=im.copy();rgba.putalpha(mask)
bb=mask.getbbox();print(bb)
rgba.crop(bb).save('assets/product.png')
bg=Image.new('RGB',rgba.crop(bb).size,(255,0,255));bg.paste(rgba.crop(bb),(0,0),rgba.crop(bb));bg.save('/tmp/check.png')
