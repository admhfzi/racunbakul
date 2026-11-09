import sys, math, subprocess
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
W,H=1080,1920
bg0=Image.open('assets/background.jpg').convert('RGB')
cw=round(bg0.height*9/16); x0=(bg0.width-cw)//2
bg=bg0.crop((x0,0,x0+cw,bg0.height)).resize((W,H),Image.LANCZOS)
prod=Image.open('assets/product.png')            # exact pixels from image 2
PS=float(sys.argv[2]) if len(sys.argv)>2 else 1.1
prod=prod.resize((round(prod.width*PS),round(prod.height*PS)),Image.LANCZOS)
hand=Image.open('assets/hand.png').crop((150,230,820,1680))
# match hand to the room's dimmer, warmer light
r,g,b,a=hand.split(); rgb=Image.merge('RGB',(r,g,b))
rgb=ImageEnhance.Brightness(rgb).enhance(0.86)
r,g,b=rgb.split(); b=b.point(lambda v:int(v*0.95))
hand=Image.merge('RGBA',(r,g,b,a))
HS=float(sys.argv[3]) if len(sys.argv)>3 else 1.25
hand=hand.resize((round(hand.width*HS),round(hand.height*HS)),Image.LANCZOS)
HROT=float(sys.argv[4]) if len(sys.argv)>4 else -28
PC=(W*0.5, H*0.47)      # product centre on screen (fixed)
HOFF=(float(sys.argv[5]) if len(sys.argv)>5 else 40, float(sys.argv[6]) if len(sys.argv)>6 else 330)  # hand-cup offset from product centre
def frame(t,T):
    ang=2.2*math.sin(2*math.pi*t/T)          # very slow tilt, degrees
    # group layer: hand behind product, both rotate together about product centre
    L=Image.new('RGBA',(W*2,H*2),(0,0,0,0)); cx,cy=W,H
    h=hand.rotate(HROT,resample=Image.BICUBIC,expand=True)
    hx=cx+HOFF[0]-h.width//2; hy=cy+HOFF[1]-h.height//3
    L.alpha_composite(h,(int(hx),int(hy)))
    # soft shadow of product on hand
    P=50; sh=Image.new('RGBA',(prod.width+2*P,prod.height+2*P),(0,0,0,0)); m=Image.new('L',sh.size,0); m.paste(prod.split()[3].point(lambda v:int(v*0.5)),(P,P)); sh.putalpha(m)
    sh=sh.filter(ImageFilter.GaussianBlur(14))
    S=Image.new('RGBA',L.size,(0,0,0,0)); S.alpha_composite(sh,(cx-prod.width//2+10-P,cy-prod.height//2+22-P))
    S.putalpha(ImageChops.multiply(S.split()[3],L.split()[3]))
    L.alpha_composite(S)
    L.alpha_composite(prod,(cx-prod.width//2,cy-prod.height//2))
    if abs(ang)>1e-3: L=L.rotate(ang,resample=Image.BICUBIC,center=(cx,cy))
    L=L.crop((cx-int(PC[0]),cy-int(PC[1]),cx-int(PC[0])+W,cy-int(PC[1])+H))
    f=bg.copy().convert('RGBA'); f.alpha_composite(L); return f.convert('RGB')
mode=sys.argv[1]
if mode=='still':
    frame(0,1).save('still.png')
else:
    T=8; fps=30; n=T*fps
    p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-',
      '-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',mode],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for i in range(n): p.stdin.write(frame(i/fps,T).tobytes())
    p.stdin.close(); p.wait(); print('done',p.returncode)
