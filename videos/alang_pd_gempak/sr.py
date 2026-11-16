import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os
class SRVGG(nn.Module):
    def __init__(s, nf=64, nc=16, up=4):
        super().__init__(); s.up = up
        body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(nf)]
        for _ in range(nc): body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(nf)]
        body += [nn.Conv2d(nf, 3 * up * up, 3, 1, 1)]
        s.body = nn.ModuleList(body); s.ps = nn.PixelShuffle(up)
    def forward(s, x):
        o = x
        for l in s.body: o = l(o)
        return s.ps(o) + F.interpolate(x, scale_factor=s.up, mode='nearest')
def load(name='realesr-animevideov3.pth', nc=16):
    m = SRVGG(nc=nc)
    sd = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', name), map_location='cpu', weights_only=True)
    sd = sd.get('params_ema', sd.get('params', sd))
    m.load_state_dict(sd); m.eval()
    return m.to(memory_format=torch.channels_last)
@torch.inference_mode()
def run(m, bgr):
    x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)[None].float().div(255).contiguous(memory_format=torch.channels_last)
    y = m(x).clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().byte().numpy()
    return np.ascontiguousarray(y[..., ::-1])
