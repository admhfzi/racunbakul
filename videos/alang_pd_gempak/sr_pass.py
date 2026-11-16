import torch, numpy as np, subprocess, sys, cv2, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
W0, H0 = 464, 832
OW, OH = 1392, 2496
m = sr.load()
src = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', os.path.join(HERE, '..', 'pdmap_in', 'src.mp4'), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
out = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', '60', '-i', '-',
                        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '8', '-pix_fmt', 'yuv444p', os.path.join(HERE, 'sr3x.mkv')], stdin=subprocess.PIPE)
n = W0 * H0 * 3; i = 0; prev = None; last = None; reused = 0; t0 = time.time()
while True:
    b = src.stdout.read(n)
    if len(b) < n: break
    f = np.frombuffer(b, np.uint8).reshape(H0, W0, 3)
    if prev is not None and np.abs(f.astype(np.int16) - prev).mean() < 0.06:
        reused += 1
    else:
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            x = torch.from_numpy(np.ascontiguousarray(f[..., ::-1])).permute(2, 0, 1)[None].float().div(255).contiguous(memory_format=torch.channels_last)
            y = m(x).float().clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().byte().numpy()
        last = cv2.resize(np.ascontiguousarray(y[..., ::-1]), (OW, OH), interpolation=cv2.INTER_AREA)
        prev = f.astype(np.int16)
        if i == 1014: cv2.imwrite(os.path.join(HERE, 'sr_check.png'), last)
    out.stdin.write(last.tobytes()); i += 1
    if i % 100 == 0: print(i, 'reused', reused, round(time.time() - t0), 's', flush=True)
out.stdin.close(); out.wait(); print('SR done', i, reused, round(time.time() - t0))
