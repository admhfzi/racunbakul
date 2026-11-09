"""Generate the 9:16 powerbank clip with Google Veo (the model behind Google Flow) via the Gemini API.

The API key is never stored here: the session's network proxy adds the
`x-goog-api-key` header for generativelanguage.googleapis.com.

Usage: python3 veo_generate.py [model] [out.mp4]
  model default: veo-3.1-fast-generate-preview  (try veo-3.1-generate-preview for top quality)
"""
import base64, json, sys, time, urllib.request, pathlib

HERE = pathlib.Path(__file__).parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "veo-3.1-fast-generate-preview"
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "veo_powerbank_9x16.mp4"
API = "https://generativelanguage.googleapis.com/v1beta"

PROMPT = (
    "Static camera, vertical POV handheld product shot. Two hands in a mustard yellow sweatshirt hold a "
    "black PINENG power bank facing the camera. The hands very slowly and gently tilt the power bank a few "
    "degrees left, then right, a soft natural sway as if showing it to a friend. The power bank stays "
    "centered and always faces the camera. The logo \"PINENG\" and the text \"22.5W\" and \"10000mAh\" stay "
    "sharp, readable and unchanged throughout. Fingers stay naturally wrapped around the sides. The "
    "background (wooden desk, grey desk lamp, white keyboard, notebook) stays exactly the same and "
    "completely still. Warm soft lamp lighting, realistic phone-camera look. No zoom, no camera movement, "
    "no text overlay. Audio: quiet room ambience only, no music, no voice."
)
NEGATIVE = ("blurry text, distorted logo, changing label, extra fingers, deformed hands, background change, "
            "new objects, camera shake, zoom, fast movement, music, voice")


def call(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


img = base64.b64encode((HERE / "pov_first_frame.png").read_bytes()).decode()
op = call(f"{API}/models/{MODEL}:predictLongRunning", {
    "instances": [{"prompt": PROMPT, "image": {"bytesBase64Encoded": img, "mimeType": "image/png"}}],
    "parameters": {"aspectRatio": "9:16", "negativePrompt": NEGATIVE, "sampleCount": 1},
})
print("started:", op["name"])
while not op.get("done"):
    time.sleep(10)
    op = call(f"{API}/{op['name']}")
    print("waiting...")
if "error" in op:
    sys.exit(f"Veo error: {op['error']}")
resp = op["response"]["generateVideoResponse"]
uri = resp["generatedSamples"][0]["video"]["uri"]
with urllib.request.urlopen(urllib.request.Request(uri), timeout=300) as r:
    OUT.write_bytes(r.read())
print("saved:", OUT)
