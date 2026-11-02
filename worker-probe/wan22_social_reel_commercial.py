# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6",
#   "diffusers @ git+https://github.com/huggingface/diffusers.git",
#   "transformers>=4.57.0",
#   "accelerate>=1.11.0",
#   "imageio[ffmpeg]>=2.37.0",
#   "Pillow>=11.3.0",
#   "requests>=2.32.5",
#   "safetensors>=0.6.2",
#   "ftfy>=6.3.1",
# ]
# ///
import base64
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import requests
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
IMAGE_URL = os.environ.get("SCENECRAFT_BENCHMARK_IMAGE_URL", "https://raw.githubusercontent.com/AnkurNeh/mythology-internal-video-controlplane/main/worker-probe/synthetic_presenter.png")
CALLBACK = "https://hindu-mythology-stories.base44.app/api/apps/6ab229601c793396d654094a/functions/manageInternalBenchmarkArtifact"
APP_ID = "6ab229601c793396d654094a"
MODEL_KEY = "wan22-ti2v-5b"
CASE_KEY = os.environ.get("SCENECRAFT_BENCHMARK_CASE_KEY", "wan22_synthetic_smoke_16x9")
LAUNCH_ID = os.environ["SCENECRAFT_BENCHMARK_LAUNCH_ID"]
NONCE = os.environ["SCENECRAFT_BENCHMARK_NONCE"]

prompt = os.environ.get("SCENECRAFT_BENCHMARK_PROMPT", "A clean stylized synthetic presenter remains centered while making a subtle natural head movement. Stable framing, no cuts, no text, no background changes.")
negative_prompt = "text, watermark, scene cut, duplicate face, distorted geometry, flashing, flicker"
WIDTH = int(os.environ.get("SCENECRAFT_BENCHMARK_WIDTH", "1280"))
HEIGHT = int(os.environ.get("SCENECRAFT_BENCHMARK_HEIGHT", "704"))
NUM_FRAMES = int(os.environ.get("SCENECRAFT_BENCHMARK_FRAMES", "17"))

started = time.time()
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
if os.environ.get("SCENECRAFT_GPU_PROFILE", "") == "a100-large":
    pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload()
image = load_image(IMAGE_URL)

torch.cuda.reset_peak_memory_stats()
gen_started = time.time()
frames = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=8,
    guidance_scale=3.0,
).frames[0]
gpu_seconds = time.time() - gen_started

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "wan22-smoke.mp4"
    export_to_video(frames, str(path), fps=24)
    data = path.read_bytes()

digest = hashlib.sha256(data).hexdigest()
peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
payload = {
    "action": "upload_artifact",
    "launch_id": LAUNCH_ID,
    "probe_nonce": NONCE,
    "model_key": MODEL_KEY,
    "case_key": CASE_KEY,
    "video_bytes_base64": base64.b64encode(data).decode(),
    "sha256": digest,
    "gpu_seconds": round(gpu_seconds, 3),
    "latency_seconds": round(time.time() - started, 3),
}
response = requests.post(
    CALLBACK,
    headers={"Content-Type": "application/json", "X-App-Id": APP_ID},
    json=payload,
    timeout=180,
    allow_redirects=False,
)
try:
    callback = response.json()
except Exception:
    callback = {}
if response.status_code < 200 or response.status_code >= 300 or callback.get("success") is not True:
    raise RuntimeError(f"SceneCraft benchmark callback failed HTTP {response.status_code}")

print(json.dumps({
    "event": "wan22_real_render",
    "passed": True,
    "model": MODEL_KEY,
    "frames": len(frames),
    "width": WIDTH,
    "height": HEIGHT,
    "fps": 24,
    "gpu_seconds": round(gpu_seconds, 3),
    "peak_cuda_gb": round(peak_gb, 3),
    "artifact_sha256": digest,
    "benchmark_result_id": callback.get("benchmark_result_id"),
    "human_review_status": callback.get("human_review_status"),
}))
