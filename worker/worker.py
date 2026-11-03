# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests==2.32.5",
#   "torch>=2.6",
#   "diffusers>=0.38.0",
#   "transformers>=4.57.0",
#   "accelerate>=1.11.0",
#   "imageio[ffmpeg]>=2.37.0",
#   "Pillow>=11.3.0",
#   "safetensors>=0.6.2",
#   "ftfy>=6.3.1",
# ]
# ///
import hashlib
import json
import os
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

APP_DOMAIN = os.environ.get("BASE44_APP_DOMAIN", "https://hindu-mythology-stories.base44.app").rstrip("/")
APP_ID = os.environ.get("BASE44_APP_ID", "6ab229601c793396d654094a").strip()
FUNCTION_URL = APP_DOMAIN + "/api/apps/" + APP_ID + "/functions/manageInternalVideoAI"
WORKER_KEY = os.environ.get("INTERNAL_VIDEO_WORKER_API_KEY", "").strip()
BOOTSTRAP_LAUNCH_ID = os.environ.get("SCENECRAFT_BOOTSTRAP_LAUNCH_ID", "").strip()
BOOTSTRAP_NONCE = os.environ.get("SCENECRAFT_BOOTSTRAP_NONCE", "").strip()
WORKER_ID = os.environ.get("INTERNAL_VIDEO_WORKER_ID", "wan22-worker-1").strip()
MODEL_KEY = os.environ.get("INTERNAL_VIDEO_MODEL_KEY", "wan22-ti2v-5b").strip()
MODEL_ID = os.environ.get("INTERNAL_VIDEO_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers").strip()
DRY_RUN = os.environ.get("INTERNAL_VIDEO_DRY_RUN", "0").strip() == "1"
REGISTER_ONLY = os.environ.get("INTERNAL_VIDEO_REGISTER_ONLY", "0").strip() == "1"
LEASE_SECONDS = int(os.environ.get("INTERNAL_VIDEO_LEASE_SECONDS", "300"))
POLL_SECONDS = float(os.environ.get("INTERNAL_VIDEO_POLL_SECONDS", "5"))
MAX_ATTEMPTS = int(os.environ.get("INTERNAL_VIDEO_MAX_ATTEMPTS", "3"))
ONE_SHOT = os.environ.get("INTERNAL_VIDEO_ONE_SHOT", "0").strip() == "1"
OUTPUT_PUT_URL = os.environ.get("INTERNAL_VIDEO_OUTPUT_PUT_URL", "").strip()
OUTPUT_PUBLIC_URL = os.environ.get("INTERNAL_VIDEO_OUTPUT_PUBLIC_URL", "").strip()
MAX_INLINE_BYTES = int(os.environ.get("INTERNAL_VIDEO_MAX_INLINE_BYTES", str(8 * 1024 * 1024)))
STOP = False

def _signal(*_):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _signal)
signal.signal(signal.SIGTERM, _signal)

@dataclass
class QueueClient:
    url: str
    worker_key: str
    worker_id: str

    def call(self, action: str, **payload: Any) -> dict:
        if not self.worker_key:
            raise RuntimeError("INTERNAL_VIDEO_WORKER_API_KEY is required")
        body = {"action": action, "worker_id": self.worker_id, **payload}
        response = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "X-Internal-Worker-Key": self.worker_key,
                "X-App-Id": APP_ID,
                "User-Agent": "mythology-internal-worker/1.0",
            },
            json=body,
            timeout=45,
            allow_redirects=False,
        )
        try:
            data = response.json()
        except Exception:
            data = {"error": response.text[:1000]}
        if response.status_code < 200 or response.status_code >= 300:
            # Do not echo an upstream body; it can contain infrastructure details.
            raise RuntimeError(f"Base44 queue HTTP {response.status_code}")
        if not isinstance(data, dict) or data.get("success") is not True:
            raise RuntimeError("Base44 queue returned an invalid success response")
        return data

class LeaseHeartbeat:
    def __init__(self, client: QueueClient, job_id: str):
        self.client = client
        self.job_id = job_id
        self.stop_event = threading.Event()
        self.cancel_requested = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        interval = max(10, min(60, LEASE_SECONDS // 3))
        while not self.stop_event.wait(interval):
            try:
                result = self.client.call(
                    "worker_heartbeat",
                    job_id=self.job_id,
                    lease_seconds=LEASE_SECONDS,
                )
                self.cancel_requested = bool(result.get("cancel_requested"))
                if self.cancel_requested:
                    return
            except Exception as exc:
                self.error = exc
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=5)


class VideoAdapter:
    def generate(self, request: dict) -> dict:
        raise NotImplementedError

class Wan22TI2VAdapter(VideoAdapter):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import WanImageToVideoPipeline
        self._pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        gpu_profile = os.environ.get("SCENECRAFT_GPU_PROFILE", "").strip().lower()
        if gpu_profile == "a100-large":
            self._pipe.to("cuda")
        else:
            self._pipe.enable_model_cpu_offload()
        return self._pipe

    def generate(self, request: dict) -> dict:
        if DRY_RUN:
            return {
                "dry_run": True,
                "sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                "gpu_seconds": 0,
                "compute_cost": 0,
            }
        import torch
        from diffusers.utils import export_to_video, load_image

        pipe = self._load()
        inputs = request.get("input") or {}
        first_frame = inputs.get("first_frame_url")
        if not first_frame:
            raise RuntimeError("first_frame_url is required for TI2V worker")
        prompt = str(request.get("prompt") or "").strip()
        duration = float(request.get("duration_seconds") or 5)
        aspect = str(request.get("aspect_ratio") or "16:9")
        fps = int(os.environ.get("INTERNAL_VIDEO_FPS", "24"))
        frames = max(17, int(round(duration * fps)))
        if frames % 4 != 1:
            frames += (1 - frames) % 4
        image = load_image(first_frame)
        generator = None
        seed = ((request.get("controls") or {}).get("seed"))
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        # Wan2.2 TI2V native geometry; reject unsupported aspect ratios rather than
        # generating distorted or unreviewed commercial outputs.
        if aspect not in ("16:9", "9:16"):
            raise RuntimeError(f"Unsupported Wan2.2 aspect ratio: {aspect}")
        width, height = (1280, 704) if aspect == "16:9" else (704, 1280)
        started = time.time()
        result = pipe(
            image=image,
            prompt=prompt,
            num_frames=frames,
            width=width,
            height=height,
            generator=generator,
        )
        output_frames = result.frames[0]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "output.mp4"
            export_to_video(output_frames, str(path), fps=fps)
            data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if OUTPUT_PUT_URL and OUTPUT_PUBLIC_URL:
            upload = requests.put(
                OUTPUT_PUT_URL,
                data=data,
                headers={"Content-Type": "video/mp4"},
                timeout=600,
            )
            upload.raise_for_status()
            return {
                "video_url": OUTPUT_PUBLIC_URL,
                "sha256": digest,
                "gpu_seconds": round(time.time() - started, 3),
                # Unknown supplier invoice cost must never be reported as zero.
            }
        if len(data) > MAX_INLINE_BYTES:
            raise RuntimeError(
                f"Rendered MP4 is {len(data)} bytes, above inline limit {MAX_INLINE_BYTES}; configure durable output upload URLs"
            )
        return {
            "video_bytes_base64": __import__("base64").b64encode(data).decode(),
            "mime_type": "video/mp4",
            "sha256": digest,
            "gpu_seconds": round(time.time() - started, 3),
            # GPU duration is telemetry; invoice reconciliation determines cost.
        }

def _bootstrap_probe_key():
    if not BOOTSTRAP_LAUNCH_ID or not BOOTSTRAP_NONCE:
        return ""
    body = {
        "action": "worker_bootstrap",
        "worker_id": WORKER_ID,
        "launch_id": BOOTSTRAP_LAUNCH_ID,
        "probe_nonce": BOOTSTRAP_NONCE,
        "runtime": "huggingface_jobs" if (os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID")) else "standalone",
        "provider": "huggingface",
    }
    response = requests.post(
        FUNCTION_URL,
        headers={"Content-Type": "application/json", "X-App-Id": APP_ID, "User-Agent": "scenecraft-worker-bootstrap/1.0"},
        json=body, timeout=45, allow_redirects=False,
    )
    try:
        data = response.json()
    except Exception:
        data = {}
    if response.status_code < 200 or response.status_code >= 300 or data.get("success") is not True:
        raise RuntimeError(f"SceneCraft worker bootstrap HTTP {response.status_code}")
    token = str(data.get("session_token") or "").strip()
    if not token:
        raise RuntimeError("SceneCraft worker bootstrap returned no session")
    return token

def main():
    worker_key = WORKER_KEY or _bootstrap_probe_key()
    client = QueueClient(FUNCTION_URL, worker_key, WORKER_ID)
    if DRY_RUN:
        # Never register as a production worker or claim/release customer jobs.
        result = client.call("worker_probe", runtime="huggingface_jobs" if (os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID")) else "standalone", provider="huggingface")
        if result.get("queue_read_verified") is not True or result.get("diagnostic_only") is not True:
            raise RuntimeError("Worker connection probe was not verified")
        print(json.dumps({"event":"scenecraft_connection_probe", "passed":True, "diagnostic_only":True}))
        return
    adapter = Wan22TI2VAdapter(MODEL_ID)
    client.call(
        "worker_register",
        runtime="huggingface_jobs" if (os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID")) else "standalone_gpu",
        provider="huggingface",
        gpu_type=os.environ.get("GPU_TYPE", "unknown"),
        gpu_count=int(os.environ.get("GPU_COUNT", "1")),
        software_version="wan22-worker-v2",
        model_keys=[MODEL_KEY],
        capabilities=["image_to_video", "character_reference", "camera_control"],
    )
    if REGISTER_ONLY:
        print(json.dumps({"event":"scenecraft_worker_register","passed":True,"register_only":True,"model_loaded":False,"customer_jobs_claimed":False}))
        return
    # A bounded one-shot mode is suitable for managed GPU Jobs. It registers,
    # polls for a limited period and exits rather than consuming unlimited GPU time.
    deadline = time.monotonic() + max(30, int(os.environ.get("INTERNAL_VIDEO_MAX_RUNTIME_SECONDS", "900")))
    while not STOP and time.monotonic() < deadline:
        claim = client.call(
            "worker_claim",
            lease_seconds=LEASE_SECONDS,
            model_keys=[MODEL_KEY],
            runtime="huggingface_jobs" if (os.environ.get("JOB_ID") or os.environ.get("HF_JOB_ID")) else "standalone_gpu",
            provider="huggingface",
            software_version="wan22-worker-v2",
            capabilities=["image_to_video", "character_reference", "camera_control"],
        )
        if not claim.get("claimed"):
            time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
            continue
        job = claim["job"]
        job_id = job["id"]
        try:
            with LeaseHeartbeat(client, job_id) as heartbeat:
                result = adapter.generate(job["input"])
                if heartbeat.cancel_requested:
                    client.call("worker_release", job_id=job_id)
                    continue
                if heartbeat.error:
                    raise RuntimeError(f"Worker lost queue heartbeat: {heartbeat.error}")
            complete = {
                "job_id": job_id,
                "sha256": result.get("sha256", ""),
                "gpu_seconds": result.get("gpu_seconds", 0),
                # Never fabricate a provider charge before invoice reconciliation.
            }
            if result.get("video_url"):
                complete["video_url"] = result["video_url"]
            else:
                complete["video_bytes_base64"] = result["video_bytes_base64"]
                complete["mime_type"] = result.get("mime_type", "video/mp4")
            client.call("worker_complete", **complete)
            if ONE_SHOT:
                break
        except Exception as exc:
            client.call(
                "worker_fail",
                job_id=job_id,
                retryable=True,
                max_attempts=MAX_ATTEMPTS,
                error_class=exc.__class__.__name__,
                error=str(exc)[:1800],
            )
            if ONE_SHOT:
                break

if __name__ == "__main__":
    main()