import base64
import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, Header, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="Mythology Internal Video Control Plane", version="1.2.0")
JOBS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()
SEEN_NONCES: OrderedDict[str, float] = OrderedDict()
MAX_NONCES = 10000
MAX_CLOCK_SKEW_SECONDS = 300


class InputSpec(BaseModel):
    first_frame_url: HttpUrl
    last_frame_url: HttpUrl | None = None
    reference_urls: list[HttpUrl] = Field(default_factory=list, max_length=8)


class Controls(BaseModel):
    seed: int | None = None
    generate_audio: bool = False


class GenerationRequest(BaseModel):
    model: str
    model_version: str = ""
    profile: str
    duration_seconds: float = Field(gt=0, le=30)
    aspect_ratio: str
    prompt: str
    input: InputSpec
    controls: Controls = Field(default_factory=Controls)
    metadata: dict = Field(default_factory=dict)


def _b64url_decode(value: str) -> bytes:
    value = value.strip()
    return base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))


def _signing_key() -> Ed25519PublicKey | None:
    raw = os.getenv("BASE44_SIGNING_PUBLIC_KEY_BASE64URL", "").strip()
    if not raw:
        return None
    return Ed25519PublicKey.from_public_bytes(_b64url_decode(raw))


def _bearer_ok(value: str) -> bool:
    key = os.getenv("INTERNAL_VIDEO_SERVICE_API_KEY", "")
    return bool(key) and value == "Bearer " + key


def _consume_nonce(nonce: str, timestamp: int) -> None:
    now = time.time()
    with LOCK:
        expired = [key for key, seen_at in SEEN_NONCES.items() if now - seen_at > MAX_CLOCK_SKEW_SECONDS * 2]
        for key in expired:
            SEEN_NONCES.pop(key, None)
        if nonce in SEEN_NONCES:
            raise HTTPException(401, "replayed signed request")
        SEEN_NONCES[nonce] = float(timestamp)
        while len(SEEN_NONCES) > MAX_NONCES:
            SEEN_NONCES.popitem(last=False)


async def authorize(request: FastAPIRequest, body: bytes = b"") -> None:
    public_key = _signing_key()
    if public_key is None:
        if _bearer_ok(request.headers.get("authorization", "")):
            return
        raise HTTPException(503, "request verifier not configured")

    key_id = request.headers.get("x-internal-key-id", "")
    algorithm = request.headers.get("x-internal-signature-alg", "")
    timestamp_raw = request.headers.get("x-internal-timestamp", "")
    nonce = request.headers.get("x-internal-nonce", "")
    signature_raw = request.headers.get("x-internal-signature", "")
    expected_key_id = os.getenv("BASE44_SIGNING_KEY_ID", "base44-derived-sync-v1")

    if key_id != expected_key_id or algorithm != "Ed25519":
        raise HTTPException(401, "invalid signing identity")
    try:
        timestamp = int(timestamp_raw)
    except ValueError:
        raise HTTPException(401, "invalid signing timestamp")
    if abs(time.time() - timestamp) > MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(401, "signed request expired")
    if not nonce or len(nonce) > 200:
        raise HTTPException(401, "invalid signing nonce")

    digest = hashlib.sha256(body).hexdigest()
    canonical = "\n".join(
        [request.method.upper(), request.url.path + (("?" + request.url.query) if request.url.query else ""), timestamp_raw, nonce, digest]
    ).encode("utf-8")
    try:
        signature = base64.b64decode(signature_raw, validate=True)
        public_key.verify(signature, canonical)
    except (ValueError, InvalidSignature):
        raise HTTPException(401, "invalid request signature")
    _consume_nonce(nonce, timestamp)


def gpu_configured() -> bool:
    return bool(os.getenv("HF_VIDEO_ENDPOINT_URL")) and bool(os.getenv("HF_VIDEO_ENDPOINT_TOKEN"))


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "version": "1.2.0"}


@app.get("/readyz")
def ready() -> dict:
    if _signing_key() is None and not os.getenv("INTERNAL_VIDEO_SERVICE_API_KEY"):
        raise HTTPException(503, "request verifier not configured")
    if not gpu_configured():
        raise HTTPException(503, "GPU endpoint not configured")
    return {"status": "ready", "gpu_endpoint": True, "signed_requests": _signing_key() is not None}


def run_generation(job_id: str, payload: dict) -> None:
    with LOCK:
        JOBS[job_id]["status"] = "processing"
    try:
        url = os.environ["HF_VIDEO_ENDPOINT_URL"].rstrip("/")
        token = os.environ["HF_VIDEO_ENDPOINT_TOKEN"]
        timeout = float(os.getenv("HF_VIDEO_TIMEOUT_SECONDS", "1800"))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                json={"inputs": payload},
            )
            response.raise_for_status()
            data = response.json()
        result = data.get("result") if isinstance(data, dict) and isinstance(data.get("result"), dict) else data
        video_bytes = result.get("video_bytes_base64") or result.get("videoBytesBase64")
        video_url = result.get("video_url") or result.get("url")
        if not video_bytes and not (isinstance(video_url, str) and video_url.startswith("https://")):
            raise RuntimeError("GPU endpoint returned neither video bytes nor durable video URL")
        output = {
            "sha256": result.get("sha256", ""),
            "gpu_seconds": float(result.get("gpu_seconds", 0) or 0),
            "compute_cost": float(result.get("compute_cost", 0) or 0),
        }
        if video_bytes:
            output.update(video_bytes_base64=video_bytes, mime_type=result.get("mime_type", "video/mp4"))
        else:
            output["video_url"] = video_url
        with LOCK:
            JOBS[job_id].update(status="completed", result=output)
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(status="failed", error={"class": "gpu_endpoint", "message": str(exc)[:2000]})


@app.post("/v1/generations")
async def create_generation(
    payload: GenerationRequest,
    request: FastAPIRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
):
    raw = await request.body()
    await authorize(request, raw)
    if not gpu_configured():
        raise HTTPException(503, "GPU endpoint not configured")
    if not idempotency_key:
        raise HTTPException(400, "Idempotency-Key required")
    with LOCK:
        for job in JOBS.values():
            if job.get("idempotency_key") == idempotency_key:
                return {"job_id": job["job_id"], "status": job["status"]}
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "submitted",
            "idempotency_key": idempotency_key,
            "created_at": time.time(),
        }
    threading.Thread(target=run_generation, args=(job_id, payload.model_dump(mode="json")), daemon=True).start()
    return {"job_id": job_id, "status": "submitted"}


@app.get("/v1/generations/{job_id}")
async def generation_status(job_id: str, request: FastAPIRequest):
    await authorize(request)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.post("/v1/generations/{job_id}/cancel")
async def cancel_generation(job_id: str, request: FastAPIRequest):
    await authorize(request)
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] == "submitted":
        job["status"] = "cancelled"
    return {"job_id": job_id, "status": job["status"]}
