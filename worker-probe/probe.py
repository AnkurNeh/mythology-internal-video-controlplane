import json
import os
import sys
import urllib.error
import urllib.request

APP_DOMAIN = os.environ.get("BASE44_APP_DOMAIN", "https://hindu-mythology-stories.base44.app").rstrip("/")
APP_ID = os.environ.get("BASE44_APP_ID", "6ab229601c793396d654094a")
WORKER_ID = os.environ["INTERNAL_VIDEO_WORKER_ID"]
LAUNCH_ID = os.environ["SCENECRAFT_BOOTSTRAP_LAUNCH_ID"]
NONCE = os.environ["SCENECRAFT_BOOTSTRAP_NONCE"]
URL = f"{APP_DOMAIN}/api/apps/{APP_ID}/functions/manageInternalVideoAI"
HEADERS = {"Content-Type": "application/json", "X-App-Id": APP_ID}

def post(body, token=None):
    headers = dict(HEADERS)
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.loads(response.read(16384))

try:
    bootstrap = post({
        "action": "worker_bootstrap",
        "worker_id": WORKER_ID,
        "launch_id": LAUNCH_ID,
        "probe_nonce": NONCE,
        "runtime": "huggingface_jobs",
        "provider": "huggingface",
    })
    session_token = bootstrap.get("session_token", "")
    if not session_token:
        raise RuntimeError("bootstrap_failed")

    result = post({
        "action": "worker_probe",
        "worker_id": WORKER_ID,
        "launch_id": LAUNCH_ID,
        "probe_nonce": NONCE,
        "runtime": "huggingface_jobs",
        "provider": "huggingface",
    }, session_token)

    ok = (
        result.get("success") is True
        and result.get("authentication_verified") is True
        and result.get("queue_read_verified") is True
        and result.get("diagnostic_only") is True
    )
    print(json.dumps({
        "event": "scenecraft_worker_probe",
        "passed": ok,
        "queue_read_verified": result.get("queue_read_verified") is True,
        "diagnostic_only": True,
    }))
    sys.exit(0 if ok else 1)
except urllib.error.HTTPError as exc:
    print(json.dumps({
        "event": "scenecraft_worker_probe",
        "passed": False,
        "http_status": exc.code,
    }))
    sys.exit(1)
except Exception as exc:
    print(json.dumps({
        "event": "scenecraft_worker_probe",
        "passed": False,
        "error": str(exc)[:120],
    }))
    sys.exit(1)
