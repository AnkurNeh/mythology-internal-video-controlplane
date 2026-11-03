import json, os, sys, urllib.request, urllib.error

APP_DOMAIN=os.environ.get("BASE44_APP_DOMAIN","https://hindu-mythology-stories.base44.app").rstrip("/")
APP_ID=os.environ.get("BASE44_APP_ID","6ab229601c793396d654094a")
WORKER_ID=os.environ["INTERNAL_VIDEO_WORKER_ID"]
LAUNCH_ID=os.environ["SCENECRAFT_BOOTSTRAP_LAUNCH_ID"]
NONCE=os.environ["SCENECRAFT_BOOTSTRAP_NONCE"]
URL=f"{APP_DOMAIN}/api/apps/{APP_ID}/functions/manageInternalVideoAI"

def post(body, token=""):
    headers={"Content-Type":"application/json","X-App-Id":APP_ID}
    if token: headers["X-Internal-Worker-Key"]=token
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers=headers,method="POST")
    with urllib.request.urlopen(req,timeout=45) as r:
        return json.loads(r.read(16384))

boot=post({"action":"worker_bootstrap","worker_id":WORKER_ID,"launch_id":LAUNCH_ID,"probe_nonce":NONCE,"runtime":"huggingface_jobs","provider":"huggingface"})
token=str(boot.get("session_token") or "")
if not token: raise RuntimeError("bootstrap_failed")
registered=post({"action":"worker_register","worker_id":WORKER_ID,"runtime":"huggingface_jobs","provider":"huggingface","gpu_type":"register_only","gpu_count":0,"software_version":"wan22-worker-v2-register-only","model_keys":["wan22-ti2v-5b"],"capabilities":["image_to_video","character_reference","camera_control"]},token)
ok=registered.get("success") is True
print(json.dumps({"event":"scenecraft_production_worker_register","passed":ok,"register_only":True,"model_loaded":False,"customer_jobs_claimed":False}))
sys.exit(0 if ok else 1)
