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
import base64, hashlib, json, os, tempfile, time
from pathlib import Path
import requests, torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MODEL_KEY="wan22-ti2v-5b"
APP_ID="6ab229601c793396d654094a"
CALLBACK="https://hindu-mythology-stories.base44.app/api/apps/6ab229601c793396d654094a/functions/manageInternalBenchmarkArtifact"
NEGATIVE="text, watermark, scene cut, duplicate face, distorted anatomy, distorted geometry, flashing, flicker"
CASES=[
 {
  "case_key":"wan22_cinematic_animation_canary_16x9_v1",
  "launch_id":"6ae9584a042172ea94e1a33e","nonce":"wan22ciman20261103a",
  "image":"https://raw.githubusercontent.com/AnkurNeh/mythology-internal-video-controlplane/main/worker-probe/synthetic_presenter.png",
  "prompt":"A stylized synthetic adult hero remains centered in a cinematic battlefield composition, turns the head slightly with a restrained emotional reaction, then settles. Stable identity, proportions, costume silhouette and background. No speech, no text, no scene cut.",
  "width":1280,"height":704,"frames":49
 },
 {
  "case_key":"wan22_cinematic_live_action_canary_16x9_v1",
  "launch_id":"6ae9584f96128c714a43e83c","nonce":"wan22cimlive20261103a",
  "image":"https://raw.githubusercontent.com/AnkurNeh/mythology-internal-video-controlplane/main/worker-probe/synthetic_presenter.png",
  "prompt":"A clean synthetic adult presenter in a realistic studio-style cinematic composition makes a subtle natural head turn and posture shift. Preserve identity, proportions, lighting and background with smooth motion and no cuts or text.",
  "width":1280,"height":704,"frames":49
 },
 {
  "case_key":"wan22_stylized_3d_canary_9x16_v1",
  "launch_id":"6ae9585644c29466ddb990d0","nonce":"wan22styl3d20261103a",
  "image":"https://raw.githubusercontent.com/AnkurNeh/mythology-internal-video-controlplane/main/worker-probe/synthetic_presenter_vertical.png",
  "prompt":"A premium stylized synthetic 3D presenter remains centered in vertical cinematic framing, makes a controlled head turn and subtle reaction. Stable identity, anatomy, proportions and background. No speech, no text, no cut.",
  "width":704,"height":1280,"frames":49
 }
]

started=time.time()
vae=AutoencoderKLWan.from_pretrained(MODEL_ID,subfolder="vae",torch_dtype=torch.float32)
pipe=WanImageToVideoPipeline.from_pretrained(MODEL_ID,vae=vae,torch_dtype=torch.bfloat16)
pipe.to("cuda")
results=[]
for case in CASES:
    image=load_image(case["image"])
    torch.cuda.reset_peak_memory_stats()
    case_started=time.time()
    frames=pipe(
      image=image,prompt=case["prompt"],negative_prompt=NEGATIVE,
      height=case["height"],width=case["width"],num_frames=case["frames"],
      num_inference_steps=8,guidance_scale=3.0
    ).frames[0]
    gpu_seconds=time.time()-case_started
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/(case["case_key"]+".mp4")
        export_to_video(frames,str(p),fps=24)
        data=p.read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    payload={
      "action":"upload_artifact","launch_id":case["launch_id"],"probe_nonce":case["nonce"],
      "model_key":MODEL_KEY,"case_key":case["case_key"],
      "video_bytes_base64":base64.b64encode(data).decode(),"sha256":digest,
      "gpu_seconds":round(gpu_seconds,3),"latency_seconds":round(time.time()-case_started,3)
    }
    response=requests.post(CALLBACK,headers={"Content-Type":"application/json","X-App-Id":APP_ID},json=payload,timeout=180,allow_redirects=False)
    try: cb=response.json()
    except Exception: cb={}
    if not (200<=response.status_code<300 and cb.get("success") is True):
        raise RuntimeError(f"SceneCraft callback failed for {case['case_key']} HTTP {response.status_code}")
    result={"case_key":case["case_key"],"passed":True,"frames":len(frames),"width":case["width"],"height":case["height"],"fps":24,
            "gpu_seconds":round(gpu_seconds,3),"peak_cuda_gb":round(torch.cuda.max_memory_allocated()/(1024**3),3),
            "artifact_sha256":digest,"benchmark_result_id":cb.get("benchmark_result_id"),"human_review_status":cb.get("human_review_status")}
    results.append(result)
    print(json.dumps({"event":"wan22_cinematic_case",**result}),flush=True)

print(json.dumps({"event":"wan22_cinematic_batch","passed":True,"elapsed_seconds":round(time.time()-started,3),"results":results}),flush=True)
