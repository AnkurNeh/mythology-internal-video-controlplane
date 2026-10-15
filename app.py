import os, time, uuid, threading
from typing import Any, Dict
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, HttpUrl

app=FastAPI(title="Mythology Internal Video Control Plane",version="1.1.0")
JOBS: Dict[str,Dict[str,Any]]={}
LOCK=threading.Lock()
class InputSpec(BaseModel):
    first_frame_url: HttpUrl
    last_frame_url: HttpUrl|None=None
    reference_urls: list[HttpUrl]=Field(default_factory=list,max_length=8)
class Controls(BaseModel):
    seed: int|None=None
    generate_audio: bool=False
class Request(BaseModel):
    model: str
    model_version: str=""
    profile: str
    duration_seconds: float=Field(gt=0,le=30)
    aspect_ratio: str
    prompt: str
    input: InputSpec
    controls: Controls=Field(default_factory=Controls)
    metadata: dict=Field(default_factory=dict)
def auth(value:str):
    key=os.getenv("INTERNAL_VIDEO_SERVICE_API_KEY","")
    if not key: raise HTTPException(503,"service key not configured")
    if value!="Bearer "+key: raise HTTPException(401,"unauthorized")
def configured(): return bool(os.getenv("HF_VIDEO_ENDPOINT_URL")) and bool(os.getenv("HF_VIDEO_ENDPOINT_TOKEN"))
@app.get('/healthz')
def health(): return {'status':'ok'}
@app.get('/readyz')
def ready():
    if not os.getenv("INTERNAL_VIDEO_SERVICE_API_KEY"): raise HTTPException(503,"service key missing")
    if not configured(): raise HTTPException(503,"GPU endpoint not configured")
    return {'status':'ready','gpu_endpoint':True}
def run(job_id:str,payload:dict):
    with LOCK: JOBS[job_id]['status']='processing'
    try:
        url=os.environ['HF_VIDEO_ENDPOINT_URL'].rstrip('/')
        token=os.environ['HF_VIDEO_ENDPOINT_TOKEN']
        with httpx.Client(timeout=float(os.getenv('HF_VIDEO_TIMEOUT_SECONDS','1800'))) as client:
            r=client.post(url,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},json={'inputs':payload})
            r.raise_for_status(); data=r.json()
        result=data.get('result') if isinstance(data,dict) and isinstance(data.get('result'),dict) else data
        video_bytes=result.get('video_bytes_base64') or result.get('videoBytesBase64')
        video_url=result.get('video_url') or result.get('url')
        if not video_bytes and not (isinstance(video_url,str) and video_url.startswith('https://')):
            raise RuntimeError('GPU endpoint returned neither video bytes nor durable video URL')
        output={'sha256':result.get('sha256',''),'gpu_seconds':float(result.get('gpu_seconds',0) or 0),'compute_cost':float(result.get('compute_cost',0) or 0)}
        if video_bytes: output.update(video_bytes_base64=video_bytes,mime_type=result.get('mime_type','video/mp4'))
        else: output['video_url']=video_url
        with LOCK: JOBS[job_id].update(status='completed',result=output)
    except Exception as exc:
        with LOCK: JOBS[job_id].update(status='failed',error={'class':'gpu_endpoint','message':str(exc)[:2000]})
@app.post('/v1/generations')
def create(payload:Request,authorization:str=Header(default=''),idempotency_key:str=Header(default='',alias='Idempotency-Key')):
    auth(authorization)
    if not configured(): raise HTTPException(503,'GPU endpoint not configured')
    if not idempotency_key: raise HTTPException(400,'Idempotency-Key required')
    with LOCK:
        for j in JOBS.values():
            if j.get('idempotency_key')==idempotency_key:return {'job_id':j['job_id'],'status':j['status']}
        job_id=str(uuid.uuid4());JOBS[job_id]={'job_id':job_id,'status':'submitted','idempotency_key':idempotency_key,'created_at':time.time()}
    threading.Thread(target=run,args=(job_id,payload.model_dump(mode='json')),daemon=True).start()
    return {'job_id':job_id,'status':'submitted'}
@app.get('/v1/generations/{job_id}')
def status(job_id:str,authorization:str=Header(default='')):
    auth(authorization);job=JOBS.get(job_id)
    if not job:raise HTTPException(404,'job not found')
    return job
@app.post('/v1/generations/{job_id}/cancel')
def cancel(job_id:str,authorization:str=Header(default='')):
    auth(authorization);job=JOBS.get(job_id)
    if not job:raise HTTPException(404,'job not found')
    if job['status']=='submitted':job['status']='cancelled'
    return {'job_id':job_id,'status':job['status']}
