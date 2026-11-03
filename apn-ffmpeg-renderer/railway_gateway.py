"""Railway-hosted APN-owned FFmpeg renderer.
Persistent state lives on /data. Base44 copies completed output into durable studio storage.
"""
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import hashlib, hmac, json, os, shutil, threading, time

from runner import compose, validate_manifest, RenderInputError
from stage_sources import stage_sources, StagingError

VERSION="apn_ffmpeg_renderer_v1"
DATA=Path(os.getenv("APN_RENDERER_DATA_DIR","/data")).resolve()
JOBS=DATA/"jobs"
TOKEN=os.getenv("APN_RENDERER_TOKEN","").strip()
SIGNING=os.getenv("APN_RENDERER_OUTPUT_SIGNING_KEY","").strip()
PUBLIC=os.getenv("APN_RENDERER_PUBLIC_URL","").rstrip("/")
REGION=os.getenv("APN_RENDERER_REGION","unknown")
LIVE=os.getenv("APN_RENDERER_LIVE_ENABLED","false").lower()=="true"
LOCK=threading.Lock()
app=FastAPI(title="APN FFmpeg Renderer",docs_url=None,redoc_url=None,openapi_url=None)

class CreateJob(BaseModel):
    tenant_id:str
    project_id:str
    owner_id:str
    quote_id:str
    idempotency_key:str
    manifest:dict

def now(): return datetime.now(timezone.utc).isoformat()
def job_dir(job_id): return (JOBS/job_id).resolve()
def record_path(job_id): return job_dir(job_id)/"job.json"
def read_record(job_id):
    p=record_path(job_id)
    if not p.exists(): raise HTTPException(404,"Render job not found")
    return json.loads(p.read_text())
def write_record(job_id,row):
    d=job_dir(job_id);d.mkdir(parents=True,exist_ok=True)
    tmp=d/"job.tmp";tmp.write_text(json.dumps(row,separators=(",",":")));os.replace(tmp,record_path(job_id))
def auth(request):
    provided=request.headers.get("Authorization","")
    if len(TOKEN)<32 or not hmac.compare_digest(provided,"Bearer "+TOKEN):
        raise HTTPException(401,"APN service authentication required")
def scope(request,row):
    for header,key in (("X-APN-Tenant","tenant_id"),("X-APN-Project","project_id")):
        value=request.headers.get(header,"")
        if not value or not hmac.compare_digest(str(value),str(row.get(key,""))):
            raise HTTPException(403,"Render job belongs to another customer scope")
def job_id_for(payload):
    raw=(payload.tenant_id+"|"+payload.project_id+"|"+payload.quote_id+"|"+payload.idempotency_key).encode()
    return hashlib.sha1(raw).hexdigest()
def signed_url(job_id,sha):
    expiry=int(time.time())+600
    sig=hmac.new(SIGNING.encode(),f"{job_id}:{expiry}:{sha}".encode(),hashlib.sha256).hexdigest()
    return f"{PUBLIC}/v1/outputs/{job_id}?expires={expiry}&signature={sig}"
def render_job(job_id):
    try:
        row=read_record(job_id)
        row.update(status="running",started_at=now());write_record(job_id,row)
        d=job_dir(job_id);inputs=d/"inputs";output=d/"output.mp4";report=d/"report.json"
        stage_sources(row["manifest"],inputs)
        compose(row["manifest"],inputs,output,report)
        data=output.read_bytes();sha=hashlib.sha256(data).hexdigest()
        technical=json.loads(report.read_text()) if report.exists() else {}
        row.update(status="done",output_sha256=sha,duration_seconds=float(row["manifest"]["duration_seconds"]),
                   completed_at=now(),technical_report=technical,failure_code="",retry_requires_operator=False)
        write_record(job_id,row)
        shutil.rmtree(inputs,ignore_errors=True)
    except Exception as e:
        try:
            row=read_record(job_id)
            row.update(status="failed",completed_at=now(),failure_code=type(e).__name__[:80],
                       retry_requires_operator=True,last_error="Render failed; inspect worker logs.")
            write_record(job_id,row)
        except Exception: pass

@app.get("/")
def home(): return {"service":"apn-ffmpeg-final-assembly","version":VERSION}

@app.get("/healthz")
def health(request:Request):
    auth(request)
    DATA.mkdir(parents=True,exist_ok=True);JOBS.mkdir(parents=True,exist_ok=True)
    writable=os.access(DATA,os.W_OK)
    ff=bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    ready=LIVE and writable and ff and len(TOKEN)>=32 and len(SIGNING)>=32 and PUBLIC.startswith("https://")
    return {"version":VERSION,"status":"ready" if ready else "staging","ffmpeg_installed":ff,
            "output_store_ready":writable,"authentication_enabled":True,"output_formats":["mp4"],
            "capabilities":{"full_hd_h264":ready,"audio_mix":ready,"continuous_picture":ready},
            "deployment_region":REGION,"production_submission_enabled":ready,"cloud_paid_job_started":False}

@app.post("/v1/jobs")
def create_job(payload:CreateJob,request:Request,background_tasks:BackgroundTasks):
    auth(request)
    if not LIVE: raise HTTPException(503,"Renderer production submission is disabled")
    if not all(4<=len(str(getattr(payload,k,"")))<=140 for k in ("tenant_id","project_id","owner_id","quote_id")):
        raise HTTPException(400,"A source-bound project-scoped render quote is required")
    if not (16<=len(payload.idempotency_key)<=128): raise HTTPException(400,"A bounded idempotency key is required")
    try: validate_manifest(payload.manifest)
    except (RenderInputError,ValueError) as e: raise HTTPException(400,"SceneCraft render manifest is invalid") from e
    if payload.manifest.get("profile")!="pilot": raise HTTPException(409,"Full masters remain disabled until separately qualified")
    job_id=job_id_for(payload)
    with LOCK:
        if record_path(job_id).exists():
            row=read_record(job_id);scope(request,row)
            return {"id":job_id,"status":row["status"],"reused":True}
        row={"id":job_id,"tenant_id":payload.tenant_id,"project_id":payload.project_id,"owner_id":payload.owner_id,
             "quote_id":payload.quote_id,"idempotency_key":payload.idempotency_key,"manifest":payload.manifest,
             "status":"queued","created_at":now(),"output_sha256":"","failure_code":"","retry_requires_operator":False}
        write_record(job_id,row)
    background_tasks.add_task(render_job,job_id)
    return {"id":job_id,"status":"queued","reused":False}

@app.get("/v1/jobs/{job_id}")
def get_job(job_id:str,request:Request):
    auth(request)
    if len(job_id)!=40 or any(c not in "0123456789abcdef" for c in job_id): raise HTTPException(400,"Invalid render job identifier")
    row=read_record(job_id);scope(request,row)
    result={"id":job_id,"status":row["status"],"output_sha256":row.get("output_sha256",""),
            "duration_seconds":row.get("duration_seconds",0),"failure_code":row.get("failure_code",""),
            "retry_requires_operator":row.get("retry_requires_operator",False)}
    if row["status"]=="done": result["url"]=signed_url(job_id,row["output_sha256"])
    return result

@app.get("/v1/outputs/{job_id}")
def get_output(job_id:str,expires:int,signature:str):
    row=read_record(job_id)
    if row.get("status")!="done": raise HTTPException(409,"Output is not ready")
    if expires<int(time.time()) or expires>int(time.time())+900: raise HTTPException(403,"Output link expired")
    expected=hmac.new(SIGNING.encode(),f"{job_id}:{expires}:{row.get('output_sha256','')}".encode(),hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature,expected): raise HTTPException(403,"Invalid output signature")
    path=job_dir(job_id)/"output.mp4"
    if not path.exists(): raise HTTPException(404,"Output missing")
    return FileResponse(path,media_type="video/mp4",filename="scenecraft-pilot.mp4")
