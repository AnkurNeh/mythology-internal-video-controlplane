"""APN-owned SceneCraft FFmpeg gateway / worker for deployment in an AU cloud project.

Production external services (Cloud Tasks, Firestore and GCS) are required.
All commercial job creation is DISABLED by default; never use this as an
anonymous rendering endpoint or an alternative payment processor.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import secrets
import tempfile
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from runner import compose, validate_manifest, RenderInputError
from stage_sources import stage_sources, validate_source_url

VERSION="apn_ffmpeg_renderer_v1"
MANIFEST_VERSION="apn_ffmpeg_manifest_v1"
COLLECTION=os.getenv("APN_RENDERER_FIRESTORE_COLLECTION","apn_render_jobs_v1")
APP_ROLE=os.getenv("APN_RENDERER_ROLE","gateway")
AUDIENCE=os.getenv("APN_RENDERER_TASK_AUDIENCE","")
MAX_BYTES=2_000_000
DEFAULT_JOB_SECONDS=30
app=FastAPI(title="APN Owned FFmpeg",version=VERSION,docs_url=None,redoc_url=None,openapi_url=None)

def settings():
    return {key:os.getenv(key,"").strip() for key in [
        "APN_RENDERER_TOKEN","APN_RENDERER_OUTPUT_SIGNING_KEY",
        "APN_RENDERER_PROJECT","APN_RENDERER_REGION","APN_RENDERER_BUCKET",
        "APN_RENDERER_QUEUE_PATH","APN_RENDERER_WORKER_URL",
        "APN_RENDERER_TASK_SERVICE_ACCOUNT","APN_RENDERER_TASK_AUDIENCE",
        "APN_RENDERER_GATEWAY_SERVICE_ACCOUNT",
        "APN_RENDERER_PUBLIC_URL","APN_RENDERER_LIVE_ENABLED",
        "APN_RENDERER_APPROVED_BUDGET_AUD","APN_RENDERER_DAILY_JOB_LIMIT"
    ]}

def production_settings_ready(env):
    strings=[
        "APN_RENDERER_TOKEN","APN_RENDERER_OUTPUT_SIGNING_KEY",
        "APN_RENDERER_PROJECT","APN_RENDERER_REGION","APN_RENDERER_BUCKET",
        "APN_RENDERER_QUEUE_PATH","APN_RENDERER_WORKER_URL",
        "APN_RENDERER_TASK_SERVICE_ACCOUNT","APN_RENDERER_TASK_AUDIENCE",
        "APN_RENDERER_GATEWAY_SERVICE_ACCOUNT",
        "APN_RENDERER_PUBLIC_URL"
    ]
    try:
        budget=float(env["APN_RENDERER_APPROVED_BUDGET_AUD"])
        daily_limit=int(env["APN_RENDERER_DAILY_JOB_LIMIT"])
    except (TypeError,ValueError):
        return False
    if not (0<budget<=100000 and 1<=daily_limit<=20):
        return False
    return all(bool(env[k]) for k in strings) and len(env["APN_RENDERER_TOKEN"])>=32 and \
      len(env["APN_RENDERER_OUTPUT_SIGNING_KEY"])>=32 and \
      env["APN_RENDERER_REGION"]=="australia-southeast1" and \
      env["APN_RENDERER_QUEUE_PATH"].startswith("projects/"+env["APN_RENDERER_PROJECT"]+
                                                 "/locations/australia-southeast1/queues/")

def authenticate_platform(request:Request):
    env=settings()
    provided=request.headers.get("Authorization","")
    expected="Bearer "+env["APN_RENDERER_TOKEN"]
    if not env["APN_RENDERER_TOKEN"] or not hmac.compare_digest(provided,expected):
        raise HTTPException(401,"APN service authentication required")
    if APP_ROLE!="gateway":
        raise HTTPException(403,"This route is only available on the APN gateway")
    return env

def authenticate_worker(request:Request,*,health_probe=False):
    if APP_ROLE!="worker":
        raise HTTPException(404,"Worker endpoint unavailable")
    provided=request.headers.get("Authorization","")
    if not provided.startswith("Bearer ") or not AUDIENCE:
        raise HTTPException(401,"Cloud Tasks service identity required")
    try:
        from google.oauth2 import id_token
        from google.auth.transport.requests import Request as GoogleRequest
        identity=id_token.verify_oauth2_token(provided[len("Bearer "):],GoogleRequest(),audience=AUDIENCE)
        expected_identity=settings()["APN_RENDERER_GATEWAY_SERVICE_ACCOUNT" if health_probe else "APN_RENDERER_TASK_SERVICE_ACCOUNT"]
        if identity.get("iss") not in ("https://accounts.google.com","accounts.google.com") or \
          not expected_identity or identity.get("email")!=expected_identity or \
          identity.get("email_verified") is not True:
            raise ValueError("Unapproved service identity")
        return identity
    except Exception:
        raise HTTPException(401,"Cloud Tasks service identity not verified") from None

def clients():
    from google.cloud import firestore, storage, tasks_v2
    env=settings()
    return firestore.Client(project=env["APN_RENDERER_PROJECT"],
                             database=os.getenv("APN_RENDERER_FIRESTORE_DATABASE","(default)")), \
           storage.Client(project=env["APN_RENDERER_PROJECT"]), \
           tasks_v2.CloudTasksClient()

def get_record(store,job_id):
    if not re.fullmatch(r"[0-9a-f]{40}",str(job_id)):
        raise HTTPException(400,"Invalid render job identifier")
    doc=store.collection(COLLECTION).document(job_id)
    snap=doc.get()
    if not snap.exists:
        raise HTTPException(404,"Render job not found")
    return doc,snap.to_dict()

def expected_scope(request,record):
    for header,key in [("X-APN-Tenant","tenant_id"),("X-APN-Project","project_id")]:
        value=request.headers.get(header,"")
        if not value or not hmac.compare_digest(str(value),str(record.get(key,""))):
            raise HTTPException(403,"Render job belongs to another customer scope")

def current_time():
    return datetime.now(timezone.utc)

def object_path(record,attempt_id=None):
    prefix="tenants/"+hashlib.sha256(record["tenant_id"].encode()).hexdigest()[:16]+\
        "/projects/"+hashlib.sha256(record["project_id"].encode()).hexdigest()[:16]+\
        "/renders/"+record["job_id"]+"/"
    if attempt_id is not None:
        if not re.fullmatch(r"[0-9a-f]{32}",str(attempt_id)):
            raise ValueError("Invalid render attempt")
        return prefix+"attempts/"+attempt_id+"/output.mp4"
    saved=record.get("output_storage")
    if saved:
        if not str(saved).startswith(prefix) or ".." in str(saved):
            raise ValueError("Output storage is outside the render scope")
        return saved
    return prefix+"output.mp4"

def signed_output(job_id,record,env):
    expiry=int(time.time())+300
    digest=str(record.get("output_sha256",""))
    message=f"{job_id}:{expiry}:{digest}".encode()
    signature=hmac.new(env["APN_RENDERER_OUTPUT_SIGNING_KEY"].encode(),message,hashlib.sha256).hexdigest()
    origin=env["APN_RENDERER_PUBLIC_URL"].rstrip("/")
    return f"{origin}/v1/outputs/{job_id}?expires={expiry}&signature={signature}"

def cloud_health():
    env=settings()
    ffmpeg_installed=bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    ready=False
    if env["APN_RENDERER_LIVE_ENABLED"].lower()=="true" and production_settings_ready(env) and ffmpeg_installed:
        try:
            db,storage,tasks=clients()
            bucket=storage.bucket(env["APN_RENDERER_BUCKET"])
            bucket_exists=bucket.exists(timeout=5)
            if bucket_exists:
                bucket.reload(timeout=5)
            bucket_australian=bucket_exists and str(bucket.location or "").lower()=="australia-southeast1"
            db.collection(COLLECTION).document("__health").get(timeout=5)
            if APP_ROLE=="worker":
                ready=bucket_australian
            else:
                tasks.get_queue(request={"name":env["APN_RENDERER_QUEUE_PATH"]},timeout=5)
                from google.oauth2 import id_token
                from google.auth.transport.requests import Request as GoogleRequest
                from urllib.request import Request, urlopen
                token=id_token.fetch_id_token(GoogleRequest(),env["APN_RENDERER_TASK_AUDIENCE"])
                url=env["APN_RENDERER_WORKER_URL"].rstrip("/")+"/internal/healthz"
                req=Request(url,headers={"Authorization":"Bearer "+token})
                with urlopen(req,timeout=8) as res:
                    data=json.load(res)
                ready=bucket_australian and data.get("worker_ready") is True
        except Exception:
            ready=False
    return {
        "version":VERSION,
        "status":"ready" if ready else "staging",
        "ffmpeg_installed":ffmpeg_installed,
        "output_store_ready":ready,
        "authentication_enabled":True,
        "output_formats":["mp4"],
        "capabilities":{"full_hd_h264":ready,"audio_mix":ready,"continuous_picture":ready},
        "deployment_region":env["APN_RENDERER_REGION"] or "not_set",
        "production_submission_enabled":ready and APP_ROLE=="gateway",
        "cloud_paid_job_started":False
    }

@app.get("/")
def home():
    return {"service":"apn-ffmpeg-final-assembly","status":"authenticated"}

@app.get("/healthz")
def health(request:Request):
    authenticate_platform(request)
    return cloud_health()

@app.get("/internal/healthz")
def internal_health(request:Request):
    authenticate_worker(request,health_probe=True)
    state=cloud_health()
    return {"worker_ready":state["status"]=="ready",
            "version":VERSION,"region":state["deployment_region"]}

class CreateJob(BaseModel):
    tenant_id:str
    project_id:str
    owner_id:str
    quote_id:str
    idempotency_key:str
    manifest:dict

def validate_new_job(payload):
    if not all(4<=len(str(getattr(payload,key,"")))<=140 for key in ("tenant_id","project_id","owner_id","quote_id")):
        raise HTTPException(400,"A source-bound, project-scoped approved customer render quote is required")
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{16,128}",payload.idempotency_key):
        raise HTTPException(400,"A bounded idempotency key is required")
    if payload.manifest.get("profile")!="pilot":
        raise HTTPException(409,"Full masters remain disabled until separately qualified")
    if payload.manifest.get("version")!=MANIFEST_VERSION:
        raise HTTPException(400,"Invalid render manifest version")
    try:
        duration=validate_manifest(payload.manifest)
        for asset in payload.manifest["video"]+payload.manifest["audio"]:
            validate_source_url(asset.get("url"))
    except (RenderInputError,ValueError) as exc:
        raise HTTPException(400,"SceneCraft render manifest is invalid or unsafe") from exc
    if duration>DEFAULT_JOB_SECONDS:
        raise HTTPException(400,"Pilot duration exceeds the owned-worker limit")
    # The Base44 server must independently validate the actual reserved quote.
    return duration

def ensure_dispatch(db,ref,tasks,env,job_id):
    """Recover the same immutable job/task after an uncertain queue response."""
    from google.cloud import firestore, tasks_v2
    from google.api_core.exceptions import AlreadyExists

    @firestore.transactional
    def record_dispatch(txn,state):
        snap=ref.get(transaction=txn)
        row=snap.to_dict() if snap.exists else {}
        # A late enqueue receipt/error must never overwrite a worker's state.
        if row.get("status") not in ("queued","enqueue_failed"):
            return row
        if row.get("dispatch_status")=="enqueued" and state!="enqueued":
            return row
        changes={"status":"queued","dispatch_status":state,
                 "last_error_code":"" if state=="enqueued" else "QUEUE_DISPATCH_UNVERIFIED"}
        txn.update(ref,changes)
        return {**row,**changes}

    row=record_dispatch(db.transaction(),"pending")
    if row.get("status")!="queued" or row.get("dispatch_status")=="enqueued":
        return row
    task_name=env["APN_RENDERER_QUEUE_PATH"]+"/tasks/apn-"+job_id
    task={
        "name":task_name,
        "http_request":{
            "http_method":tasks_v2.HttpMethod.POST,
            "url":env["APN_RENDERER_WORKER_URL"].rstrip("/")+"/internal/v1/run/"+job_id,
            "headers":{"Content-Type":"application/json"},
            "oidc_token":{"service_account_email":env["APN_RENDERER_TASK_SERVICE_ACCOUNT"],
                          "audience":env["APN_RENDERER_TASK_AUDIENCE"]}
        }
    }
    try:
        try:
            tasks.create_task(request={"parent":env["APN_RENDERER_QUEUE_PATH"],"task":task},timeout=10)
        except AlreadyExists:
            # A name tombstone is not proof a live task remains. If get_task
            # cannot verify it, retain the job and require reconciliation.
            tasks.get_task(request={"name":task_name},timeout=5)
    except Exception:
        row=record_dispatch(db.transaction(),"uncertain")
        if row.get("status") in ("running","done","failed") or row.get("dispatch_status")=="enqueued":
            return row
        raise HTTPException(503,"Job stored; queue dispatch requires reconciliation. Retry this exact idempotency key, never create a replacement job.") from None
    return record_dispatch(db.transaction(),"enqueued")

@app.post("/v1/jobs")
async def submit(request:Request):
    env=authenticate_platform(request)
    body=bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body)>MAX_BYTES:
            raise HTTPException(413,"Render manifest too large")
    try:
        payload=CreateJob.model_validate(json.loads(body))
    except (ValueError,TypeError,ValidationError):
        raise HTTPException(400,"A valid source-bound render request is required") from None
    validate_new_job(payload)
    expected_scope(request,{"tenant_id":payload.tenant_id,"project_id":payload.project_id})
    if not cloud_health()["production_submission_enabled"]:
        raise HTTPException(409,"APN renderer is not ready for billable production; no job created")
    manifest_json=json.dumps(payload.manifest,sort_keys=True,separators=(",",":"))
    digest=hashlib.sha256(manifest_json.encode()).hexdigest()
    # Encode scope as a tuple; separator characters in IDs cannot alias another customer.
    job_scope=json.dumps([payload.tenant_id,payload.project_id,payload.idempotency_key],
                         ensure_ascii=True,separators=(",",":"))
    job_id=hashlib.sha256(job_scope.encode()).hexdigest()[:40]
    db,_,tasks=clients()
    ref=db.collection(COLLECTION).document(job_id)
    record={"job_id":job_id,"status":"queued","tenant_id":payload.tenant_id,"project_id":payload.project_id,
       "owner_id":payload.owner_id,"quote_id":payload.quote_id,
       "idempotency_key":payload.idempotency_key,"source_quality_binding":payload.manifest["source_quality_binding"],
       "manifest":payload.manifest,"manifest_digest":digest,"created_at":current_time(),
       "duration_seconds":payload.manifest["duration_seconds"],"provider":"apn_ffmpeg"}
    # Atomic tenant-scoped daily quota + idempotent render creation.
    # This is a job-count ceiling, not a hard limit on the GCP invoice.
    from google.cloud import firestore
    today=current_time().strftime("%Y%m%d")
    quota_key=hashlib.sha256(payload.tenant_id.encode()).hexdigest()[:20]+"-"+today
    quota_ref=db.collection("apn_render_daily_quota_v1").document(quota_key)
    daily_limit=int(env["APN_RENDERER_DAILY_JOB_LIMIT"])
    @firestore.transactional
    def create_once(transaction):
        previous=ref.get(transaction=transaction)
        if previous.exists:
            row=previous.to_dict()
            if (row.get("manifest_digest")!=digest or row.get("quote_id")!=payload.quote_id or
                    row.get("owner_id")!=payload.owner_id or
                    row.get("tenant_id")!=payload.tenant_id or row.get("project_id")!=payload.project_id):
                return "conflict", None
            return "reused", row
        quota_snap=quota_ref.get(transaction=transaction)
        count=int(quota_snap.to_dict().get("count",0)) if quota_snap.exists else 0
        if count>=daily_limit:
            return "daily_quota_exceeded", None
        transaction.create(ref,record)
        transaction.set(quota_ref,{"count":count+1,"day_utc":today,
                          "tenant_hash":hashlib.sha256(payload.tenant_id.encode()).hexdigest()[:20]},
                        merge=True)
        return "created", record
    mode,existing=create_once(db.transaction())
    if mode=="conflict":
        raise HTTPException(409,"Idempotency key already refers to a different render")
    if mode=="daily_quota_exceeded":
        raise HTTPException(429,"APN-owned daily tenant render-job allowance exhausted")
    reused=mode=="reused"
    actual=ensure_dispatch(db,ref,tasks,env,job_id)
    return {"success":True,"id":job_id,"status":actual.get("status"),
            "reused":reused,"provider":"apn_ffmpeg","customer_charge_finalised":False}

@app.get("/v1/jobs/{job_id}")
def job_status(job_id:str,request:Request):
    env=authenticate_platform(request)
    db,_,_=clients()
    _,record=get_record(db,job_id)
    expected_scope(request,record)
    response={"id":job_id,"status":record.get("status"),
      "duration_seconds":record.get("duration_seconds"),
      "source_quality_binding":record.get("source_quality_binding"),
      "output_sha256":record.get("output_sha256") if record.get("status")=="done" else None,
      "failure_code":record.get("last_error_code") or None,
      "dispatch_status":record.get("dispatch_status","unknown"),
      "retry_requires_operator":record.get("retry_requires_operator",False)}
    if record.get("status")=="done" and record.get("output_sha256"):
        response["url"]=signed_output(job_id,record,env)
    return response

@app.get("/v1/outputs/{job_id}")
def output(job_id:str,expires:int,signature:str):
    env=settings()
    if APP_ROLE!="gateway" or not production_settings_ready(env):
        raise HTTPException(404,"Output not available")
    if abs(int(time.time())-expires)>600 or expires<int(time.time()) or expires>int(time.time())+600:
        raise HTTPException(403,"Output link expired")
    db,storage,_=clients()
    _,record=get_record(db,job_id)
    if record.get("status")!="done" or not record.get("output_sha256"):
        raise HTTPException(409,"Output is not complete")
    message=f"{job_id}:{expires}:{record['output_sha256']}".encode()
    expected=hmac.new(env["APN_RENDERER_OUTPUT_SIGNING_KEY"].encode(),message,hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(signature),expected):
        raise HTTPException(403,"Invalid output link")
    blob=storage.bucket(env["APN_RENDERER_BUCKET"]).blob(object_path(record))
    def chunks():
        with blob.open("rb",chunk_size=1024*1024) as stream:
            while True:
                segment=stream.read(1024*1024)
                if not segment:
                    break
                yield segment
    return StreamingResponse(chunks(),media_type="video/mp4",
      headers={"Cache-Control":"private, no-store","Content-Disposition":'attachment; filename="apn-render.mp4"'})

@app.post("/internal/v1/run/{job_id}")
def worker_execute(job_id:str,request:Request):
    authenticate_worker(request)
    env=settings()
    if cloud_health()["status"]!="ready":
        raise HTTPException(503,"APN worker not configured")
    db,storage,_=clients()
    ref,_=get_record(db,job_id)
    from google.cloud import firestore
    attempt_id=secrets.token_hex(16)

    @firestore.transactional
    def claim(txn):
        snap=ref.get(transaction=txn)
        if not snap.exists:
            return "missing",None
        row=snap.to_dict()
        if row.get("status") in ("done","failed"):
            return row["status"],row
        if row.get("status")=="running" and row.get("lease_until") and row["lease_until"]>current_time():
            return "running",row
        if row.get("status") not in ("queued","running"):
            return "blocked",row
        attempts=int(row.get("attempt_number",0))
        if attempts>=3:
            txn.update(ref,{"status":"failed","last_error_code":"WORKER_ATTEMPT_LIMIT",
                            "retry_requires_operator":True,"failed_at":current_time()})
            return "failed",row
        changes={"status":"running","started_at":current_time(),
                 "attempt_id":attempt_id,"attempt_number":attempts+1,
                 "lease_until":current_time()+timedelta(minutes=20)}
        txn.update(ref,changes)
        return "claimed",{**row,**changes}

    claimed,job=claim(db.transaction())
    if claimed=="running":
        # A 2xx acknowledges the Cloud Task and could strand an interrupted
        # worker forever. Keep delivery retryable until completion or takeover.
        raise HTTPException(503,"Render attempt still running; retry the same task",
                            headers={"Retry-After":"30"})
    if claimed in ("done","failed"):
        return {"status":claimed,"id":job_id,"reused":True,
                "commercial_release_approved":False}
    if claimed!="claimed":
        raise HTTPException(409,"Render job cannot be claimed for processing")

    @firestore.transactional
    def finish_owned_attempt(txn,changes):
        snap=ref.get(transaction=txn)
        row=snap.to_dict() if snap.exists else {}
        if (row.get("status")!="running" or row.get("attempt_id")!=attempt_id or
                not row.get("lease_until") or row["lease_until"]<=current_time()):
            return False
        txn.update(ref,changes)
        return True

    try:
        with tempfile.TemporaryDirectory(prefix="apn-owned-render-") as temp:
            root=Path(temp)
            manifest=job["manifest"]
            digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            if not hmac.compare_digest(digest,str(job.get("manifest_digest",""))):
                raise RuntimeError("Immutable render manifest does not match its receipt")
            stage_sources(manifest,root/"inputs")
            video_path=root/"output"/"render.mp4";video_path.parent.mkdir()
            report_path=root/"output"/"report.json"
            started=time.monotonic()
            report=compose(manifest,root/"inputs",video_path,report_path)
            actual_digest=hashlib.sha256(video_path.read_bytes()).hexdigest()
            if (not report.get("encoded_probe_passed") or report.get("commercial_release_approved") or
                    not hmac.compare_digest(str(report.get("output_sha256","")),actual_digest)):
                raise RuntimeError("The worker did not return valid technical evidence")
            bucket=storage.bucket(env["APN_RENDERER_BUCKET"])
            # Each attempt owns a distinct, create-only object. An expired
            # worker cannot overwrite the winning attempt's output bytes.
            key=object_path(job,attempt_id)
            media=bucket.blob(key)
            media.upload_from_filename(str(video_path),content_type="video/mp4",timeout=120,
                                       if_generation_match=0)
            metadata=bucket.blob(key+".report.json")
            metadata.upload_from_string(json.dumps(report),content_type="application/json",timeout=60,
                                        if_generation_match=0)
            completed=finish_owned_attempt(db.transaction(),{
              "status":"done","completed_at":current_time(),
              "output_sha256":actual_digest,"output_storage":key,
              "encoded_probe_passed":True,"full_decoded_black_gap_scan_passed":False,
              "human_review_approved":False,
              "measured_worker_wall_seconds":round(time.monotonic()-started,3),
              "last_error_code":""})
            if not completed:
                raise HTTPException(409,"Render attempt lost its lease; current job result was preserved")
            return {"status":"done","id":job_id,"encoded_probe_passed":True,
                    "commercial_release_approved":False}
    except HTTPException:
        raise
    except Exception:
        failed=finish_owned_attempt(db.transaction(),{
          "status":"failed","failed_at":current_time(),
          "last_error_code":"WORKER_EXECUTION_OR_STORAGE_FAILED","retry_requires_operator":True})
        if not failed:
            raise HTTPException(409,"Render attempt no longer owns the job; current result was preserved") from None
        raise HTTPException(503,"APN owned render failed; matching wallet quote requires safe reconciliation") from None
