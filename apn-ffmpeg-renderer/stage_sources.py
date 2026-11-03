"""Trusted APN media staging boundary for the owned FFmpeg worker.

The downloader is *only* for media URLs already authorized by SceneCraft's
server-side production quality checks. Never expose it as a public fetch proxy.
"""
import hashlib
import ipaddress
import os
from pathlib import Path
import socket
import urllib.error
import urllib.parse
import urllib.request

ALLOWED_HOSTS = frozenset({"base44.app", "media.base44.com"})
MAX_ASSET_BYTES = 250 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_COUNT = 500
TIMEOUT_SECONDS = 35


class StagingError(ValueError):
    pass


def validate_source_url(raw):
    if not isinstance(raw, str) or not 1 <= len(raw) <= 4096:
        raise StagingError("Media URL is missing or too large")
    parsed = urllib.parse.urlsplit(raw)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or
            parsed.username or parsed.password or parsed.port or parsed.fragment):
        raise StagingError("Media source must use an explicitly permitted HTTPS origin")
    if parsed.hostname == "base44.app":
        if not parsed.path.startswith("/api/apps/") or "/files/mp/public/" not in parsed.path:
            raise StagingError("Expected an approved Base44 public-media file path")
    elif not (parsed.path.startswith(("/videos/", "/audios/", "/images/", "/files/"))):
        raise StagingError("Expected a Base44 media asset path")
    if not parsed.path or "%" in parsed.path and ("%2f" in parsed.path.lower() or "%5c" in parsed.path.lower()):
        raise StagingError("Encoded URL path separator not allowed")
    return raw


def deny_nonpublic_dns(host):
    """Defense in depth; production egress firewall MUST block private networks."""
    try:
        addresses=socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as ex:
        raise StagingError("Public source DNS lookup failed") from ex
    for row in addresses:
        ip=ipaddress.ip_address(row[4][0].split("%")[0])
        if not ip.is_global:
            raise StagingError("Media hostname resolves to a non-public IP address")
    if not addresses:
        raise StagingError("No public media source address returned")


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise StagingError("Source redirect not permitted")


def _basename(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]+".media"


def stage_sources(manifest, input_directory, *, max_total_bytes=MAX_TOTAL_BYTES,
                  downloader=None, validate_dns=True):
    """Download approved media into a private, bounded, isolated job directory."""
    video=manifest.get("video", [])
    audio=manifest.get("audio", [])
    all_sources=video+audio
    if not 1 <= len(all_sources) <= MAX_SOURCE_COUNT * 2:
        raise StagingError("Manifest exceeds staging media limits")
    root=Path(input_directory).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    seen=set()
    written=0
    staged=[]
    opener=downloader or urllib.request.build_opener(NoRedirectHandler())
    for item in all_sources:
        url=validate_source_url(item.get("url"))
        if url in seen:
            continue
        seen.add(url)
        host=urllib.parse.urlsplit(url).hostname
        if validate_dns:
            deny_nonpublic_dns(host)
        request=urllib.request.Request(url, headers={"User-Agent":"APN-FFmpeg-Worker/1.0"})
        filename=_basename(url)
        dest=(root/filename).resolve()
        if not dest.is_relative_to(root):
            raise StagingError("Media staging path escaped job directory")
        try:
            with opener.open(request, timeout=TIMEOUT_SECONDS) as stream, open(dest,"xb") as out:
                content_type=str(stream.headers.get("Content-Type","")).split(";")[0].lower()
                if content_type not in {
                    "video/mp4","video/quicktime","audio/mpeg","audio/wav","audio/x-wav",
                    "audio/mp4","audio/aac","application/octet-stream"
                }:
                    raise StagingError("Source response is not an approved video/audio type")
                size=0
                while True:
                    chunk=stream.read(256*1024)
                    if not chunk:
                        break
                    size+=len(chunk)
                    written+=len(chunk)
                    if size>MAX_ASSET_BYTES or written>max_total_bytes:
                        raise StagingError("Source media exceeds per-file or total limit")
                    out.write(chunk)
                if not size:
                    raise StagingError("Empty media source")
        except StagingError:
            dest.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as ex:
            dest.unlink(missing_ok=True)
            raise StagingError("Approved source media could not be staged") from ex
        staged.append({"sha256_url_prefix":filename[:8],"bytes":size})
    return {"source_count":len(staged),"total_bytes":written,"path":str(root)}