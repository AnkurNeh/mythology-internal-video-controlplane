#!/usr/bin/env python3
"""APN FFmpeg pilot assembler. Processes an immutable, pre-staged SceneCraft manifest.

This offline runner NEVER fetches URLs, calls SceneCraft, authorises payments, or
marks production quality as approved. The trusted orchestrator stages source media
and later publishes the verified output to an authorised, durable object store.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

VERSION = "apn_ffmpeg_manifest_v1"
FPS = 24
WIDTH, HEIGHT = 1920, 1080
MAX_DURATION = 1800
MAX_CLIPS = 500
EPS = 0.03
LOOK_FILTERS = {
    "none": "",
    "boost": ",eq=contrast=1.06:saturation=1.12",
    "contrast": ",eq=contrast=1.15",
    "muted": ",eq=saturation=0.76",
    "lighten": ",eq=brightness=0.04",
    "darken": ",eq=brightness=-0.04",
}
AUDIO_EFFECTS = {"none", "fadeIn", "fadeOut", "fadeInFadeOut"}


class RenderInputError(ValueError):
    pass


def as_finite(value, label, minimum=0, maximum=MAX_DURATION):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise RenderInputError(f"{label} must be a number") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RenderInputError(f"{label} out of range")
    return number


def asset_filename(source_url):
    """Deterministic stage file basename; signed source URL is never logged."""
    if not isinstance(source_url, str):
        raise RenderInputError("Source URL missing")
    u = urlsplit(source_url)
    if (u.scheme != "https" or not u.hostname or u.username or u.password
            or u.fragment or u.port or u.hostname in ("localhost",)
            or u.hostname.endswith((".local", ".internal"))
            or not "." in u.hostname):
        raise RenderInputError("Only public HTTPS source media references are allowed")
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:32] + ".media"


def staged_input(root, source_url):
    root = Path(root).resolve()
    path = (root / asset_filename(source_url)).resolve()
    if not path.is_relative_to(root) or not path.is_file() or not path.stat().st_size:
        raise RenderInputError("Required media file has not been staged by the trusted downloader")
    return str(path)


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("version") != VERSION:
        raise RenderInputError("Unknown APN manifest contract version")
    if manifest.get("profile") not in ("pilot", "master"):
        raise RenderInputError("Only approved pilot or master output is supported")
    duration = as_finite(manifest.get("duration_seconds"), "duration", 1)
    if manifest["profile"] == "pilot" and not 15 <= duration <= 30:
        raise RenderInputError("Pilot must be between 15 and 30 seconds")
    if len(str(manifest.get("source_quality_binding", ""))) < 12:
        raise RenderInputError("The immutable production source binding is missing")
    output = manifest.get("output", {})
    if not isinstance(output, dict) or any((
        output.get("container") != "mp4",
        output.get("video_codec") != "libx264",
        output.get("audio_codec") != "aac",
        output.get("width") != WIDTH, output.get("height") != HEIGHT,
        output.get("fps") != FPS, output.get("pixel_format") != "yuv420p",
    )):
        raise RenderInputError("Only the verified Full HD H.264/AAC format is accepted")
    video, audio = manifest.get("video"), manifest.get("audio")
    if not isinstance(video, list) or not 1 <= len(video) <= MAX_CLIPS:
        raise RenderInputError("The picture timeline must contain 1–500 clips")
    if not isinstance(audio, list) or not 1 <= len(audio) <= MAX_CLIPS:
        raise RenderInputError("The audio timeline must contain 1–500 layers")
    current = 0.
    for index, clip in enumerate(video):
        if not isinstance(clip, dict) or clip.get("transition") != "cut":
            raise RenderInputError("Unapproved transitions are not allowed")
        if clip.get("look") not in LOOK_FILTERS:
            raise RenderInputError("Unsupported color finishing look")
        asset_filename(clip.get("url"))
        start = as_finite(clip.get("start_seconds"), f"picture {index} start")
        length = as_finite(clip.get("duration_seconds"), f"picture {index} length", 0.02)
        if abs(start - current) > EPS:
            raise RenderInputError("Picture clips are not continuous; do not render blank gaps")
        current = start + length
    if abs(current-duration) > EPS:
        raise RenderInputError("Picture duration differs from the exact source timeline")
    for index, item in enumerate(audio):
        if not isinstance(item, dict):
            raise RenderInputError("Invalid audio layer")
        asset_filename(item.get("url"))
        start = as_finite(item.get("start_seconds"), f"audio {index} start")
        length = as_finite(item.get("duration_seconds"), f"audio {index} length", .001)
        as_finite(item.get("trim_seconds", 0), f"audio {index} trim")
        as_finite(item.get("gain", 1), f"audio {index} gain", 0, 2)
        if item.get("effect", "none") not in AUDIO_EFFECTS:
            raise RenderInputError("Unapproved audio effect")
        if start+length > duration+EPS:
            raise RenderInputError("Audio layer extends beyond the film")
    caption = manifest.get("caption_srt", "")
    if not isinstance(caption, str) or len(caption) > 250000:
        raise RenderInputError("Invalid timed captions")
    return duration


def run(args, *, timeout=900):
    result = subprocess.run(
        args, check=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        # FFmpeg can echo media URLs/tokens in stderr: never leak raw output.
        raise RuntimeError(f"Video processor failed, code {result.returncode}; check isolated worker logs")
    return result.stdout


def probe(path):
    text = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt",
        "-of", "json", str(path)
    ], timeout=60)
    return json.loads(text)


def as_ratio(text):
    numerator, sep, denominator = str(text).partition("/")
    if not sep:
        return float(text)
    return float(numerator)/float(denominator)


def compose(manifest, input_dir, output_file, report_file):
    duration = validate_manifest(manifest)
    input_dir = Path(input_dir).resolve()
    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".mp4":
        raise RenderInputError("Output must have an .mp4 extension")

    # Validate all required media before starting FFmpeg.
    for clip in manifest["video"] + manifest["audio"]:
        staged_input(input_dir, clip["url"])

    with tempfile.TemporaryDirectory(prefix="apn_ffmpeg_") as td:
        folder = Path(td)
        segments = []
        for index, clip in enumerate(manifest["video"]):
            segment = folder / f"segment_{index:04d}.mp4"
            length = float(clip["duration_seconds"])
            frames = max(1, round(length*FPS))
            vf = (
                f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos,"
                f"setsar=1,format=yuv420p" +
                LOOK_FILTERS[clip["look"]]
            )
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-y", "-i", staged_input(input_dir, clip["url"]),
                 "-an", "-vf", vf, "-frames:v", str(frames), "-r", str(FPS),
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(segment)], timeout=600)
            segments.append(segment)

        # All video segments use an identical codec, geometry and frame cadence.
        concat_file = folder / "concat.txt"
        concat_file.write_text(
            "".join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in segments),
            encoding="utf-8"
        )
        picture_file = folder / "picture.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-an", "-c:v", "copy", str(picture_file)], timeout=900)

        # Each approved audio layer is trimmed and offset explicitly.
        audio_inputs, audio_filters, labels = [], [], []
        for index, layer in enumerate(manifest["audio"]):
            audio_inputs.extend(["-i", staged_input(input_dir, layer["url"])])
            length = float(layer["duration_seconds"])
            trim = float(layer["trim_seconds"])
            start = int(round(float(layer["start_seconds"])*1000))
            effect = layer.get("effect", "none")
            chain = (
                f"[{index}:a:0]atrim=start={trim:.3f}:duration={length:.3f},"
                f"asetpts=PTS-STARTPTS,aresample=48000,volume={float(layer['gain']):.4f}"
            )
            fade_length = min(0.25, length/4)
            if effect in ("fadeIn", "fadeInFadeOut"):
                chain += f",afade=t=in:st=0:d={fade_length:.3f}"
            if effect in ("fadeOut", "fadeInFadeOut"):
                chain += f",afade=t=out:st={max(0,length-fade_length):.3f}:d={fade_length:.3f}"
            chain += f",adelay={start}:all=1[a{index}]"
            audio_filters.append(chain)
            labels.append(f"[a{index}]")
        audio_filters.append(
            "".join(labels)+
            f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0,"
            f"atrim=duration={duration:.3f},alimiter=limit=0.95[aout]"
        )
        mixed_audio = folder / "mix.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-y", *audio_inputs, "-filter_complex", ";".join(audio_filters),
             "-map", "[aout]", "-ac", "2", "-ar", "48000",
             "-c:a", "pcm_s16le", str(mixed_audio)], timeout=900)

        # Captions are embedded as a switchable MP4 text track.
        # The SceneCraft player also exposes a separate SRT from its original
        # approved caption record; client-side burn-in remains a later QA gate.
        caption_file = folder / "captions.srt"
        has_captions = bool(manifest.get("caption_srt", "").strip())
        if has_captions:
            caption_file.write_text(manifest["caption_srt"], encoding="utf-8")
        mux = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y", "-i", str(picture_file), "-i", str(mixed_audio)
        ]
        if has_captions:
            mux.extend(["-i", str(caption_file)])
        mux.extend(["-map", "0:v:0", "-map", "1:a:0"])
        if has_captions:
            mux.extend(["-map", "2:s:0", "-c:s", "mov_text"])
        mux.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
                    "-t", f"{duration:.3f}", str(out_path)])
        run(mux, timeout=900)

    technical = probe(out_path)
    streams = technical.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    media_duration = float(technical.get("format", {}).get("duration", 0))
    passed = (
        video.get("codec_name") == "h264" and audio.get("codec_name") == "aac"
        and video.get("width") == WIDTH and video.get("height") == HEIGHT
        and video.get("pix_fmt") == "yuv420p"
        and abs(as_ratio(video.get("r_frame_rate", "0/1"))-FPS)<0.03
        and abs(media_duration-duration)<0.16
    )
    if not passed:
        raise RenderInputError("Encoded result failed the 1080p H.264/AAC duration and FPS gate")
    sha = hashlib.sha256()
    with open(out_path, "rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            sha.update(part)
    report = {
        "version": VERSION,
        "production_run_id": manifest["production_run_id"],
        "profile": manifest["profile"],
        "source_quality_binding": manifest["source_quality_binding"],
        "output_sha256": sha.hexdigest(),
        "bytes": out_path.stat().st_size,
        "width": video["width"], "height": video["height"],
        "fps": FPS, "video_codec": "h264", "audio_codec": "aac",
        "duration_seconds": round(media_duration, 3),
        "encoded_probe_passed": passed,
        "full_decoded_black_gap_scan_passed": False,
        "complete_audio_review_passed": False,
        "human_picture_and_caption_review_passed": False,
        "commercial_release_approved": False,
        "next_action": "Upload to durable authorised storage; re-read exact output and run SceneCraft full decoded technical and human QA."
    }
    Path(report_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Offline APN FFmpeg assembly prototype")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    document = json.loads(Path(options.manifest).read_text(encoding="utf-8"))
    try:
        duration = validate_manifest(document)
        if options.validate_only:
            print(json.dumps({"valid": True, "duration": duration, "video_clips":len(document["video"]),
                              "audio_layers":len(document["audio"]), "paid_job_started": False}))
            return
        result = compose(document, options.input_dir, options.output, options.report)
        print(json.dumps({k: result[k] for k in ("version","output_sha256","duration_seconds","encoded_probe_passed",
                                                   "commercial_release_approved")}))
    except (RenderInputError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"valid":False,"error":str(error)[:200],"paid_job_started":False}),file=sys.stderr)
        sys.exit(2)


if __name__=="__main__":
    main()