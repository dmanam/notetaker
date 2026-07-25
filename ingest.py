#!/usr/bin/env python3
"""
ingest.py — Download a lecture video and transcribe it with timestamps.

Usage:
  python ingest.py <input> [--output-dir DIR] [--model MODEL] [--language LANG]
                   [--transcribe {local,modal}]

<input> can be:
  - A local file path (e.g. /path/to/lecture.mp4)
  - A direct video URL
  - A YouTube URL (e.g. https://www.youtube.com/watch?v=...)

Output: <output-dir>/<video-id>/
  transcript.json   — segment-level transcript with timestamps
  audio.wav         — extracted audio (16kHz mono)
  info.json         — video metadata (title, duration, source)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def is_youtube_url(s: str) -> bool:
    return bool(re.search(r"(youtube\.com/watch|youtu\.be/)", s))


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def expand_playlist(source: str) -> list[str] | None:
    """
    If source is a playlist page (e.g. youtube.com/playlist?list=...), return
    the video URLs in playlist order; otherwise None. A watch URL that merely
    carries a &list= parameter is treated as a single video, consistent with
    the --no-playlist download behavior.
    """
    if not is_url(source) or "list=" not in source or "watch?v=" in source:
        return None
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "-J", source],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"yt-dlp failed to read playlist {source}:\n{result.stderr}")
    info = json.loads(result.stdout)
    urls = []
    for entry in info.get("entries") or []:
        u = entry.get("url") or entry.get("id")
        if u and not is_url(u):
            u = f"https://www.youtube.com/watch?v={u}"
        if u:
            urls.append(u)
    return urls or None


def slug(title: str) -> str:
    """Convert a title to a safe directory name."""
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:80]


def unique_lecture_dir(output_root: Path, title_slug: str, source: str) -> Path:
    """Directory for this lecture. If a directory with the same title-slug
    already belongs to a *different* source (series with repeated video
    titles), disambiguate with a short source hash instead of overwriting."""
    d = output_root / title_slug
    info_path = d / "info.json"
    if d.exists() and info_path.exists():
        try:
            existing = json.loads(info_path.read_text()).get("source")
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing is not None and existing != source:
            h = hashlib.sha1(source.encode()).hexdigest()[:8]
            d = output_root / f"{title_slug}-{h}"
    return d


def download_video(source: str, work_dir: Path) -> tuple[Path, dict]:
    """
    Download video to work_dir. Returns (video_path, metadata_dict).
    For local files, copies them in place.
    """
    if not is_url(source):
        # Local file
        src = Path(source).resolve()
        if not src.exists():
            sys.exit(f"File not found: {src}")
        dest = work_dir / src.name
        shutil.copy2(src, dest)
        meta = {
            "title": src.stem,
            "source": str(src),
            "source_type": "file",
        }
        return dest, meta

    # URL or YouTube — use yt-dlp for both. Output is NOT captured, so the
    # download progress bar is visible.
    print(f"Downloading: {source}")
    video_template = str(work_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "--output", video_template,
        source,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"yt-dlp failed for {source} "
                 f"(exit code {result.returncode}; see output above)")

    info_path = work_dir / "video.info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    # Find the downloaded video file (excluding the .info.json sidecar)
    videos = [p for p in work_dir.glob("video.*")
              if p.suffix.lower() != ".json"]
    if not videos:
        sys.exit("yt-dlp ran but no video file was found in work dir")
    video_path = videos[0]

    meta = {
        "title": info.get("title", video_path.stem),
        "source": source,
        "source_type": "youtube" if is_youtube_url(source) else "url",
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url"),
    }
    return video_path, meta


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono WAV from video using ffmpeg."""
    print(f"Extracting audio → {audio_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",       # quiet, but…
        "-stats",                   # …keep the live progress line
        "-i", str(video_path),
        "-vn",                      # no video
        "-ac", "1",                 # mono
        "-ar", "16000",             # 16kHz (Whisper native)
        "-acodec", "pcm_s16le",     # 16-bit PCM
        str(audio_path),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed for {video_path.name} "
                 f"(exit code {result.returncode}; see output above)")


def resolve_whisper_model(model_name: str | None, backend: str) -> str:
    """Pick a default model size per backend: a remote GPU can afford large-v3."""
    if model_name:
        return model_name
    return "large-v3" if backend == "modal" else "base"


def transcribe(audio_path: Path, model_name: str, language: str | None,
               backend: str = "local",
               source_url: str | None = None) -> tuple[list[dict], str]:
    """
    Transcribe one audio file with Whisper, locally or on a Modal GPU.
    For Modal with a remote source, the worker downloads the audio itself
    (source_url) instead of us uploading it. Returns (segments,
    detected_language) where each segment is
      { "start": float, "end": float, "text": str }
    """
    return transcribe_batch([(Path(audio_path), source_url)],
                            model_name, language, backend)[0]


def transcribe_batch(jobs: list[tuple[Path, str | None]], model_name: str,
                     language: str | None,
                     backend: str = "local") -> list[tuple[list[dict], str]]:
    """
    Transcribe several lectures. jobs is a list of (audio_path, source_url):
    source_url non-None means the Modal worker should download the audio
    itself. Local backend runs serially (one GPU); Modal fans all jobs out in
    parallel across containers, each of which loads the Whisper model once.
    Results are returned in job order as (segments, detected_language).
    """
    if backend != "modal":
        return [_transcribe_local(audio_path, model_name, language)
                for audio_path, _ in jobs]

    try:
        import modal
    except ImportError:
        sys.exit("The 'modal' package is not installed; enter the nix devshell "
                 "or run with --transcribe local.")
    from modal_transcribe import Transcriber, app

    print(f"Dispatching {len(jobs)} transcription(s) to Modal "
          f"(model '{model_name}')…")
    results: list[tuple[list[dict], str]] = []
    with modal.enable_output(), app.run():
        t = Transcriber(model_name=model_name)

        handles = []
        for audio_path, source_url in jobs:
            if source_url:
                # Worker fetches the audio itself — no local upload.
                handles.append((t.from_url.spawn(source_url, language),
                                audio_path, source_url))
            else:
                data, suffix = _compress_for_upload(audio_path)
                print(f"Uploading {len(data) / 1e6:.0f} MB of audio "
                      f"({audio_path.parent.name})…")
                handles.append((t.from_bytes.spawn(data, language, suffix),
                                audio_path, None))

        for handle, audio_path, source_url in handles:
            try:
                r = handle.get()
            except Exception as exc:
                if source_url is None:
                    raise
                print(f"Worker-side download failed for {source_url} "
                      f"({exc}); uploading the audio instead.")
                data, suffix = _compress_for_upload(audio_path)
                r = t.from_bytes.remote(data, language, suffix)
            results.append((r["segments"], r["language"]))
    return results


def _transcribe_local(audio_path: Path, model_name: str,
                      language: str | None) -> tuple[list[dict], str]:
    import whisper

    print(f"Loading Whisper model '{model_name}'…")
    model = whisper.load_model(model_name)

    print("Transcribing (this may take a while)…")
    # condition_on_previous_text=False: math lectures have long silent
    # board-writing stretches, where conditioning makes Whisper loop and
    # hallucinate.
    options = dict(language=language) if language else {}
    result = model.transcribe(str(audio_path), verbose=False,
                              condition_on_previous_text=False, **options)

    segments = [
        {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]
    detected_language = result.get("language", "unknown")
    print(f"Detected language: {detected_language}")
    return segments, detected_language


def _compress_for_upload(audio_path: Path) -> tuple[bytes, str]:
    """FLAC-compress the WAV before upload (~2x smaller, lossless). Falls
    back to raw WAV bytes if ffmpeg fails."""
    with tempfile.NamedTemporaryFile(suffix=".flac") as tmp:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-c:a", "flac", tmp.name],
            capture_output=True,
        )
        if result.returncode == 0:
            return Path(tmp.name).read_bytes(), ".flac"
    return audio_path.read_bytes(), ".wav"




def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Video file path, URL, or YouTube link")
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory (default: ./output)")
    parser.add_argument("--model", default=None,
                        choices=["tiny", "base", "small", "medium", "large",
                                 "large-v2", "large-v3", "turbo"],
                        help="Whisper model size (default: base locally, "
                             "large-v3 on Modal)")
    parser.add_argument("--language", default=None,
                        help="Force language code (e.g. 'en'). Auto-detect if omitted.")
    parser.add_argument("--transcribe", default="local",
                        choices=["local", "modal"],
                        help="Where to run Whisper: on this machine, or on a "
                             "Modal GPU (default: local)")
    args = parser.parse_args()
    whisper_model = resolve_whisper_model(args.model, args.transcribe)

    output_root = Path(args.output_dir)

    # Use a temp dir for intermediate files, then move to final location
    with tempfile.TemporaryDirectory(prefix="notetaker-") as tmp:
        tmp_dir = Path(tmp)

        # Step 1: Download / copy
        video_path, meta = download_video(args.input, tmp_dir)

        # Determine final output dir from title (source-hash suffix if a
        # different lecture already claimed this title)
        out_dir = unique_lecture_dir(output_root, slug(meta["title"]), args.input)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {out_dir}")

        # Step 2: Extract audio
        audio_path = tmp_dir / "audio.wav"
        extract_audio(video_path, audio_path)

        # Step 3: Transcribe
        segments, detected_language = transcribe(
            audio_path, whisper_model, args.language, args.transcribe,
            source_url=args.input if is_url(args.input) else None)
        meta["detected_language"] = detected_language
        meta["whisper_model"] = whisper_model
        meta["segment_count"] = len(segments)

        # Step 4: Write outputs
        audio_dest = out_dir / "audio.wav"
        shutil.move(str(audio_path), audio_dest)

        video_dest = None
        if meta["source_type"] != "file":
            video_dest = out_dir / video_path.name
            shutil.move(str(video_path), video_dest)

        transcript_dest = out_dir / "transcript.json"
        with open(transcript_dest, "w") as f:
            json.dump({"metadata": meta, "segments": segments}, f, indent=2)

        info_dest = out_dir / "info.json"
        with open(info_dest, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Transcript : {transcript_dest}")
    print(f"  Audio      : {audio_dest}")
    if video_dest:
        print(f"  Video      : {video_dest}")
    print(f"  Segments   : {len(segments)}")

    # Print a short preview
    print("\nTranscript preview (first 5 segments):")
    for seg in segments[:5]:
        t = seg["start"]
        m, s = divmod(int(t), 60)
        print(f"  [{m:02d}:{s:02d}] {seg['text']}")


if __name__ == "__main__":
    main()
