#!/usr/bin/env python3
"""
ingest.py — Download a lecture video and transcribe it with timestamps.

Usage:
  python ingest.py <input> [--output-dir DIR] [--model MODEL] [--language LANG]
                   [--transcribe {local,modal}] [--no-transcribe]

Ingest is resumable: rerunning with the same input reuses the downloaded
video/audio (and skips entirely if the transcript already exists). With
--no-transcribe it stops after download + audio extraction — useful for
pre-fetching videos while rate limits allow; transcribe later by rerunning
without the flag, or let build_course.py pick the directory up.

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


def _playlist_urls(info: dict) -> list[str]:
    urls = []
    for entry in info.get("entries") or []:
        u = entry.get("url") or entry.get("id")
        if u and not is_url(u):
            u = f"https://www.youtube.com/watch?v={u}"
        if u:
            urls.append(u)
    return urls


def expand_playlist(source: str, proxy: str | None = None,
                    via_modal: bool = False) -> list[str] | None:
    """
    If source is a playlist page (e.g. youtube.com/playlist?list=...), return
    the video URLs in playlist order; otherwise None. A watch URL that merely
    carries a &list= parameter is treated as a single video, consistent with
    the --no-playlist download behavior. With via_modal, the metadata probe
    runs on a Modal worker (falling back to a local probe on failure).
    """
    if not is_url(source) or "list=" not in source or "watch?v=" in source:
        return None

    if via_modal:
        try:
            import modal
            from modal_transcribe import app, probe_playlist
            print(f"Probing playlist via Modal: {source}")
            with modal.enable_output(), app.run():
                return _playlist_urls(probe_playlist.remote(source)) or None
        except Exception as exc:
            print(f"Modal-side playlist probe failed ({exc}); probing locally.")

    cmd = ["yt-dlp", "--flat-playlist", "-J"]
    if proxy:
        cmd += ["--proxy", proxy]
    result = subprocess.run(cmd + [source], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"yt-dlp failed to read playlist {source}:\n{result.stderr}")
    return _playlist_urls(json.loads(result.stdout)) or None


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


def _meta_from_info(info: dict, source: str, video_path: Path) -> dict:
    return {
        "title": info.get("title", video_path.stem),
        "source": source,
        "source_type": "youtube" if is_youtube_url(source) else "url",
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url"),
    }


def _download_via_modal(source: str, work_dir: Path) -> tuple[Path, dict]:
    import modal
    from modal_transcribe import app, download_media

    print(f"Downloading via Modal: {source}")
    with modal.enable_output(), app.run():
        result = download_media.remote(source)
    video_path = work_dir / result["filename"]
    video_path.write_bytes(result["video"])
    print(f"  Received {video_path.stat().st_size / 1e6:.0f} MB "
          f"({video_path.name})")
    return video_path, _meta_from_info(result["info"], source, video_path)


def download_video(source: str, work_dir: Path,
                   proxy: str | None = None,
                   via_modal: bool = False) -> tuple[Path, dict]:
    """
    Download video to work_dir. Returns (video_path, metadata_dict).
    For local files, copies them in place. proxy is passed to yt-dlp
    (e.g. socks5://127.0.0.1:1080); via_modal routes the download through a
    Modal worker's egress instead (circumvents local-IP rate limiting),
    falling back to a local download on failure.
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

    if via_modal:
        try:
            return _download_via_modal(source, work_dir)
        except Exception as exc:
            print(f"Modal-side download failed ({exc}); downloading locally.")

    # URL or YouTube — use yt-dlp for both. Output is NOT captured, so the
    # download progress bar is visible.
    print(f"Downloading: {source}")
    video_template = str(work_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "--output", video_template,
    ]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd.append(source)
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

    return video_path, _meta_from_info(info, source, video_path)


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


DEFAULT_LANGUAGE = "en"


def resolve_language(value: str | None) -> str | None:
    """CLI language code → Whisper's `language` option (None = auto-detect).

    Whisper judges the language from the first 30 seconds, which in a lecture
    is often room noise over mathematical jargon — it has been seen to settle
    on Latin for an English lecture and then decode the whole hour under that
    assumption, garbling terminology throughout. So auto-detection is opt-in
    ('auto'), not the default."""
    if value is None:
        return DEFAULT_LANGUAGE
    if value.strip().lower() in ("auto", "detect", ""):
        return None
    return value.strip()


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
    parser.add_argument("--language", default=None, metavar="LANG",
                        help="Language code for Whisper (default: en). Pass "
                             "'auto' to let Whisper detect it — it guesses "
                             "from the first 30s and gets lectures wrong.")
    parser.add_argument("--transcribe", default="local",
                        choices=["local", "modal"],
                        help="Where to run Whisper: on this machine, or on a "
                             "Modal GPU (default: local)")
    parser.add_argument("--proxy", default=None, metavar="URL",
                        help="Proxy for local yt-dlp downloads, e.g. "
                             "socks5://127.0.0.1:1080.")
    parser.add_argument("--modal-fetch", action="store_true",
                        help="Let the Modal transcription worker download "
                             "the audio itself instead of receiving the "
                             "locally-extracted audio (YouTube usually "
                             "blocks datacenter egress, hence off by "
                             "default; falls back to uploading on failure).")
    parser.add_argument("--download", default="local",
                        choices=["local", "modal"],
                        help="Where yt-dlp runs: locally, or on a Modal "
                             "worker whose egress circumvents rate limiting "
                             "of your IP (the video is shipped back; falls "
                             "back to a local download on failure).")
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Stop after download + audio extraction (no "
                             "transcription) — pre-fetch videos while rate "
                             "limits allow; transcribe later by rerunning "
                             "without this flag, or via build_course.py.")
    args = parser.parse_args()
    whisper_model = resolve_whisper_model(args.model, args.transcribe)

    output_root = Path(args.output_dir)
    source_key = (args.input if is_url(args.input)
                  else str(Path(args.input).resolve()))

    # Reuse / resume an existing ingest of this source
    out_dir = meta = None
    if output_root.exists():
        for d in output_root.iterdir():
            info_path = d / "info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text())
            except json.JSONDecodeError:
                continue
            if info.get("source") != source_key:
                continue
            if (d / "transcript.json").exists():
                print(f"Already ingested (transcript exists): {d}")
                return
            if (d / "audio.wav").exists():
                print(f"Resuming (audio already extracted): {d.name}")
                out_dir, meta = d, info
            break

    if out_dir is None:
        # Use a temp dir for intermediate files, then move to final location
        with tempfile.TemporaryDirectory(prefix="notetaker-") as tmp:
            tmp_dir = Path(tmp)

            # Step 1: Download / copy
            video_path, meta = download_video(args.input, tmp_dir, args.proxy,
                                              via_modal=args.download == "modal")

            # Determine final output dir from title (source-hash suffix if a
            # different lecture already claimed this title)
            out_dir = unique_lecture_dir(output_root, slug(meta["title"]),
                                         meta["source"])
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {out_dir}")

            # Step 2: Extract audio
            audio_path = tmp_dir / "audio.wav"
            extract_audio(video_path, audio_path)
            shutil.move(str(audio_path), out_dir / "audio.wav")

            if meta["source_type"] != "file":
                shutil.move(str(video_path), out_dir / video_path.name)

            with open(out_dir / "info.json", "w") as f:
                json.dump(meta, f, indent=2)

    if args.no_transcribe:
        print(f"\nDone (transcription skipped).")
        print(f"  Directory : {out_dir}")
        print(f"  Audio     : {out_dir / 'audio.wav'}")
        print("Transcribe later by rerunning without --no-transcribe, or via "
              "build_course.py (this counts as downloaded for "
              "--available-only).")
        return

    # Step 3: Transcribe from the persisted audio
    segments, detected_language = transcribe(
        out_dir / "audio.wav", whisper_model, resolve_language(args.language),
        args.transcribe,
        source_url=args.input
        if (args.modal_fetch and is_url(args.input)) else None)
    meta["detected_language"] = detected_language
    meta["whisper_model"] = whisper_model
    meta["segment_count"] = len(segments)

    transcript_dest = out_dir / "transcript.json"
    with open(transcript_dest, "w") as f:
        json.dump({"metadata": meta, "segments": segments}, f, indent=2)
    with open(out_dir / "info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Transcript : {transcript_dest}")
    print(f"  Audio      : {out_dir / 'audio.wav'}")
    print(f"  Segments   : {len(segments)}")

    # Print a short preview
    print("\nTranscript preview (first 5 segments):")
    for seg in segments[:5]:
        t = seg["start"]
        m, s = divmod(int(t), 60)
        print(f"  [{m:02d}:{s:02d}] {seg['text']}")


if __name__ == "__main__":
    main()
