"""
retranscribe.py — re-transcribe already-ingested lectures from their local
audio, on Modal, WITHOUT touching the network for media.

Why this exists separately from ingest.py and build_course.py: both of those
will fetch a video if one is missing, and a re-transcription is exactly the
situation where you do not want that — the videos are already on disk, and
re-downloading a course risks the source blocking you. This script has no
downloader in it. It reads `audio.wav` from each lecture directory and
uploads it; there is no code path here that can reach yt-dlp, and the Modal
worker's own `from_url` fetcher is never called.

    python retranscribe.py --language en                 # every lecture
    python retranscribe.py --language en <dir> [<dir>…]  # named ones
    python retranscribe.py --language en --missing       # only those lacking
                                                         # a transcript

`--missing` is the recovery case: a transcript deleted on purpose (to redo it
in the right language) or by accident. Existing transcripts are never
overwritten unless you name their directory explicitly or pass --force.

The output is byte-for-byte the shape build_course.py writes:
`{"metadata": …, "segments": [{"start","end","text"}, …]}`, with info.json
updated to match, so a later `--verify` or `--regen` sees nothing unusual.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "en"


def compress_for_upload(audio_path: Path) -> tuple[bytes, str]:
    """FLAC-compress the WAV before upload (~2x smaller, lossless). Falls
    back to raw WAV bytes if ffmpeg is unhappy."""
    with tempfile.NamedTemporaryFile(suffix=".flac") as tmp:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-c:a", "flac", tmp.name],
            capture_output=True)
        if r.returncode == 0:
            return Path(tmp.name).read_bytes(), ".flac"
    return audio_path.read_bytes(), ".wav"


def lectures(root: Path, names: list[str], missing_only: bool) -> list[Path]:
    if names:
        dirs = [Path(n) if Path(n).is_dir() else root / n for n in names]
    else:
        dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and (d / "audio.wav").exists())
    out = []
    for d in dirs:
        if not (d / "audio.wav").exists():
            print(f"  skip {d.name}: no audio.wav (nothing to work from)")
            continue
        if missing_only and (d / "transcript.json").exists():
            continue
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="Lecture directories (default: all)")
    ap.add_argument("--output-dir", default="output", type=Path)
    ap.add_argument("--language", default=DEFAULT_LANGUAGE,
                    help="Whisper language code (default: en). 'auto' to let "
                         "Whisper guess — it guesses from the first 30 "
                         "seconds and gets lectures wrong.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--missing", action="store_true",
                    help="Only lectures with no transcript.json.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite transcripts that already exist.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    language = None if a.language in ("auto", "detect", "") else a.language
    todo = lectures(a.output_dir, a.dirs, a.missing)
    if not a.missing and not a.force and not a.dirs:
        todo = [d for d in todo if not (d / "transcript.json").exists()]
    if not todo:
        print("Nothing to transcribe.")
        return

    print(f"Re-transcribing {len(todo)} lecture(s) from local audio "
          f"(model {a.model}, language {language or 'auto'}):")
    for d in todo:
        size = (d / "audio.wav").stat().st_size / 1e6
        print(f"  {d.name}  ({size:.0f} MB audio)")
    if a.dry_run:
        return

    import modal                                          # noqa: F401
    from modal_transcribe import Transcriber, app

    with modal.enable_output(), app.run():
        t = Transcriber(model_name=a.model)
        handles = []
        for d in todo:
            data, suffix = compress_for_upload(d / "audio.wav")
            print(f"Uploading {len(data) / 1e6:.0f} MB ({d.name})…", flush=True)
            # from_bytes only. from_url would have the worker fetch the media
            # itself, which is the thing this script exists to avoid.
            handles.append((d, t.from_bytes.spawn(data, language, suffix)))

        failures = []
        for d, handle in handles:
            try:
                result = handle.get()
            except Exception as exc:                       # one bad lecture
                print(f"  {d.name}: FAILED — {exc}")       # must not lose the
                failures.append(d.name)                    # rest of the batch
                continue
            segments = result["segments"]
            meta = {}
            info = d / "info.json"
            if info.exists():
                try:
                    meta = json.loads(info.read_text())
                except json.JSONDecodeError:
                    pass
            meta["detected_language"] = result.get("language", "unknown")
            meta["whisper_model"] = a.model
            meta["segment_count"] = len(segments)
            (d / "transcript.json").write_text(
                json.dumps({"metadata": meta, "segments": segments}, indent=2))
            info.write_text(json.dumps(meta, indent=2))
            print(f"  {d.name}: {len(segments)} segments "
                  f"({meta['detected_language']})")

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
