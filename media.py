"""
media.py — Shared helpers for lecture media: frame extraction, video lookup,
and transcript formatting. Used by generate_notes.py, build_course.py, and the
tool handlers in notes_tools.py.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v")


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        t = seg["start"]
        m, s = divmod(int(t), 60)
        lines.append(f"[{m:02d}:{s:02d}] {seg['text']}")
    return "\n".join(lines)


def _ffmpeg_frame(video_path: Path, timestamp: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "3",             # JPEG quality (2-5 is good)
        "-vf", "scale=1280:-1",  # cap width to keep tokens reasonable
        str(out_path),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def extract_frame(video_path: Path, timestamp: float) -> str | None:
    """
    Extract a single video frame at `timestamp` seconds and return it as a
    base64-encoded JPEG string. Returns None on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    if not _ffmpeg_frame(video_path, timestamp, tmp_path):
        return None

    with open(tmp_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    Path(tmp_path).unlink(missing_ok=True)
    return data


def extract_frame_file(video_path: Path, timestamp: float,
                       dest_dir: Path) -> Path | None:
    """
    Extract a single video frame at `timestamp` seconds into dest_dir and
    return the JPEG's path. Returns None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"frame_{timestamp:08.1f}s.jpg"
    if not _ffmpeg_frame(video_path, timestamp, out_path):
        return None
    return out_path


def find_video(lecture_dir: Path) -> Path | None:
    """
    Locate the lecture video for frame extraction. Checks the lecture dir
    first; for local-file sources (which ingest does not copy), falls back to
    the original path recorded in info.json.
    """
    for ext in VIDEO_EXTS:
        hits = list(lecture_dir.glob(f"video{ext}"))
        if hits:
            return hits[0]
    for p in lecture_dir.iterdir():
        if p.suffix.lower() in VIDEO_EXTS:
            return p

    info_path = lecture_dir / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            return None
        if info.get("source_type") == "file":
            src = Path(info.get("source", ""))
            if src.exists():
                return src
    return None
