"""
modal_transcribe.py — Whisper transcription on Modal GPUs.

Used by ingest.py / build_course.py when run with --transcribe modal. The app
runs ephemerally (`app.run()`), so no `modal deploy` step is needed — just a
one-time `modal setup` (or `modal token new`) to authenticate.

Transcription is a parameterized class so that:
  - the Whisper model is loaded once per container (at container start) and
    reused across lectures the container processes;
  - a lecture series fans out across containers in parallel (capped by
    max_containers) — wall-clock for a series approaches that of one lecture.

Entry points (methods on Transcriber):
  from_bytes(audio_bytes)  — audio uploaded from the local machine (FLAC/WAV)
  from_url(source_url)     — the worker downloads the audio itself with
                             yt-dlp (used for remote videos, so a slow local
                             uplink never has to upload the audio)

Model weights are cached in a persistent Modal volume, so only the first run
of a given model size pays the download.
"""

import modal

app = modal.App("notetaker-transcribe")

WHISPER_CACHE = "/cache/whisper"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("openai-whisper", "yt-dlp")
)

cache_volume = modal.Volume.from_name("notetaker-whisper-cache",
                                      create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",  # 24 GB — comfortably fits large-v3
    timeout=2 * 60 * 60,
    volumes={WHISPER_CACHE: cache_volume},
    max_containers=8,  # parallel fan-out cap for series transcription
)
class Transcriber:
    model_name: str = modal.parameter(default="large-v3")

    @modal.enter()
    def load_model(self):
        import whisper

        print(f"Loading Whisper model '{self.model_name}'…")
        self.model = whisper.load_model(self.model_name,
                                        download_root=WHISPER_CACHE)

    def _run(self, path: str, language: str | None) -> dict:
        # condition_on_previous_text=False: math lectures have long silent
        # board-writing stretches, where conditioning makes Whisper loop and
        # hallucinate.
        options = dict(language=language) if language else {}
        result = self.model.transcribe(path, verbose=False,
                                       condition_on_previous_text=False,
                                       **options)
        return {
            "segments": [
                {
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "text": seg["text"].strip(),
                }
                for seg in result["segments"]
            ],
            "language": result.get("language", "unknown"),
        }

    @modal.method()
    def from_bytes(self, audio_bytes: bytes, language: str | None = None,
                   suffix: str = ".flac") -> dict:
        """Transcribe uploaded audio bytes (16kHz mono FLAC or WAV)."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(audio_bytes)
            f.flush()
            return self._run(f.name, language)

    @modal.method()
    def from_url(self, source_url: str, language: str | None = None) -> dict:
        """Download a remote video's audio track worker-side (yt-dlp) and
        transcribe it. Raises on download failure — callers fall back to
        uploading the audio themselves."""
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            print(f"Fetching audio for: {source_url}")
            subprocess.run(
                ["yt-dlp", "--no-playlist", "-f", "bestaudio/best",
                 "--output", f"{td}/source.%(ext)s", source_url],
                check=True,
            )
            downloaded = next(p for p in Path(td).iterdir()
                              if p.name.startswith("source."))
            wav = f"{td}/audio.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(downloaded),
                 "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
                 wav],
                check=True, capture_output=True,
            )
            return self._run(wav, language)
