# notetaker

Turns recorded math lectures into typeset LaTeX notes.

A lecture (local file, URL, or YouTube link) is downloaded and transcribed
with Whisper, then an agent — Claude or GPT, running on your existing chat
subscription — writes graduate-course-style LaTeX notes. Along the way it can
look at video frames to read the board, fetch referenced papers, and queue
questions for you without stopping its work.

## Setup

```sh
nix develop        # python env, ffmpeg, yt-dlp, claude + codex CLIs
```

CUDA is off by default (transcription is expected to run on Modal); use
`nix develop .#cuda` if you want local GPU Whisper.

Authenticate the backend you want to use (once):

| Backend        | Model      | Auth                                        |
|----------------|------------|---------------------------------------------|
| `subscription` (default) | Claude Opus | `claude` → log in with your Claude Pro/Max account |
| `codex`        | GPT (gpt-5.6-sol) | `codex login` with your ChatGPT account |
| `api`          | Claude Opus | `export ANTHROPIC_API_KEY=…` (pay per token) |

Optional, for remote GPU transcription: `modal setup` (one-time Modal login).

## Single lecture

```sh
python ingest.py "https://www.youtube.com/watch?v=..."   # download + transcribe
python generate_notes.py output/<lecture-slug>            # write notes.tex
```

Useful flags: `--transcribe modal` (Whisper large-v3 on a Modal GPU — your
locally-extracted audio is FLAC-compressed and uploaded; add `--modal-fetch`
to have the worker download it itself, if YouTube permits its egress),
`--model tiny|base|…|large-v3|turbo` (Whisper size), `--backend`, `--model`,
`--reference URL_OR_ARXIV_ID` (preload a paper as context).

## Lecture series

```sh
python build_course.py "https://www.youtube.com/playlist?list=..." \
    --title "Prismatic Cohomology" --transcribe modal
```

Inputs can be individual videos, local files, `--from-file lectures.txt`
(one per line), or playlist URLs — playlists are expanded into their videos
in order. If YouTube rate-limits you, two escape hatches: `--proxy
socks5://127.0.0.1:1080` routes all local yt-dlp traffic (downloads and
playlist expansion) through a proxy, and `--download modal` runs yt-dlp on
Modal workers instead — their egress fetches the video (and probes
playlists) and ships the bytes back, falling back to a local download if the
worker is the one that gets blocked. Transcription always uploads your
locally-extracted audio by default (`--modal-fetch` opts into worker-side
fetching, with an upload fallback). And if
you're rate-limited mid-series, `--available-only` processes the downloaded
lectures up to the first missing one and stops there (rather than skipping
ahead, which would misnumber later lectures) — rerun later to continue.

You can also pre-fetch with the pipeline itself: `python ingest.py URL
--no-transcribe` downloads and extracts audio without transcribing (ingest is
resumable, so rerunning later — or running build_course — continues from the
cached files instead of re-downloading).

**Manual downloads** slot in as ordinary inputs: freeze the series order once
(`yt-dlp --flat-playlist --print url PLAYLIST > lectures.txt`), then replace
a URL line with a local file path as you fetch videos yourself. Keep each
lecture's identity stable across runs (don't swap an already-ingested URL for
its file), give files meaningful names (the stem becomes the directory slug),
and don't move them afterwards — frame extraction reads local-file videos
from their original location. (A `watch?v=…&list=…` URL counts as a single video; pass the
`playlist?list=…` page to get the whole series.)

Each lecture becomes a `\section` in one `course.tex`, written with the
previous lectures as context: the last two in full, older ones via compact
model-written summaries (`summary.md`), with the agent able to open any
earlier lecture's file when it needs an exact statement. Labels are prefixed
per lecture (`\label{thm:3:...}`) so they never collide, and the assembled
document gets a `pdflatex` compile check when TeX is on your `PATH`.

Ingest is two-phase: all lectures are downloaded first, then every
not-yet-transcribed lecture is transcribed — in parallel across Modal
containers with `--transcribe modal`, so a series takes roughly one
lecture's wall-clock time. State lives in `output/course_state.json` —
rerunning skips lectures that are already done (including downloaded-but-
untranscribed ones after a crash), so you can process a series incrementally. Add lectures at the
**end** only; if you reorder or insert, the numbering warning will tell you
which lectures to `--regen`. Transcript mishearings you confirm once (e.g.
"at all" → "étale") are passed to all later lectures automatically.

A **running bibliography** accumulates in `output/references.bib`: when a
lecture cites something the agent calls a `cite_reference` tool, which
fetches a real BibTeX entry (arXiv and DOI metadata are pulled
automatically; anything else becomes a web entry) and returns the key to
`\cite`. The same source cited from several lectures reuses one entry.
biblatex and `\printbibliography` are wired into the assembled document only
once something has actually been cited, and the compile check runs biber, so
undefined citations are reported rather than silently rendering as `[?]`.

## Accuracy: every lecture is checked by a second pass

Notes like these are reliably right about the mathematics and unreliably
right in the margins — the errors cluster in sentences the model *added*: an
invented "equivalently", a hedge (`I think`, `morally speaking`) quietly
promoted to an assertion, a claim the lecturer retracted five minutes later
kept as though it stood, a paper cited as the one they meant that was
published a year after the lecture.

Two things address this. The writing prompt names those failure modes
directly, and each lecture is given its **recording date** (so a citation can
be checked against it) and told when the transcript is degraded. Then, once a
section is written, a **verification pass** re-reads it against the full
transcript in a *fresh context* — no memory of having written it, and free to
read the transcript backwards and forwards, which is what makes a late
correction visible as superseding an early claim. It looks for statements
that are false as written, self-contradictions, unsupported additions, lost
hedges and lost corrections, propagated speech-recognition garble, and
anachronistic citations; it fixes what it is sure of with the smallest edit
that makes the text true, and weakens (rather than merely `\todo`-flags) what
it cannot settle.

It runs automatically after each new lecture (`--no-verify` to skip) and can
be run on its own over existing notes:

```sh
python build_course.py --verify <slug>    # one lecture
python build_course.py --verify all       # the whole course
```

## Questions never block the run

When the agent is unsure — a garbled transcript passage, a notation choice —
it queues a question for you and keeps working, marking the spot with
`\todo{awaiting answer #N @ hh:mm:ss}`. Prompts appear in the terminal while
it runs:

```
[Transcript unclear #1 @ 00:41:07] "the at all site"
[Question #2 @ 01:02:03 for you] Which package for the prism symbol?
```

- **clarify prompts**: Enter accepts the agent's guess, `?` defers
- **open questions**: type an answer, or Enter to defer

Every question carries the point in the lecture it came from, normalized to
`hh:mm:ss`, so you can jump straight there in the video — a question you
can't locate is a question you can't answer. The timestamp follows it into
the saved question file, the `\todo` marker in the PDF, and the answer block
the agent gets back. Transcript lines are shown to the model in the same
format (a 90-minute lecture used to read `[75:20]`), and it copies the
timestamp from the line the question arose from; `get_frame` and the frame
reader accept either `hh:mm:ss` or raw seconds. Questions queued by older
runs get their timestamps recovered on the next `--answer` pass by locating
the quoted text in the transcript.

Answers you give in time are folded in before the run ends; everything else
is saved (`*.questions.json`) and the run finishes without waiting (pass
`--wait` to block instead). Later:

```sh
python generate_notes.py output/<slug> --answer      # single lecture
python build_course.py --answer <slug>               # one course lecture
python build_course.py --answer-all                  # every open question
```

re-asks the open questions, has the agent revise the notes in place (also
sweeping any remaining `\todo` markers it can now resolve), and — for a
course — propagates the changes to all later lectures (`--no-propagate` to
skip) and reassembles `course.tex`. `--answer` is also useful with no open
questions at all, as a pure todo-sweep pass. Hand edits to a `section.tex`
are safe: assembly prefers the file on disk.

`--answer-all` walks the whole course: it first lists every lecture with
open questions or todos, then works through them in order, and finally
propagates once — each later lecture is visited a single time carrying the
changes from every revised lecture before it, rather than being rewritten
once per revision.

## Compile errors get fixed, not just reported

The compile check is a feedback loop, not a verdict. When the assembled
document fails, each error is traced back to the lecture that wrote it (by
line number, so `l.1858` becomes "line 12 of your section") and handed to
that lecture's agent to fix — content untouched, LaTeX corrected — then the
document is reassembled and re-checked, up to `--latex-fix-rounds N` times
(default 2, `0` to only report). Errors in the shared preamble go to a pass
of their own, since those come from `add_to_preamble` calls made by
different lectures that can collide (two lectures each `\newcommand`-ing the
same macro will not compile).

Undefined citations are only reported once the document otherwise compiles:
a hard error aborts the run before biber, at which point *every* citation
looks undefined, and chasing those would send agents off to "fix" keys that
were fine.

## How frames are read (and why it's cheap)

Reading video frames is the token-expensive part, so it's delegated: the main
model hands timestamps + context to a cheaper reader — a Haiku subagent
(Claude backends) or a `gpt-5.6-luna` subagent role (codex) — which studies
the frames and reports the board contents in LaTeX, including where things
are cut off, occluded, or unreadable. The main model only looks at frames
itself when a report seems off. Override the reader with `--frame-model`.

## Output layout

```
output/
  course.tex                  assembled document (build_course)
  course_state.json           per-lecture state: bodies, summaries, corrections
  references.bib              running bibliography (cited sources)
  references/                 cached fetched papers
  <lecture-slug>/
    video.*  audio.wav        media (video kept for frame extraction)
    transcript.json info.json Whisper output + metadata
    notes.tex                 standalone notes (generate_notes)
    section.tex  summary.md   course section + digest for later lectures
    *.questions.json          open/answered question state
    frames/                   extracted frames (codex backend)
```

## Notes & caveats

- **Whisper model**: local default is `base` (fast, weak on math vocabulary);
  Modal default is `large-v3`. For real use, prefer Modal or a local
  `large-v3`/`turbo` if you have the GPU.
- **Language** defaults to English (`--language de`, `fr`, … for others;
  `--language auto` restores Whisper's own detection). Detection is not the
  default because Whisper judges from the first 30 seconds — room noise over
  mathematical jargon — and a wrong guess is applied to the whole hour. An
  English lecture detected as Latin comes back with passages like *"Talila,
  ala noseps pale Similar sketch by Borno zip"* where forcing `en` gives
  *"this is a Banach space of null sequences"*. Transcripts are cached, so
  changing the language does not re-transcribe what you already have; the
  run warns and tells you which `transcript.json` files to delete.
- **Subscription limits**: a full lecture is a long agentic session; a large
  series will make a visible dent in plan usage limits.
- **Repeated video titles** in a series are handled (directories get a source
  hash suffix) — but tidy playlist titles make tidier slugs.
- **Compile check** only runs if `pdflatex` is on your `PATH` (`latexmk` is
  preferred when present, so biber runs and citations resolve). Without a TeX
  installation the check — and the fix loop — are skipped silently.
- The first Modal run downloads the Whisper weights (cached in a Modal
  volume afterwards).
