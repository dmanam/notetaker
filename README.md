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

Useful flags: `--transcribe modal` (Whisper large-v3 on a Modal GPU — for
remote videos the worker downloads the audio itself, nothing is uploaded),
`--model tiny|base|…|large-v3|turbo` (Whisper size), `--backend`, `--model`,
`--reference URL_OR_ARXIV_ID` (preload a paper as context).

## Lecture series

```sh
python build_course.py "https://www.youtube.com/playlist?list=..." \
    --title "Prismatic Cohomology" --transcribe modal
```

Inputs can be individual videos, local files, `--from-file lectures.txt`
(one per line), or playlist URLs — playlists are expanded into their videos
in order. (A `watch?v=…&list=…` URL counts as a single video; pass the
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

## Questions never block the run

When the agent is unsure — a garbled transcript passage, a notation choice —
it queues a question for you and keeps working, marking the spot with
`\todo{awaiting answer #N}`. Prompts appear in the terminal while it runs:

- **clarify prompts**: Enter accepts the agent's guess, `?` defers
- **open questions**: type an answer, or Enter to defer

Answers you give in time are folded in before the run ends; everything else
is saved (`*.questions.json`) and the run finishes without waiting (pass
`--wait` to block instead). Later:

```sh
python generate_notes.py output/<slug> --answer      # single lecture
python build_course.py --answer <slug>               # one course lecture
```

re-asks the open questions, has the agent revise the notes in place (also
sweeping any remaining `\todo` markers it can now resolve), and — for a
course — propagates the changes to all later lectures (`--no-propagate` to
skip) and reassembles `course.tex`. `--answer` is also useful with no open
questions at all, as a pure todo-sweep pass. Hand edits to a `section.tex`
are safe: assembly prefers the file on disk.

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
- **Subscription limits**: a full lecture is a long agentic session; a large
  series will make a visible dent in plan usage limits.
- **Repeated video titles** in a series are handled (directories get a source
  hash suffix) — but tidy playlist titles make tidier slugs.
- **Compile check** only runs if `pdflatex` is on your `PATH`; errors are
  reported, and an `--answer` revision pass is a convenient way to fix them.
- The first Modal run downloads the Whisper weights (cached in a Modal
  volume afterwards).
