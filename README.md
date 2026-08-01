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

Every agent that touches a section — writing it, revising it after your
answers, verifying it, propagating a change, or fixing a LaTeX error — is
given an **index of the other lectures** (title, notes path, summary path) and
a listing of **what is already in the bibliography**. The first means a fix
that turns on what an earlier lecture actually said has something to work
from; the second stops an agent re-deriving an arXiv ID or DOI, and
re-searching the web, for a source that is already cited. Both are indexes
rather than content — full summaries run to hundreds of kilobytes across a
long course — so the agent opens what it needs.

A **running bibliography** accumulates in `output/references.bib`: when a
lecture cites something the agent calls a `cite_reference` tool, which
fetches a real BibTeX entry (arXiv and DOI metadata are pulled
automatically; anything else becomes a web entry) and returns the key to
`\cite`. The same source cited from several lectures reuses one entry.
biblatex and `\printbibliography` are wired into the assembled document only
once something has actually been cited, and the compile check runs biber, so
undefined citations are reported rather than silently rendering as `[?]`.

### Who is lecturing

Notes written by a professional say "Whitlock defines" and "as Ostrand pointed
out" — surname alone — not "Dana says" or "the speaker claims". That needs a
name, and nothing in a video file reliably has one: the transcript rarely says
it and the uploader is an institution. So each new lecture's speaker is asked
once, before any model runs, with a guess offered as the default:

```sh
python build_course.py … --lecturer "Dana Whitlock"   # one speaker, no questions
python build_course.py …                              # asked per lecture
python lecturer.py output                             # just show me the guesses
```

The guess is a single cheap model call over the whole series rather than a
pattern per title. The hard part is not spotting two capitalised words but
deciding which pair is a person — `Spectral Sequences | Marek Ostrand` has two
candidates — and seeing every title at once settles it: a phrase in all of
them is the series, a name in only some is a speaker the series alternates
with. On the 24-lecture test course it gets all 24 right, including the
alternation between the two lecturers, for about $0.07 API-equivalent.

A wrong name is worse than none, because it gets printed as an attribution, so
the guesser is told to answer "unknown" rather than reach — it declines on
`Homotopy Theory 4` even when the transcript opens by discussing Quillen — and
everything unnamed falls back to the phrase "the lecturer". Press Enter to
accept a suggestion, `?` to decline it. Answers live in
`output/course_state.json` and are never re-asked; with no terminal the
suggestions are taken silently, so unattended runs still work. `--lecturer`
also applies in `--verify` and `--answer` mode, which is how a course written
before a name was recorded gets one.

## Exporting as a multi-file LaTeX project

`course.tex` is one large assembled file. To get something editable and
version-controllable instead:

```sh
python build_course.py --export ~/notes/advanced-topology [--export-compile]
```

```
main.tex                       preamble, \input lines, bibliography
lectures/01-<slug>.tex         one file per lecture (body only)
lectures/02-<slug>.tex
references.bib
```

`main.tex` builds with `latexmk -pdf main.tex` and is self-contained —
the same preamble, `\newcommand`s and bibliography wiring as the single-file
build, which it shares code with so the two cannot drift. Lecture files are
numbered by lecture order and hold only body content; filenames are
sanitized for `\input` (an underscore in a slug would otherwise be read as a
subscript). `--export-compile` compile-checks the result.

With no videos given it exports from the saved state and exits; passed
alongside a normal run, `--answer`, or `--verify`, it exports at the end.
Bodies are read fresh from each `section.tex`, so hand edits are picked up.

## Boards: recovering what was drawn

A transcript is linear audio, so everything the lecturer *drew* is lost. Every
lecture with a video is therefore segmented into board states before any notes
are written — on by default, since a lecture written from the transcript alone
gets notation wrong:

```sh
python build_course.py …                        # segmentation included
python build_course.py … --no-boards            # skip it
python build_course.py … --boards-color         # analyse in colour
python boards.py output/<slug>/video.mkv        # standalone
```

```
output/<slug>/boards/boards.json     every board, with the intervals it was current
output/<slug>/boards/board-07.jpg    a clean snapshot of that board at its fullest
```

No ML and no new dependencies — ffmpeg for pixels, numpy for the rest:

1. Sample at 1 fps, downscaled, greyscale by default (`--boards-color` if
   colour carries meaning).
2. **Remove the lecturer** with a sliding temporal median: they move, the
   writing does not, so the median over a window of seconds is the board
   alone. The saved snapshots are built the same way at full size, so the
   lecturer is not standing in front of the thing you wanted to read.
3. Reduce each frame to an *ink mask* — pixels of high local contrast, so it
   works for chalk on black or marker on white. The contrast threshold is
   computed **once for the whole video**: a per-frame threshold would mark
   the same fraction of every frame as ink, and a full board would score
   identically to a wiped one.
4. Track boards by ink **containment**, which is deliberately asymmetric.
   Adding writing leaves the old ink present, so it stays the same board;
   erasing it drops sharply, which ends the board. Frames are
   motion-compensated by phase correlation first, so a camera pan is
   recognised as motion rather than as an erasure.

That containment test is what handles the awkward cases: writing more on an
earlier board, and the camera panning away and coming back, both add an
interval to an existing board instead of inventing a new one. Returning to a
board after a different one is recorded as a genuine revisit, with the two
intervals kept separate.

Each board's snapshot is taken at its **ink peak** — the moment just before
it was erased, when it is most complete.

### The stills go to the model that writes

Once segmented, the boards are part of the prompt for every agent that writes
or checks a lecture — the writer, the reviser and the verifier. They arrive
two ways at once: an index listing each board with the intervals it was up,
and a marker spliced into the transcript at the moment each board goes up, so
the board that was current is visible right where the model is reading.

```
[00:23:12] === board 7 up: /…/boards/board-07.jpg ===
[00:23:14]  and so this map here is injective …
```

On the `api` backend the stills are attached to the first message directly,
inside the cached prefix. On the other backends they are listed by path and
the model opens them itself.

They go to the *main* model rather than to a cheap one on purpose. Reading
handwritten mathematics is under-determined by the pixels — ∂ against δ, an
`f` against an `f̃`, `↪` against `→` — and what resolves it is knowing the
subject, the lecturer's notation and the previous lectures. A summary written
by a small model is a lossy read of the densest artifact in the lecture, and
it saves about $4 across a 24-lecture course. The prompt therefore tells the
model to prefer the board over the transcript where the two disagree about a
symbol: the transcript is a guess at speech, and it mangles notation that was
never spoken aloud.

The `get_frame` tool remains available for when a snapshot is missing or
garbled, but its description now discourages it: a single raw frame may catch
a mid-erasure, a pan, or the lecturer's back.

### Diagrams: redrawn, never photographed

A photograph cannot go into the notes, and prose is a bad substitute for a
commutative diagram, so what the lecturer drew is redrawn as TikZ. `tikz-cd`
and tikz are in the default preamble, and the work splits like this:

1. the **board-locator**, on the cheap model, is asked *where* the diagram is
   and returns a box. It does not read mathematics and does not draw;
2. `crop_board` returns that region **at native resolution**. This is the
   whole point of the crop: sending a full still downscales it to the vision
   model's long-edge ceiling and a chalk arrowhead survives as a pixel or two,
   whereas the cropped region arrives un-shrunk. It never scales *up* —
   interpolation invents no chalk — which is also why stills are stored on
   disk at full resolution rather than at the ceiling;
3. the **main model** reads the diagram off the crop and writes the TikZ. An
   arrow direction is a mathematical claim, not a typesetting choice, and the
   main model is the one that knows what the lecture proves;
4. `check_diagram` compiles the snippet **alone** — so a broken diagram cannot
   take the course build down — and hands the render back to compare against
   the crop.

That division was measured, not assumed. With the cheap model doing the
drawing, it reversed the arrows on a lifting diagram three times running,
including once after twenty-four magnified looks at the board; the error was
caught only because the main model checked the reported directions against the
mathematics. Locating a region it can do reliably. Reading an arrowhead it
cannot.

If a diagram cannot be read off the board, the notes fall back to prose with a
`\todo` rather than carrying a confident, wrong diagram. Placeholder comments
left where a diagram should be (`% DIAGRAM_PLACEHOLDER`) compile silently and
are invisible in the PDF, so they are reported after every run.

## What the agents did: output/logs/

Every agent invocation is recorded, so a bad section can be traced back to
what the agent actually saw and did:

```
output/logs/index.jsonl                      one summary line per agent run
output/logs/<ts>-<role>-<lecture>.jsonl      that run's full event trace
```

The index carries role (`write`, `verify`, `revise`, `propagate`,
`fix-latex`, `fix-preamble`), lecture, backend and model, wall-clock seconds,
token usage and cost, a histogram of tool calls, and the outcome (chars
written, todos and open questions left, whether it fell back to saving its
chat output). The trace has one line per event — each tool call with its
input, a clipped result and how long it took, each thing the agent said, and
a final summary.

```sh
python build_course.py --logs      # digest: which agents ran, cost, tool counts
```

Both files are JSON Lines, so aggregating is a couple of lines of Python —
which is the point: "frame readers average 11 tool calls", "the verifier
spent 40s grepping before finding the label" is how the prompts above got
written. Logging never interferes with a run: writes are best-effort and a
failure disables logging with a note rather than raising.

## Long documents

A fetched paper can run to hundreds of thousands of characters — more than
fits in one read. Three things keep an agent from flailing at one:

- **The full text is cached**, and only the prompt copy is clipped. The clip
  says how much was shown, how much exists, where the file is, and how to
  read the rest (ranges, or `search_document`) — a bare `[truncated]` marker
  told the model something was missing but not what to do about it.
- **`outline.txt`** is written next to each cached reference: every
  `\section`/`\subsection` and theorem-like environment with its line number,
  plus the PDF bookmarks with page numbers. A 457k TeX source becomes an 8.7k
  outline (226 headings), so the agent jumps to a line instead of reading
  from the top. Existing caches get one on first reuse.
- **`search_document`** greps a cached file or a whole unpacked source tree
  and returns matching lines with line numbers and context — the fastest way
  to find a theorem or a symbol in a long paper, and available on every
  backend (the codex one has no native grep over these paths).

Agents are also told not to open `course.tex`: it is every lecture
concatenated (about a megabyte in a full course) and will be cut off long
before the part they wanted. The per-lecture files are what the lecture index
points at.

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

`--answer-all` walks the whole course in two phases. It lists every lecture
with open questions or todos and puts **all** of those questions to you in
one sitting; only then do the models start, so you are not stranded at the
terminal waiting for one lecture's revision to finish before being asked the
next question. The revision phase is unattended — questions the agents raise
along the way are queued for the next follow-up rather than waited on. It
then propagates once: each later lecture is visited a single time carrying
the changes from every revised lecture before it, rather than being rewritten
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
