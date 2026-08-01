# notetaker

Turn recorded mathematics lectures into typeset LaTeX notes.

A lecture — local file, URL, or YouTube link — is downloaded, transcribed with
Whisper, and handed to an agent (Claude or GPT, running on your existing chat
subscription) that writes mathematical notes. The agent reads the
board from video stills, redraws commutative diagrams as TikZ, fetches papers
the lecturer cites, and queues questions for you without stopping work.

A whole series becomes one `course.tex`, with cross-references between
lectures, a running bibliography, and a compile-repair loop that fixes its own
LaTeX errors.

- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Command-line options](#command-line-options)
- [Output layout](#output-layout)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [License](#license)

## Requirements

- **Python 3.10+** (`X | None` annotations and `match`), **ffmpeg** and
  **yt-dlp**.
- **An agent backend**: a Claude Pro/Max subscription, a ChatGPT subscription,
  or an Anthropic API key.
- **Optional**: a TeX installation (`latexmk` preferred, `pdflatex` accepted)
  for the compile check and the repair loop; a [Modal](https://modal.com)
  account for GPU transcription; an NVIDIA GPU for local `large-v3` Whisper.

The optional pieces degrade rather than fail — without TeX the compile check is
skipped, without Modal transcription runs locally.

## Install

<details open>
<summary><b>Manually</b></summary>

Install ffmpeg and yt-dlp from your package manager, then:

```sh
python -m venv .venv && . .venv/bin/activate
pip install numpy openai-whisper yt-dlp tqdm click requests \
            anthropic claude-agent-sdk mcp modal pymupdf pypdf
```

Then install the CLI for the backend you want: `claude`
([Claude Code](https://claude.com/claude-code)) for `subscription`, or `codex`
([OpenAI Codex](https://developers.openai.com/codex/cli)) for `codex`. The
`api` backend needs neither.

Not everything is needed for every run: `modal` only for `--transcribe modal`,
`mcp` only for the codex backend, `pymupdf`/`pypdf` only for fetching papers,
`openai-whisper` only for local transcription.

</details>

<details>
<summary><b>With Nix</b></summary>

```sh
nix develop                # python env, ffmpeg, yt-dlp, claude + codex CLIs
nix develop .#cuda         # same, with local GPU Whisper
```

</details>

Authenticate whichever backend you plan to use, once:

| Backend | Model | Authentication |
|---|---|---|
| `subscription` (default) | Claude Opus | run `claude`, log in with a Claude Pro/Max account |
| `codex` | GPT (`gpt-5.6-sol`) | `codex login`, with a ChatGPT account |
| `api` | Claude Opus | `export ANTHROPIC_API_KEY=…` (pay per token) |

For remote GPU transcription: `modal setup`.

## Quickstart

One lecture:

```sh
python ingest.py "https://www.youtube.com/watch?v=..."   # download + transcribe
python generate_notes.py output/<lecture-slug>           # write notes.tex
```

A whole series:

```sh
python build_course.py "https://www.youtube.com/playlist?list=..." \
    --title "Étale Cohomology" --transcribe modal
```

The result is `output/course.tex`, plus a `section.tex` per lecture. Run
`latexmk -pdf output/course.tex` to get the PDF.

## Usage

### Inputs

Individual videos, local files, playlist URLs (expanded in order), or
`--from-file lectures.txt` with one input per line. A `watch?v=…&list=…` URL is
a single video; pass the `playlist?list=…` page for the series.

State lives in `output/course_state.json`. Rerunning skips lectures that are
already done, so a series can be processed incrementally — including resuming
after a crash that left a lecture downloaded but untranscribed.

Add lectures at the **end** only. Reordering or inserting changes lecture
numbers; the run warns and names the lectures to `--regen`.

### Working around rate limits

```sh
--proxy socks5://127.0.0.1:1080   # route local yt-dlp through a proxy
--download modal                  # run yt-dlp on Modal workers instead
--available-only                  # process what is downloaded, then stop
```

`--available-only` stops at the first missing lecture rather than skipping
ahead, which would misnumber everything after it. Rerun later to continue.

You can also pre-fetch without transcribing (`python ingest.py URL
--no-transcribe`); ingest is resumable, so a later run continues from the
cached files.

**Manual downloads** slot in as ordinary inputs. Freeze the series order once
(`yt-dlp --flat-playlist --print url PLAYLIST > lectures.txt`), then replace a
URL line with a file path as you fetch videos yourself. Keep each lecture's
identity stable across runs, give files meaningful names (the stem becomes the
directory slug), and don't move them afterwards — frame extraction reads
local-file videos from their original location.

### Answering questions

The agent never blocks on you. When it is unsure it queues a question, marks
the spot with `\todo{awaiting answer #N @ hh:mm:ss}`, and keeps working.
Prompts appear in the terminal while it runs:

```
[Transcript unclear #1 @ 00:41:07] "the at all site"
[Question #2 @ 01:02:03 for you] Which package for the prism symbol?
```

Enter accepts the agent's guess on a clarify prompt (`?` defers); on an open
question, type an answer or press Enter to defer. Anything unanswered is saved
and the run finishes without waiting (`--wait` blocks instead). Later:

```sh
python generate_notes.py output/<slug> --answer   # single lecture
python build_course.py --answer <slug>            # one course lecture
python build_course.py --answer-all               # every lecture with anything open
```

This re-asks open questions, offers each remaining `\todo` for you to answer
(Enter leaves it for the model to sweep), has the agent revise in place,
propagates the changes to later lectures, and reassembles `course.tex`.
`--answer` is useful with no open questions at all, as a pure todo sweep.

`--answer-all` runs in two phases: it puts **all** questions from every lecture
to you in one sitting, then runs the models unattended. You are never stranded
at the terminal waiting for one revision to finish before being asked the next
question.

Hand edits to a `section.tex` are safe — assembly prefers the file on disk.

### Verifying accuracy

Every lecture is re-read by a second pass in a fresh context (see
[Accuracy](#accuracy-a-second-pass-per-lecture)). It runs automatically;
`--no-verify` skips it, and it can be run on its own:

```sh
python build_course.py --verify <slug>   # one lecture
python build_course.py --verify all      # the whole course
```

### Who is lecturing

Notes written by a professional say e.g. "Bourbaki defines", not "the speaker
claims" — surname alone. Nothing in a video file reliably carries a name, so
each new lecture's speaker is asked once, before any model runs, with a guess
offered as the default:

```sh
python build_course.py … --lecturer "Nicolas Bourbaki"  # one speaker, no questions
python build_course.py …                                # asked per lecture
python lecturer.py output                               # just show the guesses
```

Answers are saved and never re-asked. With no terminal attached the guesses are
taken silently, so unattended runs work. `--lecturer` also applies in
`--verify` and `--answer` mode, which is how a course written before a name was
recorded gets one.

### Exporting a multi-file project

`course.tex` is one large assembled file. For something editable and
version-controllable:

```sh
python build_course.py --export ~/notes/advanced-topology [--export-compile]
```

```
main.tex                    preamble, \input lines, bibliography
lectures/01-<slug>.tex      one file per lecture (body only)
lectures/02-<slug>.tex
references.bib
```

`main.tex` builds with `latexmk -pdf main.tex` and shares its preamble code
with the single-file build, so the two cannot drift. Filenames are sanitized
for `\input` (an underscore in a slug would be read as a subscript). With no
videos given it exports from saved state and exits; alongside a normal run,
`--answer` or `--verify`, it exports at the end.

### Inspecting what the agents did

```sh
python build_course.py --logs   # digest: which agents ran, cost, tool counts
```

Every agent invocation is recorded under `output/logs/`, so a bad section can
be traced to what the agent actually saw and did — one summary line per run in
`index.jsonl`, and a full event trace per run beside it. Both are JSON Lines,
so aggregating is a couple of lines of Python. Logging is best-effort and never
interferes with a run: a failure disables logging with a note rather than
raising.

## Command-line options

`build_course.py --help` is authoritative; this is the shape of it.

**Input and output**

| Flag | Effect |
|---|---|
| `--from-file FILE` | One input per line (`#` comments ignored) |
| `--output-dir DIR` | Per-lecture data (default `output/`) |
| `--output FILE` | Assembled document (default `output/course.tex`) |
| `--title TITLE` | Course title, saved in state on first run |
| `--export DIR` | Write a multi-file LaTeX project |
| `--export-compile` | Compile-check the export |

**Ingest and transcription**

| Flag | Effect |
|---|---|
| `--transcribe local\|modal` | Where Whisper runs (default `local`) |
| `--whisper-model SIZE` | `tiny`…`large-v3`, `turbo` (default `base` local, `large-v3` Modal) |
| `--language LANG` | Whisper language (default `en`; `auto` to detect) |
| `--download local\|modal` | Where yt-dlp runs |
| `--modal-fetch` | Let Modal workers fetch audio instead of receiving it |
| `--proxy URL` | Proxy for local yt-dlp traffic |
| `--available-only` | Stop at the first missing lecture |
| `--skip-ingest` | Treat every input as an already-ingested directory |
| `--regen SLUG` | Force one lecture to be rewritten |

**Agent**

| Flag | Effect |
|---|---|
| `--backend subscription\|codex\|api` | Which agent runs (default `subscription`) |
| `--model MODEL` | Override the backend's default |
| `--frame-model MODEL` | Cheaper model for reading frames |
| `--lecturer NAME` | Set the speaker for every lecture |
| `--reference URL_OR_ID` | Preload a paper as context (repeatable) |
| `--style-exemplar FILE` | Notes whose exposition style to imitate |

**Passes**

| Flag | Effect |
|---|---|
| `--answer SLUG` / `--answer-all` | Follow-up: answer questions and todos, revise |
| `--wait` | Block at the end of each lecture for queued questions |
| `--no-propagate` | Don't push revisions into later lectures |
| `--verify SLUG\|all` / `--no-verify` | Accuracy pass |
| `--no-boards` / `--boards-color` | Board segmentation off / in colour |
| `--latex-fix-rounds N` | Compile-repair rounds (default 2, `0` to only report) |
| `--logs` | Print the agent-run digest and exit |

## Output layout

```
output/
  course.tex                  assembled document
  course_state.json           per-lecture state: bodies, summaries, corrections
  references.bib              running bibliography (cited sources)
  references/                 cached fetched papers
  logs/                       one trace per agent run
  <lecture-slug>/
    video.*  audio.wav        media (video kept for frame extraction)
    transcript.json           Whisper output
    info.json                 metadata
    notes.tex                 standalone notes (generate_notes)
    references.bib            its own bibliography, alongside notes.tex
    section.tex               course section
    summary.md                digest for later lectures
    boards/                   board stills + boards.json
    *.questions.json          open/answered question state
```

## How it works

### The course is written lecture by lecture

Each lecture becomes a `\section` in one document, written with the previous
lectures as context: the last two in full, older ones as compact model-written
summaries, with the agent able to open any earlier lecture's file when it needs
an exact statement. Labels are prefixed per lecture (`\label{thm:3:...}`) so
they never collide.

Ingest is two-phase — all lectures are downloaded, then every untranscribed one
is transcribed, in parallel across Modal containers with `--transcribe modal`,
so a series takes roughly one lecture's wall-clock time.

Transcript mishearings you confirm once ("at all" → "étale") are passed to
every later lecture automatically.

Every agent that touches a section is given an **index of the other lectures**
and a listing of **what is already in the bibliography** — indexes rather than
content, since full summaries run to hundreds of kilobytes across a long
course, so the agent opens only what it needs.

A **running bibliography** accumulates in `output/references.bib`. When a
lecture cites something, the agent calls a tool that fetches a real BibTeX
entry (arXiv and DOI metadata automatically, anything else as a web entry) and
returns the key. The same source cited from several lectures reuses one entry.
biblatex is wired in only once something is actually cited, and the compile
check runs biber, so undefined citations are reported rather than silently
rendering as `[?]`.

`generate_notes.py` collects citations the same way, into a `references.bib`
next to its `notes.tex`. Since that document's preamble is written by the
model — which is told never to write bibliography machinery — the `biblatex`
lines and `\printbibliography` are attached mechanically before each compile
round, so a repair pass that rewrites the preamble cannot take them with it.

### One prompt, two entry points

The course assembler and the single-lecture writer share most of what their
system prompt says: how to treat an ASR transcript, what not to invent, when to
draw a diagram, which dash to set. That text lives in `instructions.py` and is
composed into both. What stays in each driver is what is genuinely different —
body-only output versus a whole document, lecture-numbered labels, the preamble
tool, the theorem environments that already exist. Two blocks are parameterised
rather than duplicated, since only one clause of each depends on which tools the
driver has.

This is not cosmetic. When the two prompts were maintained separately they
drifted: the course prompt banned `\ref` in favour of cleveref while the
single-lecture prompt still asked for it, and the diagram, fidelity and display
rules existed in only one of them.

### Boards: recovering what was drawn

A transcript is linear audio, so everything the lecturer *drew* is lost. Every
lecture with a video is segmented into board states before any notes are
written — on by default, because a lecture written from the transcript alone
gets notation wrong.

```sh
python boards.py output/<slug>/video.mkv    # standalone
```

No ML and no new dependencies — ffmpeg for pixels, numpy for the rest:

1. Sample at 1 fps, downscaled, greyscale by default.
2. **Remove the lecturer** with a sliding temporal median: they move, the
   writing does not. Snapshots are built the same way at full size, so the
   lecturer is not standing in front of what you wanted to read.
3. Reduce each frame to an *ink mask* — pixels of high local contrast, so it
   works for chalk on black or marker on white. The threshold is computed once
   for the whole video; a per-frame threshold would mark the same fraction of
   every frame as ink, and a full board would score identically to a wiped one.
4. Track boards by ink **containment**, which is deliberately asymmetric.
   Adding writing leaves the old ink present, so it stays the same board;
   erasing drops sharply, which ends it. Frames are motion-compensated by phase
   correlation first, so a camera pan reads as motion rather than erasure.

That asymmetry is what handles the awkward cases: writing more on an earlier
board, and the camera panning away and back, both extend an existing board
instead of inventing a new one. Returning to a board after a different one is
recorded as a genuine revisit. Each snapshot is taken at the board's **ink
peak** — the moment just before it was erased, when it is most complete.

The stills then go to *every* agent that writes or checks a lecture, two ways
at once: an index of each board with the intervals it was up, and a marker
spliced into the transcript where each board goes up, so the current board is
visible right where the model is reading.

```
[00:23:12] === board 7 up: /…/boards/board-07.jpg ===
[00:23:14]  and so this map here is injective …
```

They go to the *main* model on purpose. Reading handwritten mathematics is
under-determined by the pixels — ∂ against δ, `f` against `f̃`, `↪` against `→`
— and what resolves it is knowing the subject, the lecturer's notation and the
previous lectures. The prompt tells the model to prefer the board over the
transcript when the two disagree about a symbol, since the transcript is a
guess at speech and mangles notation that was never spoken aloud.

### Diagrams are redrawn, never photographed

A photograph cannot go into the notes, and prose is a bad substitute for a
commutative diagram, so what was drawn is redrawn as TikZ:

1. a **board-locator**, on the cheap model, is asked *where* the diagram is and
   returns a box — it does not read mathematics and does not draw;
2. `crop_board` returns that region **at native resolution**. Sending a full
   still downscales it to the vision model's long-edge ceiling and a chalk
   arrowhead survives as a pixel or two, whereas the crop arrives un-shrunk. It
   never scales *up* — interpolation invents no chalk;
3. the **main model** reads the diagram off the crop and writes the TikZ. An
   arrow direction is a mathematical claim, not a typesetting choice;
4. `check_diagram` compiles the snippet **alone**, so a broken diagram cannot
   take the course build down, and hands back the render to compare against the
   crop.

That division was measured, not assumed. With the cheap model drawing, it
reversed the arrows on a lifting diagram three times running — once after
twenty-four magnified looks at the board. Locating a region it can do reliably;
reading an arrowhead it cannot.

If a diagram cannot be read off the board, the notes fall back to prose with a
`\todo` rather than a confident, wrong diagram.

### Accuracy: a second pass per lecture

Notes like these are reliably right about the mathematics and unreliably right
in the margins. The errors cluster in sentences the model *added*: an invented
"equivalently", a hedge quietly promoted to an assertion, a claim the lecturer
retracted five minutes later kept as though it stood, a paper cited as the one
they meant that was published a year after the lecture.

The writing prompt names those failure modes directly, and each lecture is
given its recording date so a citation can be checked against it. Then a
**verification pass** re-reads the section against the full transcript in a
*fresh context* — no memory of having written it, and free to read the
transcript backwards and forwards, which is what makes a late correction
visible as superseding an early claim. It fixes what it is sure of with the
smallest edit that makes the text true, and weakens what it cannot settle
rather than merely flagging it.

### The compile check is a feedback loop

When the assembled document fails to compile, each error is traced back to the
lecture that wrote it — by line number, so `l.1858` becomes "line 12 of your
section" — and handed to that lecture's agent to fix: content untouched, LaTeX
corrected. The document is then reassembled and re-checked, up to
`--latex-fix-rounds N` times.

Errors in the shared preamble go to a pass of their own, since those come from
different lectures whose additions can collide.

Once the document compiles, the remaining rounds are spent on presentation:
overfull boxes (text printing into the margin) and section titles containing
maths that hyperref cannot put in a PDF bookmark, which need
`\texorpdfstring`. Correctness runs first, because while the document fails to
compile the line numbers on an overfull box describe a layout that will not
survive the fix.

Undefined citations are only reported once the document otherwise compiles: a
hard error aborts the run before biber, at which point *every* citation looks
undefined.

### What the assembler enforces mechanically

Some things are decidable, so they are applied rather than asked of a model:

- **Theorem numbering.** Every environment the agent declares is put on the
  shared `theorem` counter, so the document has one sequence and a `\cref` is
  unambiguous. Re-declaring an environment that already exists is dropped —
  it is a hard error, and two lectures independently declaring `claim` is an
  easy way to get there.
- **Hyperref anchors.** `\theH<env>` defaults to the bare counter with no
  section in it, so Theorem 1.1 and Theorem 2.1 both anchor at `theorem.1`,
  hyperref discards the duplicate, and every link to either lands on whichever
  came first. The section is put back into the anchor.
- **Equation numbering.** A display follows what cites it, in both directions:
  uncited displays become `equation*`, and one that a later lecture starts
  citing is numbered again on the next assembly. `\label` always stays — it is
  how a later lecture knows what to `\cref`. A display carrying an explicit
  `\tag` is left alone. What cannot be settled mechanically is reported: a
  cited label inside a starred multi-line display numbers per line via
  `\notag`, so which line the reference meant is a judgement call.

### Long documents

A fetched paper can run to hundreds of thousands of characters. Three things
keep an agent from flailing at one:

- **The full text is cached** and only the prompt copy is clipped. The clip
  says how much was shown, how much exists, where the file is, and how to read
  the rest — a bare `[truncated]` marker told the model something was missing
  but not what to do about it.
- **`outline.txt`** beside each cached reference lists every heading and
  theorem-like environment with its line number, plus PDF bookmarks with page
  numbers. A 457k TeX source becomes an 8.7k outline, so the agent jumps to a
  line instead of reading from the top.
- **`search_document`** greps a cached file or an unpacked source tree and
  returns matching lines with context — available on every backend.

Agents are told not to open `course.tex`: it is every lecture concatenated, and
will be cut off long before the part they wanted.

### Reading frames cheaply

Reading video frames is the token-expensive part, so it is delegated: the main
model hands timestamps and context to a cheaper reader — a Haiku subagent on
the Claude backends, a `gpt-5.6-luna` subagent role on codex — which reports
the board contents in LaTeX, including where things are cut off, occluded or
unreadable. The main model looks at frames itself only when a report seems off.
Override with `--frame-model`.

## Troubleshooting

**Transcription quality.** The local default is `base` — fast, and weak on
mathematical vocabulary. For real use prefer `--transcribe modal` (defaults to
`large-v3`) or a local `large-v3`/`turbo` if you have the GPU.

**Wrong language.** Language defaults to English rather than Whisper's own
detection, which judges from the first 30 seconds — room noise over
mathematical jargon — and applies a wrong guess to the whole hour. An English
lecture detected as Latin comes back with passages like *"Talila, ala noseps
pale Similar sketch by Borno zip"* where forcing `en` gives *"this is a Banach
space of null sequences"*. Use `--language de`, `fr`, … for other languages, or
`--language auto` to restore detection. Transcripts are cached, so changing the
language does not re-transcribe what you have; the run warns and names the
`transcript.json` files to delete.

**Subscription limits.** A full lecture is a long agentic session, and a large
series makes a visible dent in plan usage limits.

**No compile check.** It runs only if `pdflatex` is on your `PATH` (`latexmk`
is preferred when present, so biber runs and citations resolve). Without a TeX
installation the check and the repair loop are skipped silently.

**Repeated video titles** in a series are handled — directories get a source
hash suffix — but tidy playlist titles make tidier slugs.

**First Modal run** downloads the Whisper weights; they are cached in a Modal
volume afterwards.

## Tests

```sh
for t in tests/test_*.py; do python "$t" || echo "FAILED $t"; done
```

Plain assertion scripts, no runner and no framework. They need ffmpeg, numpy
and a TeX installation, and build their own fixtures in a temp directory — none
of them touch `output/`. No test calls a model: what is tested is the parsing,
gating and fallback around each model call, which is where the bugs have been.
See [`tests/README.md`](tests/README.md).

## License

MIT — see [LICENSE](LICENSE).
