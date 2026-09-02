#!/usr/bin/env python3
"""
build_course.py — Process a series of lecture videos into one LaTeX document.

Ingests each video (downloading + transcribing, skipping lectures that have
already been processed), then calls Claude lecture-by-lecture to write the
notes. Each new lecture is written with the full prior LaTeX in context so
Claude can cross-reference earlier material with \\ref{}/\\hyperref[]{}.

State is persisted in <output-dir>/course_state.json after every lecture, so
you can run with one lecture at a time and add more later — already-written
sections are reused as context without regenerating them.

Usage:
  python build_course.py VIDEO [VIDEO ...] [options]
  python build_course.py --from-file lectures.txt [options]

Each VIDEO can be a local file path, a direct URL, a YouTube link, or a
YouTube playlist URL (expanded into its videos, in playlist order). A
--from-file list is one input per line (blank lines and # comments ignored).

Options:
  --output-dir DIR      Root directory for per-lecture output (default: output/)
  --output FILE         Final .tex file path (default: output/course.tex)
  --title TITLE         Course title for the LaTeX document
  --whisper-model SIZE  Whisper model (default: base locally, large-v3 on Modal)
  --transcribe WHERE    Where to run Whisper: 'local' or 'modal' (default: local)
  --proxy URL           Proxy for yt-dlp, e.g. socks5://127.0.0.1:1080
  --download WHERE      Where yt-dlp runs: 'local' or 'modal' (Modal egress
                        circumvents rate limiting of your IP)
  --available-only      Process downloaded lectures up to the first missing
                        one (stops there; rerun later to continue)
  --language LANG       Whisper language code (default: en; 'auto' to detect)
  --skip-ingest         Skip ingest step (all lecture dirs must already exist)
  --regen SLUG          Force regeneration of one lecture (by its directory name)
  --backend BACKEND     'subscription' (default) = Claude via your Claude
                        subscription; 'codex' = GPT via your ChatGPT
                        subscription; 'api' = Anthropic API (ANTHROPIC_API_KEY)
  --model MODEL         Override the backend's default model
  --frame-model MODEL   Cheaper model that reads video frames for the main
                        model (default: haiku on the Claude backends)
  --answer SLUG         Answer questions left open by an earlier run of
                        lecture SLUG, and its \\todo markers, then revise that
                        section in place. Answering a \\todo is optional —
                        press Enter to leave it for the model to sweep
  --answer-all          Same, for every lecture with open questions or todos
  --latex-fix-rounds N  Rounds of model-driven repair when the assembled
                        document fails to compile (default: 2; 0 disables)
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Import helpers from sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from claude_backend import (BACKENDS, collect_followup_answers, count_todos,
                            mark_answers_applied, open_question_count,
                            questions_file_for, run_agent)
from ingest import (download_video, expand_playlist, extract_audio, is_url,
                    resolve_language, resolve_whisper_model, slug,
                    transcribe_batch, unique_lecture_dir)
from fetch import describe_assets, fetch_reference, load_cached_reference
from agent_log import summarize
from bibliography import (BIB_FILENAME, BIB_PREAMBLE, BIB_PRINT, has_entries,
                          inline_entries, list_entries, tidy_bibliography)
from style_extract import extract as extract_style
from boards import analyse
from lecturer import (ATTRIBUTION_INSTRUCTION, lecturer_note,
                      resolve as resolve_lecturers)
from equations import (ReviewItem, dangling_references, defined_labels,
                       normalize_equation_numbering, referenced_labels,
                       review_items)
from latex_check import (LatexError, check_latex, compile_document,
                         print_errors, print_warnings, tokens_of)
from media import find_video, format_timestamp, format_transcript
from timestamps import (TIMESTAMP_PREAMBLE, drop_reserved,
                        read_video_id, video_table)
from instructions import (ASK_USER_RULE, ASR_INSTRUCTION, CLARIFY_RULE,
                          CROSSREF_RULE, DISFLUENCY_RULE, DISPLAY_RULES,
                          FIDELITY_INSTRUCTION, FRAMES_RULE,
                          HOUSE_STYLE_INSTRUCTION, MACRO_BRACING_RULE,
                          READER_RULE, TIMESTAMP_RULE, TODO_RULE,
                          cite_rule, verify_prompt,
                          diagram_rules)
from notes_tools import (NotesToolContext, REGISTER_INSTRUCTION,
                         ask_user_input, style_exemplar_block)
from usage import Usage, format_usage

# ---------------------------------------------------------------------------
# LaTeX preamble (fixed; Claude writes only the body sections)
# ---------------------------------------------------------------------------

PREAMBLE_TEMPLATE = r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amsthm,amssymb}
%% thmtools must come before hyperref so cleveref learns the theorem type names
\usepackage{thmtools}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage[colorinlistoftodos,obeyDraft]{todonotes}
%(timestamps)s%% Diagrams: tikz-cd for commutative diagrams, tikz for everything drawn
\usepackage{tikz-cd}
\usetikzlibrary{arrows.meta,decorations.pathmorphing,positioning,calc,patterns}
%% Additions requested by Claude during note generation:
%(extra_preamble)s
%% hyperref before cleveref
\usepackage[
  bookmarksdepth=3,
  linktoc=all,
  bookmarksnumbered=true,
  pdfusetitle,
]{hyperref}
%% cleveref last — produces "Theorem 2.3", "Definition 1.4", etc. automatically
\usepackage[nameinlink,noabbrev,capitalise]{cleveref}
%% reference sections with \S
\crefformat{section}{#2\S#1#3}
\Crefformat{section}{#2\S#1#3}
\crefmultiformat{section}{\S\S#2#1#3}{ and~#2#1#3}{, #2#1#3}{, and~#2#1#3}
\crefrangeformat{section}{\S\S#3#1#4-#5#2#6}
\crefformat{subsection}{#2\S#1#3}
\Crefformat{subsection}{#2\S#1#3}
\crefmultiformat{subsection}{\S\S#2#1#3}{ and~#2#1#3}{, #2#1#3}{, and~#2#1#3}
\crefrangeformat{subsection}{\S\S#3#1#4-#5#2#6}
%(bibliography)s

%% Make equation and figure numbering per-section
\numberwithin{equation}{section}
\numberwithin{figure}{section}

%% Theorem environments via thmtools (\declaretheorem registers names with cleveref)
\declaretheorem[numberwithin=section,style=plain]{theorem}
\declaretheorem[sibling=theorem,style=plain]{lemma}
\declaretheorem[sibling=theorem,style=plain]{proposition}
\declaretheorem[sibling=theorem,style=plain]{corollary}
\declaretheorem[sibling=theorem,style=plain]{claim}
\declaretheorem[sibling=theorem,style=definition]{definition}
\declaretheorem[sibling=theorem,style=definition]{construction}
\declaretheorem[sibling=theorem,style=definition]{example}
\declaretheorem[sibling=theorem,style=definition]{exercise}
\declaretheorem[sibling=theorem,style=remark]{remark}
\declaretheorem[sibling=theorem,style=remark]{notation}
\declaretheorem[sibling=theorem,style=remark]{recollection}
\declaretheorem[sibling=theorem,style=remark]{goal}
\declaretheorem[sibling=theorem,style=remark]{question}
%% Theorem environments Claude asked for. These land after the built-in ones
%% because they routinely say sibling=theorem or numberlike=theorem, and
%% thmtools resolves that at declaration time — declared earlier they fail
%% with "No counter 'theorem' defined".
%(extra_theorems)s

%% \theH<env> is the anchor hyperref uses, and it defaults to the bare counter
%% with no section in it — so Theorem 1.1 and Theorem 2.1 both anchor at
%% "theorem.1", hyperref drops the duplicate, and every link to either lands on
%% whichever came first. Putting \theHsection in front makes each anchor
%% unique. Must come after every \declaretheorem above.
\renewcommand{\theHequation}{\theHsection.\arabic{equation}}
\renewcommand{\theHfigure}{\theHsection.\arabic{figure}}
%(theorem_anchors)s

\title{%(title)s}
\date{}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""

CLOSING = r"\end{document}"

BOARDS_SUBDIR = "boards"
DIAGRAMS_SUBDIR = "diagrams"

# ---------------------------------------------------------------------------
# System prompt for the note-writing step
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""You are an expert mathematical note-taker writing LaTeX sections for
a math lecture series. A fixed preamble and theorem environments have already
been set up; you output *only* the body content to be appended to the document.

""" + ASR_INSTRUCTION + "\n\n" + FIDELITY_INSTRUCTION + r"""

Rules:
- Begin each lecture with \section{<descriptive title>} and add
  \label{lec:N} immediately after it. Do not include the lecture number in the
  course name, as LaTeX will number it automatically. If the lecturer provides
  a descriptive per-lecture title, use it.
- Use the pre-defined theorem environments: theorem, lemma, proposition,
  corollary, definition, example, exercise, remark, notation.
- Label everything a later lecture might cite — you cannot see the later
  lectures, and they can only cross-reference what you labeled, so do NOT
  label only what you anticipate needing. Put a \label{} on every theorem,
  lemma, proposition, corollary, definition, example, exercise, remark, and
  notation environment you write, and number displayed equations with
  \begin{equation}\label{eq:...} whenever they state a result, definition,
  or identity worth naming (use \[ ... \] for incidental display math).
  Give labels meaningful names — \label{thm:3:tilting-equivalence}, not
  \label{thm:3:1}.
- Prefix every label you create with the lecture number so labels never
  collide across lectures: \label{eq:N:...}, \label{thm:N:...},
  \label{def:N:...}, and so on (the lecture heading itself keeps
  \label{lec:N}).
""" + TIMESTAMP_RULE + "\n" + READER_RULE + "\n" + CROSSREF_RULE + r"""
- For lecture section labels, write \cref{lec:2} to produce a clickable
  "Section 2" link, or just write "Lecture~2" as plain text if no label exists.
""" + FRAMES_RULE + "\n" + CLARIFY_RULE + r"""
- Use the add_to_preamble tool whenever you need anything in the LaTeX
  preamble that is not already there: \usepackage{...}, \newcommand{...},
  \DeclareMathOperator{...}, \declaretheorem{...}, or any other declaration.
  Call it before writing the body content that depends on it.
  Already in the preamble: geometry, amsmath, amsthm, amssymb, thmtools,
  microtype, parskip, enumitem, todonotes, marginnote, and the theorem
  environments
  theorem, lemma, proposition, corollary, definition, example, exercise,
  remark, notation.
  Note: hyperref and cleveref are loaded last and must stay last — additions
  go before them, so do not re-add either of those packages.
""" + MACRO_BRACING_RULE + "\n" + cite_rule(shared=True) + "\n" \
    + ASK_USER_RULE + "\n" \
    + TODO_RULE + r""" (todonotes is already loaded — do not
  add it via add_to_preamble.)
""" + diagram_rules(board_tools=True) + "\n" + DISPLAY_RULES + "\n" \
    + DISFLUENCY_RULE + r"""
- Write only valid LaTeX body content — no \documentclass, no \begin{document},
  no \end{document} — to the output file named in the task instructions. Do
  not put the LaTeX in your reply text."""

SYSTEM_PROMPT += (REGISTER_INSTRUCTION + HOUSE_STYLE_INSTRUCTION
                  + ATTRIBUTION_INSTRUCTION)

# ---------------------------------------------------------------------------
# Ingest: download/extract each lecture, then transcribe all pending at once
# (on Modal, the pending lectures are transcribed in parallel)
# ---------------------------------------------------------------------------

def prepare_lecture(source: str, output_root: Path,
                    proxy: str | None = None,
                    download_via_modal: bool = False) -> tuple[Path, dict | None]:
    """
    Download the lecture and extract its audio (or reuse existing files).
    Returns (lecture_dir, meta): meta is None when transcript.json already
    exists, otherwise the metadata for the pending transcription.
    """
    for d in output_root.iterdir() if output_root.exists() else []:
        info_path = d / "info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        if info.get("source") != source:
            continue
        if (d / "transcript.json").exists():
            print(f"  Reusing existing transcript: {d.name}")
            return d, None
        if (d / "audio.wav").exists():
            print(f"  Resuming (audio already extracted): {d.name}")
            return d, info

    print(f"  Ingesting: {source}")
    with tempfile.TemporaryDirectory(prefix="notetaker-") as tmp:
        tmp_dir = Path(tmp)

        video_path, meta = download_video(source, tmp_dir, proxy,
                                          via_modal=download_via_modal)

        out_dir = unique_lecture_dir(output_root, slug(meta["title"]), source)
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_path = tmp_dir / "audio.wav"
        extract_audio(video_path, audio_path)
        shutil.move(str(audio_path), out_dir / "audio.wav")

        if meta["source_type"] != "file":
            shutil.move(str(video_path), out_dir / video_path.name)

        with open(out_dir / "info.json", "w") as f:
            json.dump(meta, f, indent=2)

    return out_dir, meta


def prepare_boards(lecture_dirs: list[Path], color: bool = False) -> None:
    """Segment each lecture's video into board states (boards/boards.json).

    On by default (--no-boards to skip): everything the lecturer wrote and
    never said is in these stills, and a lecture written from the transcript
    alone gets notation wrong. Skips lectures already done, and any without a
    video — an audio-only source silently gets the old behaviour."""
    todo = [d for d in lecture_dirs
            if not (d / BOARDS_SUBDIR / "boards.json").exists()
            and find_video(d) is not None]
    if not todo:
        print("Boards: nothing to segment (all done, or no videos).")
        return
    print(f"\n=== Segmenting boards for {len(todo)} lecture(s) ===")
    for d in todo:
        try:
            analyse(find_video(d), d / BOARDS_SUBDIR, color=color,
                    progress=lambda m: print(f"  {m.strip()}"))
        except Exception as exc:
            print(f"  Warning: board segmentation failed for {d.name}: {exc}")


def transcribe_pending(pending: list[tuple[Path, dict, str]],
                       whisper_model: str, language: str | None,
                       backend: str, modal_fetch: bool = False) -> None:
    """Transcribe the pending lectures (in parallel on Modal) and write each
    transcript.json. pending entries are (lecture_dir, meta, source).
    By default Modal workers get the locally-extracted audio uploaded to
    them; modal_fetch lets them download it themselves instead (YouTube
    tends to block datacenter egress, so this is opt-in)."""
    where = ("in parallel on Modal" if backend == "modal" else "locally")
    print(f"\nTranscribing {len(pending)} lecture(s) {where}…")
    jobs = [
        (d / "audio.wav",
         src if (backend == "modal" and modal_fetch and is_url(src)) else None)
        for d, _meta, src in pending
    ]
    results = transcribe_batch(jobs, whisper_model, language, backend)
    for (d, meta, _src), (segments, detected_lang) in zip(pending, results):
        meta["detected_language"] = detected_lang
        meta["whisper_model"] = whisper_model
        meta["segment_count"] = len(segments)
        with open(d / "transcript.json", "w") as f:
            json.dump({"metadata": meta, "segments": segments}, f, indent=2)
        with open(d / "info.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  {d.name}: {len(segments)} segments ({detected_lang})")


def warn_language_mismatch(lecture_dirs: list[Path],
                           language: str | None) -> None:
    """Transcription is cached, so changing --language does not touch
    lectures that already have a transcript. Say so, loudly: a lecture
    decoded as the wrong language is garbled everywhere, and the fix is to
    delete its transcript.json and rerun."""
    if not language:
        return
    stale = []
    for d in lecture_dirs:
        path = d / "transcript.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                got = json.load(f).get("metadata", {}).get("detected_language")
        except (OSError, ValueError):
            continue
        if got and got != language:
            stale.append((d, got))
    if not stale:
        return
    print(f"\nWARNING: {len(stale)} cached transcript(s) were made in a "
          f"different language than --language {language}:")
    for d, got in stale:
        print(f"  {d.name}: transcribed as '{got}'")
    print("  Their text is likely garbled throughout. To redo them:")
    for d, _ in stale:
        print(f"    rm {d / 'transcript.json'}")
    print(f"  then rerun; add --regen <slug> to rewrite notes from the new "
          f"transcript.\n")


# ---------------------------------------------------------------------------
# Transcript corrections
# ---------------------------------------------------------------------------

VERIFY_PROMPT = verify_prompt(shared=True)


def _section_title(body: str) -> str:
    """The \\section{...} heading of a lecture body (braces may nest)."""
    m = re.search(r"\\section\*?\s*\{", body)
    if not m:
        return ""
    depth, start = 0, m.end()
    for j in range(m.end() - 1, len(body)):
        if body[j] == "{":
            depth += 1
        elif body[j] == "}":
            depth -= 1
            if depth == 0:
                return re.sub(r"\s+", " ", body[start:j]).strip()
    return ""


def load_boards(lecture_dir: Path) -> list[dict]:
    """The board records written by `--boards`, or [] if never segmented."""
    path = Path(lecture_dir) / BOARDS_SUBDIR / "boards.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for b in data.get("boards", []):
        if b.get("image") and (path.parent / b["image"]).exists():
            b = dict(b, path=(path.parent / b["image"]).resolve())
            out.append(b)
    return out


def _spans(board: dict) -> str:
    return ", ".join(f"{format_timestamp(a)}–{format_timestamp(b)}"
                     for a, b in board["intervals"])


def board_index(boards: list[dict], attached: bool = False) -> str:
    """The lecture's boards, with when each was up and where its still is.

    The transcript carries only what was said; every "this", "here" and
    "that map" in it points at something that was written and never spoken.
    These stills are the other half of the lecture, so they are listed up
    front rather than left for the model to discover.

    With attached=True the images are already in the message and the listing
    is just the key to them; otherwise the paths are the only way in and the
    model has to open them itself."""
    if not boards:
        return ""
    rows = []
    for b in boards:
        note = f" (returned to; {b['revisits'] + 1} visits)" if b["revisits"] else ""
        rows.append(f"  Board {b['id']:>2}: {_spans(b)}{note}\n"
                    f"    {b['path']}\n")
    how = (
        "The stills are attached above, in this order.\n\n" if attached else
        f"Open all {len(boards)} stills by path with your image-viewing tool "
        f"before you start writing, in order, and go back to the relevant one "
        f"as you write each part. Read every one, including boards that look "
        f"like recap: skipping any is a decision you cannot make before you "
        f"have seen it. This is not optional and not a fallback — a lecture "
        f"written from the transcript alone will be wrong about notation.\n\n"
        f"Read them YOURSELF. Do not hand batches of boards to subagents to "
        f"transcribe for you. Delegating the reading loses the lecture twice "
        f"over. Dispatching parallel readers and waiting on them is how a "
        f"turn ends in narration about the boards with the notes never "
        f"written; and what comes back is prose, which is not something you "
        f"can draw a commutative diagram from, so the diagrams go with it. "
        f"Yes, the images cost tokens. Spend them. (The one exception is the "
        f"small 'board-locator' call described below, which finds a region "
        f"and reads nothing.)\n\n"
    )
    return (
        "**Boards.** The blackboard was photographed at every distinct state. "
        "Each still is that board at its most complete, with the lecturer "
        "removed, so it shows everything ever written on it — including, for "
        "a board still being filled at a given moment, writing that comes "
        "later than the transcript line you are reading. The exact form of a "
        "definition, an index, a diagram or a piece of notation is usually on "
        "the board and not in the words. Where board and transcript disagree "
        "about a symbol, prefer the board: the transcript is a guess at "
        "speech, and it mangles notation that was never spoken aloud. A board "
        "listed with several visits was returned to later.\n\n"
        + how + "".join(rows) + "\n"
    )


_PLACEHOLDERS = (
    # A SHOUTED_TOKEN on a comment line: DIAGRAM_PLACEHOLDER, TODO, FIXME.
    # Case-sensitive on purpose — "% Additions requested by Claude" is an
    # ordinary comment, and ignoring case would swallow it.
    re.compile(r"^[ \t]*%+[ \t]*(?:[A-Z][A-Z0-9_]{3,}\b|TODO|FIXME|XXX)"
               r"[^\n]*$", re.MULTILINE),
    # Or the same intention spelled out in words.
    re.compile(r"^[ \t]*%+[ \t]*(?:insert|fill|add|paste|put|draw)\b[^\n]*"
               r"\b(?:here|later|below|to follow|goes)\b[^\n]*$",
               re.MULTILINE | re.IGNORECASE),
)


class SectionNotWritten(Exception):
    """The agent finished without writing a lecture. Never cache this: a
    cached stub is indistinguishable from a real lecture on the next run, and
    the course silently loses a chapter."""


_MIN_SECTION_CHARS = 4000
_NARRATION = re.compile(r"^\s*(?:I'll|I will|Let me|I'm going to|First,? I|"
                        r"I need to|I've|I have launched|Now I)", re.I)
_ENVIRONMENTS = re.compile(
    r"\\begin\{(?:theorem|lemma|proposition|corollary|definition|example"
    r"|exercise|remark|notation|proof|equation|align|tikzcd)\}")


def looks_like_section(text: str) -> tuple[bool, str]:
    """Is this a lecture section, or the agent talking to itself?

    A section can come back as a few hundred bytes ending "I'll wait for the
    subagents to complete before continuing": the agent dispatches board
    readers, treats the calls as asynchronous, and ends its turn. The pipeline
    sees a written file, caches it, and moves on to the next lecture. Nothing
    warns, and the lecture is simply gone from the course.

    So the written file has to be checked for being a lecture at all. These
    thresholds are deliberately crude — a real section runs to tens of
    kilobytes with dozens of environments, so anything tripping them is
    broken rather than merely terse, and the check costs nothing."""
    stripped = (text or "").strip()
    if not stripped:
        return False, "the file is empty"
    if _NARRATION.match(stripped):
        return False, ("it begins as narration (\"" + stripped[:40].strip()
                       + "…\"), not as LaTeX — the agent wrote its "
                         "commentary here instead of the notes")
    if "\\section{" not in stripped:
        return False, "there is no \\section{...} heading"
    if len(stripped) < _MIN_SECTION_CHARS:
        return False, (f"it is only {len(stripped)} characters; a written-up "
                       f"lecture runs to tens of thousands")
    envs = len(_ENVIRONMENTS.findall(stripped))
    if envs < 3:
        return False, (f"it contains {envs} theorem/equation environment(s); "
                       f"a lecture has dozens")
    return True, ""


def hand_written_references(section_file: Path) -> list[str]:
    """References the model wrote out itself instead of registering.

    Finding these is a scan, so code does it rather than asking the model to
    audit its own draft — an audit it has every reason to believe it has
    already passed. Fixing them needs to know what the source actually is,
    which is why the list is handed to the checker rather than applied here.
    """
    try:
        return inline_entries(Path(section_file).read_text())
    except OSError:
        return []


def report_hand_written_references(section_file: Path) -> list[str]:
    hits = hand_written_references(section_file)
    if hits:
        print(f"\n    Warning: {len(hits)} reference(s) written by hand in "
              f"{Path(section_file).name} — these never reached "
              f"{BIB_FILENAME}, so nothing \\cite{{}}s them:")
        for h in hits[:5]:
            print(f"      {h}")
    return hits


def hand_written_note(section_file: Path) -> str:
    """The same list, addressed to the checker."""
    hits = hand_written_references(section_file)
    if not hits:
        return ""
    return ("**References written by hand.** A scan of the file found these "
            "places where a source was written into the notes instead of "
            "registered with cite_reference. Fix each one as described in "
            "your instructions:\n\n"
            + "\n".join(f"  - {h}" for h in hits) + "\n\n")


def report_placeholders(section_file: Path) -> list[str]:
    """Comments left standing in for content the agent meant to come back to.

    An agent that delegates a diagram and then narrates "I'll insert it once
    it returns" ends its turn with a comment where the diagram should be —
    which compiles perfectly happily and is invisible in the PDF, so nothing
    downstream would ever notice."""
    try:
        text = Path(section_file).read_text()
    except OSError:
        return []
    hits, seen = [], set()
    for pattern in _PLACEHOLDERS:
        for m in pattern.finditer(text):
            line = m.group(0).strip()
            if line not in seen:       # "% TODO: fill in … here" matches both
                seen.add(line)
                hits.append(line)
    if hits:
        print(f"\n    Warning: {len(hits)} placeholder comment(s) left in "
              f"{Path(section_file).name} — content the agent meant to fill "
              f"in and did not:")
        for h in hits[:5]:
            print(f"      {h[:100]}")
    return hits


_TODO_OPEN = re.compile(r"\\todo(?:\[[^\]]*\])?\s*\{")


def _brace_group(text: str, start: int) -> tuple[str, int]:
    """Contents of the brace group opening at `start`, and where it ends.

    Brace-counting rather than a regex because a \\todo body routinely
    contains braces of its own ({\\mathbb Z}, \\text{...}), and a
    non-greedy \\todo\\{(.*?)\\} would stop at the first inner closer.
    """
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "{" and (i == start or text[i - 1] != "\\"):
            depth += 1
        elif c == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)


def todo_items(text: str) -> list[str]:
    """The text inside each \\todo{...}, in document order."""
    out, i = [], 0
    while True:
        m = _TODO_OPEN.search(text, i)
        if not m:
            return out
        body, end = _brace_group(text, m.end() - 1)
        out.append(" ".join(body.split()))
        i = end


def ask_todo_answers(todos: list[str], lecture_num: int) -> str | None:
    """Put each \\todo to the user; return a block of the ones they answered.

    A \\todo is the agent saying it could not resolve something — which is
    the same kind of gap as an open question, and the user is the one who
    can close it. Before this they were only ever swept by the model, so
    --answer-all offered no way to answer them however many there were.

    Answering is optional: an empty reply leaves the marker for the model to
    sweep as before. Non-interactive runs collect nothing (ask_user_input
    returns "" with no terminal available) and behave exactly as they did.
    """
    if not todos:
        return None
    print(f"\n  {len(todos)} \\todo marker(s) in Lecture {lecture_num}. "
          f"Answer any you can; press Enter to leave one to the model.",
          flush=True)
    answered = []
    for i, todo in enumerate(todos, 1):
        shown = todo if len(todo) <= 300 else todo[:300] + "…"
        print(f"\n  [\\todo {i}/{len(todos)}] {shown}", flush=True)
        reply = ask_user_input("  Your answer (Enter to skip): ")
        if reply is None:          # prompt cancelled — stop asking
            break
        if reply.strip():
            answered.append((todo, reply.strip()))
    if not answered:
        return None
    return "\n\n".join(f"\\todo{{{todo}}}\n  → {ans}" for todo, ans in answered)


def report_unread_boards(ctx, boards: list[dict],
                         role: str = "write") -> list[int]:
    """Which stills the agent was given and never opened.

    Only meaningful where the images are read by path (the subscription and
    codex backends); on the api backend they are in the prompt and there is
    nothing to open. Silent when nothing was skipped.

    The wording has to follow the role, or it says something false. A writer
    that skipped a board really did reconstruct it from audio, which is worth
    a warning. A *checker* is told it may consult the boards, not that it must
    read all of them, and it opens the handful bearing on the claims it
    doubts — printing the writer's warning there implies the stills went
    unread when the writing pass had already read every one."""
    seen = getattr(getattr(ctx, "log", None), "boards_seen", None)
    if not boards or not seen:
        return []
    missed = [b["id"] for b in boards if b["id"] not in seen]
    if not missed:
        return []
    if role == "write":
        print(f"\n    Warning: {len(missed)} of {len(boards)} board stills "
              f"were never opened: {', '.join(map(str, missed))}. Anything "
              f"written only on those boards was reconstructed from audio.")
    else:
        print(f"\n    ({len(seen)} of {len(boards)} board stills opened while "
              f"checking: {', '.join(map(str, sorted(seen)))} — the rest were "
              f"checked against the transcript alone.)")
    return missed


def board_marks(boards: list[dict]) -> list[tuple[float, str]]:
    """Transcript interleaves: which board is up, at the moment it goes up."""
    marks = []
    for b in boards:
        for n, (start, _end) in enumerate(b["intervals"]):
            again = " again" if n else ""
            marks.append((start, f"[{format_timestamp(start)}] "
                                 f"=== board {b['id']} up{again}: {b['path']} ==="))
    return marks


def lecture_index(output_root: Path, state: dict,
                  exclude_slug: str | None = None) -> str:
    """An index of the course's other lectures, for agents that revise or
    check a single section.

    They can already read these files, but were never told what is there —
    so a question or a fix that turns on what an earlier lecture actually
    said had nothing to work from. Full summaries would be hundreds of
    kilobytes across a long course, so this lists titles and paths and lets
    the agent open what it needs (the summary for a digest, the section for
    the exact statement or label)."""
    rows = []
    for slug in ordered_slugs(state):
        if slug == exclude_slug:
            continue
        sec = state["sections"][slug]
        num = sec["lecture_num"]
        title = _section_title(sec.get("body", "")) or slug
        # The heading already reads "Lecture N: ..." — don't say it twice.
        title = re.sub(r"^Lecture\s+\d+\s*[::.-]\s*", "", title)
        d = (output_root / slug).resolve()
        rows.append(f"  Lecture {num}: {title}\n"
                    f"    notes:   {d / 'section.tex'}\n"
                    + (f"    summary: {d / 'summary.md'}\n"
                       if (d / "summary.md").exists() else ""))
    if not rows:
        return ""
    return (
        "Other lectures in this course. Open these when what you are fixing "
        "depends on an earlier lecture — an exact statement, a label you need "
        "to \\cref, or notation established there. Read the summary for a "
        "digest, the notes file for the precise wording. Do NOT open "
        "course.tex: it is every lecture concatenated (about a megabyte) and "
        "will be cut off before you reach what you wanted — use the "
        "per-lecture files below, or search_document to find a label across "
        "them:\n\n"
        + "".join(rows) + "\n")


def bibliography_index(bib_file: Path) -> str:
    """What is already cited, so an agent reuses a key instead of re-deriving
    the identifier (and re-searching the web) for a source already in."""
    entries = list_entries(Path(bib_file))
    if not entries:
        return ""
    rows = []
    for e in entries:
        who = e["author"].split(" and ")[0] if e["author"] else ""
        if who and " and " in e["author"]:
            who += " et al."
        title = e["title"][:70] + ("…" if len(e["title"]) > 70 else "")
        bits = " — ".join(x for x in (who, title) if x)
        year = f" ({e['year']})" if e["year"] else ""
        rows.append(f"  \\cite{{{e['key']}}}{' — ' if bits else ''}{bits}{year}")
    return (
        f"Already in the course bibliography ({len(entries)} entries). Cite "
        f"any of these directly with the key shown — do NOT call "
        f"cite_reference for them, and do not look them up again:\n"
        + "\n".join(rows) + "\n\n")


def lecture_provenance(meta: dict) -> str:
    """What the model needs to know about where this transcript came from.

    The date is load-bearing: without it the model cannot tell that a paper
    it found is too recent to be the one the lecturer gestured at. The
    language note tells it how far to trust the words in front of it."""
    parts = []
    raw = str(meta.get("upload_date") or "")
    if len(raw) == 8 and raw.isdigit():
        parts.append(
            f"This lecture was recorded/published on "
            f"{raw[:4]}-{raw[4:6]}-{raw[6:]}. Nothing published after that "
            f"date can be what the lecturer was referring to — check any "
            f"reference you attribute to them against it.")
    lang = meta.get("detected_language")
    if lang and lang != "en":
        parts.append(
            f"WARNING: speech recognition ran with language '{lang}', not "
            f"English. The transcript is therefore garbled well beyond the "
            f"usual level — expect mangled technical terms and whole "
            f"sentences of nonsense. Be correspondingly more willing to "
            f"flag passages with \\todo or clarify_transcript instead of "
            f"reconstructing them.")
    return ("\n".join(parts) + "\n\n") if parts else ""


def corrections_block(corrections: dict[str, str]) -> str:
    """Render recorded mishearings for the model to apply with judgment.
    (Deliberately not a mechanical str.replace: a short mishearing like
    "at all" → "étale" must only be fixed where it really is the mishearing.)"""
    entries = "\n".join(f'- "{wrong}" → "{right}"'
                        for wrong, right in corrections.items()
                        if wrong and right and wrong != right)
    if not entries:
        return ""
    return (
        "**Known recurring mishearings** — the speech recogniser has "
        "consistently misheard these in earlier lectures. Apply each "
        "correction wherever the transcript below actually contains the "
        "mishearing — use judgment, since the literal phrase may also occur "
        "legitimately — without asking again:\n" + entries + "\n\n"
    )


# ---------------------------------------------------------------------------
# Generate one lecture section, with prior LaTeX in context
# ---------------------------------------------------------------------------

# How many immediately-preceding lectures are included in full; everything
# older is included as its model-written summary (with a file path the agent
# can read on demand). Keeps context from growing linearly with the series.
FULL_CONTEXT_LECTURES = 2


def build_prior_context(state: dict, lecture_dirs: list[Path],
                        upto: int) -> str:
    """Context for writing lecture number `upto` (1-based position)."""
    prior = [d for d in lecture_dirs[:upto - 1] if d.name in state["sections"]]
    parts = []
    for idx, d in enumerate(prior):
        sec = state["sections"][d.name]
        path = (d / "section.tex").resolve()
        recent = idx >= len(prior) - FULL_CONTEXT_LECTURES
        if recent or not sec.get("summary"):
            parts.append(f"--- Lecture {sec['lecture_num']} — full text "
                         f"(file: {path}) ---\n{sec['body']}")
        else:
            parts.append(f"--- Lecture {sec['lecture_num']} — summary "
                         f"(full text at {path}) ---\n{sec['summary']}")
    return "\n\n".join(parts)

def generate_section(
    lecture_num: int,
    lecture_dir: Path,
    prior_latex: str,
    corrections: dict[str, str],
    references: list[dict],
    refs_dir: Path,
    existing_preamble_additions: list[str],
    style_exemplars: list | None = None,
    lecturer: str | None = None,
    backend: str = "subscription",
    model: str | None = None,
    frame_model: str | None = None,
    wait: bool = False,
) -> tuple[str, dict[str, str], list[str], Usage]:
    """
    Call Claude to write the LaTeX section for this lecture.
    references is a list of already-loaded {url, title, text} dicts.
    refs_dir is where new fetches (via the fetch_document tool) are cached.
    Returns (section_text, new_corrections, new_preamble_additions, usage).
    """
    with open(lecture_dir / "transcript.json") as f:
        data = json.load(f)
    segments = data["segments"]
    meta = data.get("metadata", {})
    title = meta.get("title", lecture_dir.name)
    total_duration = segments[-1]["end"] if segments else 0
    boards = load_boards(lecture_dir)
    transcript_text = format_transcript(segments, board_marks(boards))

    video_path = find_video(lecture_dir)
    ctx = NotesToolContext(
        refs_dir=refs_dir,
        video_path=video_path,
        total_duration=total_duration,
        transcript_path=lecture_dir / "transcript.json",
        enable_preamble=True,
        existing_preamble=list(existing_preamble_additions),
        read_roots=[refs_dir.parent.resolve()],
        bib_file=refs_dir.parent / BIB_FILENAME,
        boards=boards,
        diagrams_dir=lecture_dir / DIAGRAMS_SUBDIR,
    )

    # Build the user message
    context_block = ""
    if prior_latex:
        context_block = (
            "Earlier lectures are provided below — the most recent in full, "
            "older ones as summaries. You may reference definitions, "
            "theorems, equations, and labels from all of them. When you need "
            "the exact contents of a summarized lecture (a precise statement "
            "or label), read its file at the listed path with your "
            "file-reading tool.\n\n"
            "<prior_lectures>\n" + prior_latex + "\n</prior_lectures>\n\n"
        )

    corrections_note = corrections_block(corrections) if corrections else ""

    refs_block = ""
    if references:
        parts = []
        for ref in references:
            parts.append(
                f"--- Reference: {ref['title']} ---\n"
                f"URL: {ref['url']}\n"
                f"{describe_assets(ref, refs_dir.parent)}\n"
                f"{ref['text']}"
            )
        refs_block = "\n\n".join(parts) + "\n\n"

    user_text = (
        f"{context_block}"
        f"{refs_block}"
        f"{style_exemplar_block(style_exemplars)}"
        f"{bibliography_index(ctx.bib_file)}"
        f"Now write **Lecture {lecture_num}** (source title: \"{title}\").\n\n"
        f"{lecturer_note(lecturer)}"
        f"{lecture_provenance(meta)}"
        f"{corrections_note}"
        f"{board_index(boards, attached=backend == 'api')}"
        f"**Transcript:**\n\n{transcript_text}"
    )

    def write(extra: str = "", role: str = "write") -> str:
        return run_agent(
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text + extra,
            ctx=ctx,
            output_file=lecture_dir / "section.tex",
            backend=backend,
            model=model,
            frame_model=frame_model,
            wait_for_answers=wait,
            summary_file=lecture_dir / "summary.md",
            images=[(b["path"], f"Board {b['id']} ({_spans(b)})")
                    for b in boards],
            role=role, log_dir=refs_dir.parent / LOG_SUBDIR,
        )

    section_text = write()
    ok, why = looks_like_section(section_text)
    if not ok:
        # One retry, naming the mistake. The observed failure was the agent
        # ending its turn while "waiting" for subagents it had dispatched, so
        # say plainly that there is nothing to wait for.
        print(f"\n    Warning: the written section is not a lecture — {why}. "
              f"Retrying once.")
        section_text = write(
            "\n\n---\n**Your previous attempt did not write the notes.** The "
            f"file held only your own commentary: {why}. Note that every tool "
            "and subagent call returns its result to you inside this same "
            "turn — there is nothing to wait for and no later turn in which "
            "to continue, so never end your reply pending a result. Dispatch "
            "what you need, use the results as they come back, and write the "
            "complete LaTeX body to the output file before you finish.",
            role="write-retry")
        ok, why = looks_like_section(section_text)
        if not ok:
            print(f"    Warning: the retry also failed — {why}. This lecture "
                  f"will NOT be cached; rerun to try again.")
            raise SectionNotWritten(f"{lecture_dir.name}: {why}")

    if ctx.frame_requests:
        print(f"\n    ({ctx.frame_requests} frame(s) fetched)", end="")
    report_unread_boards(ctx, boards)
    report_placeholders(lecture_dir / "section.tex")
    report_hand_written_references(lecture_dir / "section.tex")
    # Questions are asked *during* the write pass, and the agent has already
    # used the answers in the text it just wrote — so they are applied, and
    # must be recorded as such. Without this every answer given while writing
    # stays applied=False for ever, and each later --answer/--answer-all
    # announces it as "an answer from an earlier run that was never applied"
    # and re-delivers it to the model, on every run, indefinitely.
    mark_answers_applied(ctx, lecture_dir / "section.tex")
    return section_text, ctx.new_corrections, ctx.new_preamble_additions, ctx.usage


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

STATE_FILE = "course_state.json"

def load_state(output_root: Path) -> dict:
    """
    State schema:
      {
        "title": str,
        "sections": {
          "<lecture-dir-name>": {
            "lecture_num": int,
            "body": str          # the LaTeX body written for this lecture
          }
        },
        "lecturers": {"<lecture-dir-name>": str}   # who spoke, as answered
      }
    Sections are ordered by lecture_num, but stored as a dict keyed by the
    lecture directory name so we can look up whether a lecture is already done.
    """
    path = output_root / STATE_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"title": None, "sections": {}, "corrections": {}, "references": [],
            "preamble_additions": [], "lecturers": {}}


def save_state(output_root: Path, state: dict) -> None:
    path = output_root / STATE_FILE
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def current_body(output_root: Path, state: dict, slug: str) -> str:
    """Body used for assembly. Prefers the on-disk section.tex — so hand
    edits survive — and syncs it back into state."""
    f = output_root / slug / "section.tex"
    if f.exists():
        body = f.read_text().strip()
        if body:
            state["sections"][slug]["body"] = body
            return body
    return state["sections"][slug]["body"]


def lecture_videos(output_root: Path, state: dict) -> dict[int, str]:
    """Each lecture's YouTube id, keyed by lecture number.

    This is how a margin timestamp becomes a link into the video at the
    moment it marks, without the note-taker knowing anything about it: it
    writes \\ts{hh:mm:ss} and nothing else, and the preamble assembled here
    says which video each lecture's marks belong to. A lecture that came from
    a local file contributes nothing, and its marks stay unlinked rather than
    pointing into some other lecture's video.
    """
    out: dict[int, str] = {}
    for slug, section in (state.get("sections") or {}).items():
        video = read_video_id(Path(output_root) / slug)
        if video and section.get("lecture_num") is not None:
            out[section["lecture_num"]] = video
    return out


def course_preamble(title: str, state: dict, with_bib: bool,
                    videos: dict[int, str] | None = None) -> tuple[str, str]:
    """(preamble, closing bibliography block) — shared by the single-file
    build and the multi-file export so the two cannot drift apart."""
    bib_preamble = BIB_PREAMBLE % BIB_FILENAME if with_bib else ""
    early, late = split_preamble(state.get("preamble_additions", []))
    # Both of these are enforced here rather than asked for in the prompt:
    # they are mechanically decidable, so a model that forgets one should not
    # be able to produce a document that is wrong.
    early, late = drop_reserved(early), drop_reserved(late)
    late = drop_duplicate_theorems(normalize_theorem_decls(late))
    preamble = PREAMBLE_TEMPLATE % {
        "title": title,
        "timestamps": TIMESTAMP_PREAMBLE + video_table(videos),
        "extra_preamble": "\n".join(early),
        "extra_theorems": "\n".join(late),
        "theorem_anchors": theorem_anchor_block(late),
        "bibliography": bib_preamble,
    }
    return preamble, ("\n\n" + BIB_PRINT if with_bib else "")


_LATE_PREAMBLE = re.compile(r"^\s*\\(?:declaretheorem|theoremstyle"
                            r"|newtheorem|Crefname|crefname)\b")

# \declaretheorem[opts]{name} and \newtheorem{name}[shared]{Title}[within]
_DECLARETHEOREM = re.compile(r"(\\declaretheorem\s*)(?:\[([^\]]*)\])?\s*\{(\w+\*?)\}")
_NEWTHEOREM = re.compile(r"(\\newtheorem\s*)\{(\w+\*?)\}\s*(?:\[(\w+)\])?\s*"
                         r"\{([^{}]*)\}\s*(?:\[(\w+)\])?")

# thmtools keys that decide which counter an environment uses. Any of them
# means "number this independently or off something else"; all are replaced by
# sibling=theorem so the whole document shares one sequence.
_NUMBERING_KEYS = ("sibling", "numberlike", "numberwithin", "parent",
                   "within")

BASE_THEOREM = "theorem"


def _strip_numbering(opts: str) -> list[str]:
    """The thmtools options with every counter-choosing key removed."""
    kept = []
    for opt in opts.split(","):
        opt = opt.strip()
        if not opt:
            continue
        key = opt.split("=", 1)[0].strip()
        if key not in _NUMBERING_KEYS:
            kept.append(opt)
    return kept


def normalize_theorem_decls(lines: list[str]) -> list[str]:
    """Put every model-declared theorem environment on the shared counter.

    The built-in environments all say sibling=theorem, so Theorem 1.1 is
    followed by Lemma 1.2 and Definition 1.3 — one sequence a reader can
    scan. An environment the model declares without it starts its own
    sequence, so the document then has two Claim 1.1s and no way to tell
    which "1.1" a \\cref means. An unnumbered environment has no counter at
    all and cannot be a sibling, so it is left alone.
    """
    out = []
    for line in lines:
        def fix_declare(m):
            name, opts = m.group(3), m.group(2) or ""
            if name == BASE_THEOREM or "unnumbered" in opts:
                return m.group(0)
            kept = _strip_numbering(opts) + [f"sibling={BASE_THEOREM}"]
            return f"{m.group(1)}[{','.join(kept)}]{{{name}}}"

        def fix_newtheorem(m):
            head, name, shared, title, within = m.groups()
            if name == BASE_THEOREM:
                return m.group(0)
            # [shared] and the trailing [within] are mutually exclusive in
            # LaTeX; forcing the shared form drops the trailing one.
            return f"{head}{{{name}}}[{BASE_THEOREM}]{{{title}}}"

        line = _DECLARETHEOREM.sub(fix_declare, line)
        line = _NEWTHEOREM.sub(fix_newtheorem, line)
        out.append(line)
    return out


def declared_names(source: str) -> list[str]:
    """Theorem environment names declared in a chunk of preamble, either way
    round (\\declaretheorem or \\newtheorem)."""
    names = []
    for pattern, group in ((_DECLARETHEOREM, 3), (_NEWTHEOREM, 2)):
        for m in pattern.finditer(source):
            name = m.group(group).rstrip("*")
            if name not in names:
                names.append(name)
    return names


def drop_duplicate_theorems(late: list[str]) -> list[str]:
    """Remove declarations of an environment that already exists.

    Re-declaring one is a hard error ("Command \\c@claim already defined")
    that takes the whole document down, and there are two easy ways to get
    there: the model declares the same environment while writing two
    different lectures, or an environment it used to have to declare is later
    promoted into the fixed template. Both are decidable here, so neither
    should be able to reach a compile.
    """
    seen = set(declared_names(PREAMBLE_TEMPLATE))
    kept = []
    for line in late:
        names = declared_names(line)
        if names and all(n in seen for n in names):
            continue
        seen.update(names)
        kept.append(line)
    return kept


def theorem_env_names(late: list[str]) -> list[str]:
    """Every environment declared with \\declaretheorem: the built-in ones
    plus whatever the model added, in declaration order, without repeats.

    \\newtheorem environments are deliberately absent — hyperref hooks
    \\newtheorem itself and gives those a working \\theH already.
    """
    names = []
    for source in (PREAMBLE_TEMPLATE, "\n".join(late)):
        for m in _DECLARETHEOREM.finditer(source):
            name = m.group(3).rstrip("*")
            if name not in names:
                names.append(name)
    return names


def theorem_anchor_block(late: list[str]) -> str:
    """\\theH redefinitions that put the section into every theorem anchor.

    \\theHtheorem defaults to the bare counter, with no section in it, so
    Theorem 1.1 and Theorem 2.1 both anchor at "theorem.1" — hyperref drops
    the duplicate and every link to either one lands on whichever came first.
    Adding \\theHsection makes the anchor unique; pointing the siblings at
    \\theHtheorem keeps them consistent with it, since they share the counter.
    """
    lines = [r"\renewcommand{\theHtheorem}{\theHsection.\arabic{theorem}}"]
    lines += [f"\\renewcommand{{\\theH{name}}}{{\\theH{BASE_THEOREM}}}"
              for name in theorem_env_names(late) if name != BASE_THEOREM]
    return "\n".join(lines)


def split_preamble(additions: list) -> tuple[list, list]:
    """(before hyperref, after the theorem block).

    Two conflicting constraints. A \\usepackage has to load before hyperref,
    which must be second-to-last with cleveref last. But a \\declaretheorem
    that says sibling=theorem needs the built-in theorem environments to
    exist already, and those are declared after cleveref. So the additions
    are split rather than dropped in one place: packages and macros early,
    theorem declarations and cross-reference names late."""
    early, late = [], []
    for entry in additions:
        for line in str(entry).splitlines():
            (late if _LATE_PREAMBLE.match(line) else early).append(line)
    return early, late


def render_document(title: str, body_parts: list[str], state: dict,
                    output_root: Path, output_tex: Path
                    ) -> tuple[str, list[tuple[int, int]]]:
    """Assemble the full LaTeX document, wiring in the running bibliography
    when anything has been cited.

    Returns (document, spans), where spans[i] is the 1-based (first, last)
    line range that body_parts[i] occupies in the document — compile errors
    carry a line number, and that is how they get attributed back to the
    lecture that wrote the offending source.
    """
    bib_src = output_root / BIB_FILENAME
    with_bib = has_entries(bib_src)
    if with_bib:
        # Entries written before this ran, and entries a hand edit reinstated.
        tidy_bibliography(bib_src)
        # biblatex resolves \addbibresource relative to the .tex file.
        if bib_src.resolve() != (output_tex.parent / BIB_FILENAME).resolve():
            shutil.copy2(bib_src, output_tex.parent / BIB_FILENAME)
    preamble, bib_print = course_preamble(
        title, state, with_bib, lecture_videos(output_root, state))

    doc = preamble + "\n\n"
    spans: list[tuple[int, int]] = []
    for i, body in enumerate(body_parts):
        start = doc.count("\n") + 1
        doc += body
        spans.append((start, doc.count("\n") + 1))
        if i < len(body_parts) - 1:
            doc += "\n\n"
    doc += bib_print + "\n\n" + CLOSING + "\n"
    return doc, spans


LECTURE_SUBDIR = "lectures"
MAIN_FILENAME = "main.tex"
LOG_SUBDIR = "logs"


def _tex_stem(num: int, slug: str) -> str:
    """A filename safe to hand to \\input: ASCII, no spaces, no underscores
    (TeX would read one as a subscript in the argument)."""
    safe = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-") or "lecture"
    return f"{num:02d}-{safe}"


def export_project(output_root: Path, state: dict, dest: Path,
                   slugs: list[str] | None = None,
                   title: str | None = None) -> Path:
    """Write the course as an editable multi-file LaTeX project:

        dest/main.tex                 preamble, \\input lines, bibliography
        dest/lectures/NN-<slug>.tex   one file per lecture, body only
        dest/references.bib           the running bibliography (if non-empty)

    Separate from the single assembled course.tex: this is the form you hand
    to a co-author or put under version control, where a per-lecture diff is
    worth more than one 200k-character file."""
    if slugs is None:
        slugs = ordered_slugs(state)
    dest = Path(dest)
    (dest / LECTURE_SUBDIR).mkdir(parents=True, exist_ok=True)
    title = title or state.get("title") or "Lecture Notes"

    bib_src = output_root / BIB_FILENAME
    with_bib = has_entries(bib_src)
    if with_bib:
        # Entries written before this ran, and entries a hand edit reinstated.
        tidy_bibliography(bib_src)
        shutil.copy2(bib_src, dest / BIB_FILENAME)

    inputs, seen = [], set()
    for slug in slugs:
        num = state["sections"][slug]["lecture_num"]
        stem = _tex_stem(num, slug)
        while stem in seen:            # distinct slugs can sanitize alike
            stem += "-b"
        seen.add(stem)
        body = current_body(output_root, state, slug).strip()
        (dest / LECTURE_SUBDIR / f"{stem}.tex").write_text(body + "\n")
        inputs.append(f"\\input{{{LECTURE_SUBDIR}/{stem}}}")

    preamble, bib_print = course_preamble(
        title, state, with_bib, lecture_videos(output_root, state))
    header = (
        "%% Generated by build_course.py --export. Edit the per-lecture files\n"
        f"%% in {LECTURE_SUBDIR}/; regenerating overwrites them.\n"
        "%% Build with:  latexmk -pdf main.tex\n")
    (dest / MAIN_FILENAME).write_text(
        header + preamble + "\n\n" + "\n".join(inputs)
        + bib_print + "\n\n" + CLOSING + "\n")

    save_state(output_root, state)
    print(f"Exported {len(slugs)} lecture(s) to {dest}/: {MAIN_FILENAME}, "
          f"{LECTURE_SUBDIR}/, "
          + (f"{BIB_FILENAME}" if with_bib else "(no bibliography)"))
    return dest / MAIN_FILENAME


def merge_section_usage(state: dict, slug: str, usage: Usage) -> None:
    """Accumulate a run's usage into the section's stored (lifetime) usage."""
    prior = Usage.from_dict(state["sections"][slug].get("usage") or {})
    prior.add(usage)
    state["sections"][slug]["usage"] = prior.to_dict()


def print_usage_totals(run_usage: Usage, state: dict) -> None:
    if run_usage.any():
        print(f"LLM usage this run: {format_usage(run_usage)}")
    course = Usage()
    for sec in state["sections"].values():
        course.add(Usage.from_dict(sec.get("usage") or {}))
    if course.any():
        print(f"Course total (all recorded runs): {format_usage(course)}")


def ordered_slugs(state: dict) -> list[str]:
    return sorted(state["sections"],
                  key=lambda s: state["sections"][s]["lecture_num"])


def normalize_equations(output_root: Path, state: dict,
                        slugs: list[str]) -> list[ReviewItem]:
    """Unnumber every display nothing cites, and report the ones a person has
    to settle. Returns the review items.

    Course-wide by construction: the referenced set is gathered from every
    section before any section is rewritten, because lecture 9 routinely cites
    an equation from lecture 3 and a per-section view would unnumber exactly
    the equations that carry the course.
    """
    bodies = {slug: current_body(output_root, state, slug) for slug in slugs}
    whole = "\n".join(bodies.values())
    referenced = referenced_labels(whole)
    defined = defined_labels(whole)

    starred = numbered = 0
    for slug, body in bodies.items():
        new_body, off, on = normalize_equation_numbering(body, referenced)
        if not (off or on):
            continue
        starred += off
        numbered += on
        state["sections"][slug]["body"] = new_body.strip()
        section_file = output_root / slug / "section.tex"
        if section_file.exists():
            section_file.write_text(new_body)
        bodies[slug] = new_body
    if starred or numbered:
        bits = []
        if starred:
            bits.append(f"{starred} uncited display(s) unnumbered")
        if numbered:
            bits.append(f"{numbered} newly cited display(s) numbered")
        print("Equation numbering: " + ", ".join(bits) + ".")

    items = []
    for slug, body in bodies.items():
        for item in review_items(body, referenced):
            items.append((state["sections"][slug]["lecture_num"], slug, item))
    items += [(0, "", d) for d in
              dangling_references("\n".join(bodies.values()), defined)]
    if items:
        print(f"\nEquation numbering needs a reviewer ({len(items)} item(s)) "
              f"— referenced displays that cannot produce a number:")
        for num, slug, item in sorted(items, key=lambda r: (r[0], r[2].label)):
            where = f"Lecture {num}" if num else "course-wide"
            print(f"  {where}: {item}")
    return [item for _, _, item in items]


def write_document(output_root: Path, state: dict, output_tex: Path,
                   slugs: list[str], title: str | None = None
                   ) -> tuple[str, list[tuple[int, int]]]:
    """Assemble and write the course document. Returns (text, line spans)."""
    normalize_equations(output_root, state, slugs)
    body_parts = [current_body(output_root, state, s) for s in slugs]
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    doc, spans = render_document(title or state.get("title") or "Lecture Notes",
                                 body_parts, state, output_root, output_tex)
    output_tex.write_text(doc)
    save_state(output_root, state)
    print(f"Written: {output_tex}  ({len(doc):,} chars)")
    return doc, spans


# ---------------------------------------------------------------------------
# Compile errors: attribute them to the lecture that wrote them, then fix
# ---------------------------------------------------------------------------

PREAMBLE_SLUG = "\0preamble"   # sentinel key, cannot collide with a directory


def attribute_errors(errors: list[LatexError], doc: str,
                     spans: list[tuple[int, int]], slugs: list[str],
                     state: dict, output_root: Path
                     ) -> tuple[dict[str, list[LatexError]], list[LatexError]]:
    """Group compile errors by the source that produced them.

    Line-numbered errors are placed by line range (anything above the first
    section belongs to the model-supplied preamble additions — the rest of
    the preamble is a fixed, known-good template). Undefined citations are
    placed by searching the section bodies for the offending \\cite key, and
    errors with no usable line — a missing package, an undefined environment
    — by searching for the identifier they name.

    Returns (errors_by_slug, unattributed)."""
    by_slug: dict[str, list[LatexError]] = {}
    unattributed: list[LatexError] = []
    additions = "\n".join(state.get("preamble_additions", []))
    first_body_line = spans[0][0] if spans else None
    bodies = {s: current_body(output_root, state, s) for s in slugs}
    # Preamble additions first: a stray \usepackage there explains far more
    # errors than the same word appearing in a section body.
    sources = ([(PREAMBLE_SLUG, additions)] if additions.strip() else []) \
        + [(s, bodies[s]) for s in slugs]

    def by_identifier(err: LatexError) -> bool:
        for tok in tokens_of(err):
            for name, text in sources:
                if tok in text:
                    by_slug.setdefault(name, []).append(err)
                    return True
        return False

    for err in errors:
        if err.citations:
            hit = False
            for key in err.citations:
                # \cite{a,b} / \cite[p. 3]{a} — match the key inside braces.
                pat = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\s*(\[[^\]]*\]\s*)*"
                                 r"\{[^}]*\b" + re.escape(key) + r"\b[^}]*\}")
                for slug_ in slugs:
                    if pat.search(bodies[slug_]):
                        one = LatexError(f"undefined citation: {key}",
                                         citations=[key])
                        by_slug.setdefault(slug_, []).append(one)
                        hit = True
            if not hit:
                unattributed.append(err)
            continue

        if err.line is None:
            if not by_identifier(err):
                unattributed.append(err)
            continue
        if first_body_line is not None and err.line < first_body_line:
            # Preamble region. Only the additions are ours to fix.
            if additions.strip():
                by_slug.setdefault(PREAMBLE_SLUG, []).append(err)
            elif not by_identifier(err):
                unattributed.append(err)
            continue
        placed = False
        for slug_, (lo, hi) in zip(slugs, spans):
            if lo <= err.line <= hi:
                by_slug.setdefault(slug_, []).append(err)
                placed = True
                break
        if not placed and not by_identifier(err):
            unattributed.append(err)
    return by_slug, unattributed


def _localize(errors: list[LatexError], span: tuple[int, int] | None,
              doc_lines: list[str]) -> str:
    """Render errors for a repair prompt, with document line numbers
    translated into lines of the section file the model will edit."""
    out = []
    for err in errors:
        head = err.message
        if err.line is not None:
            src = (doc_lines[err.line - 1].strip()
                   if 0 < err.line <= len(doc_lines) else "")
            where = (f"line {err.line - span[0] + 1} of your section"
                     if span else f"line {err.line} of the document")
            head += f"\n  at {where}" + (f": {src}" if src else "")
        if err.detail:
            head += "\n  LaTeX said:\n" + "\n".join(
                "    " + ln for ln in err.detail.splitlines())
        out.append("- " + head)
    return "\n".join(out)


PREAMBLE_ADDITIONS_FILE = "preamble_additions.tex"


def _fix_preamble(output_root: Path, state: dict, errors: list[LatexError],
                  doc_lines: list[str], backend: str, model: str | None,
                  frame_model: str | None, run_usage: Usage) -> None:
    """Repair a broken entry in the shared preamble (added at some point by
    add_to_preamble). The additions are round-tripped through a file so the
    model can edit them with its normal tools."""
    path = (output_root / PREAMBLE_ADDITIONS_FILE).resolve()
    path.write_text("\n".join(state.get("preamble_additions", [])) + "\n")
    ctx = NotesToolContext(refs_dir=output_root / "references",
                           read_roots=[output_root.resolve()])
    user_text = (
        f"The course preamble additions in `{path}` break the LaTeX build:\n\n"
        f"{_localize(errors, None, doc_lines)}\n\n"
        f"These lines are inserted into a fixed preamble that already loads "
        f"amsmath, amsthm, amssymb, thmtools, enumitem, todonotes, hyperref "
        f"and cleveref (in that order), and defines the theorem environments. "
        f"Read the file and fix it — correct or delete whatever is broken or "
        f"redundant, keeping every macro that the lecture sections actually "
        f"use. Edit the file in place; change nothing else.")
    print(f"\n[latex-fix → preamble] {len(errors)} error(s)", flush=True)
    text = run_agent(system_prompt=SYSTEM_PROMPT, user_text=user_text, ctx=ctx,
                     output_file=path, backend=backend, model=model,
                     frame_model=frame_model, revise=True,
                     role="fix-preamble", log_dir=output_root / LOG_SUBDIR)
    run_usage.add(ctx.usage)
    cleaned = text.strip()
    if not cleaned:
        print("  (the preamble came back empty — keeping the previous one)")
    else:
        # The list is only ever joined with newlines, so collapsing it to one
        # entry is safe; dedup of future additions compares whole strings.
        state["preamble_additions"] = [cleaned]
        save_state(output_root, state)
    # A scratch file for the model to edit — not an input to later runs, so
    # don't leave it (or run_agent's sidecars for it) lying around looking
    # like one. Nothing re-reads them: the preamble lives in the state file.
    leftover = open_question_count(path)
    if leftover:
        print(f"  ({leftover} question(s) asked during the preamble fix go "
              f"unanswered — there is no follow-up pass for the preamble)")
    for junk in (path, path.with_name(path.name + ".bak"),
                 questions_file_for(path)):
        junk.unlink(missing_ok=True)


def double_script_note(errors: list[LatexError]) -> str:
    """Extra guidance when a section reports a stacked sub/superscript.

    "Double subscript" is almost never a defect of the line that reports it:
    the macro's own definition ends in an unbraced script, so it breaks at
    every call site that adds one. Left to itself the model braces the call
    site in front of it — which fixes this lecture and leaves the definition
    broken for every other one that uses the macro.
    """
    if not any("Double subscript" in e.message
               or "Double superscript" in e.message for e in errors):
        return ""
    return (
        "\nA \"Double subscript\"/\"Double superscript\" error usually means "
        "the macro's definition already ends in a script — \\Gm expanding to "
        "\\mathbb{G}_{m}, so \\Gm_{A} stacks two subscripts. Fix the "
        "definition, not the call site: add a \\renewcommand with the whole "
        "body braced (\\renewcommand{\\Gm}{{\\mathbb{G}_{m}}}) via "
        "add_to_preamble, which repairs every lecture that uses it at once. "
        "Only brace the call site if the definition is genuinely fine.\n")


POLISH_INSTRUCTION = """
Fix each one in the file, and change nothing else. The mathematics is
finished — you are adjusting how it is set, not what it says.

Overfull \\hbox — something is wider than the text block and is printing into
the margin. The usual causes, in the order they are usually the answer:
- A long display that is really one line: break it with \\begin{align} or
  \\begin{multline}, or insert \\allowbreak / \\quad at a natural point.
- An inline formula too long to break: move it into a display.
- A long \\texttt, URL or unhyphenatable word: allow a break (\\allowbreak,
  \\-, or \\sloppy for that paragraph only).
- A wide tabular, tikzcd or array: shrink the column spacing (@{}, \\arraycolsep,
  column sep=small) or wrap it in \\resizebox{\\textwidth}{!}{...}.
Do not fix one by deleting content, and do not wrap the whole file in
\\sloppy — that trades a visible overflow for ugly inter-word spacing
everywhere.

"Token not allowed in a PDF string" — a \\section, \\subsection or caption
contains maths, and hyperref cannot put maths in a PDF bookmark. Wrap the
mathematical part in \\texorpdfstring{<the maths>}{<a plain-text version>},
e.g. \\section{The \\texorpdfstring{$p$-adic}{p-adic} case}. The second
argument is read by a PDF reader's bookmark pane, so it must be plain text:
no macros, no $, no backslashes. Spell the symbol out when there is no ASCII
for it (\\texorpdfstring{$\\mathbb{Z}_\\ell$}{Z_l}).
"""


def _fix_section(output_root: Path, state: dict, slug: str,
                 errors: list[LatexError], span: tuple[int, int],
                 doc_lines: list[str], backend: str, model: str | None,
                 frame_model: str | None, run_usage: Usage,
                 polish: bool = False) -> None:
    lecture_dir = output_root / slug
    section_file = ensure_section_file(output_root, state, slug)
    num = state["sections"][slug]["lecture_num"]
    ctx = NotesToolContext(
        refs_dir=output_root / "references",
        video_path=find_video(lecture_dir),
        total_duration=lecture_duration(lecture_dir),
        transcript_path=lecture_dir / "transcript.json",
        enable_preamble=True,
        existing_preamble=list(state.get("preamble_additions", [])),
        read_roots=[output_root.resolve()],
        bib_file=output_root / BIB_FILENAME,
    )
    cite_note = ""
    if any(e.citations for e in errors):
        cite_note = (
            "\nAn undefined citation means the key is not in the "
            "bibliography. Either you invented the key — replace it with the "
            "right one from the list above — or the source was never "
            "registered: call cite_reference for it (with title, author and "
            "year for anything that is not an arXiv ID or DOI) and use the "
            "key it returns.\n")
    script_note = double_script_note(errors)
    if polish:
        user_text = (
            f"The course document compiles, but these presentation problems "
            f"come from your section, Lecture {num}, in `{section_file}`:\n\n"
            f"{_localize(errors, span, doc_lines)}\n"
            + POLISH_INSTRUCTION)
    else:
        user_text = (
            f"The assembled course document does not compile. These errors "
            f"come from your section, Lecture {num}, in `{section_file}`:\n\n"
            f"{_localize(errors, span, doc_lines)}\n"
            f"{cite_note}"
            f"{script_note}\n"
            f"{bibliography_index(output_root / BIB_FILENAME) if any(e.citations for e in errors) else ''}"
            f"Read the file and fix them. Keep the mathematics exactly as it "
            f"is — you are correcting LaTeX, not rewriting content. Note that "
            f"the preamble is fixed and shared: if a macro or environment is "
            f"genuinely missing, define it with add_to_preamble rather than "
            f"working around it, and remember that the packages listed in "
            f"your instructions are already loaded. Edit the file in place.")
    print(f"\n[latex-{'polish' if polish else 'fix'} → Lecture {num} "
          f"({slug})] {len(errors)} "
          f"{'item' if polish else 'error'}(s)", flush=True)
    # No summary_file: this pass corrects LaTeX, it does not change what the
    # lecture says, so the existing summary stays valid.
    body = run_agent(system_prompt=SYSTEM_PROMPT, user_text=user_text, ctx=ctx,
                     output_file=section_file, backend=backend, model=model,
                     frame_model=frame_model, revise=True,
                     role="fix-latex", log_dir=output_root / LOG_SUBDIR)
    state["sections"][slug]["body"] = body.strip()
    merge_section_usage(state, slug, ctx.usage)
    run_usage.add(ctx.usage)
    for entry in ctx.new_preamble_additions:
        if entry not in state.setdefault("preamble_additions", []):
            state["preamble_additions"].append(entry)
    save_state(output_root, state)


def assemble_from_state(output_root: Path, state: dict, output_tex: Path,
                        slugs: list[str] | None = None,
                        title: str | None = None,
                        backend: str = "subscription",
                        model: str | None = None,
                        frame_model: str | None = None,
                        fix_rounds: int = 0,
                        run_usage: Usage | None = None) -> None:
    """Assemble the course document from state (and on-disk section files),
    then compile-check it. When it does not compile and fix_rounds > 0, each
    error is handed back to the lecture that produced it and the document is
    reassembled — up to fix_rounds times."""
    if slugs is None:
        slugs = ordered_slugs(state)
    if run_usage is None:
        run_usage = Usage()
    doc, spans = write_document(output_root, state, output_tex, slugs, title)

    for attempt in range(fix_rounds + 1):
        errors, warnings = compile_document(output_tex)
        if errors is None:
            print("(no LaTeX toolchain found on PATH — skipping compile check)")
            return
        if not errors and not warnings:
            print(f"Compile check OK: {output_tex.name}")
            return
        # Correctness before appearance. While the document does not compile,
        # TeX stops early and reflows nothing after the failure, so the line
        # numbers on an overfull box are measured against a layout that will
        # not exist once the errors are gone.
        polish = not errors
        items = warnings if polish else errors
        if polish:
            print_warnings(output_tex, warnings)
        else:
            print_errors(output_tex, errors)
        if attempt == fix_rounds:
            break

        doc_lines = doc.splitlines()
        by_slug, unattributed = attribute_errors(items, doc, spans, slugs,
                                                 state, output_root)
        noun = "polish item" if polish else "error"
        if not by_slug:
            print(f"  (could not attribute these {noun}s to a lecture — "
                  "leaving them for a manual pass)")
            break
        if unattributed:
            print(f"  ({len(unattributed)} {noun}(s) not attributable to a "
                  f"single lecture — not sent for repair)")
        print(f"\n{'Polishing' if polish else 'Fixing'} "
              f"(round {attempt + 1}/{fix_rounds}): "
              f"{len(items) - len(unattributed)} {noun}(s) across "
              f"{len(by_slug)} source(s).")
        if PREAMBLE_SLUG in by_slug:
            _fix_preamble(output_root, state, by_slug.pop(PREAMBLE_SLUG),
                          doc_lines, backend, model, frame_model, run_usage)
        span_of = dict(zip(slugs, spans))
        for slug_ in sorted(by_slug,
                            key=lambda s: state["sections"][s]["lecture_num"]):
            _fix_section(output_root, state, slug_, by_slug[slug_],
                         span_of[slug_], doc_lines, backend, model,
                         frame_model, run_usage, polish=polish)
        doc, spans = write_document(output_root, state, output_tex, slugs,
                                    title)

    print("  Remaining items need a manual look "
          "(or another --latex-fix-rounds pass).")


# ---------------------------------------------------------------------------
# Follow-up: answer open questions / resolve todos for one lecture
# ---------------------------------------------------------------------------

def lecture_segments(lecture_dir: Path) -> list[dict]:
    transcript_path = lecture_dir / "transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            return json.load(f)["segments"]
    return []


def lecture_duration(lecture_dir: Path) -> float:
    segments = lecture_segments(lecture_dir)
    return segments[-1]["end"] if segments else 0.0


def ensure_section_file(output_root: Path, state: dict, slug: str) -> Path:
    """The on-disk section file (recreated from state if missing)."""
    section_file = (output_root / slug / "section.tex").resolve()
    if not section_file.exists():
        section_file.write_text(state["sections"][slug]["body"])
    return section_file


def describe_revision(lecture_num: int, answers_block: str | None,
                      new_corrections: dict[str, str]) -> str | None:
    """What changed in one revised lecture, for the propagation prompt."""
    changes = []
    if answers_block:
        changes.append(f"Answers applied to Lecture {lecture_num}:\n"
                       f"{answers_block}")
    if new_corrections:
        lines = "\n".join(f'- "{w}" → "{r}"'
                          for w, r in new_corrections.items())
        changes.append(f"Transcript corrections confirmed while revising "
                       f"Lecture {lecture_num}:\n{lines}")
    return "\n\n".join(changes) or None


def propagate_revision(output_root: Path, state: dict, revised_slug: str,
                       answers_block: str | None,
                       new_corrections: dict[str, str],
                       backend: str, model: str | None,
                       frame_model: str | None, wait: bool,
                       run_usage: Usage | None = None) -> None:
    """After lecture N is revised in a follow-up, sweep every later lecture
    for inherited material affected by the changes."""
    num = state["sections"][revised_slug]["lecture_num"]
    changes = describe_revision(num, answers_block, new_corrections)
    if changes:
        propagate_revisions(output_root, state, {num: changes}, backend, model,
                            frame_model, wait, run_usage)


def propagate_revisions(output_root: Path, state: dict,
                        changes_by_num: dict[int, str],
                        backend: str, model: str | None,
                        frame_model: str | None, wait: bool,
                        run_usage: Usage | None = None) -> None:
    """Sweep each lecture for inherited material affected by revisions made
    to *earlier* lectures (restated definitions, notation, cross-references,
    recurring mishearings).

    Every lecture is visited at most once, carrying the changes from all
    revised lectures before it — so answering a whole course costs one pass,
    not one pass per revised lecture."""
    if not changes_by_num:
        return
    first_revised = min(changes_by_num)
    later = sorted((s["lecture_num"], sl)
                   for sl, s in state["sections"].items()
                   if s["lecture_num"] > first_revised)
    if not later:
        return

    print(f"\nPropagating {len(changes_by_num)} revision(s) to "
          f"{len(later)} later lecture(s)…")
    for num2, slug2 in later:
        relevant = [changes_by_num[n] for n in sorted(changes_by_num)
                    if n < num2]
        if not relevant:
            continue
        changes_text = "\n\n".join(relevant)
        revised_list = ", ".join(f"Lecture {n}" for n in sorted(changes_by_num)
                                 if n < num2)
        lecture_dir2 = output_root / slug2
        section_file2 = ensure_section_file(output_root, state, slug2)
        ctx2 = NotesToolContext(
            refs_dir=output_root / "references",
            video_path=find_video(lecture_dir2),
            total_duration=lecture_duration(lecture_dir2),
            transcript_path=lecture_dir2 / "transcript.json",
            enable_preamble=True,
            existing_preamble=list(state.get("preamble_additions", [])),
            read_roots=[output_root.resolve()],
            bib_file=output_root / BIB_FILENAME,
        )
        user_text = (
            f"{revised_list} of this series {'was' if len(relevant) == 1 else 'were'} "
            f"just revised in a follow-up:\n\n{changes_text}\n\n"
            f"{lecture_index(output_root, state, slug2)}"
            f"{bibliography_index(output_root / BIB_FILENAME)}"
            f"Your section is Lecture {num2}, in `{section_file2}`. Read it "
            f"and update anything affected by the changes above — material "
            f"inherited from those lectures (restated definitions, notation, "
            f"cross-references) and any occurrence of the misheard phrases "
            f"(use judgment: fix only genuine mishearings, not legitimate "
            f"uses of the same words). If nothing applies, make no edits and "
            f"reply 'no changes needed'."
        )
        print(f"\n[propagate → Lecture {num2} ({slug2})]", flush=True)
        body2 = run_agent(
            system_prompt=SYSTEM_PROMPT,
            user_text=user_text,
            ctx=ctx2,
            output_file=section_file2,
            backend=backend,
            model=model,
            frame_model=frame_model,
            revise=True,
            wait_for_answers=wait,
            summary_file=lecture_dir2 / "summary.md",
            role="propagate", log_dir=output_root / LOG_SUBDIR,
        )
        state["sections"][slug2]["body"] = body2.strip()
        merge_section_usage(state, slug2, ctx2.usage)
        if run_usage is not None:
            run_usage.add(ctx2.usage)
        summary_path2 = lecture_dir2 / "summary.md"
        if summary_path2.exists():
            state["sections"][slug2]["summary"] = \
                summary_path2.read_text().strip()
        state.setdefault("corrections", {}).update(ctx2.new_corrections)
        for entry in ctx2.new_preamble_additions:
            if entry not in state.setdefault("preamble_additions", []):
                state["preamble_additions"].append(entry)
        save_state(output_root, state)


def collect_lecture_answers(output_root: Path, state: dict, slug: str
                            ) -> tuple[NotesToolContext, str | None, int,
                                       str | None]:
    """Ask the user this lecture's open questions and \\todo markers. No model
    runs here.

    Kept separate from revise_lecture so a whole-course follow-up can put
    every question to the user in one sitting and only then start the models
    — rather than making them wait at the terminal between lectures.

    The context is handed back rather than rebuilt later because answering a
    clarify question records a transcript correction on it, which the
    revision and the propagation both need."""
    lecture_dir = output_root / slug
    section_file = ensure_section_file(output_root, state, slug)
    ctx = NotesToolContext(
        refs_dir=output_root / "references",
        video_path=find_video(lecture_dir),
        total_duration=lecture_duration(lecture_dir),
        transcript_path=lecture_dir / "transcript.json",
        enable_preamble=True,
        existing_preamble=list(state.get("preamble_additions", [])),
        read_roots=[output_root.resolve()],
        bib_file=output_root / BIB_FILENAME,
        boards=load_boards(lecture_dir),
        diagrams_dir=lecture_dir / DIAGRAMS_SUBDIR,
    )
    answers_block = collect_followup_answers(ctx, section_file,
                                             lecture_segments(lecture_dir))
    body = section_file.read_text()
    todos = count_todos(body)
    todo_answers = ask_todo_answers(
        todo_items(body), state["sections"][slug]["lecture_num"])
    return ctx, answers_block, todos, todo_answers


def revise_lecture(output_root: Path, state: dict, slug: str,
                   backend: str, model: str | None, frame_model: str | None,
                   wait: bool, run_usage: Usage, ctx: NotesToolContext,
                   answers_block: str | None, todos: int,
                   todo_answers: str | None = None
                   ) -> tuple[str | None, dict[str, str]]:
    """Have the agent revise this section in place, applying the answers
    already collected and sweeping remaining \\todo markers. Returns
    (answers_block, new_corrections); (None, {}) if there was nothing to do."""
    lecture_dir = output_root / slug
    section_file = ensure_section_file(output_root, state, slug)
    if not answers_block and not todo_answers and todos == 0:
        print("No open questions and no \\todo markers — nothing to do.")
        return None, {}

    boards = ctx.boards
    parts = [f"You previously wrote the LaTeX body for Lecture "
             f"{state['sections'][slug]['lecture_num']} to `{section_file}`.",
             lecturer_note(state.get("lecturers", {}).get(slug)).rstrip(),
             lecture_index(output_root, state, slug).rstrip(),
             bibliography_index(output_root / BIB_FILENAME).rstrip(),
             board_index(boards, attached=backend == "api").rstrip()]
    parts = [p for p in parts if p]
    if answers_block:
        parts.append("The user has now answered previously open "
                     f"questions:\n\n{answers_block}")
    if todo_answers:
        parts.append(
            "The user has answered some of the \\todo markers in the file. "
            "Apply each answer and remove that marker; the answer is "
            "authoritative, so do not second-guess it or re-flag the "
            f"point:\n\n{todo_answers}")
    parts.append(
        f"The file currently contains {todos} \\todo marker(s). Read the "
        f"file, then: apply the answers above, and review each remaining "
        f"\\todo — resolve those you now can (using the answers, the video "
        f"frames, or your other tools; ask the user again if needed) and "
        f"remove the resolved markers. Leave genuinely unresolved ones in "
        f"place.")

    section = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_text="\n\n".join(parts),
        ctx=ctx,
        output_file=section_file,
        backend=backend,
        model=model,
        frame_model=frame_model,
        revise=True,
        wait_for_answers=wait,
        summary_file=lecture_dir / "summary.md",
        images=[(b["path"], f"Board {b['id']} ({_spans(b)})") for b in boards],
        role="revise", log_dir=output_root / LOG_SUBDIR,
    )
    report_unread_boards(ctx, boards, role="revise")
    report_placeholders(section_file)
    report_hand_written_references(section_file)

    # The notes now contain these answers; don't re-deliver them next run.
    mark_answers_applied(ctx, section_file)
    state["sections"][slug]["body"] = section.strip()
    run_usage.add(ctx.usage)
    merge_section_usage(state, slug, ctx.usage)
    summary_path = lecture_dir / "summary.md"
    if summary_path.exists():
        state["sections"][slug]["summary"] = summary_path.read_text().strip()
    state.setdefault("corrections", {}).update(ctx.new_corrections)
    for entry in ctx.new_preamble_additions:
        if entry not in state.setdefault("preamble_additions", []):
            state["preamble_additions"].append(entry)
    save_state(output_root, state)
    print(f"\nRevised lecture '{slug}'.")
    return answers_block, dict(ctx.new_corrections)


def verify_lecture(output_root: Path, state: dict, slug: str, backend: str,
                   model: str | None, frame_model: str | None,
                   run_usage: Usage) -> None:
    """Re-read a finished section against its transcript, with fresh eyes.

    The writing pass builds the notes forward, segment by segment, and never
    sees the finished text as a text — which is exactly when an invented
    "equivalently" or a hedge quietly promoted to an assertion goes
    unnoticed. So this runs as its own context: no memory of having written
    the section, no attachment to it, and the whole transcript available to
    read backwards and forwards."""
    lecture_dir = output_root / slug
    section_file = ensure_section_file(output_root, state, slug)
    num = state["sections"][slug]["lecture_num"]
    segments = lecture_segments(lecture_dir)
    if not segments:
        print(f"  (no transcript for {slug} — skipping verification)")
        return

    with open(lecture_dir / "transcript.json") as f:
        meta = json.load(f).get("metadata", {})

    ctx = NotesToolContext(
        refs_dir=output_root / "references",
        video_path=find_video(lecture_dir),
        total_duration=lecture_duration(lecture_dir),
        transcript_path=lecture_dir / "transcript.json",
        enable_preamble=True,
        existing_preamble=list(state.get("preamble_additions", [])),
        read_roots=[output_root.resolve()],
        bib_file=output_root / BIB_FILENAME,
        boards=load_boards(lecture_dir),
        diagrams_dir=lecture_dir / DIAGRAMS_SUBDIR,
    )
    boards = ctx.boards
    user_text = (
        f"Check the notes for **Lecture {num}** of this course, in "
        f"`{section_file}`.\n\n"
        f"{lecturer_note(state.get('lecturers', {}).get(slug))}"
        f"{lecture_provenance(meta)}"
        f"You may consult the video frames to check anything read off the "
        f"board.\n\n"
        f"{lecture_index(output_root, state, slug)}"
        f"{bibliography_index(output_root / BIB_FILENAME)}"
        f"{hand_written_note(section_file)}"
        f"{board_index(boards, attached=backend == 'api')}"
        f"**Transcript:**\n\n"
        f"{format_transcript(segments, board_marks(boards))}"
    )
    print(f"\n[verify → Lecture {num} ({slug})]", flush=True)
    body = run_agent(
        system_prompt=VERIFY_PROMPT,
        user_text=user_text,
        ctx=ctx,
        output_file=section_file,
        backend=backend,
        model=model,
        frame_model=frame_model,
        revise=True,
        summary_file=lecture_dir / "summary.md",
        images=[(b["path"], f"Board {b['id']} ({_spans(b)})") for b in boards],
        role="verify", log_dir=output_root / LOG_SUBDIR,
    )
    report_unread_boards(ctx, boards, role="verify")
    report_placeholders(section_file)
    report_hand_written_references(section_file)
    state["sections"][slug]["body"] = body.strip()
    merge_section_usage(state, slug, ctx.usage)
    run_usage.add(ctx.usage)
    summary_path = lecture_dir / "summary.md"
    if summary_path.exists():
        state["sections"][slug]["summary"] = summary_path.read_text().strip()
    for entry in ctx.new_preamble_additions:
        if entry not in state.setdefault("preamble_additions", []):
            state["preamble_additions"].append(entry)
    save_state(output_root, state)


def pending_questions(output_root: Path, state: dict) -> list[tuple[int, str,
                                                                    int, int]]:
    """(lecture_num, slug, open questions, todos) for lectures with either."""
    out = []
    for slug in ordered_slugs(state):
        section_file = output_root / slug / "section.tex"
        body = (section_file.read_text() if section_file.exists()
                else state["sections"][slug].get("body", ""))
        opens = open_question_count(section_file)
        todos = count_todos(body)
        if opens or todos:
            out.append((state["sections"][slug]["lecture_num"], slug,
                        opens, todos))
    return out


def answer_lectures(output_root: Path, slugs: list[str] | None,
                    output_tex: Path, backend: str, model: str | None,
                    frame_model: str | None, wait: bool = False,
                    propagate: bool = True, fix_rounds: int = 0) -> None:
    """Follow-up run over one lecture (slugs=[slug]) or the whole course
    (slugs=None): answer open questions, revise, propagate, reassemble.

    Questions for every lecture are put to the user first, in one sitting;
    only then do the models start. Interleaving them would strand the user at
    the terminal between lectures, waiting for one revision to finish before
    being asked the next question."""
    state = load_state(output_root)
    if slugs is None:
        pending = pending_questions(output_root, state)
        if not pending:
            print("No open questions and no \\todo markers in any lecture — "
                  "nothing to do.")
            return
        print(f"{len(pending)} lecture(s) with open questions or todos:")
        for num, slug, opens, todos in pending:
            bits = []
            if opens:
                bits.append(f"{opens} open question(s)")
            if todos:
                bits.append(f"{todos} todo(s)")
            print(f"  Lecture {num} ({slug}): {', '.join(bits)}")
        slugs = [slug for _, slug, _, _ in pending]
    else:
        for slug in slugs:
            if slug not in state["sections"]:
                known = ", ".join(sorted(state["sections"])) or "(none)"
                sys.exit(f"No generated section for '{slug}'. Known: {known}")

    run_usage = Usage()

    # Phase 1 — everything that needs you, up front.
    collected = []
    for i, slug in enumerate(slugs, 1):
        num = state["sections"][slug]["lecture_num"]
        if len(slugs) > 1:
            print(f"\n=== Questions [{i}/{len(slugs)}]: "
                  f"Lecture {num} ({slug}) ===")
        ctx, answers_block, todos, todo_answers = collect_lecture_answers(
            output_root, state, slug)
        collected.append((slug, num, ctx, answers_block, todos, todo_answers))

    # Indexed rather than star-unpacked: these rows have grown before, and a
    # `for *_, block, todos in` silently reads the wrong fields when they do.
    todo_total = sum(row[4] for row in collected)
    answered = sum(1 for row in collected if row[3] or row[5])
    if any(row[3] or row[4] or row[5] for row in collected):
        if len(slugs) > 1:
            print(f"\n=== All questions collected ({answered} lecture(s) with "
                  f"answers, {todo_total} \\todo marker(s) to sweep) ===")
        print("Running the models now. You can leave this unattended: any new "
              "question raised during the revisions is queued, not waited on"
              + (" (--wait overrides that)." if not wait else
                 ", but --wait will block for it at the end.") + "\n",
              flush=True)

    # Phase 2 — unattended.
    changes_by_num: dict[int, str] = {}
    for i, (slug, num, ctx, answers_block, todos,
            todo_answers) in enumerate(collected, 1):
        if len(slugs) > 1:
            print(f"\n=== [{i}/{len(slugs)}] Revising Lecture {num} "
                  f"({slug}) ===")
        answers_block, new_corrections = revise_lecture(
            output_root, state, slug, backend, model, frame_model, wait,
            run_usage, ctx, answers_block, todos, todo_answers)
        changes = describe_revision(num, answers_block, new_corrections)
        if changes:
            changes_by_num[num] = changes

    if propagate and changes_by_num:
        # One sweep for the whole run: each later lecture is visited once,
        # carrying every earlier revision.
        propagate_revisions(output_root, state, changes_by_num, backend,
                            model, frame_model, wait, run_usage)

    print("\nReassembling the course document.")
    assemble_from_state(output_root, state, output_tex, backend=backend,
                        model=model, frame_model=frame_model,
                        fix_rounds=fix_rounds, run_usage=run_usage)
    print_usage_totals(run_usage, state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def filter_available(inputs: list[str], output_root: Path) -> list[str]:
    """Take the leading run of inputs that need no network: local files, and
    URL sources whose lecture directory already exists (downloaded, with at
    least the audio extracted). Used by --available-only when rate-limited.

    This stops at the FIRST unavailable lecture rather than skipping over it:
    lecture numbers and cross-references follow input order, so processing a
    later lecture before an earlier one would misnumber both it and every
    lecture after, and force a --regen once the gap is filled.
    """
    def downloaded(source: str) -> bool:
        if not output_root.exists():
            return False
        for d in output_root.iterdir():
            info_path = d / "info.json"
            if not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text())
            except json.JSONDecodeError:
                continue
            if info.get("source") == source and (
                    (d / "transcript.json").exists()
                    or (d / "audio.wav").exists()):
                return True
        return False

    kept = []
    for src in inputs:
        if is_url(src) and not downloaded(src):
            break
        kept.append(src)

    remaining = len(inputs) - len(kept)
    if remaining:
        print(f"--available-only: stopping after lecture {len(kept)} — "
              f"{inputs[len(kept)]} is not downloaded yet.")
        later = sum(1 for s in inputs[len(kept) + 1:]
                    if not is_url(s) or downloaded(s))
        if later:
            print(f"  ({later} later lecture(s) are downloaded but held back "
                  f"to keep lecture numbering in order.)")
        print(f"  {remaining} lecture(s) left for a later run.")
    if not kept:
        sys.exit("--available-only: the first lecture is not downloaded yet, "
                 "so there is nothing to process in order.")
    return kept


def parse_inputs(args) -> list[str]:
    inputs = list(args.videos)
    if args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            sys.exit(f"--from-file: not found: {path}")
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                inputs.append(line)
    if not inputs:
        sys.exit("No lecture inputs provided.")

    # Expand playlist URLs into their videos, in playlist order.
    expanded = []
    for src in inputs:
        urls = expand_playlist(src, args.proxy,
                               via_modal=args.download == "modal")
        if urls:
            print(f"Expanded playlist into {len(urls)} video(s): {src}")
            expanded.extend(urls)
        else:
            expanded.append(src)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("videos", nargs="*",
                        help="Video files, URLs, or YouTube links")
    parser.add_argument("--from-file", metavar="FILE",
                        help="Text file with one video input per line")
    parser.add_argument("--output-dir", default="output",
                        help="Root directory for per-lecture data (default: output/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .tex file (default: <output-dir>/course.tex)")
    parser.add_argument("--title", default=None,
                        help="Course title (saved in state on first run)")
    parser.add_argument("--whisper-model", default=None,
                        choices=["tiny","base","small","medium","large",
                                 "large-v2","large-v3","turbo"],
                        help="Whisper model size (default: base locally, "
                             "large-v3 on Modal)")
    parser.add_argument("--transcribe", default="local",
                        choices=["local", "modal"],
                        help="Where to run Whisper: on this machine, or on a "
                             "Modal GPU (default: local)")
    parser.add_argument("--proxy", default=None, metavar="URL",
                        help="Proxy for local yt-dlp traffic (downloads and "
                             "playlist expansion), e.g. "
                             "socks5://127.0.0.1:1080.")
    parser.add_argument("--modal-fetch", action="store_true",
                        help="Let Modal transcription workers download the "
                             "audio themselves instead of receiving the "
                             "locally-extracted audio (YouTube usually "
                             "blocks datacenter egress, hence off by "
                             "default; falls back to uploading on failure).")
    parser.add_argument("--download", default="local",
                        choices=["local", "modal"],
                        help="Where yt-dlp runs: locally, or on Modal "
                             "workers whose egress circumvents rate limiting "
                             "of your IP (videos are shipped back; also "
                             "routes playlist expansion; falls back to local "
                             "downloads on failure).")
    parser.add_argument("--available-only", action="store_true",
                        help="Process the already-downloaded lectures up to "
                             "the first missing one (stops there rather than "
                             "skipping ahead, so lecture numbering stays in "
                             "order) — useful when rate-limited; rerun "
                             "without this flag later to pick up the rest.")
    parser.add_argument("--language", default=None, metavar="LANG",
                        help="Language code for Whisper (default: en). Pass "
                             "'auto' to let Whisper detect it — it guesses "
                             "from the first 30s and gets lectures wrong.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingest; treat each input as an already-ingested directory")
    parser.add_argument("--regen", metavar="SLUG",
                        help="Force regeneration of one lecture by its directory name")
    parser.add_argument("--reference", metavar="URL_OR_ID", action="append",
                        default=[],
                        help="Pre-load a reference document (URL or arXiv ID). "
                             "Passed as context to every lecture. May be repeated. "
                             "Saved in state so it persists across runs.")
    parser.add_argument("--backend", default="subscription", choices=BACKENDS,
                        help="'subscription' = Claude via your Claude "
                             "subscription (default); 'codex' = GPT via your "
                             "ChatGPT subscription; 'api' = Anthropic API "
                             "with ANTHROPIC_API_KEY.")
    parser.add_argument("--model", default=None,
                        help="Override the backend's default model.")
    parser.add_argument("--frame-model", default=None,
                        help="Cheaper model that reads video frames on the "
                             "main model's behalf (default: haiku on the "
                             "Claude backends; inherits the main model on "
                             "codex).")
    parser.add_argument("--wait", action="store_true",
                        help="Block at the end of each lecture until every "
                             "queued question is answered (default: "
                             "unanswered questions defer to --answer).")
    parser.add_argument("--answer", metavar="SLUG", default=None,
                        help="Follow-up mode: answer questions left open by "
                             "an earlier run of lecture SLUG (its directory "
                             "name) and have the agent revise that section "
                             "(also sweeps remaining \\todo markers), "
                             "propagate the changes to later lectures, then "
                             "reassemble the course document.")
    parser.add_argument("--answer-all", action="store_true",
                        help="Follow-up mode over the whole course: put every "
                             "lecture's open questions to you first, in one "
                             "sitting, then run the revisions unattended, "
                             "propagate and reassemble.")
    parser.add_argument("--no-propagate", action="store_true",
                        help="With --answer/--answer-all: skip updating later "
                             "lectures after the revisions.")
    parser.add_argument("--no-boards", dest="boards", action="store_false",
                        help="Skip board segmentation. By default every "
                             "lecture with a video is segmented into board "
                             "states (boards/boards.json plus a clean, "
                             "lecturer-free snapshot of each), and the "
                             "stills go to the model that writes the notes.")
    parser.add_argument("--boards-color", action="store_true",
                        help="Analyse boards in colour (default: greyscale).")
    # Segmentation used to be opt-in. Without this, argparse's prefix matching
    # would quietly read a leftover `--boards` as `--boards-color`.
    parser.add_argument("--boards", dest="boards_legacy",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lecturer", metavar="NAME", default=None,
                        help="Who is lecturing, for all lectures in this run "
                             "(the usual case: one speaker). The notes refer "
                             "to them by surname, as published notes do. "
                             "Without this, each new lecture's speaker is "
                             "asked once, with a guess from the titles offered "
                             "as the default; the answers are remembered. Pass "
                             "\"the lecturer\" to keep the notes anonymous.")
    parser.add_argument("--style-exemplar", metavar="FILE", action="append",
                        default=[],
                        help="A .tex/.md file whose writing style the notes "
                             "should imitate (register only, never content). "
                             "Repeatable; remembered across runs.")
    parser.add_argument("--logs", action="store_true",
                        help="Print a digest of what the agents have done "
                             "(from output/logs/) and exit.")
    parser.add_argument("--export", metavar="DIR", default=None,
                        help="Also write the course to DIR as a multi-file "
                             "LaTeX project: main.tex, one .tex per lecture "
                             "under lectures/, and references.bib. With no "
                             "videos given, exports from the saved state and "
                             "exits.")
    parser.add_argument("--export-compile", action="store_true",
                        help="With --export: compile-check the exported "
                             "main.tex as well.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the accuracy-verification pass that "
                             "re-reads each newly written lecture against its "
                             "transcript with a fresh context.")
    parser.add_argument("--verify", metavar="SLUG", default=None,
                        help="Run only the verification pass, on lecture SLUG "
                             "('all' for every lecture), then reassemble.")
    parser.add_argument("--latex-fix-rounds", type=int, default=2,
                        metavar="N",
                        help="When the assembled document fails to compile, "
                             "hand each error back to the lecture that caused "
                             "it and reassemble, up to N times "
                             "(default: 2; 0 to only report errors).")
    args = parser.parse_args()
    whisper_model = resolve_whisper_model(args.whisper_model, args.transcribe)
    language = resolve_language(args.language)

    if args.logs:
        print(summarize(Path(args.output_dir) / LOG_SUBDIR))
        return

    def do_export(state: dict, slugs: list[str] | None = None,
                  title: str | None = None) -> None:
        main_tex = export_project(Path(args.output_dir), state,
                                  Path(args.export), slugs, title)
        if args.export_compile:
            errors = check_latex(main_tex)
            if errors is None:
                print("(no LaTeX toolchain found on PATH — "
                      "skipping compile check)")
            elif errors:
                print_errors(main_tex, errors)
            else:
                print(f"Compile check OK: {main_tex.name}")

    # Export-only: no inputs to ingest, just re-emit the saved course.
    if args.export and not (args.videos or args.from_file or args.answer
                            or args.answer_all or args.verify):
        do_export(load_state(Path(args.output_dir)))
        return

    if args.verify:
        output_root = Path(args.output_dir)
        output_tex = (Path(args.output) if args.output
                      else output_root / "course.tex")
        state = load_state(output_root)
        if args.verify == "all":
            slugs = ordered_slugs(state)
        elif args.verify in state["sections"]:
            slugs = [args.verify]
        else:
            known = ", ".join(sorted(state["sections"])) or "(none)"
            sys.exit(f"No generated section for '{args.verify}'. "
                     f"Known: {known} (or 'all').")
        # No questions in this mode — the notes already exist — but --lecturer
        # still applies, so a course written before a name was recorded can be
        # given one and re-checked.
        if args.lecturer:
            resolve_lecturers([output_root / s for s in slugs], state,
                              forced=args.lecturer)
            save_state(output_root, state)
        run_usage = Usage()
        for i, slug in enumerate(slugs, 1):
            print(f"\n=== [{i}/{len(slugs)}] verifying {slug} ===")
            verify_lecture(output_root, state, slug, args.backend, args.model,
                           args.frame_model, run_usage)
        print("\nReassembling the course document.")
        assemble_from_state(output_root, state, output_tex,
                            backend=args.backend, model=args.model,
                            frame_model=args.frame_model,
                            fix_rounds=args.latex_fix_rounds,
                            run_usage=run_usage)
        if args.export:
            do_export(state)
        print_usage_totals(run_usage, state)
        return

    if args.answer or args.answer_all:
        if args.answer and args.answer_all:
            sys.exit("Use either --answer SLUG or --answer-all, not both.")
        output_root = Path(args.output_dir)
        output_tex = (Path(args.output) if args.output
                      else output_root / "course.tex")
        if args.lecturer:
            saved = load_state(output_root)
            resolve_lecturers([output_root / s for s in ordered_slugs(saved)],
                              saved, forced=args.lecturer)
            save_state(output_root, saved)
        answer_lectures(output_root,
                        [args.answer] if args.answer else None,
                        output_tex, args.backend, args.model,
                        args.frame_model, args.wait,
                        propagate=not args.no_propagate,
                        fix_rounds=args.latex_fix_rounds)
        if args.export:
            do_export(load_state(output_root))
        return

    inputs = parse_inputs(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.available_only:
        inputs = filter_available(inputs, output_root)
    output_tex = Path(args.output) if args.output else output_root / "course.tex"
    refs_dir = output_root / "references"

    # ------------------------------------------------------------------
    # Load (or initialise) persistent state
    # ------------------------------------------------------------------
    state = load_state(output_root)
    if args.title:
        state["title"] = args.title
    if args.style_exemplar:
        state["style_exemplars"] = [str(Path(f).resolve())
                                    for f in args.style_exemplar]
        state.pop("style_passages", None)      # re-extract for a new exemplar
    if state.get("style_exemplars") and "style_passages" not in state:
        # Done once per course and cached: a model reads each exemplar whole,
        # picks passages from across it, and rewrites them to stand alone;
        # each rewrite is then compiled against the original and kept only if
        # it renders the same. See style_extract.
        passages = []
        for path in state["style_exemplars"]:
            try:
                passages += extract_style(
                    Path(path), output_root / "style",
                    backend=args.backend, model=args.model,
                    log_dir=output_root / LOG_SUBDIR)
            except Exception as exc:
                print(f"  Warning: style extraction failed for "
                      f"{Path(path).name}: {exc}")
        state["style_passages"] = passages
        save_state(output_root, state)
        if not passages:
            print("  Warning: no style passage survived verification — the "
                  "notes will be written without an exemplar.")
    title = state.get("title") or "Lecture Notes"

    if args.regen:
        if args.regen in state["sections"]:
            print(f"Dropping cached section for '{args.regen}' (will regenerate).")
            del state["sections"][args.regen]
        else:
            print(f"Warning: --regen '{args.regen}' not found in state; will generate normally.")

    # ------------------------------------------------------------------
    # Step 0: Pre-fetch / load references
    # ------------------------------------------------------------------
    # Build a dict keyed by original input for deduplication.
    state.setdefault("references", [])
    ref_by_original = {r["original"]: r for r in state["references"]}

    for url_or_id in args.reference:
        if url_or_id not in ref_by_original:
            print(f"Fetching reference: {url_or_id}")
            try:
                ref = fetch_reference(url_or_id, refs_dir)
                meta = {k: v for k, v in ref.items() if k != "text"}
                state["references"].append(meta)
                ref_by_original[url_or_id] = meta
                save_state(output_root, state)
                print(f"  → \"{ref['title']}\"")
            except Exception as exc:
                print(f"  Warning: could not fetch {url_or_id}: {exc}")
        else:
            print(f"Reference already cached: {url_or_id}")

    # Load text for all state references (may be read from disk)
    loaded_refs = [
        load_cached_reference(dict(r), output_root)
        for r in state["references"]
    ]

    # ------------------------------------------------------------------
    # Step 1: Ingest all lectures
    # ------------------------------------------------------------------
    print(f"=== Step 1: Ingest ({len(inputs)} lecture(s)) ===")
    lecture_dirs: list[Path] = []
    pending: list[tuple[Path, dict, str]] = []

    for i, src in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] {src}")
        if args.skip_ingest:
            d = Path(src).resolve()
            if not (d / "transcript.json").exists():
                sys.exit(f"--skip-ingest: no transcript.json in {d}")
            lecture_dirs.append(d)
        else:
            d, meta = prepare_lecture(src, output_root, args.proxy,
                                      args.download == "modal")
            lecture_dirs.append(d)
            if meta is not None:
                pending.append((d, meta, src))

    if pending:
        transcribe_pending(pending, whisper_model, language,
                           args.transcribe, args.modal_fetch)
    warn_language_mismatch(lecture_dirs, language)
    if args.boards_legacy:
        print("Note: --boards is now the default; the flag does nothing. "
              "Use --no-boards to skip segmentation.")
    if args.boards:
        prepare_boards(lecture_dirs, color=args.boards_color)

    # ------------------------------------------------------------------
    # Step 2: Generate LaTeX sections lecture by lecture
    # ------------------------------------------------------------------
    cached  = [d for d in lecture_dirs if d.name in state["sections"]]
    pending = [d for d in lecture_dirs if d.name not in state["sections"]]
    print(f"\n=== Step 2: Generate notes "
          f"({len(cached)} cached, {len(pending)} to write) ===")
    run_usage = Usage()

    # Who is speaking, asked before any model runs — one sitting, like the
    # follow-up questions. Only for lectures about to be written: a name
    # cannot change notes that already exist (--regen drops them, so those
    # get asked again).
    lecturers = resolve_lecturers(
        lecture_dirs, state, forced=args.lecturer, ask_for=pending,
        backend=args.backend, model=args.model, frame_model=args.frame_model,
        work_dir=output_root / "lecturers", log_dir=output_root / LOG_SUBDIR)
    save_state(output_root, state)

    # Cached section bodies bake in "Lecture N" headings and \label{lec:N}
    # labels. If the input order changed (insertion/reorder), those numbers no
    # longer match the assembled document — warn instead of silently
    # misnumbering.
    mismatched = [
        (d.name, state["sections"][d.name]["lecture_num"], i)
        for i, d in enumerate(lecture_dirs, 1)
        if d.name in state["sections"]
        and state["sections"][d.name]["lecture_num"] != i
    ]
    if mismatched:
        print("\nWARNING: the lecture order has changed since these sections "
              "were generated:")
        for name, old_num, new_num in mismatched:
            print(f"  {name}: written as Lecture {old_num}, "
                  f"now at position {new_num}")
        print("  Their cached bodies keep the old numbering and labels. "
              "Use --regen <slug> to rewrite them.\n")

    unwritten: list[str] = []
    for i, ldir in enumerate(lecture_dirs, 1):
        key = ldir.name
        if key in state["sections"]:
            print(f"\n[{i}/{len(lecture_dirs)}] {ldir.name} — using cached section.")
        else:
            # Prior context: recent lectures in full, older ones summarized.
            prior_latex = build_prior_context(state, lecture_dirs, i)

            print(f"\n[{i}/{len(lecture_dirs)}] Writing lecture {i} ({ldir.name})…",
                  end="", flush=True)
            try:
                section, new_corrections, new_preamble, usage = \
                    generate_section(
                        i, ldir, prior_latex, state.get("corrections", {}),
                        loaded_refs, refs_dir,
                        state.get("preamble_additions", []),
                        state.get("style_passages", []),
                        lecturer=lecturers.get(key),
                        backend=args.backend, model=args.model,
                        frame_model=args.frame_model, wait=args.wait,
                    )
            except SectionNotWritten as exc:
                # Not cached, so a rerun retries it — and the run carries on
                # rather than losing the other 23 lectures to one bad turn.
                print(f" FAILED: {exc}")
                unwritten.append(key)
                continue
            run_usage.add(usage)
            summary_path = ldir / "summary.md"
            summary = (summary_path.read_text().strip()
                       if summary_path.exists() else "")
            if not summary:
                print(" (no summary.md written — full text will be used as "
                      "context for later lectures)", end="")
            state["sections"][key] = {"lecture_num": i,
                                      "body": section.strip(),
                                      "summary": summary,
                                      "usage": usage.to_dict()}
            state.setdefault("corrections", {}).update(new_corrections)
            for entry in new_preamble:
                if entry not in state.setdefault("preamble_additions", []):
                    state["preamble_additions"].append(entry)
            save_state(output_root, state)
            notes = []
            if new_corrections:
                notes.append(f"{len(new_corrections)} correction(s)")
            if new_preamble:
                notes.append(f"{len(new_preamble)} preamble addition(s)")
            print(f" done." + (f" ({', '.join(notes)})" if notes else ""))

            if not args.no_verify:
                verify_lecture(output_root, state, key, args.backend,
                               args.model, args.frame_model, run_usage)

    if unwritten:
        # Loud, and at the end where it will be seen: the assembled document
        # is missing these lectures entirely.
        print(f"\n*** {len(unwritten)} lecture(s) were NOT written and are "
              f"missing from the course: {', '.join(unwritten)}")
        print("*** Rerun to retry them (they were deliberately not cached).")

    # ------------------------------------------------------------------
    # Step 3: Assemble final document (always, so it reflects latest state)
    # ------------------------------------------------------------------
    print(f"\n=== Step 3: Assembling {output_tex} ===")
    # write_document prefers section.tex on disk, so hand edits survive; if
    # the result does not compile, the errors go back to their lectures.
    assemble_from_state(output_root, state, output_tex,
                        slugs=[d.name for d in lecture_dirs], title=title,
                        backend=args.backend, model=args.model,
                        frame_model=args.frame_model,
                        fix_rounds=args.latex_fix_rounds, run_usage=run_usage)
    if args.export:
        do_export(state, [d.name for d in lecture_dirs], title)
    print_usage_totals(run_usage, state)


if __name__ == "__main__":
    main()
