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
                        lecture SLUG and revise that section in place
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
                            open_question_count, questions_file_for, run_agent)
from ingest import (download_video, expand_playlist, extract_audio, is_url,
                    resolve_language, resolve_whisper_model, slug,
                    transcribe_batch, unique_lecture_dir)
from fetch import describe_assets, fetch_reference, load_cached_reference
from bibliography import has_entries
from latex_check import LatexError, check_latex, print_errors, tokens_of
from media import find_video, format_transcript
from notes_tools import NotesToolContext
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
%% Additions requested by Claude during note generation:
%(extra_preamble)s
%% hyperref before cleveref; colorlinks keeps the PDF clean
\usepackage[
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=green!50!black,
  urlcolor=blue!70!black,
  bookmarksnumbered=true,
  pdfusetitle,
]{hyperref}
%% cleveref last — produces "Theorem 2.3", "Definition 1.4", etc. automatically
\usepackage[nameinlink,noabbrev]{cleveref}
%(bibliography)s

%% Theorem environments via thmtools (\declaretheorem registers names with cleveref)
\declaretheorem[numberwithin=section,style=plain]{theorem}
\declaretheorem[sibling=theorem,style=plain]{lemma}
\declaretheorem[sibling=theorem,style=plain]{proposition}
\declaretheorem[sibling=theorem,style=plain]{corollary}
\declaretheorem[sibling=theorem,style=definition]{definition}
\declaretheorem[sibling=theorem,style=definition]{example}
\declaretheorem[sibling=theorem,style=definition]{exercise}
\declaretheorem[sibling=theorem,style=remark]{remark}
\declaretheorem[sibling=theorem,style=remark]{notation}

\title{%(title)s}
\date{}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""

CLOSING = r"\end{document}"

BIB_PREAMBLE = ("%% biblatex loads after hyperref; the running bibliography\n"
                "\\usepackage[backend=biber,style=alphabetic]{biblatex}\n"
                "\\addbibresource{%s}")
BIB_PRINT = "\\printbibliography[heading=bibintoc]"
BIB_FILENAME = "references.bib"

# ---------------------------------------------------------------------------
# System prompt for the note-writing step
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""You are an expert mathematical note-taker writing LaTeX sections for
a math lecture series. A fixed preamble and theorem environments have already
been set up; you output *only* the body content to be appended to the document.

The transcript was produced by automatic speech recognition and may contain errors:
misheared words, mangled technical terms, or nonsensical phrases where the speaker
said something the recogniser could not handle. Treat the transcript as a rough guide,
not a verbatim record. If a passage does not make mathematical sense, it is likely a
transcription error — use the clarify_transcript tool rather than reproducing the
garbled text.

Fidelity. Notes like these fail in characteristic ways, and all of them come
from writing more than the lecture supports:
- Material you add that the lecturer did not say — a justification, an
  "equivalently", a slicker proof, a historical attribution, an illustrative
  example — is where errors concentrate. Add it only where you are certain,
  and never present your own reasoning as the lecturer's. An equivalence is a
  mathematical claim: if the lecturer did not state it, either verify it
  properly or leave it out. If your own gloss genuinely helps, mark it as
  yours ("Editorially: ...") so a reader can weigh it separately.
- Preserve the lecturer's confidence. "I think", "morally speaking", "I
  forgot", "I don't know", "this might be wrong", "I'm not sure how you'd
  define it" are content, not disfluency — keep them. Never convert a hedge
  into an assertion, and never state as settled something the lecturer
  flagged as open, conjectural, or half-remembered.
- A correction supersedes what it corrects. Lecturers correct themselves and
  audiences correct them, sometimes many minutes later. Write what the
  lecture concluded, not what was first said: do not restate a claim that was
  retracted, and do not reuse an example that was refuted as though it still
  supported the point. You have the whole transcript — when a passage sounds
  hesitant or draws a question, read ahead before writing it up.
- Never attribute to the lecturer a reference they did not give. If they only
  gestured at one ("I think there's a paper by X"), anything you cite must
  have existed at the time: check it against the lecture date given in the
  task. A later paper can still be worth citing, but as your own pointer
  ("see also"), never as the work they had in mind.
- A \todo does not license a false statement. Flagging a missing reference
  while asserting the claim is backwards: assert only the part you are sure
  of, and put the uncertainty inside the \todo.

Rules:
- Begin each lecture with \section{Lecture N: <descriptive title>} and add
  \label{lec:N} immediately after it.
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
- Use \cref{label} for ALL cross-references (mid-sentence) and \Cref{label}
  at the start of a sentence. cleveref automatically produces the correct
  type name and number, e.g. "Theorem 2.3", "Definition 1.4", "Lecture 2".
  Never use \ref or \hyperref for cross-references.
- For lecture section labels, write \cref{lec:2} to produce a clickable
  "Section 2" link, or just write "Lecture~2" as plain text if no label exists.
- Whenever the transcript mentions something drawn, written, or shown
  visually, consult the video frames (using the frame tools or subagent
  available to you) so you can transcribe the mathematics accurately.
- Use the clarify_transcript tool when a word or phrase in the transcript seems
  garbled, misheared, or mathematically nonsensical — provide the exact garbled
  text, the surrounding context, and your best guess. Do not reproduce garbled
  text in the notes.
- Use the add_to_preamble tool whenever you need anything in the LaTeX
  preamble that is not already there: \usepackage{...}, \newcommand{...},
  \DeclareMathOperator{...}, \declaretheorem{...}, or any other declaration.
  Call it before writing the body content that depends on it.
  Already in the preamble: geometry, amsmath, amsthm, amssymb, thmtools,
  microtype, parskip, enumitem, todonotes, and the theorem environments
  theorem, lemma, proposition, corollary, definition, example, exercise,
  remark, notation.
  Note: hyperref and cleveref are loaded last and must stay last — additions
  go before them, so do not re-add either of those packages.
- Cite sources with the cite_reference tool: give it an arXiv ID, DOI, or
  URL and it returns a key for \cite{key}, adding the entry to the course's
  shared bibliography (safe to call again for the same source). For arXiv
  IDs and DOIs the metadata is fetched for you; for anything else (lecture
  notes, a book, a web page) also pass title, author, and year — look them
  up in the document itself if you must, since an entry without an author
  cannot get a proper [Sch19]-style citation label. Cite papers and books
  the lecturer names, and references you consulted for a definition or
  notation. Never write bibliography entries, \bibitem, or
  \printbibliography yourself — the bibliography is assembled automatically.
- Use the ask_user tool whenever you are uncertain how to typeset a specific
  symbol or notation — for example, a symbol that requires a niche package,
  non-standard blackboard bold, or field-specific convention you are not
  confident about. Ask instead of silently guessing — then continue
  provisionally with your best rendering (marked with \todo) until the
  answer arrives.
- Use \todo{...} inline to flag any location where you are uncertain about
  mathematical content rather than typesetting: for example, a formula you
  could only partially read from a frame, a logical step that seems incomplete,
  or a passage where your best-effort reconstruction may be wrong. Prefer
  \todo{} over silently guessing; it lets the human reviewer find and fix
  uncertain spots in the compiled PDF. (todonotes is already loaded — do not
  add it via add_to_preamble.)
- Clean up speech disfluencies but preserve the mathematical content faithfully.
- Write only valid LaTeX body content — no \documentclass, no \begin{document},
  no \end{document} — to the output file named in the task instructions. Do
  not put the LaTeX in your reply text."""

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

VERIFY_PROMPT = r"""You are checking a written-up set of LaTeX lecture notes against the
transcript of the lecture they were written from. You did not write them; read
them as a skeptical reader who has the recording to hand.

The question you are asking is NOT "does this read well?" — it reads well. It
is "is each statement here true, and did the lecture actually support it?"
Notes like these are usually faithful in their main content and wrong in the
material added around it, so weight your effort towards explanation rather
than towards the theorems themselves — but do not treat a definition or a
theorem as safe, because added material hides inside them too.

In particular, check EVERY "equivalently", "i.e.", "in other words", "that
is", and "(equivalently, ...)" in the file, wherever it occurs — including in
the middle of a definition, where it wears the definition's authority. Each
one asserts that two conditions are the same, which is a real mathematical
claim and is the single most common place a false statement hides here.
Confirm each such claim independently, and delete any clause you cannot
confirm: the surrounding statement is almost always fine without it.

Look for exactly these, in order:

1. FALSE AS WRITTEN. Any definition, theorem, or proof step that is untrue as
   stated — dropped hypotheses (non-emptiness, finiteness, boundedness),
   quantifiers in the wrong place, a map or implication pointing the wrong
   way, an "equivalently" joining two genuinely different conditions, an
   identity that does not hold, a wrong construction (a pushout where a
   disjoint union is meant). Check the arithmetic and check the adjunctions:
   the exactness of a left adjoint is not the exactness of its right adjoint.
2. SELF-CONTRADICTION. A claim that contradicts another part of the same
   notes. These are strong signals — one of the two is wrong.
3. UNSUPPORTED ADDITIONS. Justifications, equivalences, examples, or
   attributions with no basis in the transcript. Some are correct and
   harmless; some are inventions. Verify each, and treat "the model made this
   up and it happens to be true" differently from "the model made this up and
   it is false".
4. LOST HEDGES AND LOST CORRECTIONS. Places where the lecturer said "I think"
   / "I forgot" / "morally" / "I don't know" and the notes assert flatly; and
   places where the lecturer or the audience corrected something and the
   notes preserve the superseded version, or reuse a refuted example.
5. GARBLE PROPAGATED. Speech-recognition nonsense reproduced as if it were
   mathematics ("very closed maps", "corner terms", "from M to M"). The
   transcript is unreliable, so distinguish three cases: the notes are wrong;
   the notes correctly REPAIRED a garble (leave it alone — that is the system
   working); the notes carried a garble through (fix or flag it).
6. ANACHRONISTIC OR WRONG CITATIONS. A cited work that postdates the lecture
   cannot be what the lecturer meant. Check names and attributions against
   the literature; you have web search and fetch.

Then fix what you found, editing the file in place:
- Fix anything you are confident is wrong, with the smallest edit that makes
  it true. Do not restructure, do not rewrite prose you merely dislike, and
  do not delete correct mathematics.
- Where you suspect a problem but cannot settle it, leave the text and add a
  \todo{...} saying precisely what you doubt. Do not assert and flag: if the
  claim may be false, weaken the claim.
- Preserve every \label{} — later lectures cite them.

Finally, reply with a short report: one line per change made, and one line
per doubt you flagged. If the notes are clean, say so; do not invent work."""


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
    transcript_text = format_transcript(segments)

    video_path = find_video(lecture_dir)
    ctx = NotesToolContext(
        refs_dir=refs_dir,
        video_path=video_path,
        total_duration=total_duration,
        enable_preamble=True,
        existing_preamble=list(existing_preamble_additions),
        read_roots=[refs_dir.parent.resolve()],
        bib_file=refs_dir.parent / BIB_FILENAME,
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
        f"Now write **Lecture {lecture_num}** (source title: \"{title}\").\n\n"
        f"{lecture_provenance(meta)}"
        f"{corrections_note}"
        f"**Transcript:**\n\n{transcript_text}"
    )

    section_text = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        ctx=ctx,
        output_file=lecture_dir / "section.tex",
        backend=backend,
        model=model,
        frame_model=frame_model,
        wait_for_answers=wait,
        summary_file=lecture_dir / "summary.md",
    )

    if ctx.frame_requests:
        print(f"\n    ({ctx.frame_requests} frame(s) fetched)", end="")
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
        }
      }
    Sections are ordered by lecture_num, but stored as a dict keyed by the
    lecture directory name so we can look up whether a lecture is already done.
    """
    path = output_root / STATE_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"title": None, "sections": {}, "corrections": {}, "references": [],
            "preamble_additions": []}


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
    bib_preamble = bib_print = ""
    if has_entries(bib_src):
        # biblatex resolves \addbibresource relative to the .tex file.
        if bib_src.resolve() != (output_tex.parent / BIB_FILENAME).resolve():
            shutil.copy2(bib_src, output_tex.parent / BIB_FILENAME)
        bib_preamble = BIB_PREAMBLE % BIB_FILENAME
        bib_print = "\n\n" + BIB_PRINT

    preamble = PREAMBLE_TEMPLATE % {
        "title": title,
        "extra_preamble": "\n".join(state.get("preamble_additions", [])),
        "bibliography": bib_preamble,
    }

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


def write_document(output_root: Path, state: dict, output_tex: Path,
                   slugs: list[str], title: str | None = None
                   ) -> tuple[str, list[tuple[int, int]]]:
    """Assemble and write the course document. Returns (text, line spans)."""
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
                     frame_model=frame_model, revise=True)
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


def _fix_section(output_root: Path, state: dict, slug: str,
                 errors: list[LatexError], span: tuple[int, int],
                 doc_lines: list[str], backend: str, model: str | None,
                 frame_model: str | None, run_usage: Usage) -> None:
    lecture_dir = output_root / slug
    section_file = ensure_section_file(output_root, state, slug)
    num = state["sections"][slug]["lecture_num"]
    ctx = NotesToolContext(
        refs_dir=output_root / "references",
        video_path=find_video(lecture_dir),
        total_duration=lecture_duration(lecture_dir),
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
            "right one — or the source was never registered: call "
            "cite_reference for it (with title, author and year for anything "
            "that is not an arXiv ID or DOI) and use the key it returns.\n")
    user_text = (
        f"The assembled course document does not compile. These errors come "
        f"from your section, Lecture {num}, in `{section_file}`:\n\n"
        f"{_localize(errors, span, doc_lines)}\n"
        f"{cite_note}\n"
        f"Read the file and fix them. Keep the mathematics exactly as it is — "
        f"you are correcting LaTeX, not rewriting content. Note that the "
        f"preamble is fixed and shared: if a macro or environment is genuinely "
        f"missing, define it with add_to_preamble rather than working around "
        f"it, and remember that the packages listed in your instructions are "
        f"already loaded. Edit the file in place.")
    print(f"\n[latex-fix → Lecture {num} ({slug})] {len(errors)} error(s)",
          flush=True)
    # No summary_file: this pass corrects LaTeX, it does not change what the
    # lecture says, so the existing summary stays valid.
    body = run_agent(system_prompt=SYSTEM_PROMPT, user_text=user_text, ctx=ctx,
                     output_file=section_file, backend=backend, model=model,
                     frame_model=frame_model, revise=True)
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
        errors = check_latex(output_tex)
        if errors is None:
            print("(no LaTeX toolchain found on PATH — skipping compile check)")
            return
        if not errors:
            print(f"Compile check OK: {output_tex.name}")
            return
        print_errors(output_tex, errors)
        if attempt == fix_rounds:
            break

        doc_lines = doc.splitlines()
        by_slug, unattributed = attribute_errors(errors, doc, spans, slugs,
                                                 state, output_root)
        if not by_slug:
            print("  (could not attribute these errors to a lecture — "
                  "leaving them for a manual pass)")
            break
        if unattributed:
            print(f"  ({len(unattributed)} error(s) not attributable to a "
                  f"single lecture — not sent for repair)")
        print(f"\nFixing (round {attempt + 1}/{fix_rounds}): "
              f"{len(errors) - len(unattributed)} error(s) across "
              f"{len(by_slug)} source(s).")
        if PREAMBLE_SLUG in by_slug:
            _fix_preamble(output_root, state, by_slug.pop(PREAMBLE_SLUG),
                          doc_lines, backend, model, frame_model, run_usage)
        span_of = dict(zip(slugs, spans))
        for slug_ in sorted(by_slug,
                            key=lambda s: state["sections"][s]["lecture_num"]):
            _fix_section(output_root, state, slug_, by_slug[slug_],
                         span_of[slug_], doc_lines, backend, model,
                         frame_model, run_usage)
        doc, spans = write_document(output_root, state, output_tex, slugs,
                                    title)

    print("  Remaining errors need a manual look "
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
            enable_preamble=True,
            existing_preamble=list(state.get("preamble_additions", [])),
            read_roots=[output_root.resolve()],
            bib_file=output_root / BIB_FILENAME,
        )
        user_text = (
            f"{revised_list} of this series {'was' if len(relevant) == 1 else 'were'} "
            f"just revised in a follow-up:\n\n{changes_text}\n\n"
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


def revise_lecture(output_root: Path, state: dict, slug: str,
                   backend: str, model: str | None, frame_model: str | None,
                   wait: bool, run_usage: Usage) -> tuple[str | None,
                                                          dict[str, str]]:
    """Re-ask this lecture's open questions and have the agent revise its
    section in place (also sweeping remaining \\todo markers). Returns
    (answers_block, new_corrections); (None, {}) if there was nothing to do."""
    lecture_dir = output_root / slug
    section_file = ensure_section_file(output_root, state, slug)
    total_duration = lecture_duration(lecture_dir)

    ctx = NotesToolContext(
        refs_dir=output_root / "references",
        video_path=find_video(lecture_dir),
        total_duration=total_duration,
        enable_preamble=True,
        existing_preamble=list(state.get("preamble_additions", [])),
        read_roots=[output_root.resolve()],
        bib_file=output_root / BIB_FILENAME,
    )

    answers_block = collect_followup_answers(ctx, section_file,
                                             lecture_segments(lecture_dir))
    todos = count_todos(section_file.read_text())
    if not answers_block and todos == 0:
        print("No open questions and no \\todo markers — nothing to do.")
        return None, {}

    parts = [f"You previously wrote the LaTeX body for Lecture "
             f"{state['sections'][slug]['lecture_num']} to `{section_file}`."]
    if answers_block:
        parts.append("The user has now answered previously open "
                     f"questions:\n\n{answers_block}")
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
    )

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
        enable_preamble=True,
        existing_preamble=list(state.get("preamble_additions", [])),
        read_roots=[output_root.resolve()],
        bib_file=output_root / BIB_FILENAME,
    )
    user_text = (
        f"Check the notes for **Lecture {num}** of this course, in "
        f"`{section_file}`.\n\n"
        f"{lecture_provenance(meta)}"
        f"You may consult the video frames (to check anything read off the "
        f"board) and the bibliography tools, and you may read the other "
        f"lectures under {output_root.resolve()} if a cross-reference needs "
        f"checking.\n\n"
        f"**Transcript:**\n\n{format_transcript(segments)}"
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
    )
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
    (slugs=None): answer open questions, revise, propagate, reassemble."""
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
    changes_by_num: dict[int, str] = {}
    for i, slug in enumerate(slugs, 1):
        num = state["sections"][slug]["lecture_num"]
        if len(slugs) > 1:
            print(f"\n=== [{i}/{len(slugs)}] Lecture {num} ({slug}) ===")
        answers_block, new_corrections = revise_lecture(
            output_root, state, slug, backend, model, frame_model, wait,
            run_usage)
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
                        help="Follow-up mode over the whole course: work "
                             "through every lecture that has open questions "
                             "or \\todo markers, in order, then propagate and "
                             "reassemble.")
    parser.add_argument("--no-propagate", action="store_true",
                        help="With --answer/--answer-all: skip updating later "
                             "lectures after the revisions.")
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
        print_usage_totals(run_usage, state)
        return

    if args.answer or args.answer_all:
        if args.answer and args.answer_all:
            sys.exit("Use either --answer SLUG or --answer-all, not both.")
        output_root = Path(args.output_dir)
        output_tex = (Path(args.output) if args.output
                      else output_root / "course.tex")
        answer_lectures(output_root,
                        [args.answer] if args.answer else None,
                        output_tex, args.backend, args.model,
                        args.frame_model, args.wait,
                        propagate=not args.no_propagate,
                        fix_rounds=args.latex_fix_rounds)
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

    # ------------------------------------------------------------------
    # Step 2: Generate LaTeX sections lecture by lecture
    # ------------------------------------------------------------------
    cached  = [d for d in lecture_dirs if d.name in state["sections"]]
    pending = [d for d in lecture_dirs if d.name not in state["sections"]]
    print(f"\n=== Step 2: Generate notes "
          f"({len(cached)} cached, {len(pending)} to write) ===")
    run_usage = Usage()

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

    for i, ldir in enumerate(lecture_dirs, 1):
        key = ldir.name
        if key in state["sections"]:
            print(f"\n[{i}/{len(lecture_dirs)}] {ldir.name} — using cached section.")
        else:
            # Prior context: recent lectures in full, older ones summarized.
            prior_latex = build_prior_context(state, lecture_dirs, i)

            print(f"\n[{i}/{len(lecture_dirs)}] Writing lecture {i} ({ldir.name})…",
                  end="", flush=True)
            section, new_corrections, new_preamble, usage = generate_section(
                i, ldir, prior_latex, state.get("corrections", {}),
                loaded_refs, refs_dir, state.get("preamble_additions", []),
                backend=args.backend, model=args.model,
                frame_model=args.frame_model, wait=args.wait,
            )
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
    print_usage_totals(run_usage, state)


if __name__ == "__main__":
    main()
