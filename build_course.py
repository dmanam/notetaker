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
  --language LANG       Force transcript language code (default: auto-detect)
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
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Import helpers from sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from claude_backend import (BACKENDS, collect_followup_answers, count_todos,
                            run_agent)
from ingest import (download_video, expand_playlist, extract_audio, is_url,
                    resolve_whisper_model, slug, transcribe_batch,
                    unique_lecture_dir)
from fetch import describe_assets, fetch_reference, load_cached_reference
from latex_check import report_latex_check
from media import find_video, format_transcript
from notes_tools import NotesToolContext

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

Rules:
- Begin each lecture with \section{Lecture N: <descriptive title>} and add
  \label{lec:N} immediately after it.
- Use the pre-defined theorem environments: theorem, lemma, proposition,
  corollary, definition, example, exercise, remark, notation.
- Number displayed equations with \begin{equation}\label{eq:...} when you
  anticipate referencing them; use \[ ... \] otherwise.
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

def prepare_lecture(source: str, output_root: Path) -> tuple[Path, dict | None]:
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

        video_path, meta = download_video(source, tmp_dir)

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
                       backend: str) -> None:
    """Transcribe the pending lectures (in parallel on Modal) and write each
    transcript.json. pending entries are (lecture_dir, meta, source)."""
    where = ("in parallel on Modal" if backend == "modal" else "locally")
    print(f"\nTranscribing {len(pending)} lecture(s) {where}…")
    jobs = [
        (d / "audio.wav",
         src if (backend == "modal" and is_url(src)) else None)
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


# ---------------------------------------------------------------------------
# Transcript corrections
# ---------------------------------------------------------------------------

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
) -> tuple[str, dict[str, str], list[str]]:
    """
    Call Claude to write the LaTeX section for this lecture.
    references is a list of already-loaded {url, title, text} dicts.
    refs_dir is where new fetches (via the fetch_document tool) are cached.
    Returns (section_text, new_corrections, new_preamble_additions).
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
    return section_text, ctx.new_corrections, ctx.new_preamble_additions


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


def assemble_from_state(output_root: Path, state: dict,
                        output_tex: Path) -> None:
    """Assemble the course document from state (and on-disk section files),
    ordered by lecture_num (used by --answer runs, where no input list is
    given)."""
    slugs = sorted(state["sections"],
                   key=lambda s: state["sections"][s]["lecture_num"])
    body_parts = [current_body(output_root, state, s) for s in slugs]
    extra_preamble = "\n".join(state.get("preamble_additions", []))
    preamble = PREAMBLE_TEMPLATE % {
        "title": state.get("title") or "Lecture Notes",
        "extra_preamble": extra_preamble,
    }
    full_doc = (preamble + "\n\n" + "\n\n".join(body_parts)
                + "\n\n" + CLOSING + "\n")
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(full_doc)
    save_state(output_root, state)
    print(f"Written: {output_tex}  ({len(full_doc):,} chars)")
    report_latex_check(output_tex)


# ---------------------------------------------------------------------------
# Follow-up: answer open questions / resolve todos for one lecture
# ---------------------------------------------------------------------------

def lecture_duration(lecture_dir: Path) -> float:
    transcript_path = lecture_dir / "transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            segments = json.load(f)["segments"]
        return segments[-1]["end"] if segments else 0.0
    return 0.0


def ensure_section_file(output_root: Path, state: dict, slug: str) -> Path:
    """The on-disk section file (recreated from state if missing)."""
    section_file = (output_root / slug / "section.tex").resolve()
    if not section_file.exists():
        section_file.write_text(state["sections"][slug]["body"])
    return section_file


def propagate_revision(output_root: Path, state: dict, revised_slug: str,
                       answers_block: str | None,
                       new_corrections: dict[str, str],
                       backend: str, model: str | None,
                       frame_model: str | None, wait: bool) -> None:
    """After lecture N is revised in a follow-up, sweep every later lecture
    for inherited material affected by the changes (restated definitions,
    notation, cross-references, recurring mishearings)."""
    revised_num = state["sections"][revised_slug]["lecture_num"]
    later = sorted((s["lecture_num"], sl)
                   for sl, s in state["sections"].items()
                   if s["lecture_num"] > revised_num)
    changes = []
    if answers_block:
        changes.append(f"Answers applied to Lecture {revised_num}:\n"
                       f"{answers_block}")
    if new_corrections:
        lines = "\n".join(f'- "{w}" → "{r}"'
                          for w, r in new_corrections.items())
        changes.append(f"Newly confirmed transcript corrections:\n{lines}")
    if not later or not changes:
        return
    changes_text = "\n\n".join(changes)

    print(f"\nPropagating the revision to {len(later)} later lecture(s)…")
    for num2, slug2 in later:
        lecture_dir2 = output_root / slug2
        section_file2 = ensure_section_file(output_root, state, slug2)
        ctx2 = NotesToolContext(
            refs_dir=output_root / "references",
            video_path=find_video(lecture_dir2),
            total_duration=lecture_duration(lecture_dir2),
            enable_preamble=True,
            existing_preamble=list(state.get("preamble_additions", [])),
            read_roots=[output_root.resolve()],
        )
        user_text = (
            f"Lecture {revised_num} of this series was just revised in a "
            f"follow-up:\n\n{changes_text}\n\n"
            f"Your section is Lecture {num2}, in `{section_file2}`. Read it "
            f"and update anything affected by the changes above — material "
            f"inherited from Lecture {revised_num} (restated definitions, "
            f"notation, cross-references) and any occurrence of the misheard "
            f"phrases (use judgment: fix only genuine mishearings, not "
            f"legitimate uses of the same words). If nothing applies, make "
            f"no edits and reply 'no changes needed'."
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
        summary_path2 = lecture_dir2 / "summary.md"
        if summary_path2.exists():
            state["sections"][slug2]["summary"] = \
                summary_path2.read_text().strip()
        state.setdefault("corrections", {}).update(ctx2.new_corrections)
        for entry in ctx2.new_preamble_additions:
            if entry not in state.setdefault("preamble_additions", []):
                state["preamble_additions"].append(entry)
        save_state(output_root, state)


def answer_lecture(output_root: Path, slug: str, output_tex: Path,
                   backend: str, model: str | None,
                   frame_model: str | None, wait: bool = False,
                   propagate: bool = True) -> None:
    state = load_state(output_root)
    if slug not in state["sections"]:
        known = ", ".join(sorted(state["sections"])) or "(none)"
        sys.exit(f"No generated section for '{slug}'. Known: {known}")

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
    )

    answers_block = collect_followup_answers(ctx, section_file)
    todos = count_todos(section_file.read_text())
    if not answers_block and todos == 0:
        print("No open questions and no \\todo markers — nothing to do.")
        return

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
    summary_path = lecture_dir / "summary.md"
    if summary_path.exists():
        state["sections"][slug]["summary"] = summary_path.read_text().strip()
    state.setdefault("corrections", {}).update(ctx.new_corrections)
    for entry in ctx.new_preamble_additions:
        if entry not in state.setdefault("preamble_additions", []):
            state["preamble_additions"].append(entry)
    save_state(output_root, state)
    print(f"\nRevised lecture '{slug}'.")

    if propagate:
        propagate_revision(output_root, state, slug, answers_block,
                           ctx.new_corrections, backend, model,
                           frame_model, wait)

    print("Reassembling the course document.")
    assemble_from_state(output_root, state, output_tex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        urls = expand_playlist(src)
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
    parser.add_argument("--language", default=None,
                        help="Force transcript language, e.g. 'en'")
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
    parser.add_argument("--no-propagate", action="store_true",
                        help="With --answer: skip updating later lectures "
                             "after the revision.")
    args = parser.parse_args()
    whisper_model = resolve_whisper_model(args.whisper_model, args.transcribe)

    if args.answer:
        output_root = Path(args.output_dir)
        output_tex = (Path(args.output) if args.output
                      else output_root / "course.tex")
        answer_lecture(output_root, args.answer, output_tex,
                       args.backend, args.model, args.frame_model, args.wait,
                       propagate=not args.no_propagate)
        return

    inputs = parse_inputs(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
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
            d, meta = prepare_lecture(src, output_root)
            lecture_dirs.append(d)
            if meta is not None:
                pending.append((d, meta, src))

    if pending:
        transcribe_pending(pending, whisper_model, args.language,
                           args.transcribe)

    # ------------------------------------------------------------------
    # Step 2: Generate LaTeX sections lecture by lecture
    # ------------------------------------------------------------------
    cached  = [d for d in lecture_dirs if d.name in state["sections"]]
    pending = [d for d in lecture_dirs if d.name not in state["sections"]]
    print(f"\n=== Step 2: Generate notes "
          f"({len(cached)} cached, {len(pending)} to write) ===")

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
            section, new_corrections, new_preamble = generate_section(
                i, ldir, prior_latex, state.get("corrections", {}),
                loaded_refs, refs_dir, state.get("preamble_additions", []),
                backend=args.backend, model=args.model,
                frame_model=args.frame_model, wait=args.wait,
            )
            summary_path = ldir / "summary.md"
            summary = (summary_path.read_text().strip()
                       if summary_path.exists() else "")
            if not summary:
                print(" (no summary.md written — full text will be used as "
                      "context for later lectures)", end="")
            state["sections"][key] = {"lecture_num": i,
                                      "body": section.strip(),
                                      "summary": summary}
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

    # ------------------------------------------------------------------
    # Step 3: Assemble final document (always, so it reflects latest state)
    # ------------------------------------------------------------------
    print(f"\n=== Step 3: Assembling {output_tex} ===")
    # current_body prefers section.tex on disk, so hand edits survive.
    body_parts = [current_body(output_root, state, d.name)
                  for d in lecture_dirs]
    extra_preamble = "\n".join(state.get("preamble_additions", []))
    preamble = PREAMBLE_TEMPLATE % {"title": title, "extra_preamble": extra_preamble}
    full_doc = preamble + "\n\n" + "\n\n".join(body_parts) + "\n\n" + CLOSING + "\n"

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(full_doc)
    save_state(output_root, state)
    print(f"Written: {output_tex}  ({len(full_doc):,} chars)")
    report_latex_check(output_tex)


if __name__ == "__main__":
    main()
