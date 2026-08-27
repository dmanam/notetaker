#!/usr/bin/env python3
"""
generate_notes.py — Turn a lecture transcript into typeset LaTeX notes.

Loads the transcript produced by ingest.py and runs an agent with a
`get_frame` tool so it can inspect any video frame for visual context
(board diagrams, slides, written notation). The agent writes the LaTeX
directly to the output file with its file tools.

Three backends are supported (see claude_backend.py):
  --backend subscription  (default) Claude via the Claude Agent SDK + Claude
                          Code CLI, authenticated with your Claude subscription
  --backend codex         GPT via the OpenAI Codex CLI, authenticated with
                          your ChatGPT subscription
  --backend api           the Anthropic API (ANTHROPIC_API_KEY, pay-per-token)

Usage:
  python generate_notes.py <lecture-dir> [--output notes.tex] [--title "Lecture 1"]
                           [--backend {subscription,codex,api}] [--model MODEL]

<lecture-dir> is the output directory produced by ingest.py, e.g.:
  output/my-lecture-title/

It must contain:
  transcript.json   (segment-level transcript with timestamps)
  info.json         (video metadata)
and, optionally:
  video.*           (any video file — required for frame extraction)
  audio.wav         (not used here)
"""

import argparse
import json
import sys
from pathlib import Path

from bibliography import (BIB_FILENAME, attach_to_document,
                          tidy_bibliography)
from timestamps import attach_macro, read_video_id
from claude_backend import (BACKENDS, collect_followup_answers, count_todos,
                            mark_answers_applied, run_agent)
from latex_check import check_latex, print_errors
from fetch import describe_assets, fetch_reference
from instructions import (ASK_USER_RULE, ASR_INSTRUCTION, CLARIFY_RULE,
                          CROSSREF_RULE, DISFLUENCY_RULE, DISPLAY_RULES,
                          FIDELITY_INSTRUCTION, FRAMES_RULE,
                          HOUSE_STYLE_INSTRUCTION, MACRO_BRACING_RULE,
                          TIMESTAMP_RULE, TODO_RULE, cite_rule,
                          diagram_rules)
from media import find_video, format_transcript
from notes_tools import (NotesToolContext, REGISTER_INSTRUCTION,
                         style_exemplar_block)
from lecturer import (ATTRIBUTION_INSTRUCTION, lecturer_note,
                      resolve as resolve_lecturers)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (r"""You are an expert mathematical note-taker. Your task is to convert
a raw lecture transcript into polished, well-structured LaTeX lecture notes in the
style of Stanford/Berkeley graduate math course notes (e.g. the style at
https://math.berkeley.edu/~fengt/stanford_course.html).

""" + ASR_INSTRUCTION + "\n\n" + FIDELITY_INSTRUCTION + r"""

Guidelines:
- Produce a *complete*, standalone LaTeX document with a preamble.
- Start with at least: amsmath, amsthm, amssymb, geometry (1in margins),
  microtype, parskip, tikz, tikz-cd, todonotes, then hyperref, and
  cleveref last (it must load after hyperref). Add any further
  \usepackage{...}, \newcommand{...}, \DeclareMathOperator{...}, or
  \newtheorem{...} declarations that the content requires directly in the
  preamble.
- Define theorem environments: theorem, lemma, proposition, corollary,
  definition, example, remark, proof (use amsthm).
- Organize the content into sections and subsections mirroring the lecture's
  logical flow.
- Render all mathematics properly in LaTeX: inline math with $...$,
  displayed equations with \[ ... \] or align environments.
- Put a \label{} on anything the notes refer back to — every theorem, lemma,
  proposition, corollary, definition and numbered equation you expect to
  cite. Give labels meaningful names: \label{thm:tilting-equivalence}, not
  \label{thm:1}.
""" + TIMESTAMP_RULE + "\n" + CROSSREF_RULE + "\n" + FRAMES_RULE + "\n" + CLARIFY_RULE + r"""
- Define macros with care.
""" + MACRO_BRACING_RULE + "\n" + cite_rule(shared=False) + "\n" \
    + ASK_USER_RULE + "\n" \
    + TODO_RULE + r""" (Load todonotes in the preamble
  with \usepackage[colorinlistoftodos]{todonotes}.)
""" + diagram_rules(board_tools=False) + "\n" + DISPLAY_RULES + "\n" \
    + DISFLUENCY_RULE + r"""
- The transcript provides timestamps [hh:mm:ss] before each segment. Use them
  to decide when to call get_frame, to stamp any question you queue for the
  user, and to write the margin marks described above.

Write the complete LaTeX document (starting with \documentclass) to the output
file named in the task instructions. Do not put the LaTeX source in your reply
text.""")

SYSTEM_PROMPT += (REGISTER_INSTRUCTION + HOUSE_STYLE_INSTRUCTION
                  + ATTRIBUTION_INSTRUCTION)


# ---------------------------------------------------------------------------
# Compile check, with the model fixing what it broke
# ---------------------------------------------------------------------------

def check_and_fix(output_path: Path, ctx_factory, backend: str,
                  model: str | None, frame_model: str | None,
                  fix_rounds: int = 2) -> None:
    """Compile-check the notes; on failure hand the errors back to the model
    and re-check, up to fix_rounds times."""
    for attempt in range(fix_rounds + 1):
        # Before every compile, not once at the end: the model is told to
        # write the \cite and \ts marks but none of the machinery behind
        # them, so if a fix round rewrites the preamble the \addbibresource
        # and the \ts definition go with it. Every \cite then turns into a
        # "Citation undefined" and every mark into an undefined control
        # sequence, which the next round tries to fix by deleting them.
        tidy_bibliography(output_path.parent / BIB_FILENAME)
        if attach_to_document(output_path, output_path.parent / BIB_FILENAME):
            print(f"  Wired {BIB_FILENAME} into {output_path.name} "
                  f"(biblatex + \\printbibliography).")
        if attach_macro(output_path,
                         read_video_id(output_path.parent)):
            print(f"  Defined \\ts in {output_path.name} "
                  f"(margin timestamps).")
        errors = check_latex(output_path)
        if errors is None:
            print("(no LaTeX toolchain found on PATH — skipping compile check)")
            return
        if not errors:
            print(f"Compile check OK: {output_path.name}")
            return
        print_errors(output_path, errors)
        if attempt == fix_rounds:
            break
        lines = output_path.read_text().splitlines()
        listed = []
        for err in errors:
            item = f"- {err.message}"
            if err.line is not None:
                src = (lines[err.line - 1].strip()
                       if 0 < err.line <= len(lines) else "")
                item += f"\n  at line {err.line}" + (f": {src}" if src else "")
            if err.detail:
                item += "\n  LaTeX said:\n" + "\n".join(
                    "    " + ln for ln in err.detail.splitlines())
            listed.append(item)
        print(f"\nFixing (round {attempt + 1}/{fix_rounds})…", flush=True)
        ctx = ctx_factory()
        run_agent(
            system_prompt=SYSTEM_PROMPT,
            user_text=(
                f"`{output_path}` does not compile:\n\n" + "\n".join(listed)
                + "\n\nRead the file and fix these errors. Keep the "
                  "mathematics exactly as it is — you are correcting LaTeX, "
                  "not rewriting content. If a macro or environment is "
                  "genuinely missing, add the definition (or the package) to "
                  "the preamble. Edit the file in place."),
            ctx=ctx,
            output_file=output_path,
            backend=backend,
            model=model,
            frame_model=frame_model,
            revise=True,
            role="fix-latex", log_dir=output_path.parent / "logs",
        )
    print("  Remaining errors need a manual look "
          "(or another --latex-fix-rounds pass).")


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(lecture_dir: Path, title: str | None, output_path: Path,
             references: list[dict] | None = None,
             backend: str = "subscription", model: str | None = None,
             frame_model: str | None = None, wait: bool = False,
             fix_rounds: int = 2,
             style_exemplars: list | None = None,
             lecturer: str | None = None) -> None:
    # Load transcript
    transcript_path = lecture_dir / "transcript.json"
    if not transcript_path.exists():
        sys.exit(f"transcript.json not found in {lecture_dir}")
    with open(transcript_path) as f:
        data = json.load(f)
    segments = data["segments"]
    meta = data.get("metadata", {})

    if title is None:
        title = meta.get("title", lecture_dir.name)

    transcript_text = format_transcript(segments)
    total_duration = segments[-1]["end"] if segments else 0

    video_path = find_video(lecture_dir)
    if video_path is None:
        print("Warning: no video file found — get_frame tool will be unavailable.")

    # Next to the .tex, because biblatex resolves \addbibresource relative to
    # the document — and the document is the thing that will be moved around.
    bib_file = output_path.parent / BIB_FILENAME

    def make_ctx() -> NotesToolContext:
        return NotesToolContext(
            refs_dir=lecture_dir / "references",
            video_path=video_path,
            total_duration=total_duration,
            transcript_path=transcript_path,
            bib_file=bib_file,
        )

    ctx = make_ctx()

    refs_block = ""
    if references:
        parts = []
        for ref in references:
            parts.append(
                f"--- Reference: {ref['title']} ---\n"
                f"URL: {ref['url']}\n"
                f"{describe_assets(ref, lecture_dir)}\n"
                f"{ref['text']}"
            )
        refs_block = "\n\n".join(parts) + "\n\n"

    user_text = (
        f"{style_exemplar_block(style_exemplars)}"
        f"{refs_block}"
        f"Please write up the following lecture as LaTeX notes.\n\n"
        f"**Title:** {title}\n\n"
        f"{lecturer_note(lecturer)}"
        f"**Transcript:**\n\n{transcript_text}"
    )

    print(f"Sending transcript ({len(segments)} segments, "
          f"{len(transcript_text)} chars) to the {backend} backend…")
    if video_path:
        print(f"Video available for frame extraction: {video_path.name}")

    run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        ctx=ctx,
        output_file=output_path,
        backend=backend,
        model=model,
        frame_model=frame_model,
        wait_for_answers=wait,
        role="write", log_dir=lecture_dir / "logs",
    )

    print(f"\nDone. Frame requests: {ctx.frame_requests}")
    print(f"LaTeX saved to: {output_path}")
    check_and_fix(output_path, make_ctx, backend, model, frame_model,
                  fix_rounds)


# ---------------------------------------------------------------------------
# Follow-up: answer open questions / resolve remaining todos from a past run
# ---------------------------------------------------------------------------

def answer_followup(lecture_dir: Path, output_path: Path,
                    backend: str, model: str | None,
                    frame_model: str | None, wait: bool = False,
                    fix_rounds: int = 2) -> None:
    if not output_path.exists():
        sys.exit(f"No notes file at {output_path} — generate the notes first.")

    segments: list[dict] = []
    total_duration = 0.0
    transcript_path = lecture_dir / "transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            segments = json.load(f)["segments"]
        total_duration = segments[-1]["end"] if segments else 0.0

    def make_ctx() -> NotesToolContext:
        return NotesToolContext(
            refs_dir=lecture_dir / "references",
            video_path=find_video(lecture_dir),
            total_duration=total_duration,
            transcript_path=lecture_dir / "transcript.json",
            bib_file=output_path.parent / BIB_FILENAME,
        )

    ctx = make_ctx()

    answers_block = collect_followup_answers(ctx, output_path, segments)
    todos = count_todos(output_path.read_text())
    if not answers_block and todos == 0:
        print("No open questions and no \\todo markers — nothing to do.")
        return

    parts = [f"You previously wrote LaTeX lecture notes to `{output_path}`."]
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

    run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_text="\n\n".join(parts),
        ctx=ctx,
        output_file=output_path,
        backend=backend,
        model=model,
        frame_model=frame_model,
        revise=True,
        wait_for_answers=wait,
        role="revise", log_dir=lecture_dir / "logs",
    )
    mark_answers_applied(ctx, output_path)
    print(f"\nRevised: {output_path}")
    check_and_fix(output_path, make_ctx, backend, model, frame_model,
                  fix_rounds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("lecture_dir",
                        help="Directory produced by ingest.py (contains transcript.json)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .tex file (default: <lecture_dir>/notes.tex)")
    parser.add_argument("--title", default=None,
                        help="Lecture title for the LaTeX document")
    parser.add_argument("--reference", metavar="URL_OR_ID", action="append",
                        default=[],
                        help="Pre-load a reference (URL or arXiv ID). May be repeated.")
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
                        help="Block at the end of the run until every queued "
                             "question is answered (default: unanswered "
                             "questions defer to a --answer follow-up run).")
    parser.add_argument("--answer", action="store_true",
                        help="Follow-up mode: answer questions left open by "
                             "an earlier run and have the agent revise the "
                             "existing notes (also sweeps remaining \\todo "
                             "markers).")
    parser.add_argument("--lecturer", metavar="NAME", default=None,
                        help="Who gave this lecture. The notes refer to them "
                             "by surname, as published notes do. Without this "
                             "you are asked once, with a guess from the title "
                             "offered as the default.")
    parser.add_argument("--style-exemplar", metavar="FILE", action="append",
                        default=[],
                        help="A file whose writing style the notes should "
                             "imitate (register only, never content). Repeatable.")
    parser.add_argument("--latex-fix-rounds", type=int, default=2, metavar="N",
                        help="When the notes fail to compile, hand the errors "
                             "back to the model and re-check, up to N times "
                             "(default: 2; 0 to only report errors).")
    args = parser.parse_args()

    lecture_dir = Path(args.lecture_dir).resolve()
    if not lecture_dir.is_dir():
        sys.exit(f"Not a directory: {lecture_dir}")

    output_path = Path(args.output) if args.output else lecture_dir / "notes.tex"
    refs_dir = lecture_dir / "references"

    if args.answer:
        answer_followup(lecture_dir, output_path.resolve(), args.backend,
                        args.model, args.frame_model, args.wait,
                        args.latex_fix_rounds)
        return

    references = []
    for url_or_id in args.reference:
        print(f"Fetching reference: {url_or_id}")
        try:
            ref = fetch_reference(url_or_id, refs_dir)
            references.append(ref)
            print(f"  → \"{ref['title']}\"")
        except Exception as exc:
            print(f"  Warning: could not fetch {url_or_id}: {exc}")

    names = resolve_lecturers([lecture_dir], {}, forced=args.lecturer,
                              backend=args.backend, model=args.model,
                              frame_model=args.frame_model,
                              work_dir=lecture_dir / "lecturers",
                              log_dir=lecture_dir / "logs")
    generate(lecture_dir, args.title, output_path, references,
             args.backend, args.model, args.frame_model, args.wait,
             args.latex_fix_rounds, args.style_exemplar,
             lecturer=names.get(lecture_dir.name))


if __name__ == "__main__":
    main()
