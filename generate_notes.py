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

from claude_backend import (BACKENDS, collect_followup_answers, count_todos,
                            run_agent)
from latex_check import report_latex_check
from fetch import describe_assets, fetch_reference
from media import find_video, format_transcript
from notes_tools import NotesToolContext

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""You are an expert mathematical note-taker. Your task is to convert
a raw lecture transcript into polished, well-structured LaTeX lecture notes in the
style of Stanford/Berkeley graduate math course notes (e.g. the style at
https://math.berkeley.edu/~fengt/stanford_course.html).

The transcript was produced by automatic speech recognition and may contain errors:
misheared words, mangled technical terms, or nonsensical phrases where the speaker
said something the recogniser could not handle. Treat the transcript as a rough guide,
not a verbatim record. If a passage does not make mathematical sense, it is likely a
transcription error — use the clarify_transcript tool rather than reproducing the
garbled text.

Guidelines:
- Produce a *complete*, standalone LaTeX document with a preamble.
- Start with at least: amsmath, amsthm, amssymb, hyperref, geometry
  (1in margins), microtype, parskip. Add any further \usepackage{...},
  \newcommand{...}, \DeclareMathOperator{...}, or \newtheorem{...}
  declarations that the content requires directly in the preamble.
- Define theorem environments: theorem, lemma, proposition, corollary,
  definition, example, remark, proof (use amsthm).
- Organize the content into sections and subsections mirroring the lecture's
  logical flow.
- Render all mathematics properly in LaTeX: inline math with $...$,
  displayed equations with \[ ... \] or align environments.
- When the speaker writes or draws something, consult the video frames at
  that moment (using the frame tools or subagent available to you) and
  transcribe the board/slide content accurately.
- Use the clarify_transcript tool when a word or phrase in the transcript seems
  garbled, misheared, or mathematically nonsensical — provide the exact garbled
  text, the surrounding context, and your best guess. Do not reproduce garbled
  text in the notes.
- Use the ask_user tool whenever you are uncertain how to typeset a specific
  symbol or notation — for example, a symbol that requires a niche package,
  non-standard blackboard bold, or field-specific convention you are not
  confident about. Ask instead of silently guessing — then continue
  provisionally with your best rendering (marked with \todo) until the
  answer arrives.
- Use \todo{...} inline to flag any location where you are uncertain about
  mathematical content rather than typesetting: for example, a formula you
  could only partially read from a frame, a logical step that seems incomplete,
  or a passage where your best-effort reconstruction may be wrong. Include
  \usepackage[colorinlistoftodos]{todonotes} in the preamble. Prefer \todo{}
  over silently guessing; it lets the human reviewer find and fix uncertain
  spots in the compiled PDF.
- Clean up speech disfluencies (um, uh, repetitions) but preserve the
  lecturer's explanations faithfully.
- Add \label{} and \ref{} cross-references where appropriate.
- Do not invent mathematics not present in the lecture.
- The transcript provides timestamps [MM:SS] before each segment. Use them
  only to decide *when* to call get_frame — do not include them in the output.

Write the complete LaTeX document (starting with \documentclass) to the output
file named in the task instructions. Do not put the LaTeX source in your reply
text."""


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(lecture_dir: Path, title: str | None, output_path: Path,
             references: list[dict] | None = None,
             backend: str = "subscription", model: str | None = None,
             frame_model: str | None = None, wait: bool = False) -> None:
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

    ctx = NotesToolContext(
        refs_dir=lecture_dir / "references",
        video_path=video_path,
        total_duration=total_duration,
    )

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
        f"{refs_block}"
        f"Please write up the following lecture as LaTeX notes.\n\n"
        f"**Title:** {title}\n\n"
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
    )

    print(f"\nDone. Frame requests: {ctx.frame_requests}")
    print(f"LaTeX saved to: {output_path}")
    report_latex_check(output_path)


# ---------------------------------------------------------------------------
# Follow-up: answer open questions / resolve remaining todos from a past run
# ---------------------------------------------------------------------------

def answer_followup(lecture_dir: Path, output_path: Path,
                    backend: str, model: str | None,
                    frame_model: str | None, wait: bool = False) -> None:
    if not output_path.exists():
        sys.exit(f"No notes file at {output_path} — generate the notes first.")

    total_duration = 0.0
    transcript_path = lecture_dir / "transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            segments = json.load(f)["segments"]
        total_duration = segments[-1]["end"] if segments else 0.0

    ctx = NotesToolContext(
        refs_dir=lecture_dir / "references",
        video_path=find_video(lecture_dir),
        total_duration=total_duration,
    )

    answers_block = collect_followup_answers(ctx, output_path)
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
    )
    print(f"\nRevised: {output_path}")
    report_latex_check(output_path)


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
    args = parser.parse_args()

    lecture_dir = Path(args.lecture_dir).resolve()
    if not lecture_dir.is_dir():
        sys.exit(f"Not a directory: {lecture_dir}")

    output_path = Path(args.output) if args.output else lecture_dir / "notes.tex"
    refs_dir = lecture_dir / "references"

    if args.answer:
        answer_followup(lecture_dir, output_path.resolve(), args.backend,
                        args.model, args.frame_model, args.wait)
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

    generate(lecture_dir, args.title, output_path, references,
             args.backend, args.model, args.frame_model, args.wait)


if __name__ == "__main__":
    main()
