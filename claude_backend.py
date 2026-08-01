"""
claude_backend.py — Run the note-writing agent against one of three backends.

Backends:
  subscription (default) — Claude, via the Claude Agent SDK driving the local
      Claude Code CLI. Authenticates with your Claude subscription (Pro/Max):
      log in once with `claude`; no API key. The agent writes the output file
      with its native Write/Edit tools and may use WebFetch/WebSearch.
  codex — GPT, via the OpenAI Codex CLI. Authenticates with your ChatGPT
      subscription: log in once with `codex login`. Custom tools are served
      to Codex over a stdio MCP server (notes_mcp_server.py); the agent
      writes the output file with its own file tools and may use web search.
  api — the Anthropic API with the `anthropic` SDK. Requires
      ANTHROPIC_API_KEY and bills per token. The agent writes the output file
      through a write_notes tool and may use the server-side web fetch/search
      tools.

Common contract: run_agent() sends a system prompt plus user message, lets the
agent call the tools defined in notes_tools.py (frame extraction, transcript
clarification, ...), and instructs it to write the final LaTeX to
`output_file` rather than into its reply — chat commentary therefore never
pollutes the .tex output. The file's content is returned. Recorded
corrections, preamble additions, and the frame counter land on the passed
NotesToolContext.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path

from agent_log import start_log
from media import extract_frame, format_timestamp, parse_timestamp
from notes_tools import (FRAME_READER_PROMPT, NotesToolContext, ToolResult,
                         backfill_question_timestamps, build_handlers,
                         build_tools, ensure_broker, format_answers, is_open)
from usage import Usage, format_usage, usage_from_claude_code

BACKENDS = ("subscription", "codex", "api")

API_MODEL = "claude-opus-5"
# Aliases resolved by the respective CLIs:
SUBSCRIPTION_MODEL = "opus"
CODEX_MODEL = "gpt-5.6-sol"

# Cheap models that study video frames on the main model's behalf.
SUBSCRIPTION_FRAME_MODEL = "haiku"
API_FRAME_MODEL = "claude-haiku-4-5"
CODEX_FRAME_MODEL = "gpt-5.6-luna"


ASYNC_QA_INSTRUCTION = """

Questions to the user (ask_user, clarify_transcript) are asynchronous: the
tool queues the question and returns immediately, and the user answers while
you keep working. Do not stop and wait for an answer. Adopt your best
provisional version, mark the spot with
\\todo{awaiting answer #N @ hh:mm:ss}, and continue. Always give the question
a timestamp, copied from the transcript line it arose from: it is how the
user finds the moment in the video, and a question they cannot locate is a
question they cannot answer. Tell ask_user what your provisional choice was,
and clarify_transcript your best guess — the answer may come back in a
follow-up run, where you are a fresh context with no memory of either, and
a reply of "yes, that one" is only usable if you are told what you proposed. Call get_user_answers before you finish, to incorporate answers
that have already arrived; answers that arrive by the end of your pass are
delivered in a follow-up turn, in which you revise the file (apply the
answers and remove the resolved \\todo markers) rather than rewriting it.
Questions the user has not answered (or has deferred) by then stay open —
keep their \\todo markers; a later follow-up run resolves them."""

RESEARCH_INSTRUCTION = """

Web research: you have web search and fetch tools — use them freely and
autonomously, without asking, whenever outside material would improve the
notes: the lecturer names a paper, book, or set of notes; a term or theorem
attribution sounds off; you are unsure of the standard notation, definition,
or convention for something niche or recent. Do not rely on memory for
material you are not certain about — checking is cheap, a wrong definition in
the notes is not. For arXiv papers and PDFs prefer the fetch_document tool; use
web search when you do not have a URL yet. Fetched papers are cached on disk
and the tool result lists the local paths — the TeX source tree, the
rendered PDF, sometimes HTML. Open those files directly with your own file
tools instead of re-fetching: read the source for exact macros and
notation, and open the PDF (view_pdf_page, or your native PDF reading if you
have it) whenever you need the typeset form — in particular to check the
resolved theorem/equation numbers, which raw TeX source cannot show, and
whenever extracted text or HTML looks garbled."""

# How many ask-revise cycles to run after the main pass (each cycle delivers
# outstanding answers; the agent may ask new questions during revision).
MAX_REVISION_ROUNDS = 3

# Tools whose handlers already print their own progress lines.
_SELF_REPORTING_TOOLS = {"fetch_document", "clarify_transcript", "ask_user",
                         "get_user_answers", "get_frame", "view_pdf_page",
                         "add_to_preamble", "analyze_frames"}


def _tool_line(name: str, tool_input) -> str | None:
    """One concise console line for a model tool call, so the user can see
    what the agent is doing. Returns None for tools that report themselves."""
    short = name.removeprefix("mcp__notes__")
    if short in _SELF_REPORTING_TOOLS:
        return None
    arg = ""
    if isinstance(tool_input, dict):
        for key in ("file_path", "path", "query", "url", "url_or_id",
                    "description", "prompt", "pattern", "command", "mode"):
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                arg = value
                break
        else:
            arg = next((v for v in tool_input.values()
                        if isinstance(v, str) and v), "")
    arg = " ".join(arg.split())
    if len(arg) > 70:
        arg = arg[:70] + "…"
    return f"  [{short}{': ' if arg else ''}{arg}]"


# ---------------------------------------------------------------------------
# Question persistence across runs (enables --answer follow-up runs)
# ---------------------------------------------------------------------------

def questions_file_for(output_file: Path) -> Path:
    return output_file.with_name(output_file.name + ".questions.json")


def load_saved_questions(ctx: NotesToolContext, output_file: Path) -> None:
    qf = questions_file_for(output_file)
    if qf.exists():
        d = json.loads(qf.read_text())
        ctx.questions = d.get("questions", [])
        ctx.question_seq = max(d.get("question_seq", 1),
                               ctx.question_seq)


def save_questions(ctx: NotesToolContext, output_file: Path) -> None:
    questions_file_for(output_file).write_text(json.dumps({
        "questions": ctx.questions,
        "question_seq": ctx.question_seq,
    }, indent=2))


def open_question_count(output_file: Path) -> int:
    """How many questions an earlier run left open for this file (no
    prompting — used to survey a course before a follow-up pass)."""
    qf = questions_file_for(output_file)
    if not qf.exists():
        return 0
    try:
        saved = json.loads(qf.read_text()).get("questions", [])
    except (OSError, ValueError):
        return 0
    return sum(1 for q in saved if is_open(q))


def count_todos(text: str) -> int:
    return len(re.findall(r"\\todo\b", text))


def collect_followup_answers(ctx: NotesToolContext, output_file: Path,
                             segments: list[dict] | None = None) -> str | None:
    """Re-ask the questions left open by earlier runs. Returns a formatted
    answers block for the ones the user answered now (re-deferred ones stay
    open), or None if there were no open questions."""
    load_saved_questions(ctx, output_file)
    if segments:
        # Questions queued before timestamps were recorded can still be
        # located in the transcript.
        filled = backfill_question_timestamps(ctx.questions, segments)
        if filled:
            print(f"(recovered timestamps for {filled} earlier question(s))",
                  file=sys.stderr)
    open_qs = [q for q in ctx.questions if is_open(q)]
    # Answers collected by an earlier run that never reached the model — the
    # run was killed between asking and revising. They must be re-delivered,
    # not re-asked: "delivered" only records that we took the answer off the
    # user, "applied" records that the notes were actually updated with it.
    unapplied = [q for q in ctx.questions
                 if not is_open(q) and q.get("answer") is not None
                 and not q.get("applied")]
    if not open_qs and not unapplied:
        return None
    if unapplied:
        print(f"{len(unapplied)} answer(s) from an earlier run were never "
              f"applied — re-delivering them (you will not be asked again).",
              file=sys.stderr, flush=True)
    if open_qs:
        print(f"{len(open_qs)} question(s) left open by earlier runs:",
              file=sys.stderr, flush=True)
    for q in open_qs:
        q["answer"] = None
        q["deferred"] = False
        q["delivered"] = False
    broker = ensure_broker(ctx)
    broker.finish()
    deliverable = list(unapplied)
    for q in broker.drain_new():
        if q.get("deferred"):
            q["delivered"] = False  # stays open for the next follow-up
        else:
            deliverable.append(q)
    save_questions(ctx, output_file)
    deliverable = sorted({q["id"]: q for q in deliverable}.values(),
                         key=lambda q: q["id"])
    return format_answers(deliverable) if deliverable else None


def mark_answers_applied(ctx: NotesToolContext, output_file: Path) -> None:
    """Record that the revision agent has run, so these answers are not
    re-delivered next time. Called only after the agent finishes."""
    changed = False
    for q in ctx.questions:
        if q.get("answer") is not None and not q.get("applied"):
            q["applied"] = True
            changed = True
    if changed:
        save_questions(ctx, output_file)


def _revision_message(items: list[dict]) -> str:
    return (
        "The user has answered your earlier questions:\n\n"
        + format_answers(items)
        + "\n\nRevise the notes file accordingly: apply the corrections, "
          "remove the \\todo markers that are now resolved, and leave "
          "everything else untouched. Reply with a one-line summary of what "
          "you changed."
    )


def run_agent(
    *,
    system_prompt: str,
    user_text: str,
    ctx: NotesToolContext,
    output_file: Path,
    backend: str = "subscription",
    model: str | None = None,
    frame_model: str | None = None,
    max_tokens: int = 64000,
    revise: bool = False,
    wait_for_answers: bool = False,
    summary_file: Path | None = None,
    images: list[tuple[Path, str]] | None = None,
    role: str = "agent",
    log_dir: Path | None = None,
) -> str:
    """Run the agentic loop; return the LaTeX the agent wrote to output_file.

    With revise=True the existing file is kept and the agent is instructed to
    edit it in place (used by --answer follow-up runs).

    By default the run does NOT block on unanswered user questions at the
    end — they are deferred to a follow-up run (--answer). Pass
    wait_for_answers=True to prompt for all of them before finishing."""
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Question continuity across runs (ids, delivered/deferred flags).
    load_saved_questions(ctx, output_file)
    backup = output_file.with_name(output_file.name + ".bak")
    if output_file.exists():
        if revise:
            shutil.copy2(output_file, backup)
        else:
            # Keep the previous version and let us detect the agent's write.
            output_file.rename(backup)

    ctx.usage = Usage()  # filled by the backend; read by callers afterwards
    resolved_model = model or {"subscription": SUBSCRIPTION_MODEL,
                               "codex": CODEX_MODEL}.get(backend, API_MODEL)
    ctx.log = start_log(
        log_dir, role=role, lecture=output_file.parent.name,
        backend=backend, model=resolved_model,
        frame_model=frame_model, revise=revise,
        output_file=str(output_file),
        system_prompt_chars=len(system_prompt),
        user_prompt_chars=len(user_text),
        # The opening lines say which task this was; the body is usually a
        # whole transcript, which does not belong in a log.
        user_prompt_head=user_text[:1500])
    if log_dir and getattr(ctx.log, "path", None):
        # The codex backend runs tools in an MCP subprocess; it needs the
        # path to append to the same trace.
        ctx.log_file = ctx.log.path
    system_prompt = system_prompt + ASYNC_QA_INSTRUCTION + RESEARCH_INSTRUCTION
    full_user = user_text + _write_instruction(backend, output_file, revise)
    if summary_file is not None:
        summary_file = Path(summary_file).resolve()
        ctx.summary_file = summary_file  # picked up by the api backend
        verb = ("Update" if (revise and summary_file.exists()) else "Write")
        full_user += (
            f"\n\n{verb} a compact summary of this lecture for downstream "
            f"models at `{summary_file}`. This summary is the ONLY view of "
            f"this lecture that later lectures get by default, so an omitted "
            f"item is invisible to them. Required structure:\n"
            f"1. Overview: 2–3 sentences on what the lecture covers.\n"
            f"2. Labeled items: one line per \\label in the section — every "
            f"single one — in the form "
            f"`\\label{{key}} (theorem/definition/equation/…): one-line "
            f"statement in LaTeX`. Do not paraphrase away hypotheses that a "
            f"later lecture would need to cite the result correctly.\n"
            f"3. Notation: every symbol, macro, or naming convention "
            f"introduced, with its meaning.\n"
            f"4. Depends on: the labels/results from earlier lectures that "
            f"this lecture uses.\n"
            f"Terse and machine-oriented; no prose polish. Before finishing, "
            f"cross-check against the section file: any \\label present "
            f"there but missing from the summary is a bug — fix it."
        )
        if backend == "api":
            full_user += " Use the write_summary tool."

    if backend == "subscription":
        fallback = _run_subscription(system_prompt, full_user, ctx, output_file,
                                     model or SUBSCRIPTION_MODEL,
                                     frame_model or SUBSCRIPTION_FRAME_MODEL,
                                     wait_for_answers)
    elif backend == "codex":
        fallback = _run_codex(system_prompt, full_user, ctx, output_file,
                              model or CODEX_MODEL,
                              frame_model or CODEX_FRAME_MODEL,
                              wait_for_answers)
    elif backend == "api":
        fallback = _run_api(system_prompt, full_user, ctx, output_file,
                            model or API_MODEL,
                            frame_model or API_FRAME_MODEL, max_tokens,
                            wait_for_answers, images=images)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected one of {BACKENDS})")

    # Stop prompting; anything still unanswered stays open for --answer.
    ensure_broker(ctx).close()

    if output_file.exists() and output_file.stat().st_size > 0:
        text = output_file.read_text()
    else:
        print(f"\nWarning: the agent never wrote {output_file.name}; saving "
              f"its chat output there instead — review it before compiling.",
              file=sys.stderr)
        output_file.write_text(fallback)
        text = fallback

    save_questions(ctx, output_file)
    if ctx.usage.any():
        print(f"\nLLM usage: {format_usage(ctx.usage)}", file=sys.stderr)
    n_open = sum(1 for q in ctx.questions if is_open(q))
    n_todo = count_todos(text)
    ctx.log.close(
        cost=format_usage(ctx.usage) if ctx.usage.any() else "",
        usage=ctx.usage.to_dict(),
        wrote_output=output_file.exists() and output_file.stat().st_size > 0,
        used_fallback=not (output_file.exists()
                           and output_file.stat().st_size > 0),
        output_chars=len(text),
        open_questions=n_open,
        todos=n_todo,
        questions_asked=len(ctx.questions),
        frames=ctx.frame_requests,
        corrections=list(ctx.new_corrections),
        reply=fallback[-600:] if fallback else "")
    if n_open or n_todo:
        print(f"\nNote: {n_open} open question(s) and {n_todo} \\todo "
              f"marker(s) remain (any prompt still on screen was cancelled). "
              f"Resolve them later with a follow-up run (--answer).",
              file=sys.stderr)
    return text


def _write_instruction(backend: str, output_file: Path,
                       revise: bool = False) -> str:
    if revise:
        base = (
            f"\n\n---\n**Output**: revise the existing file `{output_file}` "
            f"in place — read it first, then apply targeted edits rather than "
            f"rewriting the whole file, and do NOT include LaTeX in your "
            f"reply text. Reply with a one-line summary of the edits."
        )
        if backend == "api":
            return base + (" Use read_notes and edit_notes (write_notes only "
                           "if a full rewrite is unavoidable).")
        if backend == "subscription":
            return base + " Use your Read and Edit tools."
        return base
    base = (
        f"\n\n---\n**Output**: write the final LaTeX to the file "
        f"`{output_file}` — do NOT include the LaTeX source in your reply "
        f"text. Build the file incrementally: work through the transcript in "
        f"order, segment by segment (10–15 minutes of lecture at a time), "
        f"appending each completed part before moving on — the final third "
        f"of the lecture deserves the same detail as the first. Once the "
        f"file is complete, reply with a one-line confirmation."
    )
    if backend == "api":
        return base + (
            " Use the write_notes tool: one call with mode=\"overwrite\" "
            "first; if the document does not fit in a single call, continue "
            "with mode=\"append\" calls."
        )
    if backend == "subscription":
        return base + " Use your Write tool (and Edit for corrections)."
    return base  # codex has its own file tools


# ---------------------------------------------------------------------------
# Backend: Anthropic API (pay-per-token, ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def _write_notes_tool(output_file: Path) -> dict:
    return {
        "name": "write_notes",
        "description": (
            f"Write LaTeX to the output file ({output_file.name}). Call with "
            "mode='overwrite' for the first chunk; call again with "
            "mode='append' if the document needs more than one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latex": {"type": "string",
                          "description": "The LaTeX text to write."},
                "mode": {"type": "string", "enum": ["overwrite", "append"],
                         "description": "overwrite (default) or append."},
            },
            "required": ["latex"],
        },
    }


def _analyze_frames_tool(frame_model: str) -> dict:
    return {
        "name": "analyze_frames",
        "description": (
            f"Have a cheaper model ({frame_model}) study video frames and "
            "report what is on the board/slides. Extracts frames at the given "
            "timestamps and sends them — with your context — to the fast "
            "model, which returns a faithful LaTeX transcription of the "
            "visible content. Prefer this over get_frame: frames are "
            "token-expensive for you. Call get_frame directly only when the "
            "report is ambiguous, incomplete, or mathematically implausible "
            "and the passage is important."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamps": {
                    "type": "array",
                    "items": {"type": ["number", "string"]},
                    "description": ("Up to 8 positions in the video, as "
                                    "hh:mm:ss (as the transcript gives them) "
                                    "or seconds. Include a few nearby moments "
                                    "for a clear view of the board."),
                },
                "question": {
                    "type": "string",
                    "description": "What to look for / transcribe.",
                },
                "context": {
                    "type": "string",
                    "description": ("What the transcript says around this "
                                    "moment, so the reader knows the topic "
                                    "and notation."),
                },
            },
            "required": ["timestamps", "question"],
        },
    }


FRAME_DELEGATION_API = """

Frame analysis: prefer the analyze_frames tool — a cheaper model studies the
frames with the context you supply and reports back. Call get_frame directly
only when its report is ambiguous, incomplete, or mathematically implausible
and the passage is important."""


BOARD_LOCATOR_PROMPT = """\
You find things on a photograph of a lecturer's blackboard. You do not read
the mathematics and you do not draw anything — you say WHERE something is.

You are given the still and a description of what is wanted: usually one
diagram, sometimes a formula or a corner of the board. Your job is to return
the tightest box that contains all of it.

The box is fractional, relative to the whole still: x and y are the top-left
corner, width and height the extent, each between 0 and 1.

- Include every part of the thing asked for. A diagram's stray arrow off to
  one side, a label written above it, an annotation in the margin joined to it
  by an arrow — all inside the box. A box that clips an arrow is worse than a
  box that is too big.
- Do not include the whole board out of caution either. The point of the crop
  is that the region arrives at full resolution instead of being shrunk with
  the rest of the slate, and a box covering everything buys nothing.
- Lecture-hall boards are often several sliding panels. If what was asked for
  is on one panel, box that panel, not the frame.
- Use crop_board if you need a closer look to place the edges. That is what it
  is for; it does not sharpen anything, it only crops.

Reply with the box as JSON and nothing else:

  {"x": 0.5, "y": 0.18, "width": 0.48, "height": 0.5, "note": "right panel"}

`note` is one short phrase saying what you boxed, so the reader can tell you
understood the request. If you cannot find what was asked for anywhere on the
board, reply {"error": "..."} saying what you do see instead."""


BOARD_LOCATOR_SUBSCRIPTION = """

The board still is a file on disk; its path is in your instructions. Open it
with the Read tool — that is the only way you get to see it. Your final
message is the JSON box and nothing else; it is parsed, not read."""


DIAGRAM_INSTRUCTION = """

Diagrams, in detail. You draw them yourself — you are the one who knows what
the lecture proves, and an arrow direction is a mathematical claim, not a
typesetting choice.

There are two kinds, and only the first needs the board. A diagram you are
COMPOSING — because a square, a span, a lifting problem or an exact sequence
reads better drawn than described, whether or not the lecturer drew it — you
write straight from the mathematics: skip to step 4, and pass the objects you
intend as `objects` so the check still catches one you meant to include and
dropped. A diagram you are REPRODUCING off a board goes through all of it.
Composing is a presentation decision and is encouraged; it is not licence to
assert a map the lecture does not.

For a diagram off a board:

1. Ask the 'board-locator' subagent (via the Task tool) for the region of the
   board still holding the diagram: give it the still's absolute path and a
   short description of what to box. It returns a JSON box. It is cheap and
   it only finds things; it does not read mathematics.
2. Call crop_board with that box. You get the region back at full resolution
   instead of the whole slate shrunk to fit, which is the difference between
   being able to see an arrowhead and guessing at it. (If you already know
   where the diagram is, skip step 1 and pass the box yourself. A crop cannot
   add detail that is not in the frame, so cropping tighter and tighter past
   the diagram buys nothing.)
3. Before drawing anything, list what is in the crop: first every object,
   then every arrow with its direction, its style (↪ hook, ↠ two heads,
   dashed, dotted), its label and which side the label is on. This is not
   ceremony, and it is not for you — you will pass this list to
   check_diagram, which diffs it against what you actually drew. The mistake
   that survives every other check is the object you never noticed, because
   nothing in your own diagram points at its absence; a declared list is the
   only thing that can point at it. So list what is ON THE BOARD, not what
   you intend to draw. Take directions off the picture, not off what the
   mathematics "ought" to say; if the two disagree, stop, because either you
   misread an arrowhead or the passage you are writing is wrong. Resolve it
   before drawing, and say how in a \todo if you are not certain.
4. Write the tikzcd (objects and arrows) or tikzpicture (things genuinely
   drawn: a space, a region, a covering, a sketch), and call check_diagram
   with the board number, the `objects` you listed in step 3, and their
   `arrows` if you have them. It compiles the snippet on its own — so a
   broken diagram cannot take the whole course build down — refuses any
   diagram that does not contain everything you said was on the board,
   reports structural defects compiling does not catch, hands the render
   back, and gives you a provenance comment. Expect to go round more than
   once. One clean compile is not a match; compiling only means it is valid
   TikZ. Put the provenance comment directly above the diagram in the notes,
   so the drawing can always be checked against the board it came from.
5. Reproduce the mathematics, not the photograph: same objects, arrows,
   labels and directions, but laid out cleanly rather than where they
   happened to fall on the slate. Do not drop a map that is drawn off in a
   corner if the diagram exists in order to lift it.
6. Then smell-test the finished diagram AS MATHEMATICS, forgetting the
   photograph for a moment. Reading chalk fails in specific ways — an
   arrowhead is a few strokes, a superscript is a smudge, an object off to
   one side gets missed — and the mathematics is the one check the picture
   cannot argue with. Ask: could every arrow possibly go between the objects
   it connects? Do the composites compose? Does a lift point from the thing
   being constructed towards the thing it maps into? Does a surjection run
   from the big object onto the small one? Is an "op" or other variance
   decoration consistent with how the functor is used elsewhere in the
   lecture, or did it appear from nowhere? Does the diagram say the same
   thing as the sentence you are about to put above it? If the mathematics
   and your reading of the board disagree, you have misread an arrowhead —
   look again at that arrow specifically, and if it is still not clear,
   follow the mathematics and note the doubt in a \todo. A reversed arrow is
   a false theorem drawn beautifully, and it will pass every check that only
   looks at the picture.

Never write a placeholder comment meaning to come back and insert a diagram
later. Nothing is running in the background; by the time you stop there is no
later, and a comment where a diagram should be compiles silently. If you
genuinely cannot read a diagram off the board, write the content as prose and
mark it with \todo — that is a much cheaper failure than a confident diagram
with an arrow reversed."""


_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}


def _image_block(path: Path) -> dict | None:
    """An image content block for the Messages API, or None if unreadable —
    a missing snapshot must not abort a whole lecture."""
    path = Path(path)
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if not media_type:
        return None
    try:
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        print(f"\nWarning: could not attach {path.name}: {exc}", file=sys.stderr)
        return None
    return {"type": "image",
            "source": {"type": "base64", "media_type": media_type,
                       "data": data}}


def _board_by_id(ctx: NotesToolContext, want) -> dict | None:
    try:
        want = int(want)
    except (TypeError, ValueError):
        return None
    return next((b for b in ctx.boards if b["id"] == want), None)


def _locate_diagram_tool(ctx: NotesToolContext, frame_model: str) -> dict:
    ids = ", ".join(str(b["id"]) for b in ctx.boards)
    return {
        "name": "locate_diagram",
        "description": (
            f"Ask a cheaper model ({frame_model}) where something is on a "
            f"board still, and get that region back cropped at full "
            f"resolution. Use it before drawing a diagram: the whole slate "
            f"is downscaled to fit the vision ceiling, which leaves a chalk "
            f"arrowhead a pixel or two wide, whereas the cropped region "
            f"arrives un-shrunk. You draw the diagram yourself from the "
            f"crop. Boards: {ids}."),
        "input_schema": {
            "type": "object",
            "properties": {
                "board": {"type": ["integer", "string"],
                          "description": "Which board to look at."},
                "find": {
                    "type": "string",
                    "description": (
                        "What to box, in plain terms — 'the commutative "
                        "diagram on the right-hand panel', 'the displayed "
                        "formula under the word Proof'. The locator does not "
                        "read mathematics; describe the thing by where and "
                        "what it looks like."),
                },
            },
            "required": ["board", "find"],
        },
    }


def _locate(client, model: str, ctx: NotesToolContext, board: dict,
            find: str) -> tuple[dict | None, str]:
    """Cheap model → a fractional box on the board still."""
    photo = _image_block(Path(board["path"]))
    if photo is None:
        return None, f"board {board['id']} has no readable still"
    content = [{"type": "text", "text": f"Board {board['id']}:"}, photo,
               {"type": "text", "text": f"Box this: {find}"}]
    reply = client.messages.create(
        model=model, max_tokens=1000, system=BOARD_LOCATOR_PROMPT,
        messages=[{"role": "user", "content": content}])
    ctx.usage.add_anthropic_response(model, reply.usage)
    text = "".join(b.text for b in reply.content if b.type == "text")
    return parse_box(text)


def parse_box(text: str) -> tuple[dict | None, str]:
    """The locator's JSON box out of whatever it actually replied with."""
    match = re.search(r"\{[^{}]*\}", text or "", re.DOTALL)
    if not match:
        return None, (text or "").strip()[:200] or "no box returned"
    try:
        box = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, match.group(0)[:200]
    if box.get("error"):
        return None, str(box["error"])[:200]
    try:
        out = {k: float(box[k]) for k in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None, f"box missing x/y/width/height: {match.group(0)[:150]}"
    out["note"] = str(box.get("note", ""))[:120]
    return out, ""


def _run_api(system_prompt: str, user_text: str, ctx: NotesToolContext,
             output_file: Path, model: str, frame_model: str,
             max_tokens: int, wait_for_answers: bool,
             images: list[tuple[Path, str]] | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    handlers = build_handlers(ctx)

    def write_notes(inp: dict) -> ToolResult:
        mode = inp.get("mode", "overwrite")
        with open(output_file, "w" if mode == "overwrite" else "a") as f:
            f.write(inp["latex"])
        return ToolResult(f"Wrote {len(inp['latex'])} characters ({mode}).")

    def edit_notes(inp: dict) -> ToolResult:
        if not output_file.exists():
            return ToolResult("Error: the notes file does not exist yet — "
                              "use write_notes first.", is_error=True)
        old, new = inp["old_str"], inp["new_str"]
        text = output_file.read_text()
        n = text.count(old)
        if n != 1:
            return ToolResult(
                f"Error: old_str matched {n} times; it must match exactly "
                f"once. Include more surrounding context.", is_error=True)
        output_file.write_text(text.replace(old, new, 1))
        return ToolResult("Edited.")

    def read_notes(inp: dict) -> ToolResult:
        if not output_file.exists():
            return ToolResult("Error: the notes file does not exist yet.",
                              is_error=True)
        return ToolResult(output_file.read_text())

    handlers["write_notes"] = write_notes
    handlers["edit_notes"] = edit_notes
    handlers["read_notes"] = read_notes

    extra_tools = []
    summary_file = getattr(ctx, "summary_file", None)
    if summary_file is not None:
        def write_summary(inp: dict) -> ToolResult:
            Path(summary_file).write_text(inp["text"])
            return ToolResult("Summary written.")
        handlers["write_summary"] = write_summary
        extra_tools.append({
            "name": "write_summary",
            "description": (f"Write the lecture summary to "
                            f"{Path(summary_file).name} (overwrites)."),
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        })

    if ctx.read_roots:
        roots = [Path(r).resolve() for r in ctx.read_roots]

        def read_file(inp: dict) -> ToolResult:
            p = Path(inp["path"]).resolve()
            if not any(p == r or r in p.parents for r in roots):
                return ToolResult(
                    "Error: path is outside the course directory.",
                    is_error=True)
            if not p.is_file():
                return ToolResult(f"Error: no such file: {p}", is_error=True)
            text = p.read_text(errors="replace")
            if len(text) > 150_000:
                text = text[:150_000] + "\n\n[… truncated at 150,000 chars]"
            return ToolResult(text)

        handlers["read_file"] = read_file
        extra_tools.append({
            "name": "read_file",
            "description": ("Read a file from the course directory — e.g. an "
                            "earlier lecture's section.tex when its summary "
                            "is not detailed enough."),
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string",
                                        "description": "Absolute path."}},
                "required": ["path"],
            },
        })

    def analyze_frames(inp: dict) -> ToolResult:
        stamps = [t for t in (parse_timestamp(x) for x in inp["timestamps"])
                  if t is not None][:8]
        if not stamps:
            return ToolResult("Error: no readable timestamps — use hh:mm:ss "
                              "or a number of seconds.", is_error=True)
        print(f"\n  [analyze_frames @ "
              f"{', '.join(format_timestamp(t) for t in stamps)}"
              f" → {frame_model}]", end="", flush=True)
        content = []
        for ts in stamps:
            b64 = extract_frame(ctx.video_path, ts)
            ctx.frame_requests += 1
            if b64:
                content.append({"type": "text",
                                "text": f"Frame at {format_timestamp(ts)}:"})
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": b64},
                })
        if not content:
            return ToolResult("Error: could not extract any of those frames.",
                              is_error=True)
        context = inp.get("context", "")
        content.append({"type": "text", "text": (
            (f"Lecture context: {context}\n\n" if context else "")
            + f"Task: {inp['question']}"
        )})
        reply = client.messages.create(
            model=frame_model,
            # Generous — a full multi-board report with layout notes fits well
            # under this; the cap only guards against repetition loops.
            max_tokens=8000,
            system=FRAME_READER_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        ctx.usage.add_anthropic_response(frame_model, reply.usage)
        text = "".join(b.text for b in reply.content if b.type == "text")
        return ToolResult(f"Frame report ({len(stamps)} frame(s)):\n{text}")

    def locate_diagram(inp: dict) -> ToolResult:
        board = _board_by_id(ctx, inp.get("board"))
        if board is None:
            have = ", ".join(str(b["id"]) for b in ctx.boards) or "none"
            return ToolResult(f"Error: no board {inp.get('board')!r}. "
                              f"Boards for this lecture: {have}.",
                              is_error=True)
        print(f"\n  [locate_diagram board {board['id']} → {frame_model}]",
              end="", flush=True)
        box, why = _locate(client, frame_model, ctx, board, inp["find"])
        if box is None:
            return ToolResult(
                f"The locator could not box that: {why}\nCrop it yourself "
                f"with crop_board if you can see where it is.", is_error=True)
        # Straight on to the crop: the box on its own is of no use to anyone.
        return build_handlers(ctx)["crop_board"]({
            "board": board["id"], "x": box["x"], "y": box["y"],
            "width": box["width"], "height": box["height"]})

    if ctx.boards and ctx.diagrams_dir is not None:
        handlers["locate_diagram"] = locate_diagram
        extra_tools.append(_locate_diagram_tool(ctx, frame_model))
        system_prompt = system_prompt + DIAGRAM_INSTRUCTION

    tools = build_tools(ctx) + extra_tools + [
        _write_notes_tool(output_file),
        {
            "name": "edit_notes",
            "description": (f"Edit the output file ({output_file.name}) by "
                            "exact string replacement. old_str must match the "
                            "current file contents exactly once."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["old_str", "new_str"],
            },
        },
        {
            "name": "read_notes",
            "description": (f"Read the current contents of the output file "
                            f"({output_file.name})."),
            "input_schema": {"type": "object", "properties": {}},
        },
        # Server-side tools — run on Anthropic's infrastructure.
        {"type": "web_fetch_20260209", "name": "web_fetch"},
        {"type": "web_search_20260209", "name": "web_search"},
    ]
    if ctx.video_path:
        handlers["analyze_frames"] = analyze_frames
        tools.append(_analyze_frames_tool(frame_model))
        system_prompt = system_prompt + FRAME_DELEGATION_API
    content: list[dict] = []
    for path, caption in images or []:
        block = _image_block(path)
        if block:                      # a caption, so a bare image is never
            content.append({"type": "text", "text": caption})   # unattributed
            content.append(block)
    content.append({"type": "text", "text": user_text})
    # cache_control on the last block of the (large) initial user turn and on
    # the system prompt, so the whole stable prefix — images included — is
    # cached across tool-call rounds.
    content[-1]["cache_control"] = {"type": "ephemeral"}
    messages = [{"role": "user", "content": content}]

    broker = ensure_broker(ctx)
    revision_rounds = 0
    truncations = 0
    collected = ""
    while True:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        ) as stream:
            wrote_text = False
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                wrote_text = True
            if wrote_text:
                print(flush=True)
            response = stream.get_final_message()

        ctx.usage.add_anthropic_response(model, response.usage)
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "text":
                collected += block.text
                if block.text.strip():
                    ctx.log.event("say", text=block.text.strip())
            elif block.type in ("tool_use", "server_tool_use"):
                line = _tool_line(block.name, block.input)
                if line:
                    print(line, flush=True)
                if block.name not in handlers:   # ours are logged in-handler
                    ctx.log.tool(block.name, block.input)

        # Execute any tool calls present, regardless of stop_reason (a
        # max_tokens-truncated turn may still carry complete tool_use blocks).
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if tool_uses:
            tool_results = []
            for block in tool_uses:
                handler = handlers.get(block.name)
                if handler is None:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}",
                        "is_error": True,
                    })
                    continue
                result = handler(block.input)
                tr = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.content,
                }
                if result.is_error:
                    tr["is_error"] = True
                tool_results.append(tr)
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "pause_turn":
            # Server-side tool loop paused a long turn; re-send to resume.
            continue

        if response.stop_reason == "max_tokens":
            truncations += 1
            if truncations > 10:
                print("\nToo many truncated turns; giving up on continuation.")
                break
            print("\n  [output-token limit hit — asking the model to continue]",
                  flush=True)
            messages.append({
                "role": "user",
                "content": ("Your previous response was cut off by the "
                            "output-token limit. Continue exactly where you "
                            "stopped; re-issue any interrupted tool call, and "
                            "prefer smaller write_notes chunks (append mode)."),
            })
            continue

        if response.stop_reason != "end_turn":
            print(f"\nUnexpected stop_reason: {response.stop_reason}")
            break

        # Main pass done — deliver answers that have arrived. By default we
        # do not block on unanswered questions (they defer to --answer).
        if wait_for_answers:
            broker.finish()
        items = broker.drain_new()
        if items and revision_rounds < MAX_REVISION_ROUNDS:
            revision_rounds += 1
            print(f"\n  [revision round {revision_rounds}: delivering "
                  f"{len(items)} answer(s)]", flush=True)
            messages.append({"role": "user", "content": _revision_message(items)})
            continue
        break

    return collected


# ---------------------------------------------------------------------------
# Backend: Claude subscription (Claude Agent SDK → Claude Code CLI)
# ---------------------------------------------------------------------------

def _to_mcp_content(content: str | list[dict]) -> list[dict]:
    """Convert a ToolResult's content to MCP content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for b in content:
        if b.get("type") == "image":
            src = b.get("source", {})
            blocks.append({
                "type": "image",
                "data": src.get("data", ""),
                "mimeType": src.get("media_type", "image/jpeg"),
            })
        elif b.get("type") == "text":
            blocks.append({"type": "text", "text": b.get("text", "")})
        else:
            blocks.append({"type": "text", "text": json.dumps(b)})
    return blocks


FRAME_DELEGATION_SUBSCRIPTION = """

Frame analysis: video frames are token-expensive for you. By default, delegate
frame reading to the 'frame-reader' subagent (via the Task tool): tell it the
timestamp(s), what the transcript says around that moment, and what to look
for; it will fetch and study the frames and report the board/slide contents.
Only call get_frame yourself when the subagent's report is ambiguous,
incomplete, or mathematically implausible and the passage is important."""


def _run_subscription(system_prompt: str, user_text: str,
                      ctx: NotesToolContext, output_file: Path,
                      model: str, frame_model: str,
                      wait_for_answers: bool) -> str:
    try:
        import anyio
        from claude_agent_sdk import (
            AgentDefinition,
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            CLINotFoundError,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            create_sdk_mcp_server,
            tool as sdk_tool,
        )
    except ImportError as exc:
        raise SystemExit(
            f"claude-agent-sdk is not installed ({exc}). "
            "Enter the nix devshell (it provides claude-agent-sdk and the "
            "claude CLI), or run with --backend api."
        )

    tools = build_tools(ctx)
    handlers = build_handlers(ctx)

    def make_sdk_tool(spec: dict):
        name = spec["name"]
        handler = handlers[name]

        @sdk_tool(name, spec["description"], spec["input_schema"])
        async def _tool(args: dict) -> dict:
            # Handlers may block (input(), ffmpeg) — keep the event loop alive.
            result = await anyio.to_thread.run_sync(handler, args)
            out = {"content": _to_mcp_content(result.content)}
            if result.is_error:
                out["is_error"] = True
            return out

        return _tool

    server = create_sdk_mcp_server(
        name="notes",
        version="1.0.0",
        tools=[make_sdk_tool(spec) for spec in tools],
    )

    allowed = ([f"mcp__notes__{spec['name']}" for spec in tools]
               + ["Write", "Edit", "Read", "WebFetch", "WebSearch"])
    disallowed = ["Bash", "Glob", "Grep", "TodoWrite", "NotebookEdit"]

    agents = None
    if ctx.video_path:
        # Frames are read by a cheaper subagent by default; the main model
        # escalates to get_frame itself only when the report is insufficient.
        agents = {
            "frame-reader": AgentDefinition(
                description=(
                    "Reads lecture-video frames. Give it the timestamp(s), "
                    "the surrounding transcript, and what to look for; it "
                    "fetches frames via get_frame and reports the board/slide "
                    "contents in LaTeX."
                ),
                prompt=FRAME_READER_PROMPT,
                tools=["mcp__notes__get_frame"],
                mcpServers=["notes"],
                model=frame_model,
            ),
        }
        allowed.append("Task")
        system_prompt = system_prompt + FRAME_DELEGATION_SUBSCRIPTION
    else:
        disallowed.append("Task")

    if ctx.boards and ctx.diagrams_dir is not None:
        # The cheap model only finds things. Reading a diagram off a board is
        # a mathematical judgement — an arrow direction is a claim, not a
        # typesetting choice — and measurement said the cheap model gets it
        # wrong even when magnified. So the main model draws, and this one
        # answers the single question it can answer reliably: where is it.
        agents = dict(agents or {})
        agents["board-locator"] = AgentDefinition(
            description=(
                "Finds a region on a board photograph. Give it the still's "
                "path and a description of what to box; it returns a JSON "
                "box {x, y, width, height} in fractions of the image. It "
                "does not read mathematics and does not draw."
            ),
            prompt=BOARD_LOCATOR_PROMPT + BOARD_LOCATOR_SUBSCRIPTION,
            tools=["mcp__notes__crop_board", "Read"],
            mcpServers=["notes"],
            model=frame_model,
        )
        if "Task" not in allowed:
            allowed.append("Task")
        if "Task" in disallowed:
            disallowed.remove("Task")
        system_prompt = system_prompt + DIAGRAM_INSTRUCTION

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"notes": server},
        cwd=str(output_file.parent),
        # Our tools plus the native file and web tools are auto-approved;
        # everything else built into Claude Code is disallowed.
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        agents=agents,
        model=model,
    )

    broker = ensure_broker(ctx)

    async def _main() -> str:
        parts: list[str] = []
        final_result: list = [None]  # latest ResultMessage (cumulative usage)

        async def drain(client) -> None:
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                            if block.text.strip():
                                print(block.text.strip(), flush=True)
                                ctx.log.event("say", text=block.text.strip())
                        elif isinstance(block, ToolUseBlock):
                            line = _tool_line(block.name, block.input)
                            if line:
                                print(line, flush=True)
                            # Native tools (Read/Edit/WebSearch) never reach
                            # our handlers, so record them here.
                            if not block.name.startswith("mcp__notes__"):
                                ctx.log.tool(block.name, block.input)
                elif isinstance(message, ResultMessage):
                    final_result[0] = message
                    if message.is_error:
                        raise RuntimeError(
                            f"Claude Code returned an error: {message.result}")

        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_text)
            await drain(client)
            # Main pass done — deliver answers that have arrived; revision
            # turns run in the same session. By default we do not block on
            # unanswered questions (they defer to --answer).
            for round_num in range(1, MAX_REVISION_ROUNDS + 1):
                if wait_for_answers:
                    await anyio.to_thread.run_sync(broker.finish)
                items = broker.drain_new()
                if not items:
                    break
                print(f"\n  [revision round {round_num}: delivering "
                      f"{len(items)} answer(s)]", flush=True)
                await client.query(_revision_message(items))
                await drain(client)

        # The last ResultMessage carries cumulative usage for the session
        # and the API-equivalent cost as computed by Claude Code.
        if final_result[0] is not None:
            rm = final_result[0]
            ctx.usage.add(usage_from_claude_code(rm.usage, rm.total_cost_usd))
        return "".join(parts)

    try:
        return anyio.run(_main)
    except CLINotFoundError:
        raise SystemExit(
            "Claude Code CLI not found. The 'subscription' backend needs the "
            "`claude` binary on PATH (the nix devshell provides it) and a "
            "one-time login with your Claude subscription (`claude login`). "
            "Alternatively, run with --backend codex or --backend api."
        )


# ---------------------------------------------------------------------------
# Backend: ChatGPT subscription (OpenAI Codex CLI)
# ---------------------------------------------------------------------------

def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


FRAME_DELEGATION_CODEX = """

Frame analysis: video frames are token-expensive. By default, spawn a
'frame_reader' subagent to do frame reading: point it at the timestamp(s),
tell it what the transcript says around that moment and what to look for, and
have it report the board/slide contents back. get_frame saves each frame as a
JPEG in ./frames/ and returns the path — view the file with the image-viewing
tool. Only read frames yourself when the subagent's report is ambiguous,
incomplete, or mathematically implausible and the passage is important."""


def _stream_codex(cmd: list[str], prompt: str) -> tuple[str | None, int | None]:
    """Run a codex command, streaming its output through, feeding the prompt
    on stdin, and capturing the session id from the header plus the token
    count from the footer. Returns (session_id, tokens_used)."""
    import re
    import subprocess
    import threading

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)

    def feed():
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except BrokenPipeError:
            pass

    # Feed stdin from a thread so a large prompt can't deadlock against
    # unread stdout.
    threading.Thread(target=feed, daemon=True).start()

    session_id = None
    tokens = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if session_id is None:
            m = re.search(r"session id: ([0-9a-fA-F-]{16,})", line)
            if m:
                session_id = m.group(1)
        m = re.search(r"[Tt]okens used:?\s*([\d,]+)", line)
        if m:
            tokens = int(m.group(1).replace(",", ""))
    if proc.wait() != 0:
        raise SystemExit(f"codex failed (exit code {proc.returncode})")
    return session_id, tokens


def _run_codex(system_prompt: str, user_text: str, ctx: NotesToolContext,
               output_file: Path, model: str,
               frame_model: str | None, wait_for_answers: bool) -> str:
    import shutil
    import tempfile

    codex = shutil.which("codex")
    if codex is None:
        raise SystemExit(
            "codex CLI not found. The 'codex' backend needs the `codex` "
            "binary on PATH (the nix devshell provides it) and a one-time "
            "`codex login` with your ChatGPT subscription. Alternatively, "
            "run with --backend subscription or --backend api."
        )

    server_script = Path(__file__).resolve().parent / "notes_mcp_server.py"

    with tempfile.TemporaryDirectory(prefix="notetaker-codex-") as td:
        tmp = Path(td)
        ctx.state_file = tmp / "tool_state.json"
        if ctx.video_path:
            # Frames are saved as files; codex views local images natively.
            ctx.frames_dir = output_file.parent / "frames"
        ctx_file = tmp / "context.json"
        ctx.dump(ctx_file)
        last_msg = tmp / "last_message.txt"

        server_args = "[" + ", ".join(
            _toml_str(a) for a in
            [str(server_script), "--context", str(ctx_file)]
        ) + "]"

        # Options valid on both `codex exec` and `codex exec resume`.
        config_flags = [
            "--model", model,
            "--skip-git-repo-check",
            "-c", f"mcp_servers.notes.command={_toml_str(sys.executable)}",
            "-c", f"mcp_servers.notes.args={server_args}",
            # PATH so the MCP server subprocess can find ffmpeg.
            "-c", "mcp_servers.notes.env={PATH="
                  + _toml_str(os.environ.get("PATH", "")) + "}",
            "-c", 'web_search="live"',
        ]

        if ctx.video_path:
            # Frame reading is delegated to a subagent (codex subagents
            # inherit the parent's full context, so it already knows the
            # lecture). A role config layer pins it to a cheaper model.
            config_flags += [
                "--enable", "multi_agent",
                "-c", "agents.frame_reader.description=" + _toml_str(
                    "Reads lecture-video frames: give it timestamps and what "
                    "to look for; it fetches frames with get_frame, views the "
                    "saved images, and reports the board/slide contents in "
                    "LaTeX."),
            ]
            if frame_model:
                role_cfg = tmp / "frame_reader.toml"
                role_cfg.write_text(f"model = {_toml_str(frame_model)}\n")
                config_flags += ["-c", "agents.frame_reader.config_file="
                                 + _toml_str(str(role_cfg))]
            system_prompt = system_prompt + FRAME_DELEGATION_CODEX

        # Codex has no separate system-prompt channel in exec mode.
        prompt = f"<instructions>\n{system_prompt}\n</instructions>\n\n{user_text}"
        session_id, tokens = _stream_codex(
            [codex, "exec",
             "--sandbox", "workspace-write",
             "--cd", str(output_file.parent)]
            + config_flags
            + ["--output-last-message", str(last_msg), "-"],
            prompt,
        )
        total_tokens = tokens or 0
        ctx.merge_state()

        # Revision rounds: deliver answers that arrived during the run by
        # resuming the codex session. With wait_for_answers, prompt for the
        # outstanding ones ourselves first (the MCP subprocess is gone).
        # Sandbox and cwd are inherited from the resumed session.
        broker = ensure_broker(ctx)
        for round_num in range(1, MAX_REVISION_ROUNDS + 1):
            if wait_for_answers:
                broker.finish()
            items = broker.drain_new()
            if not items:
                break
            if session_id is None:
                print("\nWarning: could not capture the codex session id; "
                      "skipping the revision round. Answers:\n"
                      + format_answers(items), file=sys.stderr)
                break
            print(f"\n  [revision round {round_num}: delivering "
                  f"{len(items)} answer(s)]", flush=True)
            ctx.dump(ctx_file)  # refresh question_seq for the resumed server
            _, tokens = _stream_codex(
                [codex, "exec", "resume"] + config_flags
                + ["--output-last-message", str(last_msg), session_id, "-"],
                _revision_message(items),
            )
            total_tokens += tokens or 0
            ctx.merge_state()

        if total_tokens:
            ctx.usage.note = (f"codex reported {total_tokens:,} total tokens; "
                              f"no price table for GPT models")

        return last_msg.read_text() if last_msg.exists() else ""
