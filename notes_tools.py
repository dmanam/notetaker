"""
notes_tools.py — Tool definitions and handlers for the note-writing agent,
shared by every backend (Claude subscription, Codex, Anthropic API).

The tool surface is parameterized by a NotesToolContext:
  - get_frame is offered only when a video is available;
  - add_to_preamble is offered only in course mode (build_course.py).

Handlers are synchronous and may block (stdin prompts, ffmpeg). They mutate
the context (recorded corrections, preamble additions, frame counter); when
ctx.state_file is set — as in the MCP subprocess spawned for the Codex
backend — every mutation is also persisted there so the parent process can
pick it up after the run.

Diagnostics go to stderr: inside the Codex MCP subprocess, stdout is the
protocol channel and must stay clean. Interactive prompts fall back to
/dev/tty when stdin is not a terminal (again: the MCP subprocess).
"""

from __future__ import annotations

import base64
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fetch import describe_assets, fetch_reference
from media import (extract_frame, extract_frame_file, format_timestamp,
                   parse_timestamp)


@dataclass
class ToolResult:
    """content is either a string or a list of Anthropic-style content blocks
    (text / base64-image blocks)."""
    content: str | list[dict]
    is_error: bool = False


Handler = Callable[[dict], ToolResult]

# System prompt for the cheap model that studies video frames on behalf of
# the main note-writer (subagent on the subscription/codex backends, direct
# call on the api backend).
FRAME_READER_PROMPT = """You are a frame-reading assistant for math lecture videos. You are given
lecture context and one or more video frames (or a get_frame tool to fetch
them — it may return the image directly, or save it to a file for you to open
with your image-viewing tool).

Produce a *report* on what is shown that is relevant to the context — not
bare LaTeX:
- Transcribe all visible mathematics in LaTeX.
- Describe the layout in prose: which board/column/slide region each piece
  occupies, how items are connected (arrows, boxes, underlines, cross-outs),
  and what any diagrams depict, precisely.
- Say explicitly where content is cut off by the frame edge, occluded (by the
  lecturer, glare), partially erased, or too blurry to read — and which parts
  of your transcription are uncertain because of it.
- Quote labels, captions, and marginal remarks.

Frames may be mid-erasure or mid-transition; when you can fetch frames
yourself, try nearby timestamps to get a clearer view before reporting.
Report only what is visible — never invent or complete mathematics. Flagging
something as unreadable is always better than guessing."""


REGISTER_INSTRUCTION = """
Register: these are seminar notes for professional mathematicians — your
readers are peers, not students. Write concisely and lead with the concepts:
say what the idea is and why it works, keep the mathematics in the foreground,
and let routine verifications stay routine. Assume the standard graduate
background of the field and do not re-explain what such a reader already
knows. Concise does not mean incomplete: every result, hypothesis and
construction the lecturer gave still belongs in the notes."""


def style_exemplar_block(passages: list) -> str:
    """Passages of notes whose *register* the agent should imitate.

    Supplied by the user per course, not baked in: the right exemplar for a
    seminar on analytic number theory is not the right one for homotopy
    theory, and pinning one document would bias the writing towards its
    subject as well as its style.

    These arrive already chosen and rewritten by style_extract — spread
    through the source rather than taken off the front, private macros
    expanded, and each one checked to render the same as the original. Taking
    the first few thousand characters of a file, which is what this used to
    do, samples the preface: the least representative page in any set of
    notes, and the one most likely to be a list of conventions.

    The instruction deliberately does NOT claim the exemplar is unrelated to
    the lecture. The best exemplar is often the closest one — the same
    author's own written-up notes on neighbouring material, which is where
    their register is most visible — and telling the model that a document
    plainly about the same subject is "unrelated" is both false and
    self-defeating: a model that spots the falsehood has reason to discount
    the whole instruction. What has to be said instead is the true thing,
    which is stronger anyway: the exemplar is a different course, so what it
    contains is not evidence about what THIS lecture said."""
    parts = [str(p).strip() for p in passages or [] if str(p).strip()]
    if not parts:
        return ""
    parts = [f"--- style passage {n} ---\n{p}" for n, p in enumerate(parts, 1)]
    return (
        "The excerpts below are here for their WRITING STYLE only. Match "
        "their register: sentence density, how much detail is spelled out, "
        "what is left to the reader, how results and proofs are laid out.\n\n"
        "Take no mathematical content, notation, terminology, or choice of "
        "definition from them. They may well cover material close to this "
        "lecture's — possibly by the same author — and that makes this rule "
        "more important, not less: they are a DIFFERENT exposition, so what "
        "they contain is not evidence about what this lecture said. Where the "
        "two overlap, follow the lecture: if the exemplar states a result "
        "more generally, defines a term differently, or names an object with "
        "another symbol, the lecture wins and the difference is not an error "
        "to correct. Anything you take from these excerpts because it looked "
        "relevant is an unsupported addition, and will be treated as one.\n\n"
        "Their machinery is theirs, not yours: the \\cite keys, \\label names "
        "and \\ref targets in them belong to that document and do not exist "
        "here. Follow this course's own rules for citing and "
        "cross-referencing, whatever the excerpts happen to do.\n\n"
        + "\n\n".join(parts) + "\n\n")


@dataclass
class NotesToolContext:
    refs_dir: Path
    video_path: Path | None = None
    total_duration: float = 0.0
    enable_preamble: bool = False
    existing_preamble: list = field(default_factory=list)
    state_file: Path | None = None
    # Trace file the Codex MCP subprocess appends its tool calls to (the
    # parent process holds the AgentLog object itself).
    log_file: Path | None = None
    # Lets the clarify tool check a quoted passage's real position.
    transcript_path: Path | None = None
    # When set, get_frame saves each frame as a JPEG here and returns its path
    # (for agents that view local images — the codex backend) instead of
    # returning the image inline.
    frames_dir: Path | None = None
    # Directories the api backend's read_file tool may read from (course
    # root, so the agent can consult earlier lecture files). The
    # subscription/codex backends read files natively.
    read_roots: list = field(default_factory=list)
    # When set, the cite_reference tool is offered and entries accumulate in
    # this .bib file (course mode's running bibliography).
    bib_file: Path | None = None
    # Board stills for this lecture ({id, path, …}) and where compiled
    # diagrams are built. Together these enable check_diagram/draw_diagram.
    boards: list = field(default_factory=list)
    diagrams_dir: Path | None = None
    # Populated during the run:
    new_corrections: dict = field(default_factory=dict)
    new_preamble_additions: list = field(default_factory=list)
    frame_requests: int = 0
    # Asynchronous user questions: dicts with keys
    #   id, kind ("clarify" | "ask_user"), text, context, guess,
    #   answer (None = not answered yet, "" = skipped/guess accepted),
    #   delivered (bool — answer already handed to the agent)
    questions: list = field(default_factory=list)
    question_seq: int = 1

    # -- serialization (for the Codex MCP subprocess) -----------------------

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({
            "refs_dir": str(self.refs_dir),
            "video_path": str(self.video_path) if self.video_path else None,
            "total_duration": self.total_duration,
            "enable_preamble": self.enable_preamble,
            "existing_preamble": self.existing_preamble,
            "state_file": str(self.state_file) if self.state_file else None,
            "log_file": str(self.log_file) if self.log_file else None,
            "transcript_path": (str(self.transcript_path)
                                if self.transcript_path else None),
            "frames_dir": str(self.frames_dir) if self.frames_dir else None,
            "question_seq": self.question_seq,
            "bib_file": str(self.bib_file) if self.bib_file else None,
            "read_roots": [str(r) for r in self.read_roots],
        }))

    @classmethod
    def load(cls, path: Path) -> "NotesToolContext":
        d = json.loads(path.read_text())
        return cls(
            refs_dir=Path(d["refs_dir"]),
            video_path=Path(d["video_path"]) if d["video_path"] else None,
            total_duration=d["total_duration"],
            enable_preamble=d["enable_preamble"],
            existing_preamble=d["existing_preamble"],
            state_file=Path(d["state_file"]) if d["state_file"] else None,
            log_file=Path(d["log_file"]) if d.get("log_file") else None,
            transcript_path=(Path(d["transcript_path"])
                             if d.get("transcript_path") else None),
            frames_dir=Path(d["frames_dir"]) if d.get("frames_dir") else None,
            question_seq=d.get("question_seq", 1),
            bib_file=Path(d["bib_file"]) if d.get("bib_file") else None,
            read_roots=[Path(r) for r in d.get("read_roots", [])],
        )

    def save_state(self) -> None:
        if self.state_file:
            self.state_file.write_text(json.dumps({
                "new_corrections": self.new_corrections,
                "new_preamble_additions": self.new_preamble_additions,
                "frame_requests": self.frame_requests,
                "questions": self.questions,
                "question_seq": self.question_seq,
            }))

    def merge_state(self) -> None:
        """Merge mutations persisted by a subprocess back into this context."""
        if not (self.state_file and self.state_file.exists()):
            return
        d = json.loads(self.state_file.read_text())
        self.new_corrections.update(d.get("new_corrections", {}))
        for entry in d.get("new_preamble_additions", []):
            if entry not in self.new_preamble_additions:
                self.new_preamble_additions.append(entry)
        self.frame_requests += d.get("frame_requests", 0)
        by_id = {q["id"]: q for q in self.questions}
        for q in d.get("questions", []):
            cur = by_id.get(q["id"])
            if cur is None:
                self.questions.append(q)
            else:
                if cur.get("answer") is None and q.get("answer") is not None:
                    cur["answer"] = q["answer"]
                    cur["deferred"] = q.get("deferred", False)
                cur["delivered"] = cur.get("delivered") or q.get("delivered")
        self.question_seq = max(self.question_seq, d.get("question_seq", 1))


# ---------------------------------------------------------------------------
# Terminal I/O that works both in-process and inside the MCP subprocess
# ---------------------------------------------------------------------------

def emit(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def ask_user_input(prompt_text: str, should_abort=None) -> str | None:
    """Read a line from the user. Falls back to /dev/tty when stdin is not a
    terminal (e.g. inside the MCP server whose stdin is the protocol stream).
    Returns "" if no terminal is available, or None if should_abort() became
    true before any input arrived (the wait is polled, so a stale prompt can
    be cancelled instead of stealing a later prompt's input)."""
    import select

    opened = None
    if sys.stdin.isatty():
        stream = sys.stdin
        print(prompt_text, file=sys.stderr, end="", flush=True)
    else:
        try:
            opened = open("/dev/tty", "r+")
        except OSError:
            return ""
        stream = opened
        stream.write(prompt_text)
        stream.flush()
    try:
        while True:
            if should_abort is not None and should_abort():
                return None
            ready, _, _ = select.select([stream], [], [], 0.25)
            if ready:
                return stream.readline().strip()
    finally:
        if opened:
            opened.close()


# ---------------------------------------------------------------------------
# Asynchronous user questions
# ---------------------------------------------------------------------------

class QuestionBroker:
    """Queues questions for the user and prompts for answers from a background
    thread, so the agent never blocks on the terminal. Answers are recorded on
    the context (and persisted via save_state) and handed back to the agent
    either mid-run (get_user_answers) or in a revision turn afterwards."""

    def __init__(self, ctx: NotesToolContext):
        self.ctx = ctx
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    def ask(self, kind: str, text: str, context: str = "",
            guess: str = "", timestamp: str | None = None) -> dict:
        with self._lock:
            q = {"id": self.ctx.question_seq, "kind": kind, "text": text,
                 "context": context, "guess": guess, "timestamp": timestamp,
                 "answer": None, "delivered": False}
            self.ctx.question_seq += 1
            self.ctx.questions.append(q)
            self.ctx.save_state()
            self._ensure_thread()
        return q

    def drain_new(self) -> list[dict]:
        """Answered-but-not-yet-delivered questions; marks them delivered."""
        with self._lock:
            items = [q for q in self.ctx.questions
                     if q["answer"] is not None and not q["delivered"]]
            for q in items:
                q["delivered"] = True
            if items:
                self.ctx.save_state()
        return items

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for q in self.ctx.questions if q["answer"] is None)

    def finish(self) -> None:
        """Block until every queued question has an answer (prompting the
        user for any that are still open)."""
        while True:
            with self._lock:
                if self._closed or not any(q["answer"] is None
                                           for q in self.ctx.questions):
                    return
                self._ensure_thread()
                t = self._thread
            t.join()

    def close(self) -> None:
        """Stop prompting; unanswered questions stay open for a follow-up
        run. Cancels any prompt currently waiting on the terminal."""
        with self._lock:
            self._closed = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
                q = next((x for x in self.ctx.questions
                          if x["answer"] is None), None)
            if q is None:
                return
            if not self._prompt_one(q):
                return  # aborted by close()

    def _prompt_one(self, q: dict) -> bool:
        """Prompt for one question. Returns False if aborted by close()."""
        aborted = lambda: self._closed
        at = f" @ {q['timestamp']}" if q.get("timestamp") else ""
        if q["kind"] == "clarify":
            emit(f'\n  [Transcript unclear #{q["id"]}{at}] "{q["text"]}"')
            if q["context"]:
                emit(f"  Context: {q['context']}")
            if q["guess"]:
                emit(f"  Model's guess: \"{q['guess']}\"")
            ans = ask_user_input(
                "  Correct text (Enter accepts the guess, '?' defers): ",
                should_abort=aborted)
            if ans is None:
                return False
            with self._lock:
                if ans == "?":
                    q["answer"] = ""
                    q["deferred"] = True
                else:
                    final = ans or q["guess"]
                    q["answer"] = ans  # "" = guess accepted
                    q["deferred"] = not final
                    if final:
                        self.ctx.new_corrections[q["text"]] = final
                self.ctx.save_state()
        else:
            emit(f"\n  [Question #{q['id']}{at} for you] {q['text']}")
            if q.get("guess"):
                emit(f"  Provisionally using: {q['guess']}")
            ans = ask_user_input("  Your answer (Enter to defer): ",
                                 should_abort=aborted)
            if ans is None:
                return False
            with self._lock:
                q["answer"] = ans
                q["deferred"] = not ans  # "" = deferred to a follow-up run
                self.ctx.save_state()
        return True


def ensure_broker(ctx: NotesToolContext) -> QuestionBroker:
    broker = getattr(ctx, "_broker", None)
    if broker is None:
        broker = QuestionBroker(ctx)
        ctx._broker = broker
    return broker


_DEF_RE = re.compile(
    r"\\(?:new|renew|provide)command\s*\*?\s*\{?\s*(\\[a-zA-Z@]+)\s*\}?"
    r"|\\DeclareMathOperator\s*\*?\s*\{\s*(\\[a-zA-Z@]+)\s*\}"
    r"|\\def\s*(\\[a-zA-Z@]+)")


def macro_definitions(latex: str) -> dict[str, tuple[str, str]]:
    """Macros defined by a preamble block: name -> (definition body, line).

    The body is what follows the declaration, so \\newcommand{\\Q}{\\Q} and
    \\providecommand{\\Q}{\\Q} compare equal — one lecture re-declaring
    exactly what another already declared is harmless."""
    out: dict[str, tuple[str, str]] = {}
    for line in latex.splitlines():
        m = _DEF_RE.search(line)
        if not m:
            continue
        name = m.group(1) or m.group(2) or m.group(3)
        out.setdefault(name, (line[m.end():].strip(), line.strip()))
    return out


def is_open(q: dict) -> bool:
    """A question that a follow-up run should (re-)ask the user."""
    return q["answer"] is None or bool(q.get("deferred"))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def locate_quote(segments: list[dict], quote: str) -> float | None:
    """When in the lecture a quoted passage was said, or None if not found.

    Matching is on a normalized word stream so it survives punctuation and
    the '...' elisions models put in quotes; longest prefix first."""
    if not segments or not quote:
        return None
    stream: list[str] = []
    starts: list[float] = []
    for seg in segments:
        for w in _words(seg.get("text", "")):
            stream.append(w)
            starts.append(seg["start"])
    target = _words(quote)
    for n in range(min(8, len(target)), 2, -1):
        probe = target[:n]
        for i in range(len(stream) - n + 1):
            if stream[i:i + n] == probe:
                return starts[i]
    return None


def backfill_question_timestamps(questions: list[dict],
                                 segments: list[dict]) -> int:
    """Give timestamps to questions queued before timestamps were recorded.

    A clarify question quotes the transcript verbatim, so its position can be
    recovered exactly by locating that quote — no guessing. Matching is on a
    normalized word stream (the quote may elide with '...' or differ in
    punctuation), longest prefix first; a question that cannot be located is
    left alone."""
    todo = [q for q in questions if not q.get("timestamp") and q.get("text")]
    if not todo or not segments:
        return 0

    stream: list[str] = []
    starts: list[float] = []
    for seg in segments:
        for w in _words(seg.get("text", "")):
            stream.append(w)
            starts.append(seg["start"])

    filled = 0
    for q in todo:
        target = _words(q["text"])
        for n in range(min(8, len(target)), 2, -1):
            probe = target[:n]
            for i in range(len(stream) - n + 1):
                if stream[i:i + n] == probe:
                    q["timestamp"] = format_timestamp(starts[i])
                    filled += 1
                    break
            else:
                continue
            break
    return filled


def format_answers(items: list[dict]) -> str:
    """Render answered questions for delivery to the agent.

    A follow-up run is a fresh context: the agent no longer remembers what it
    asked or what it provisionally wrote, and only has this block plus the
    \\todo marker to work from. So every answer restates the question and the
    agent's own guess — without them a bare answer like "yes, the second one"
    is unusable."""
    lines = []
    for q in items:
        at = f" [{q['timestamp']}]" if q.get("timestamp") else ""
        guess = q.get("guess") or ""
        if q["kind"] == "clarify":
            head = f"- Answer #{q['id']}{at} — transcript read \"{q['text']}\""
            head += (f"; your guess was \"{guess}\"." if guess
                     else " (you made no guess).")
        else:
            head = f"- Answer #{q['id']}{at} — you asked: \"{q['text']}\""
            head += (f"; provisionally you used: {guess}." if guess else ".")
        lines.append(head)

        if q.get("deferred"):
            lines.append("  DEFERRED by the user — keep your provisional "
                         "version and its \\todo; a later follow-up run may "
                         "resolve it.")
        elif q["kind"] == "clarify":
            final = q["answer"] or guess
            if final == guess:
                lines.append(f"  CONFIRMED: it reads \"{final}\". Remove the "
                             f"matching \\todo.")
            else:
                lines.append(f"  CORRECTED: it should read \"{final}\", not "
                             f"\"{guess}\". Fix the text and remove the "
                             f"matching \\todo.")
        else:
            lines.append(f"  User's answer: {q['answer']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool format; converted per backend)
# ---------------------------------------------------------------------------

def build_tools(ctx: NotesToolContext) -> list[dict]:
    tools = [
        {
            "name": "fetch_document",
            "description": (
                "Fetch a web page, PDF, or arXiv paper by URL or arXiv ID. "
                "For arXiv references (e.g. '2310.12345', "
                "'arxiv:2310.12345', or any arxiv.org URL) the returned text "
                "is the paper's actual TeX source when available (exact "
                "macros and notation), and ALL artifacts are cached as local "
                "files whose paths are listed in the result: the unpacked "
                "source tree and the rendered PDF (open the PDF — via your "
                "file tools or view_pdf_page — to see resolved "
                "theorem/equation numbers, or if extracted text looks "
                "garbled). Prefer this for arXiv papers; your built-in web "
                "fetch/search (if available) is fine for general pages."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url_or_id": {
                        "type": "string",
                        "description": (
                            "A URL (https://...) or arXiv ID "
                            "(e.g. '2310.12345' or 'arxiv:2310.12345')."
                        ),
                    }
                },
                "required": ["url_or_id"],
            },
        },
        {
            "name": "clarify_transcript",
            "description": (
                "Use when a word or phrase in the transcript appears garbled, "
                "misheared, or makes no mathematical sense in context. "
                "Provide the exact unclear text, the surrounding context, and "
                "your best guess at what was said. The question is queued for "
                "the user and answered asynchronously — proceed with your "
                "guess (marked with \\todo) and pick the verdict up later via "
                "get_user_answers. Confirmed corrections are passed to future "
                "lectures so the mishearing gets fixed wherever it recurs."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "transcript_text": {
                        "type": "string",
                        "description": "The exact garbled or unclear text from the transcript.",
                    },
                    "context": {
                        "type": "string",
                        "description": "One or two surrounding sentences for context.",
                    },
                    "guess": {
                        "type": "string",
                        "description": "Your best guess at the correct word or phrase.",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": (
                            "When this is said, as hh:mm:ss — copy the "
                            "timestamp of the transcript line it comes from. "
                            "The user uses it to jump to that point in the "
                            "video, so it must be the real position."
                        ),
                    },
                },
                "required": ["transcript_text", "context", "timestamp"],
            },
        },
        {
            "name": "ask_user",
            "description": (
                "Ask the user for help with a LaTeX typesetting question you are "
                "not confident about — e.g. which package and command to use for "
                "a non-standard symbol, field-specific notation, or unusual "
                "mathematical construct. Use this rather than silently guessing "
                "or omitting. The question is queued and answered "
                "asynchronously — proceed provisionally (marked with \\todo) "
                "and collect the answer later via get_user_answers."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A precise question for the user, e.g. "
                            "'What LaTeX package and command should I use for "
                            "the prism symbol in prismatic cohomology?'"
                        ),
                    },
                    "timestamp": {
                        "type": "string",
                        "description": (
                            "The point in the lecture the question is about, "
                            "as hh:mm:ss (from the transcript line, or the "
                            "frame you were looking at). Supply it whenever "
                            "the question comes from a specific moment — "
                            "nearly always — so the user can go and look."
                        ),
                    },
                    "provisional": {
                        "type": "string",
                        "description": (
                            "What you are doing in the meantime, e.g. "
                            "'\\square from amssymb'. This is handed back to "
                            "you with the answer — by then you will be a "
                            "fresh context that no longer remembers what you "
                            "chose, and the user's reply may well be just "
                            "'yes, that works'."
                        ),
                    },
                },
                "required": ["question", "timestamp"],
            },
        },
        {
            "name": "search_document",
            "description": (
                "Search a cached file — or a whole directory of them, e.g. a "
                "paper's unpacked TeX source — for a regular expression, and "
                "get back the matching lines with their line numbers and "
                "surrounding context. Use this instead of reading a long "
                "document from the top: fetched papers can run to hundreds of "
                "thousands of characters, more than one read returns, and the "
                "part you want is usually findable by a term ('solid', "
                "\\\\begin{theorem}, a symbol, an author name). Having found "
                "the line number, read just that range."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File or directory to search — a path from a "
                            "'Cached locally' listing, or any file in the "
                            "course output directory."),
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Python regular expression, case-insensitive, "
                            "matched line by line."),
                    },
                    "context": {
                        "type": "integer",
                        "description": "Lines of context each side (default 2, max 20).",
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Stop after this many matches (default 30, max 200).",
                    },
                },
                "required": ["path", "pattern"],
            },
        },
        {
            "name": "view_pdf_page",
            "description": (
                "Render one page of a locally cached PDF (e.g. a fetched "
                "paper's paper.pdf) as an image, so you see the typeset page "
                "— including the resolved theorem/equation numbering that "
                "raw TeX source cannot show. Pages are 1-indexed. (If your "
                "native file-reading tool can open PDFs directly, that works "
                "too.)"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Absolute path to a local PDF."},
                    "page": {"type": "integer",
                             "description": "1-indexed page number."},
                },
                "required": ["path", "page"],
            },
        },
        {
            "name": "get_user_answers",
            "description": (
                "Fetch any answers the user has provided to your earlier "
                "ask_user / clarify_transcript questions (each marked "
                "confirmed, corrected, or skipped). Call this before "
                "finalizing the document so you can incorporate answers that "
                "arrived while you worked; answers still outstanding will be "
                "delivered to you afterwards for a revision pass."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    if ctx.enable_preamble:
        tools.insert(0, {
            "name": "add_to_preamble",
            "description": (
                "Add one or more lines to the LaTeX document preamble — "
                "e.g. \\usepackage{bbm}, \\newcommand{\\Prism}{...}, "
                "\\DeclareMathOperator{\\Tr}{Tr}, or \\declaretheorem{...}. "
                "Call this before writing body content that depends on it. "
                "Do not add hyperref or cleveref (already loaded last). The "
                "preamble is shared by every lecture in the course, so a "
                "macro another lecture already defined cannot be redefined — "
                "you will be told what it is already bound to."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "latex": {
                        "type": "string",
                        "description": (
                            "One or more LaTeX lines to add to the preamble, "
                            "e.g. '\\usepackage{tikz-cd}\\n"
                            "\\newcommand{\\cat}[1]{\\mathbf{#1}}'."
                        ),
                    }
                },
                "required": ["latex"],
            },
        })

    if ctx.bib_file:
        tools.append({
            "name": "cite_reference",
            "description": (
                "Add a paper, book, or web page to the course bibliography "
                "and get back its cite key, for use as \\cite{key} in the "
                "notes. For an arXiv ID/URL or DOI the full metadata is "
                "fetched automatically — pass just the identifier. For "
                "anything else (lecture notes, a book without a DOI, a web "
                "page) ALSO pass title, author, and year: you almost always "
                "know or can look them up, and without an author the entry "
                "gets an ugly placeholder citation label instead of a proper "
                "one like [Sch19]. Safe to call repeatedly for the same "
                "source — it returns the existing key. Cite the sources the "
                "lecturer names, and the references you consulted for "
                "definitions or notation. Never hand-write bibliography "
                "entries or \\printbibliography: the bibliography is "
                "assembled for you."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url_or_id": {
                        "type": "string",
                        "description": ("arXiv ID (e.g. '1905.08229'), DOI, "
                                        "or URL."),
                    },
                    "title": {
                        "type": "string",
                        "description": ("Title. Needed only when metadata "
                                        "cannot be fetched automatically "
                                        "(i.e. not an arXiv paper or DOI)."),
                    },
                    "author": {
                        "type": "string",
                        "description": (
                            "Author(s), BibTeX style: 'Marek Ostrand' or "
                            "'Dana Whitlock and Marek Ostrand'. Supply this "
                            "whenever the source is not an arXiv paper or "
                            "DOI — check the document itself if unsure."),
                    },
                    "year": {
                        "type": "string",
                        "description": ("Publication year, e.g. '2019'. "
                                        "Supply alongside author."),
                    },
                },
                "required": ["url_or_id"],
            },
        })

    if ctx.video_path:
        delivery = (
            "The frame is saved as a JPEG file and its absolute path is "
            "returned — open it with your image-viewing tool to inspect it. "
            if ctx.frames_dir else ""
        )
        tools.append({
            "name": "get_frame",
            "description": (
                "Extract a single frame from the lecture video at a given "
                f"timestamp. {delivery}"
                "Use this SPARINGLY, as a fallback. Where board snapshots are "
                "provided they are already the best view of each board — "
                "taken at the moment it was most complete and with the "
                "lecturer edited out — so reach for a raw frame only when a "
                "snapshot is missing, garbled, or plainly does not cover the "
                "moment you need. A single frame is a poor substitute: it may "
                "catch a mid-erasure, a slide transition, a camera pan, or the "
                "lecturer standing in front of the very thing you want to "
                "read. If you do use it, take several nearby timestamps and "
                "reconcile them."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": ["number", "string"],
                        "description": (
                            f"Position in the video, from 00:00:00 to "
                            f"{format_timestamp(ctx.total_duration)}. Either "
                            f"hh:mm:ss — exactly as the transcript's [hh:mm:ss] "
                            f"markers give it — or a number of seconds."
                        ),
                    }
                },
                "required": ["timestamp"],
            },
        })

    if ctx.boards and ctx.diagrams_dir is not None:
        ids = ", ".join(str(b["id"]) for b in ctx.boards)
        tools.append({
            "name": "crop_board",
            "description": (
                f"Get part of a board still at full resolution. Sending a "
                f"whole slate downscales it to the vision ceiling, which "
                f"leaves a chalk stroke a pixel or two wide — too little to "
                f"tell one arrowhead from another or read a subscript. A crop "
                f"of the same region arrives un-shrunk. Do this before you "
                f"commit to what a diagram's arrows do. It does not sharpen "
                f"anything and never scales up, so cropping tighter and "
                f"tighter past the thing you want buys nothing. "
                f"Boards: {ids}."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "board": {"type": ["integer", "string"],
                              "description": "Which board to crop."},
                    "x": {"type": "number",
                          "description": "Left edge, 0 (left) to 1 (right)."},
                    "y": {"type": "number",
                          "description": "Top edge, 0 (top) to 1 (bottom)."},
                    "width": {"type": "number",
                              "description": "Box width as a fraction, 0–1."},
                    "height": {"type": "number",
                               "description": "Box height as a fraction, 0–1."},
                },
                "required": ["board", "x", "y", "width", "height"],
            },
        })

    if ctx.diagrams_dir is not None:
        tools.append({
            "name": "check_diagram",
            "description": (
                "Compile one tikz-cd or tikz diagram on its own and render it "
                "to a PNG, so you can look at your own drawing next to the "
                "board and see whether it matches. Returns the compiler's "
                "errors if it does not build. Use this on every diagram "
                "before putting it in the notes: a diagram that fails to "
                "compile takes the whole course build down with it, and one "
                "that compiles but has an arrow reversed is worse than no "
                "diagram at all."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "latex": {
                        "type": "string",
                        "description": ("The diagram alone — a \\begin{tikzcd}"
                                        "…\\end{tikzcd} or \\begin{tikzpicture}"
                                        "…\\end{tikzpicture} block, with no "
                                        "surrounding document or figure."),
                    },
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Every object that belongs in this diagram, in "
                            "LaTeX — e.g. [\"M_\\\\infty\", \"M_0\", \"S\"]. "
                            "Required. For a diagram read off a board this is "
                            "your reading of the slate, listed BEFORE you "
                            "compare anything: name what is there even if you "
                            "have not drawn it, and the two get diffed for "
                            "you. For a diagram you are composing from the "
                            "mathematics, it is the objects you mean to "
                            "include. Either way an object you meant and "
                            "omitted is invisible to every other check, "
                            "because the diagram has nothing there to point "
                            "at — this is the only check that finds it."),
                    },
                    "arrows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                            },
                        },
                        "description": (
                            "Optional, and worth doing: the arrows you read "
                            "off the board, each {from, to}. Diffed against "
                            "the diagram the same way."),
                    },
                    "name": {
                        "type": "string",
                        "description": ("Short slug identifying the diagram, "
                                        "e.g. 'pushout-square'. Used for the "
                                        "file name."),
                    },
                    "board": {
                        "type": ["integer", "string"],
                        "description": ("Which board this diagram came off. "
                                        "Pass it and you get back the "
                                        "provenance comment to put above the "
                                        "diagram in the notes."),
                    },
                },
                "required": ["latex", "objects"],
            },
        })

    return tools


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def build_handlers(ctx: NotesToolContext) -> dict[str, Handler]:
    broker = ensure_broker(ctx)
    # In files mode (the codex MCP server), binary results are written to
    # disk and returned as paths for the agent's own viewers.
    files_mode = getattr(ctx, "files_mode", False)

    def _file_roots() -> list[Path]:
        roots = [Path(r).resolve() for r in ctx.read_roots]
        roots.append(Path(ctx.refs_dir).resolve().parent)
        return roots

    def fetch_document(inp: dict) -> ToolResult:
        url_or_id = inp["url_or_id"]
        emit(f"  [fetch {url_or_id}]")
        try:
            ref = fetch_reference(url_or_id, ctx.refs_dir)
            assets = describe_assets(ref, Path(ctx.refs_dir).resolve().parent)
            return ToolResult(
                f"Title: {ref['title']}\nURL: {ref['url']}\n"
                f"{assets}\n{ref['text']}")
        except Exception as exc:
            return ToolResult(f"Error fetching document: {exc}", is_error=True)

    def cite_reference(inp: dict) -> ToolResult:
        from bibliography import cite
        url_or_id = inp["url_or_id"]
        try:
            key, added = cite(Path(ctx.bib_file), url_or_id,
                              inp.get("title") or None,
                              inp.get("author") or None,
                              inp.get("year") or None)
        except Exception as exc:
            return ToolResult(f"Error adding to bibliography: {exc}",
                              is_error=True)
        emit(f"  [cite {url_or_id} → \\cite{{{key}}}"
             f"{'' if added else ', already present'}]")
        return ToolResult(
            f"{'Added to' if added else 'Already in'} the bibliography. "
            f"Cite it as \\cite{{{key}}}.")

    TEXT_SUFFIXES = {".tex", ".txt", ".md", ".html", ".bib", ".json", ".bbl",
                     ".sty", ".cls"}

    def search_document(inp: dict) -> ToolResult:
        pattern = inp["pattern"]
        p = Path(inp["path"]).resolve()
        if not any(p == r or r in p.parents for r in _file_roots()):
            return ToolResult(
                "Error: path is outside the working/course directories.",
                is_error=True)
        if not p.exists():
            return ToolResult(f"Error: no such path: {p}", is_error=True)
        try:
            rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            return ToolResult(f"Error: bad regular expression: {exc}",
                              is_error=True)

        ctx_lines = max(0, min(int(inp.get("context", 2)), 20))
        limit = max(1, min(int(inp.get("max_matches", 30)), 200))
        files = ([p] if p.is_file()
                 else sorted(f for f in p.rglob("*")
                             if f.is_file() and f.suffix.lower()
                             in TEXT_SUFFIXES))
        emit(f"  [search {pattern!r} in {p.name}]")

        out, hits, truncated = [], 0, False
        for f in files:
            try:
                lines = f.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if not rx.search(line):
                    continue
                if hits >= limit:
                    truncated = True
                    break
                hits += 1
                lo, hi = max(0, i - ctx_lines), min(len(lines), i + ctx_lines + 1)
                out.append(f"--- {f} line {i + 1} ---")
                for n in range(lo, hi):
                    mark = ">" if n == i else " "
                    out.append(f"{mark} {n + 1:>6}  {lines[n]}")
            if truncated:
                break
        if not hits:
            return ToolResult(f"No match for {pattern!r} in {p}.")
        head = (f"{hits} match(es) for {pattern!r}"
                + (f" (stopped at {limit}; narrow the pattern for more)"
                   if truncated else "") + ":\n")
        return ToolResult(head + "\n".join(out))

    def view_pdf_page(inp: dict) -> ToolResult:
        p = Path(inp["path"]).resolve()
        if not any(p == r or r in p.parents for r in _file_roots()):
            return ToolResult(
                "Error: path is outside the working/course directories.",
                is_error=True)
        if not p.is_file():
            return ToolResult(f"Error: no such file: {p}", is_error=True)
        try:
            import pymupdf
        except ImportError:
            return ToolResult("Error: pymupdf is not installed.",
                              is_error=True)
        doc = pymupdf.open(p)
        page_no = int(inp["page"])
        if not 1 <= page_no <= doc.page_count:
            return ToolResult(
                f"Error: page {page_no} out of range — "
                f"{p.name} has {doc.page_count} pages.", is_error=True)
        emit(f"  [view_pdf_page {p.name} p{page_no}]")
        png = doc[page_no - 1].get_pixmap(dpi=150).tobytes("png")
        if files_mode:
            out = p.parent / "pages" / f"{p.stem}_p{page_no:04d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png)
            return ToolResult(
                f"Rendered page {page_no}/{doc.page_count} of {p.name} to "
                f"{out}. Open it with your image-viewing tool.")
        import base64
        return ToolResult([
            {"type": "text",
             "text": f"Page {page_no}/{doc.page_count} of {p.name}:"},
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(png).decode()}},
        ])

    def _transcript_segments(c: NotesToolContext) -> list[dict]:
        cached = getattr(c, "_segments", None)
        if cached is None:
            cached = []
            if c.transcript_path and Path(c.transcript_path).exists():
                try:
                    with open(c.transcript_path) as f:
                        cached = json.load(f).get("segments", [])
                except (OSError, ValueError):
                    cached = []
            c._segments = cached
        return cached

    def _question_time(inp: dict) -> tuple[str | None, str]:
        """Normalize the timestamp the model supplied. Returns (hh:mm:ss or
        None, a note to append to the tool result)."""
        raw = inp.get("timestamp")
        if raw in (None, ""):
            return None, (
                " No timestamp was recorded, so the user has nothing to jump "
                "to in the video — pass one (hh:mm:ss) next time.")
        seconds = parse_timestamp(raw)
        if seconds is None:
            return None, (
                f" Could not read the timestamp {raw!r}, so none was recorded"
                f" — use hh:mm:ss.")
        stamp = format_timestamp(seconds)
        if ctx.total_duration and seconds > ctx.total_duration + 1:
            return stamp, (
                f" Note: {stamp} is past the end of this lecture "
                f"({format_timestamp(ctx.total_duration)}) — check you used "
                f"hh:mm:ss and not some other unit.")
        return stamp, ""

    def _marker(q: dict) -> str:
        at = f" @ {q['timestamp']}" if q.get("timestamp") else ""
        return f"\\todo{{awaiting answer #{q['id']}{at}}}"

    def clarify_transcript(inp: dict) -> ToolResult:
        stamp, note = _question_time(inp)
        # The quoted text is verbatim from the transcript, so its position is
        # a fact we can look up rather than trust. Models misread the hour
        # field — reading [01:33:41] and writing 00:33:44 — which sends the
        # user an hour away in a two-hour video.
        true_at = locate_quote(_transcript_segments(ctx),
                               inp.get("transcript_text", ""))
        if true_at is not None:
            correct = format_timestamp(true_at)
            given = parse_timestamp(stamp) if stamp else None
            if given is None or abs(given - true_at) > 60:
                if stamp and given is not None:
                    note += (f" (Timestamp corrected to {correct}: you gave "
                             f"{stamp}, but that passage is at {correct} in "
                             f"the transcript — check the hour field.)")
                stamp = correct
        q = broker.ask("clarify", text=inp["transcript_text"],
                       context=inp.get("context", ""),
                       guess=inp.get("guess", ""), timestamp=stamp)
        return ToolResult(
            f"Question #{q['id']} queued for the user; the answer arrives "
            f"asynchronously. Proceed with your best guess for now, mark the "
            f"spot with {_marker(q)}, and keep working. Call get_user_answers "
            f"later to pick up the verdict." + note)

    def ask_user(inp: dict) -> ToolResult:
        stamp, note = _question_time(inp)
        q = broker.ask("ask_user", text=inp["question"], timestamp=stamp,
                       guess=inp.get("provisional", ""))
        return ToolResult(
            f"Question #{q['id']} queued for the user; the answer arrives "
            f"asynchronously. Proceed with your best provisional choice, mark "
            f"the spot with {_marker(q)}, and keep working. Call "
            f"get_user_answers later to pick up the answer." + note)

    def get_user_answers(inp: dict) -> ToolResult:
        items = broker.drain_new()
        pending = broker.pending_count()
        pending_note = (f" {pending} question(s) still awaiting the user."
                        if pending else "")
        if not items:
            return ToolResult("No new answers yet." + pending_note
                              + ("" if pending else " No questions pending."))
        return ToolResult(format_answers(items) + pending_note)

    def add_to_preamble(inp: dict) -> ToolResult:
        latex = inp["latex"].strip()
        if not latex:
            return ToolResult("Nothing to add.")
        if (latex in ctx.existing_preamble
                or latex in ctx.new_preamble_additions):
            return ToolResult("Already in the preamble.")

        # The preamble is shared across the whole course, so a macro this
        # lecture defines may already have been defined by another one.
        # Redefining it with \newcommand breaks the build; redefining it with
        # \providecommand silently keeps the *old* meaning, which is worse —
        # the notes then render with notation the lecture never intended.
        defined = {}
        for block in list(ctx.existing_preamble) + list(
                ctx.new_preamble_additions):
            for name, (body, line) in macro_definitions(block).items():
                defined.setdefault(name, (body, line))

        kept, clashes, superseded = [], [], []
        for line in latex.splitlines():
            defs = macro_definitions(line)
            if not defs:
                kept.append(line)
                continue
            name, (body, _) = next(iter(defs.items()))
            prior = defined.get(name)
            if prior is None:
                kept.append(line)
                defined[name] = (body, line.strip())
            elif prior[0] == body:
                continue          # same definition, already there
            elif line.lstrip().startswith("\\providecommand"):
                superseded.append((name, prior[1]))
            else:
                clashes.append((name, prior[1], line.strip()))

        if clashes:
            detail = "\n".join(
                f"  {n} is already defined as: {old}\n"
                f"    your version: {new}" for n, old, new in clashes)
            return ToolResult(
                "Nothing was added — these macros are already defined in the "
                "shared course preamble with a different meaning, and "
                "redefining them would break the build:\n" + detail +
                "\nUse the existing definition, or pick a different name for "
                "yours.", is_error=True)

        note = ""
        if superseded:
            note = "\n" + "\n".join(
                f"Note: {n} already exists as `{old}` — your "
                f"\\providecommand did not take effect, so it will render "
                f"with the existing meaning. Use a different name if you "
                f"need different notation." for n, old in superseded)

        block = "\n".join(kept).strip()
        if not block:
            return ToolResult("Nothing new to add." + note)
        ctx.new_preamble_additions.append(block)
        ctx.save_state()
        emit(f"    [preamble] {block[:60]}{'…' if len(block) > 60 else ''}")
        return ToolResult("Added to preamble." + note)

    def get_frame(inp: dict) -> ToolResult:
        ts = parse_timestamp(inp["timestamp"])
        if ts is None:
            return ToolResult(
                f"Error: could not read the timestamp "
                f"{inp['timestamp']!r} — use hh:mm:ss or a number of seconds.",
                is_error=True)
        ctx.frame_requests += 1
        ctx.save_state()
        emit(f"  [get_frame @ {format_timestamp(ts)}]")
        if ctx.frames_dir:
            path = extract_frame_file(ctx.video_path, ts, ctx.frames_dir)
            if path:
                return ToolResult(
                    f"Saved the frame at {format_timestamp(ts)} to {path}. "
                    f"Open it with your image-viewing tool to inspect it.")
        else:
            b64 = extract_frame(ctx.video_path, ts)
            if b64:
                return ToolResult([{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                }])
        return ToolResult("Error: could not extract frame at that timestamp.",
                          is_error=True)

    def _image_result(path: Path, text: str) -> ToolResult:
        """An image back to the caller — inline where it takes images, by
        path where it opens files itself (codex, and the frame-file mode)."""
        if files_mode or getattr(ctx, "frames_dir", None):
            return ToolResult(f"{text}\nSaved at {path} — open it with your "
                              f"image-viewing tool.")
        try:
            data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return ToolResult(f"{text}\nSaved at {path}.")
        suffix = path.suffix.lower()
        media = "image/png" if suffix == ".png" else "image/jpeg"
        return ToolResult([
            {"type": "text", "text": text},
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media, "data": data}},
        ])

    def crop_board(inp: dict) -> ToolResult:
        from boards import zoom

        board = next((b for b in ctx.boards
                      if str(b["id"]) == str(inp.get("board"))), None)
        if board is None:
            have = ", ".join(str(b["id"]) for b in ctx.boards) or "none"
            return ToolResult(f"Error: no board {inp.get('board')!r}. "
                              f"Available: {have}.", is_error=True)
        box = (inp.get("x", 0), inp.get("y", 0),
               inp.get("width", 1), inp.get("height", 1))
        root = Path(ctx.diagrams_dir) / "crops"
        n = len(list(root.glob("crop-*.jpg"))) + 1 if root.exists() else 1
        dest = root / f"crop-{n:03d}.jpg"
        emit(f"  [crop_board {board['id']} {box}]")
        out = zoom(Path(board["path"]), dest, box)
        if out is None:
            return ToolResult(
                "Error: that crop is unusable — the box is off the image or "
                "smaller than 4% of it in one direction. Give x, y, width and "
                "height as fractions of the whole still.", is_error=True)
        path, mag = out
        return _image_result(path, (
            f"Board {board['id']}, the region x={box[0]:.2f} y={box[1]:.2f} "
            f"w={box[2]:.2f} h={box[3]:.2f}, at native resolution — {mag:.1f}x "
            f"the detail you would get from the whole still. If what you "
            f"wanted is not in frame, crop again with a different box; if it "
            f"is in frame but still unreadable, it is unreadable, and "
            f"cropping tighter will not help."))

    def check_diagram(inp: dict) -> ToolResult:
        from diagrams import (check_inventory, compile_snippet, lint,
                              looks_like_diagram, strip_fences)

        latex = (inp.get("latex") or "").strip()
        if not latex:
            return ToolResult("Error: no diagram given.", is_error=True)
        objects = [o for o in (inp.get("objects") or []) if str(o).strip()]
        if not objects and "\\begin{tikzcd}" in strip_fences(latex):
            return ToolResult(
                "Error: list the objects first. `objects` must be every "
                "object you read off the board for this diagram — your "
                "reading of the slate, not a description of what you drew. "
                "It is checked against the diagram, and it is the only way a "
                "dropped object gets caught: one you never noticed leaves no "
                "trace in the drawing for any other check to find.",
                is_error=True)
        if not looks_like_diagram(strip_fences(latex)):
            # Prose compiles perfectly well, so this has to be caught before
            # the compiler says yes to something that is not a diagram.
            return ToolResult(
                "Error: no tikzcd or tikzpicture environment here. Pass the "
                "diagram itself — a \\begin{tikzcd}…\\end{tikzcd} or "
                "\\begin{tikzpicture}…\\end{tikzpicture} block — not the "
                "surrounding text.", is_error=True)
        slug = re.sub(r"[^a-z0-9-]+", "-",
                      (inp.get("name") or "diagram").lower()).strip("-")
        root = Path(ctx.diagrams_dir)
        n = 1
        while (root / f"{slug or 'diagram'}-{n:02d}").exists():
            n += 1
        workdir = root / f"{slug or 'diagram'}-{n:02d}"
        emit(f"  [check_diagram {workdir.name}]")

        # The course preamble has to go in. The notes define \Nb, \utri,
        # \Zhat and the rest, and the agent is told to; a diagram written
        # with them is correct in the document and fails here without them —
        # "Undefined control sequence" for a macro that is in fact defined.
        # Measured cost of omitting it: the gate rejected valid diagrams,
        # reported no parsed errors at all, and the agent either inlined every
        # macro by hand or gave up and wrote prose instead of the diagram.
        # Includes what was added during this run, since the agent may have
        # declared a macro moments ago and used it here.
        preamble = "\n".join(list(ctx.existing_preamble)
                             + list(ctx.new_preamble_additions))
        result = compile_snippet(latex, workdir, preamble)
        if not result.ok:
            return ToolResult(
                f"The diagram does not compile. Fix it and check again "
                f"(line numbers are into your snippet):\n"
                f"{result.describe()}", is_error=True)

        # The inventory diff comes first and is fatal: a diagram missing an
        # object the model itself says is on the board is wrong, and letting
        # it through with a note is how the last one got written.
        body = strip_fences(latex)
        missing = check_inventory(body, objects, inp.get("arrows"))
        if missing:
            return ToolResult(
                "It compiles, but it does not match your own reading of the "
                "board:\n" + "\n".join(f"  - {m}" for m in missing)
                + "\nFix the diagram (or the list, if the list was wrong) "
                  "and check again. Do not write this into the notes as it "
                  "stands.", is_error=True)

        # Compiling proves nothing about fidelity. These two defects are
        # what dropped objects and mis-hung arrows look like structurally,
        # and they cost nothing to find.
        problems = lint(body)
        note = ""
        if problems:
            note = ("\n\nStructural problems, which compiling does not "
                    "catch:\n" + "\n".join(f"  - {p}" for p in problems)
                    + "\nCheck each against the board before you use this.")

        board = next((b for b in ctx.boards
                      if str(b["id"]) == str(inp.get("board"))), None)
        if board is not None:
            at = format_timestamp(board.get("best_at", 0))
            note += (f"\n\nPut this line above the diagram in the notes, so "
                     f"it can be checked against the source later:\n"
                     f"  % board {board['id']} @ {at} — {board['path']}")

        if result.image is None:
            return ToolResult(
                f"Compiles. {result.note} You can use it, but nobody has "
                f"seen it drawn — check it by eye against the board.{note}")
        return _image_result(result.image, (
            "Compiles. Now compare this render against the board — every "
            "object, every arrow, every label, every direction. Anything on "
            "the board and not in the render is the failure that is hardest "
            "to notice, because nothing points at it: go back to your list "
            "of what is in the crop and check each item is here."
            + note))

    handlers: dict[str, Handler] = {
        "fetch_document": fetch_document,
        "search_document": search_document,
        "view_pdf_page": view_pdf_page,
        "clarify_transcript": clarify_transcript,
        "ask_user": ask_user,
        "get_user_answers": get_user_answers,
    }
    if ctx.bib_file:
        handlers["cite_reference"] = cite_reference
    if ctx.enable_preamble:
        handlers["add_to_preamble"] = add_to_preamble
    if ctx.video_path:
        handlers["get_frame"] = get_frame
    if ctx.diagrams_dir is not None:
        handlers["check_diagram"] = check_diagram
        if ctx.boards:
            handlers["crop_board"] = crop_board
    return {name: _logged(ctx, name, fn) for name, fn in handlers.items()}


def _logged(ctx: NotesToolContext, name: str, fn: Handler) -> Handler:
    """Record every call to one of our tools. Wrapping here rather than in
    each backend covers all three of them, and the Codex MCP subprocess too."""
    def wrapper(inp: dict) -> ToolResult:
        log = _ctx_log(ctx)
        started = time.time()
        try:
            result = fn(inp)
        except Exception as exc:
            log.tool(name, inp, f"{type(exc).__name__}: {exc}",
                     time.time() - started, is_error=True)
            raise
        summary = (result.content if isinstance(result.content, str)
                   else f"[{len(result.content)} content block(s)]")
        log.tool(name, inp, summary, round(time.time() - started, 2),
                 is_error=result.is_error)
        return result
    return wrapper


def _ctx_log(ctx: NotesToolContext):
    """The run's log. In the Codex MCP subprocess there is no in-memory log
    object, only the trace path handed over in the serialized context, so
    attach to that file instead."""
    log = getattr(ctx, "log", None)
    if log is not None:
        return log
    from agent_log import AgentLog, NullLog
    if not ctx.log_file:
        return NullLog()
    log = AgentLog.__new__(AgentLog)     # append to the parent's trace
    log.path = Path(ctx.log_file)
    log.index_path = log.path.parent / "index.jsonl"
    log.meta, log.t0, log.tool_counts = {}, time.time(), {}
    log.n_events, log._broken = 0, False
    ctx.log = log
    return log
