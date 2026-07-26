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

import json
import re
import sys
import threading
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


@dataclass
class NotesToolContext:
    refs_dir: Path
    video_path: Path | None = None
    total_duration: float = 0.0
    enable_preamble: bool = False
    existing_preamble: list = field(default_factory=list)
    state_file: Path | None = None
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
                            "Author(s), BibTeX style: 'Peter Scholze' or "
                            "'Dustin Clausen and Peter Scholze'. Supply this "
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
                "A single frame may not be enough: the board may be mid-erasure, "
                "a slide may be transitioning, or the camera may be panning. "
                "Call this tool multiple times with nearby timestamps (e.g. a few "
                "seconds before and after) to build up a clear picture of what is "
                "being shown before transcribing it."
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

    handlers: dict[str, Handler] = {
        "fetch_document": fetch_document,
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
    return handlers
