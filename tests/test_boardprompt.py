"""Board stills reaching the model: index, transcript interleave, attachment,
unread-board detection, and placeholder detection."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import agent_log
import build_course as BC
from claude_backend import _image_block
from media import format_transcript

root = Path(tempfile.mkdtemp(prefix="boardprompt-"))
lec = root / "lecture-1"
(lec / "boards").mkdir(parents=True)

# Two boards, the second returned to later, plus one whose still never got
# written (segmentation can fail per board) — it must drop, not crash.
(lec / "boards" / "board-01.jpg").write_bytes(b"\xff\xd8\xff\xe0fake jpeg")
(lec / "boards" / "board-02.jpg").write_bytes(b"\xff\xd8\xff\xe0fake jpeg 2")
(lec / "boards" / "boards.json").write_text(json.dumps({"boards": [
    {"id": 1, "intervals": [[0, 300]], "revisits": 0, "seconds": 300,
     "ink": 0.02, "best_at": 200, "image": "board-01.jpg"},
    {"id": 2, "intervals": [[300, 900], [3720, 3900]], "revisits": 1,
     "seconds": 780, "ink": 0.02, "best_at": 800, "image": "board-02.jpg"},
    {"id": 3, "intervals": [[900, 1000]], "revisits": 0, "seconds": 100,
     "ink": 0.01, "best_at": 950, "image": "board-03.jpg"},   # missing file
]}))

boards = BC.load_boards(lec)
assert [b["id"] for b in boards] == [1, 2], "a board with no still must drop"
assert boards[0]["path"].is_absolute()
print(f"loaded {len(boards)} boards (1 dropped for a missing still)")

# --- the index --------------------------------------------------------------
idx = BC.board_index(boards)
assert "01:02:00–01:05:00" in idx, "revisit interval missing or misformatted"
assert "2 visits" in idx and str(boards[1]["path"]) in idx
assert "Open all 2 stills" in idx, "non-api backends need the read instruction"
assert "attached above" in BC.board_index(boards, attached=True)
assert "Open all" not in BC.board_index(boards, attached=True), \
    "don't tell the api backend to re-read images it already has"
assert BC.board_index([]) == "", "no boards => no block at all"
print("index ok")

# --- interleaving into the transcript ---------------------------------------
segments = [{"start": t, "end": t + 30, "text": f"line at {t}"}
            for t in range(0, 4200, 30)]
marks = BC.board_marks(boards)
assert len(marks) == 3, f"one mark per interval, got {len(marks)}"
text = format_transcript(segments, marks)
lines = text.splitlines()
for at, mark in marks:
    i = lines.index(mark)
    assert i + 1 < len(lines), "mark must not be stranded at the end"
    assert lines[i + 1].startswith("[")
    assert i == 0 or not lines[i - 1].startswith("=== "), "marks bunched up"
assert "board 2 up again" in text, "the second visit must say so"
assert text.count("=== board") == 3
assert len(lines) == len(segments) + 3
assert format_transcript(segments) == format_transcript(segments, [])
# a mark past the end of the transcript still gets emitted
assert format_transcript(segments[:2],
                         [(99999.0, "=== late ===")]).endswith("=== late ===")
print("interleave ok")

# --- direct attachment (api backend) ----------------------------------------
blk = _image_block(boards[0]["path"])
assert blk["type"] == "image" and blk["source"]["media_type"] == "image/jpeg"
assert blk["source"]["data"], "empty payload"
assert _image_block(lec / "nope.jpg") is None, "missing file must not raise"
assert _image_block(lec / "boards" / "boards.json") is None, "not an image"
print("attachment ok: base64 jpeg block, missing/non-image => None")

# --- unread-board detection --------------------------------------------------
# The model claimed it had read all the stills and had skipped one.
log = agent_log.AgentLog(root / "t.jsonl", root / "i.jsonl", {"role": "write"})
log.tool("Read", {"file_path": str(boards[0]["path"])})
log.tool("Read", {"file_path": "/elsewhere/notes.tex"})     # not a board
missed = BC.report_unread_boards(SimpleNamespace(log=log), boards)
assert missed == [2], f"board 2 was never opened; got {missed}"
assert log.close()["boards_read"] == [1]
# api backend: no reads at all, so nothing to report rather than "all missed"
assert BC.report_unread_boards(SimpleNamespace(log=agent_log.NullLog()),
                               boards) == []
print("unread-board detection ok")

# --- placeholder detection ---------------------------------------------------
# The failure: the agent delegates a diagram, narrates "I'll insert it once it
# returns", and stops. The comment compiles fine and is invisible in the PDF.
sec = root / "sec.tex"
sec.write_text(r"""\begin{theorem}\label{t}Statement.\end{theorem}
The lecturer's diagram:

% DIAGRAM_PLACEHOLDER

and then some prose.
% TODO: fill in the pushout square here
%% Additions requested by Claude during note generation:
% tikz-cd for commutative diagrams, tikz for everything drawn
Ordinary text with a 100% margin and \todo{a real todo} in it.
""")
hits = BC.report_placeholders(sec)
joined = " | ".join(hits)
assert any("DIAGRAM_PLACEHOLDER" in h for h in hits), joined
assert any("TODO" in h for h in hits), joined
assert not any("Additions requested" in h for h in hits), joined
assert not any("tikz-cd for commutative" in h for h in hits), joined
assert not any("100%" in h for h in hits), joined
assert BC.report_placeholders(root / "nope.tex") == []
sec.write_text("Clean notes with no placeholders.\n")
assert BC.report_placeholders(sec) == []
print(f"placeholder detection ok: caught {len(hits)}, no false positives")

# --- the unread-board warning must not lie about which pass skipped them ----
# It once did: the same message ran after the write, revise and verify passes.
# A checker opens the handful of stills bearing on the claims it doubts, so
# after verification the writer's wording ("reconstructed from audio") implied
# 38 of 45 boards went unread when the writing pass had read every one.
import io, contextlib
import build_course as _B


class _Log:
    def __init__(self, seen): self.boards_seen = set(seen)


class _Ctx:
    def __init__(self, seen): self.log = _Log(seen)


_boards = [{"id": i} for i in range(1, 6)]


def _say(role, seen):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        missed = _B.report_unread_boards(_Ctx(seen), _boards, role=role)
    return missed, buf.getvalue()

missed, out = _say("write", [1, 2])
assert missed == [3, 4, 5]
assert "reconstructed from audio" in out and "3, 4, 5" in out

for role in ("verify", "revise"):
    missed, out = _say(role, [1, 2])
    assert missed == [3, 4, 5], "the return value is the same either way"
    assert "reconstructed from audio" not in out, \
        f"{role} must not claim the notes were written without those boards"
    assert "2 of 5" in out and "1, 2" in out, out

# Nothing skipped, nothing said — in either role.
for role in ("write", "verify"):
    assert _say(role, range(1, 6)) == ([], "")
# No log, or no boards: also silent.
assert _B.report_unread_boards(_Ctx([]), _boards) == []
assert _B.report_unread_boards(_Ctx([1]), []) == []
print("the unread-board warning says the right thing for each pass")

# --- the board listing must forbid delegated transcription ------------------
# Measured over a course: write passes that handed batches of boards to
# subagents failed outright a third of the time (the turn ended with narration
# where the notes should be, losing the lecture), while every pass that read
# the stills itself succeeded. It also cost diagrams — prose about a board is
# not something you can draw a tikzcd from.
_boards_for_prompt = [{"id": 1, "path": "/tmp/board-01.jpg", "revisits": 0,
                       "intervals": [[0.0, 60.0]]}]
_txt = _B.board_index(_boards_for_prompt, attached=False)
assert "Read them YOURSELF" in _txt
assert "subagents" in _txt and "third of the time" in _txt
# The locator is a deliberate exception and must stay allowed.
assert "board-locator" in _txt
# With the images already in the message there is nothing to open, so the
# instruction would be nonsense there.
_att = _B.board_index(_boards_for_prompt, attached=True)
assert "Read them YOURSELF" not in _att and "attached above" in _att
print("board listing forbids delegated transcription, keeps the locator")

print("\nALL OK")
