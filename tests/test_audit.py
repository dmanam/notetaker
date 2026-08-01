"""The run audit: columns line up, and zero is distinguished from unknown.

Both properties are here because both broke. A patch updated the header row
and silently failed to update the row that prints the values, so every number
appeared under the wrong heading — and the table was read as fact before
anyone noticed. A misaligned diagnostic is worse than no diagnostic.
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import audit_run as A

root = Path(tempfile.mkdtemp(prefix="audit-"))
(root / "logs").mkdir()

SECTION = r"""\section{Lecture 1: Test}
\begin{equation}\label{eq:1:a}x=y\end{equation}
% board 4 @ 00:10:00 — boards/board-04.jpg
\begin{tikzcd} A \arrow[r] & B \end{tikzcd}
\todo{check this}
"""

d = root / "lec-one"
(d / "boards").mkdir(parents=True)
(d / "section.tex").write_text(SECTION)
(d / "boards" / "boards.json").write_text(json.dumps(
    {"boards": [{"id": i} for i in range(1, 6)]}))
(d / "transcript.json").write_text(json.dumps({"segments": [
    {"text": "Let me draw the diagram here."},
    {"text": "This diagram commutes, and a diagram chase finishes it."}]}))
(root / "course_state.json").write_text(json.dumps(
    {"sections": {"lec-one": {"lecture_num": 1, "body": SECTION}}}))

# --- 0 opened and no-log-at-all must not share a symbol ----------------------
assert A.boards_opened(root / "logs", "lec-one", "verify") is None
(root / "logs" / "20260730-000000-verify-lec-one.jsonl").write_text(
    json.dumps({"kind": "tool", "name": "Read"}) + "\n"
    + json.dumps({"usage": {"in": 1}, "tools": {}}) + "\n")
assert A.boards_opened(root / "logs", "lec-one", "verify") == 0, \
    "a closed log with no boards_read is a real zero, not missing data"
(root / "logs" / "20260730-000001-write-lec-one.jsonl").write_text(
    json.dumps({"boards_read": [1, 2, 3, "… [+2 more]"]}) + "\n"
    + json.dumps({"usage": {}}) + "\n")
assert A.boards_opened(root / "logs", "lec-one", "write") == 5, \
    "the clipped '+N more' marker carries the rest of the count"
print("boards_opened: None, 0 and a clipped list are three distinct answers")

# --- the cue counter --------------------------------------------------------
assert A.diagram_cues(d) == 4, A.diagram_cues(d)     # draw, the diagram, this diagram, diagram chase
noop = root / "empty"; noop.mkdir()
assert A.diagram_cues(noop) == 0
# A distinguished triangle is not a drawing; this course says it constantly.
(noop / "transcript.json").write_text(json.dumps({"segments": [
    {"text": "We get a distinguished triangle and a fibre triangle."}]}))
assert A.diagram_cues(noop) == 0, "'triangle' must not count as a drawing"
print("diagram_cues: counts pointing-at-the-board, ignores triangles")

# --- header and rows must have the same number of columns -------------------
out = subprocess.run([sys.executable, str(ROOT / "audit_run.py"),
                      "--output-dir", str(root)],
                     capture_output=True, text=True, cwd=ROOT)
assert out.returncode == 0, out.stderr
lines = [l for l in out.stdout.splitlines() if l.strip()]
header = lines[0].split()
row = lines[1].split()
# A slug never contains whitespace, so the row splits into exactly as many
# fields as the header. (The "!" flags append to a field rather than adding
# one, which is what keeps that true when a row is flagged.)
assert header[0] == "#" and header[1] == "lecture"
assert len(row) == len(header), \
    f"header has {len(header)} columns, row has {len(row)}:\n{lines[0]}\n{lines[1]}"
got = dict(zip(header[2:], row[2:]))
assert got["cue"] == "4", got
assert got["diag"] == "1", got
assert got["prov"] == "1", got
assert got["eq"] == "1", got
assert got["todo"] == "1", got
assert got["W"] == "5/5", got
assert got["V"] == "0", got
print(f"audit table columns align: {got}")

# --- the flag fires on exactly the failure it is for ------------------------
(d / "transcript.json").write_text(json.dumps({"segments": [
    {"text": "let me draw the diagram"}, {"text": "this diagram, that diagram"}]}))
(d / "section.tex").write_text("\\section{No diagrams}\n")
(root / "course_state.json").write_text(json.dumps(
    {"sections": {"lec-one": {"lecture_num": 1, "body": "no diagrams"}}}))
out = subprocess.run([sys.executable, str(ROOT / "audit_run.py"),
                      "--output-dir", str(root)],
                     capture_output=True, text=True, cwd=ROOT)
assert "the diagrams were lost" in out.stdout, out.stdout
assert "Every writing pass opened every board still." in out.stdout
print("the lost-diagram flag fires when cues are many and diagrams are none")

print("\nALL OK")
