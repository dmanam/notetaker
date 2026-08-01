"""The declared-inventory gate on check_diagram.

The failure this exists for: reading a board, the model missed one object
entirely, hung the map that should have started there onto a neighbouring
object instead, left an empty cell where the missing one belonged, called
check_diagram once at the very end, was told about the empty cell, and wrote
the notes anyway. Most of its arrow errors were downstream of that single
missed object.

Declaring the reading first is the only check that can catch that: every
other one compares the diagram to itself or to a photograph, and an object
nobody noticed is absent from the diagram, so there is nothing in it to point
at the absence.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diagrams import check_inventory
from notes_tools import NotesToolContext, build_handlers, build_tools

root = Path(tempfile.mkdtemp(prefix="inv-"))
img = root / "board-11.jpg"
img.write_bytes(b"\xff\xd8\xff\xe0fake")
ctx = NotesToolContext(refs_dir=root / "refs", diagrams_dir=root / "dgm",
                       boards=[{"id": 11, "path": img, "best_at": 3212.0,
                                "intervals": [[3196.0, 3401.0]],
                                "revisits": 0}])
check = build_handlers(ctx)["check_diagram"]

# Verbatim what the pipeline produced for one board, and the reading it should
# have declared. S is on the board and missing from the drawing.
DREW = r"""\begin{tikzcd}
M_\infty \ar[r] & \cdots \ar[r] & M_2 \ar[r] & M_1 \ar[r, two heads] & M_0 \\
S_\infty \ar[u, dashed, "\exists?"] \ar[r] & \cdots \ar[r] & S_2 \ar[u, dashed] \ar[r] & S_1 \ar[u, dashed] \ar[ur] &
\end{tikzcd}"""
READING = ["M_\\infty", "M_2", "M_1", "M_0", "S_\\infty", "S_2", "S_1", "S"]

problems = check_inventory(DREW, READING)
assert any("'S'" in p and "not in the diagram" in p for p in problems), problems
print(f"catches the dropped object: {problems[0][:64]}…")

# Fatal through the tool, not a footnote on a success: letting it through
# with a warning is exactly how the bad diagram got written.
r = check({"latex": DREW, "objects": READING, "board": 11, "name": "b11"})
assert r.is_error, "a diagram missing a declared object must be refused"
assert "does not match your own reading" in r.content and "S" in r.content
print("tool refuses it outright")

undeclared = check({"latex": DREW, "board": 11})
assert undeclared.is_error and "list the objects first" in undeclared.content
assert check({"latex": DREW, "objects": [], "board": 11}).is_error
assert check({"latex": DREW, "objects": ["  "], "board": 11}).is_error
print("an undeclared reading is refused")

FIXED = DREW.replace(r"\ar[ur] &", r"\ar[r] & S")
ok = check({"latex": FIXED, "objects": READING, "board": 11, "name": "fix"})
assert not ok.is_error, ok.content
print("passes once the object is drawn")

# \cdots is layout, not an undeclared object.
assert not [p for p in check_inventory(FIXED, READING) if "cdots" in p]
# Declared names are normalised exactly as drawn ones are, so the board's
# "S \in Pro_N(Fin)" and a diagram's bare S are the same object.
assert check_inventory(
    FIXED, [o for o in READING if o != "S"]
    + ["S \\in \\mathrm{Pro}_{\\mathbb N}(\\mathrm{Fin})"]) == []
print("filler ignored; declared names normalised like drawn ones")

# Drawn but never declared is reported the other way round — either the
# reading missed it or the diagram invented it.
assert any("not in your list" in p for p in check_inventory(FIXED, READING[:-1]))

# Optional arrows are diffed too. This is the mis-hung "given" map: the board
# has S -> M_0 and the diagram hung it off S_1.
assert any("no such arrow" in p
           for p in check_inventory(FIXED, READING, [{"from": "S",
                                                      "to": "M_0"}]))
assert check_inventory(FIXED, READING, [{"from": "M_1", "to": "M_0"}]) == []
print("declared arrows are diffed when given")

# A drawn picture has no node/arrow grid, so the gate must stay out of it.
PIC = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
assert check_inventory(PIC, []) == []
assert not check({"latex": PIC, "objects": [], "name": "pic"}).is_error
print("tikzpicture exempt")

spec = next(t for t in build_tools(ctx) if t["name"] == "check_diagram")
assert spec["input_schema"]["required"] == ["latex", "objects"]
print("objects is required in the schema")

print("\nALL OK")
