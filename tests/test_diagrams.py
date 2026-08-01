"""Compiling, rendering, linting and parsing one diagram in isolation."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_course as BC
import diagrams as D
from notes_tools import NotesToolContext, build_handlers, build_tools

root = Path(tempfile.mkdtemp(prefix="dgm-test-"))

GOOD = r"""\begin{tikzcd}
A \arrow[r, "f"] \arrow[d, hook] & B \arrow[d, two heads, "g"] \\
C \arrow[r, dashed] & D
\end{tikzcd}"""
BAD = "\\begin{tikzcd}\nA \\arrow[r, \"f\" & B\n\\end{tikzcd}"
PROSE = r"Let $A \to B$ be a map."
PIC = (r"\begin{tikzpicture}\draw[->] (0,0) -- (2,0) node[right] {$x$};"
       r"\end{tikzpicture}")

# --- compile + render -------------------------------------------------------
r = D.compile_snippet(GOOD, root / "good")
assert r.ok, r.describe()
assert r.image and r.image.exists() and r.image.stat().st_size > 500
print(f"good: compiled, rendered {r.image.stat().st_size}B PNG")

r2 = D.compile_snippet(BAD, root / "bad")
assert not r2.ok and r2.errors, "a malformed diagram must be rejected"
lines = [e.line for e in r2.errors if e.line is not None]
assert not lines or max(lines) <= GOOD.count("\n") + 6, \
    f"line numbers not rebased onto the snippet: {lines}"
print(f"bad : rejected — {r2.describe().splitlines()[0]}")

assert D.strip_fences("```latex\n" + GOOD + "\n```") == GOOD
assert D.strip_fences("```\n" + GOOD + "\n```") == GOOD
assert not D.compile_snippet("   ").ok
assert D.looks_like_diagram(GOOD) and not D.looks_like_diagram(PROSE)
assert D.compile_snippet(PIC, root / "pic").ok and D.looks_like_diagram(PIC)
print("fences, empties, tikzpicture ok")

# --- parsing a tikzcd into a graph ------------------------------------------
# The real board-11 output: arrows resolve through r / u / ur / rrrr.
REAL = r"""\begin{tikzcd}[row sep=3.4em, column sep=2.0em]
M_\infty \arrow[r] \arrow[rrrr, bend left=20, "\mathrm{surj}?"] & \cdots \arrow[r] & M_2 \arrow[r] & M_1 \arrow[r, two heads] & M_0 \\
S_\infty \arrow[u, dashed, "\exists?"] \arrow[r] & \cdots \arrow[r] & S_2 \arrow[u, dashed, "\exists"'] \arrow[r] & S_1 \arrow[u, dashed, "\exists"'] \arrow[ur, "\mathrm{given}"'] & {}
\end{tikzcd}"""
grid, edges = D.parse_tikzcd(REAL)
assert len(grid) == 2 and len(grid[0]) == 5, grid
by = {(e["from"], e["to"]) for e in edges}
assert (r"S_\infty", r"M_\infty") in by, "an up arrow did not resolve"
assert (r"M_\infty", "M_0") in by, "rrrr did not resolve"
assert ("S_1", "M_0") in by, "ur did not resolve"
epi = [e for e in edges if "two heads" in e["style"]]
assert epi and (epi[0]["from"], epi[0]["to"]) == ("M_1", "M_0")
assert {r"\exists?", r"\mathrm{surj}?"} <= {e["label"] for e in edges}
print(f"parse: {len(grid)}x{len(grid[0])} grid, {len(edges)} edges resolved")

# --- the structural lint ----------------------------------------------------
problems = D.lint(REAL)
assert any("empty" in p for p in problems), problems     # the dropped object
OFF = ("\\begin{tikzcd}\nA \\arrow[r, \"f\"] & B \\arrow[rr, \"g\"] \\\\ "
       "C & D\n\\end{tikzcd}")
assert any("off the edge" in p for p in D.lint(OFF)), D.lint(OFF)
assert D.lint(GOOD) == [], D.lint(GOOD)                  # no crying wolf
assert D.lint(PIC) == []                                 # not a grid
print(f"lint: {len(problems)} problem(s) on the real output, clean on a clean one")

# --- the tool as the model sees it ------------------------------------------
img = root / "b.jpg"
img.write_bytes(b"\xff\xd8\xff\xe0fake")
ctx = NotesToolContext(refs_dir=root / "refs", diagrams_dir=root / "dgm",
                       boards=[{"id": 3, "path": img, "best_at": 120.0,
                                "intervals": [[0, 200]], "revisits": 0}])
assert {"check_diagram", "crop_board"} <= {t["name"] for t in build_tools(ctx)}
h = build_handlers(ctx)["check_diagram"]

ok = h({"latex": GOOD, "name": "Pushout Square!", "objects": list("ABCD"),
        "board": 3})
assert not ok.is_error and isinstance(ok.content, list)
assert [b["type"] for b in ok.content] == ["text", "image"], \
    "the render must come back as an image, not just a path"
assert "% board 3 @ 00:02:00" in ok.content[0]["text"], ok.content[0]["text"]
assert (root / "dgm" / "pushout-square-01").is_dir()
h({"latex": GOOD, "name": "Pushout Square!", "objects": list("ABCD")})
assert (root / "dgm" / "pushout-square-02").is_dir(), "name collision"
print("tool ok: image + provenance returned, slug sanitised, collisions numbered")

assert h({"latex": BAD, "objects": ["A", "B"]}).is_error
nod = h({"latex": PROSE, "objects": ["A"]})
assert nod.is_error and "no tikzcd or tikzpicture" in nod.content, \
    "prose compiles fine, so it has to be caught before the compiler"
assert h({"latex": "", "objects": ["A"]}).is_error

bare = NotesToolContext(refs_dir=root / "refs")
assert "check_diagram" not in {t["name"] for t in build_tools(bare)}
assert "check_diagram" not in build_handlers(bare)
print("absent without diagrams_dir")

# --- the preamble the notes compile against ---------------------------------
pre, _ = BC.course_preamble("T", {"preamble_additions": []}, with_bib=False)
assert "tikz-cd" in pre and "usetikzlibrary" in pre
assert pre.index(r"\usepackage{tikz-cd}") < pre.index(r"]{hyperref}"), \
    "packages must load before hyperref/cleveref"
print("course preamble loads tikz-cd before hyperref")

# --- notation that is not a map ---------------------------------------------
# tikz-cd draws a pushout/pullback corner with a phantom arrow. It renders
# nothing; counting it as a map invented an arrow in the benchmark.
PUSHOUT = r"""\begin{tikzcd}
S_\infty \arrow[r, hook] \arrow[d] & S \arrow[d] \\
\{\infty\} \arrow[r, hook] & \mathbb N\cup\{\infty\} \arrow[ul, phantom, "\ulcorner", very near start]
\end{tikzcd}"""
_, pushout_edges = D.parse_tikzcd(PUSHOUT)
pairs = {(e["from"], e["to"]) for e in pushout_edges}
assert len(pushout_edges) == 4, [(e["from"], e["to"]) for e in pushout_edges]
assert (r"\mathbb N\cup\{\infty\}", r"S_\infty") not in pairs, \
    "the corner mark is not an arrow"
assert (r"S_\infty", "S") in pairs and (r"S_\infty", r"\{\infty\}") in pairs
assert D.lint(PUSHOUT) == [], D.lint(PUSHOUT)
print("phantom corner marks are not counted as maps")

# --- the compile gate must see the notes' own macros ------------------------
# It did not, and the cost was real: the notes define macros of their own (the
# agent is instructed to), a diagram written with them failed to compile in
# the gate, and -file-line-error meant the failure parsed to zero errors — so
# the agent was told "pdflatex failed." with no detail. It spent its attempts
# blind and then wrote prose instead of the diagram.
import tempfile as _tf
from pathlib import Path as _P
import diagrams as _D
from latex_check import _parse_errors as _pe
from notes_tools import NotesToolContext as _Ctx, build_handlers as _bh

_COURSE = (r"\newcommand{\Nb}{\mathbb{N}}" "\n"
           r"\newcommand{\Zhat}{\widehat{\mathbb{Z}}}" "\n"
           r"\newcommand{\utri}[1]{{#1}^{\triangleright}}")
_SNIPPET = (r"\begin{tikzcd} \utri{A} \arrow[r] & \Zhat \\ "
            r"\Nb \arrow[u] & B \arrow[l] \end{tikzcd}")

_r = _D.compile_snippet(_SNIPPET, _P(_tf.mkdtemp()), _COURSE)
assert _r.ok, f"course macros must compile when the preamble is passed: {_r.describe()}"

_r = _D.compile_snippet(_SNIPPET, _P(_tf.mkdtemp()), "")
assert not _r.ok
# Whatever else it says, it must not be an information-free failure.
assert _r.describe() != "pdflatex failed.", "a failure must say what went wrong"
assert "utri" in _r.describe(), _r.describe()
print("compile gate: course macros compile; a bare failure names the cause")

# --- the parser understands both of TeX's error formats ---------------------
_PLAIN = "! Undefined control sequence.\nl.7 \\begin{tikzcd} \\utri\n"
_FILELINE = "./diagram.tex:7: Undefined control sequence.\nl.7 \\begin{tikzcd} \\utri\n"
for _name, _log in (("plain", _PLAIN), ("-file-line-error", _FILELINE)):
    _errs = _pe(_log)
    assert len(_errs) == 1, f"{_name}: got {_errs}"
    assert _errs[0].message == "Undefined control sequence."
    assert _errs[0].line == 7, f"{_name}: line {_errs[0].line}"
# Both spellings of one error must not be counted twice.
assert len(_pe(_PLAIN + _FILELINE)) == 1
# An ordinary log line carrying a colon and a number is not an error.
for _benign in ("(/nix/store/abc/tex/latex/base/article.cls",
                "Package hyperref Warning: Token not allowed: 3",
                "Output written on diagram.pdf (1 page, 12345 bytes).",
                "LaTeX Font Info:    Font shape `OMS/cmr/m/n' in size <10>"):
    assert _pe(_benign) == [], f"false positive on: {_benign}"
print("error parser: reads both TeX formats, dedupes, no false positives")

# --- and check_diagram actually hands the preamble over ---------------------
_ctx = _Ctx(refs_dir=_P(_tf.mkdtemp()), diagrams_dir=_P(_tf.mkdtemp()),
            enable_preamble=True, existing_preamble=[_COURSE])
_h = _bh(_ctx)["check_diagram"]
_out = _h({"latex": _SNIPPET, "objects": ["\\utri{A}", "\\Zhat", "\\Nb", "B"]})
assert not getattr(_out, "is_error", False), \
    f"the tool must compile a diagram using course macros: {_out}"
# A macro added mid-run, not yet in existing_preamble, must work too.
_ctx2 = _Ctx(refs_dir=_P(_tf.mkdtemp()), diagrams_dir=_P(_tf.mkdtemp()),
             enable_preamble=True, existing_preamble=[])
_ctx2.new_preamble_additions.append(r"\newcommand{\Freshly}{\mathbb{F}}")
_h2 = _bh(_ctx2)["check_diagram"]
_out2 = _h2({"latex": r"\begin{tikzcd} \Freshly \arrow[r] & B \end{tikzcd}",
             "objects": ["\\Freshly", "B"]})
assert not getattr(_out2, "is_error", False), \
    f"a macro declared during this run must count too: {_out2}"
print("check_diagram: passes both the stored and the just-added preamble")

print("\nALL OK")
