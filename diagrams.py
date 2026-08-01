"""
diagrams.py — turn a board photograph into TikZ that compiles and matches it.

A photograph cannot go into the notes, and prose is a poor substitute for a
diagram, so the drawing has to be redrawn. This module owns the mechanical
half of that: compile a snippet in isolation, rasterise the result, and hand
both the render and the original board back to whoever is judging them.

The loop it supports is:

    cheap model — says WHERE on the board the diagram is (a box)
    crop        — that region at native resolution, instead of the whole
                  slate shrunk to the vision model's ceiling
    main model  — reads the diagram off the crop and writes the TikZ; it is
                  the one that knows what the lecture proves, and an arrow
                  direction is a mathematical claim, not a typesetting choice
    compile     — standalone document, real pdflatex, real errors
    render      — PDF to PNG via PyMuPDF (already a dependency; no poppler,
                  no ghostscript, no ImageMagick)
    main model  — compares its render against the crop and fixes what differs

The division of labour is not an assumption. A cheap model will draw the same
diagram with its arrows reversed however many magnified looks at the board it
is given. Locating a region it can do; reading an arrowhead it cannot.

Two things are deliberate. The snippet is compiled *alone*, so a broken
diagram is caught here rather than taking the whole course build down with
it. And a diagram that will not come right is not silently emitted: the notes
fall back to prose with a \\todo, because a confident-looking commutative
diagram with an arrow reversed is worse than no diagram at all.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from latex_check import LatexError, _parse_errors

# The standalone wrapper. `border` keeps a little whitespace so the render is
# not clipped; varwidth lets a wide diagram lay out rather than run off.
WRAPPER = r"""\documentclass[border=6pt,varwidth=20cm]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz-cd}
\usetikzlibrary{arrows.meta,decorations.pathmorphing,positioning,calc,patterns}
%(preamble)s
\begin{document}
%(body)s
\end{document}
"""

RENDER_DPI = 160
TIMEOUT = 120


def _log_tail(text: str, lines: int = 12) -> str:
    """The log's own complaint, for a failure _parse_errors could not read.

    Prefers the lines TeX marks as trouble ("!", "l.NN", a missing file) and
    falls back to the end of the log, which is where it gave up."""
    marked = [l.rstrip() for l in (text or "").splitlines()
              if l.startswith("!") or l.startswith("l.")
              or "not found" in l or "Emergency stop" in l]
    if not marked:
        marked = [l.rstrip() for l in (text or "").splitlines() if l.strip()][-lines:]
    if not marked:
        return "pdflatex failed and wrote no log."
    return ("pdflatex failed. Its log says:\n"
            + "\n".join(f"  {l}" for l in marked[:lines]))


@dataclass
class DiagramResult:
    ok: bool
    latex: str
    errors: list[LatexError] = field(default_factory=list)
    image: Path | None = None          # the rendered PNG, when it compiled
    note: str = ""

    def describe(self) -> str:
        if self.ok:
            return "Compiled cleanly."
        if not self.errors:
            return self.note or "Did not compile (no errors parsed)."
        return "\n".join(e.describe() for e in self.errors)


_ENVS = ("tikzcd", "tikzpicture")


def looks_like_diagram(latex: str) -> bool:
    return any(f"\\begin{{{e}}}" in latex for e in _ENVS)


def strip_fences(latex: str) -> str:
    """Models like to wrap code in ``` fences; TeX does not."""
    text = latex.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


# Arrow styles that mean the same thing, mapped to one spelling.
_STYLE_ALIASES = {
    "two heads": "epi", "twoheadrightarrow": "epi", "->>": "epi",
    "hook": "mono", "hookrightarrow": "mono", "right hook": "mono",
    "dashed": "dashed", "dotted": "dashed",      # a board rarely distinguishes
    "bend left": "", "bend right": "", "": "",
}

_FONTS = r"mathrm|mathbb|mathcal|mathscr|mathfrak|mathsf|operatorname|text|mathop"
_DROP = re.compile(rf"\\(?:{_FONTS})\s*\{{([^{{}}]*)\}}")
# The same commands written without braces: \mathbb N, \mathrm Fin. Dropping
# the font conflates \mathcal{F} with F, which is a real if rare collision;
# a case that needs the distinction should use different letters.
_DROP_BARE = re.compile(rf"\\(?:{_FONTS})\b\s*")


_NEWCOMMAND = re.compile(
    r"\\(?:re)?newcommand\s*\*?\s*\{?\s*\\(\w+)\s*\}?\s*(?:\[(\d)\])?"
    r"(?:\[[^\]]*\])?\s*\{", re.M)
_MATHOP = re.compile(r"\\DeclareMathOperator\s*\*?\s*\{?\s*\\(\w+)\s*\}?\s*"
                     r"\{([^{}]*)\}")


def _balanced(text: str, start: int) -> tuple[str, int]:
    """The contents of the brace group beginning at `start`, and where it ends."""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "{" and (i == start or text[i - 1] != "\\"):
            depth += 1
        elif c == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)


def macro_table(preamble: str) -> dict:
    """{name: (arity, body)} for the \\newcommand and \\DeclareMathOperator in
    a preamble.

    The scorer needs this because the notes are *supposed* to define macros —
    the agent is told to put \\newcommand{\\Nb}{\\mathbb{N}} in the preamble
    rather than spelling it out everywhere. Comparing the raw source against
    ground truth written in plain LaTeX then reports correct mathematics as
    invented: \\Nb\\cup\\infty and \\mathbb{N}\\cup\\infty are the same object
    and normalise to different strings. That is a defect in the measurement,
    and it fails in the direction that matters — it makes a good run look bad."""
    table: dict[str, tuple[int, str]] = {}
    for m in _MATHOP.finditer(preamble or ""):
        table[m.group(1)] = (0, rf"\mathrm{{{m.group(2)}}}")
    for m in _NEWCOMMAND.finditer(preamble or ""):
        body, _end = _balanced(preamble, m.end() - 1)
        table[m.group(1)] = (int(m.group(2) or 0), body)
    return table


def expand(text: str, table: dict, rounds: int = 4) -> str:
    """Substitute the macros in `table`, including one-argument ones.

    Iterated, since a macro body may use another macro (\\Zhat in terms of
    \\Zb). Unknown macros are left alone: an unexpanded \\foo compared against
    an unexpanded \\foo still matches, so ignorance is safe here."""
    if not text or not table:
        return text or ""
    for _ in range(rounds):
        before = text
        for name, (arity, body) in table.items():
            pat = re.compile(rf"\\{re.escape(name)}(?![a-zA-Z])")
            if arity == 0:
                text = pat.sub(lambda _m: body, text)
                continue
            out, pos = [], 0
            for m in pat.finditer(text):
                if m.start() < pos:
                    continue
                out.append(text[pos:m.start()])
                rest = text[m.end():]
                stripped = rest.lstrip()
                pad = len(rest) - len(stripped)
                if stripped.startswith("{"):
                    arg, end = _balanced(stripped, 0)
                    out.append(body.replace("#1", arg))
                    pos = m.end() + pad + end
                elif stripped:                     # \utri A — one token
                    out.append(body.replace("#1", stripped[0]))
                    pos = m.end() + pad + 1
                else:
                    out.append(m.group(0))
                    pos = m.end()
            out.append(text[pos:])
            text = "".join(out)
        if text == before:
            break
    return text


# What makes an expression a construction rather than a name for one: a
# binary operation, a limit, a bracketed argument. "S", "S'", "M_0",
# "\tilde N" are names; "2\Nb\cup\{\infty\}", "\varprojlim_n S_n" and
# "\Pro(\Fin)" are not.
_CONSTRUCTION = re.compile(
    r"\\(?:cup|cap|amalg|sqcup|oplus|otimes|times|coprod|prod|sum"
    r"|varprojlim|projlim|varinjlim|injlim|lim|colim)\b|[+(]")


def _is_name(text: str) -> bool:
    """Does this read as a label for an object rather than a construction?"""
    return not _CONSTRUCTION.search(text or "")


def normalise(text: str | None, macros: dict | None = None) -> str:
    """One spelling for an object or label, so cosmetic differences between
    two correct diagrams do not read as errors."""
    if not text:
        return ""
    s = text.strip().strip("$").strip()
    if macros:
        s = expand(s, macros)
    for _ in range(3):                       # \mathrm{Pro}_{\mathbb N}(...)
        s2 = _DROP.sub(r"\1", s)
        if s2 == s:
            break
        s = s2
    s = _DROP_BARE.sub("", s)
    # A node written "S \in Pro_N(Fin)" names the object S and says where it
    # lives; the identity is S, and boards annotate objects this way
    # constantly. This has to happen while the spaces are still here: once
    # they are gone, "\in Pro" and "\infty" both read as \in followed by a
    # letter and cannot be told apart.
    parts = re.split(r"\\in(?![a-zA-Z])", s)
    if len(parts) > 1 and parts[0].strip():
        s = parts[0]
    else:
        parts = re.split(r"\\ni(?![a-zA-Z])", s)
        if len(parts) > 1 and parts[-1].strip():
            s = parts[-1]
    # Likewise "M_\infty = \varprojlim(...)" or "S' = (2N u {inf}) u ...":
    # the board names an object and then says what it is. The node is the
    # name. (Labels are not scored, so splitting on = cannot damage them.)
    #
    # But the two halves come in either order. A board writes both
    # "S_\infty = \varprojlim S_n" and "\Nb\cup\{\infty\} = S", and taking the
    # left side blindly picks the description in the second case. Worse, it
    # then collapses two distinct nodes onto one spelling: a diagram with
    # \Nb\cup\{\infty\} in one cell and \Nb\cup\{\infty\}=S in another lost
    # the second node entirely, and the arrow between them became a self-loop
    # — reported as a missing object and an invented arrow, both spurious.
    # So prefer whichever side is a *name*: short, and free of the operators
    # that make an expression a construction rather than a label.
    if "=" in s:
        sides = [p.strip() for p in s.split("=")]
        sides = [p for p in sides if p]
        if sides:
            named = [p for p in sides if _is_name(p)]
            s = (named[0] if len(named) == 1 else sides[0])
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\varprojlim|\\projlim|\\lim", "lim", s)
    # \widetilde N and \tilde N are the same object; so are \widehat/\hat.
    s = s.replace("\\widetilde", "\\tilde").replace("\\widehat", "\\hat")
    s = s.replace("\\{", "").replace("\\}", "")   # before the brace strip,
    # NOT the prime: S' and S are different objects, and conflating them
    # would let the benchmark score a diagram right for drawing the wrong
    # one. (tikzcd's label-placement quote lives in the arrow spec, which is
    # stripped before this, so nothing else needs it gone.)
    s = re.sub(r"[{}\s$~]", "", s)                # or a lone \ is left behind
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    # Trailing sentence punctuation: a board writes "N u {inf}." at the end
    # of a line and the node is still N u {inf}.
    s = s.rstrip(".,;:")
    return s.lower()


def normalise_style(styles: list[str]) -> str:
    out = {_STYLE_ALIASES.get(s.strip().split("=")[0].strip(), "")
           for s in styles}
    return ",".join(sorted(x for x in out if x))


def edge_key(e: dict) -> tuple:
    return (normalise(e.get("from")), normalise(e.get("to")),
            normalise_style(e.get("style") or []))


_DIRS = {"r": (0, 1), "l": (0, -1), "d": (1, 0), "u": (-1, 0)}
_ARROW = re.compile(r"\\(?:arrow|ar)\[([^\[\]]*(?:\[[^\]]*\][^\[\]]*)*)\]")


def parse_tikzcd(latex: str) -> tuple[list[list[str]], list[dict]]:
    """A tikzcd body as (grid of node texts, edges).

    Arrows in tikzcd are written inside the cell they leave, with a direction
    like `r`, `ur`, `rrrr`; resolving those against the grid recovers an
    actual graph. That is what makes a diagram checkable by machine rather
    than only by eye — and omissions, which are what self-review misses, show
    up here as a node that is empty or an arrow that lands nowhere."""
    m = re.search(r"\\begin\{tikzcd\}(?:\[[^\]]*\])?(.*)\\end\{tikzcd\}",
                  latex, re.DOTALL)
    inner = m.group(1) if m else latex
    grid: list[list[str]] = []
    specs: list[tuple[int, int, str]] = []
    for row in re.split(r"\\\\", inner):
        if not row.strip():
            continue
        cells = []
        for j, cell in enumerate(row.split("&")):
            found: list[str] = []
            text = _ARROW.sub(lambda mm: found.append(mm.group(1)) or "", cell)
            cells.append(text.strip())
            specs.extend((len(grid), j, s) for s in found)
        grid.append(cells)

    def at(r: int, c: int) -> str | None:
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
            return grid[r][c].strip()
        return None

    edges = []
    for r, c, spec in specs:
        # `phantom` draws no arrow: it is how tikz-cd places the corner mark
        # of a pushout or pullback square. Counting it as a map invents one.
        if "phantom" in spec:
            continue
        head = spec.split(",")[0].strip()
        if not head or not set(head) <= set("rlud"):
            continue          # bend/loop syntax we do not resolve; not an error
        dr = sum(_DIRS[ch][0] for ch in head)
        dc = sum(_DIRS[ch][1] for ch in head)
        label = re.search(r'"((?:[^"\\]|\\.)*)"', spec)
        edges.append({
            "from": at(r, c), "to": at(r + dr, c + dc),
            "off_grid": at(r + dr, c + dc) is None,
            "label": label.group(1) if label else None,
            "style": [w.strip() for w in spec.split(",")[1:]
                      if w.strip() and not w.strip().startswith('"')],
        })
    return grid, edges


_FILLER = {"cdots", "dots", "ldots", "vdots", "ddots", "hdots", ""}


def is_filler(text: str | None) -> bool:
    """\\cdots and friends stand for "and so on", not for an object."""
    return normalise(text).lstrip("\\") in _FILLER


def check_inventory(latex: str, objects: list, arrows: list | None = None
                    ) -> list[str]:
    """Diff what the model says it read off the board against what it drew.

    This is the one check that catches an omission. Everything else compares
    the diagram to itself or to a picture, and an object nobody noticed
    leaves no trace in either — it is absent from the drawing, so the drawing
    has nothing to point at, and self-review confirms what is there rather
    than finding what is not. Declaring the reading first turns the omission
    into a mismatch between two lists, which is mechanical."""
    if "\\begin{tikzcd}" not in latex:
        return []                      # tikzpicture has no node/arrow grid
    grid, edges = parse_tikzcd(latex)
    drawn = {normalise(c) for row in grid for c in row
             if normalise(c) and not is_filler(c)}
    declared = {normalise(o) for o in objects if not is_filler(o)}

    problems = []
    for name, o in sorted((normalise(o), o) for o in objects):
        if name and not is_filler(o) and name not in drawn:
            problems.append(
                f"you read {o!r} off the board but it is not in the diagram — "
                f"either draw it, or say why it does not belong.")
    for name in sorted(drawn - declared):
        problems.append(
            f"the diagram contains {name!r}, which is not in your list of "
            f"what is on the board. Either it belongs and your reading "
            f"missed it, or you invented it.")

    if arrows:
        drawn_pairs = {(e["from"], e["to"]) for e in edges}
        drawn_pairs = {(normalise(a), normalise(b)) for a, b in drawn_pairs
                       if not is_filler(a) and not is_filler(b)}
        for a in arrows:
            pair = (normalise(a.get("from")), normalise(a.get("to")))
            if pair not in drawn_pairs:
                problems.append(
                    f"you read an arrow {a.get('from')} -> {a.get('to')} off "
                    f"the board; the diagram has no such arrow. A missing "
                    f"arrow and one hung off the wrong object look the same "
                    f"here — check which it is.")
    return problems


_EMPTY_CELL = {"", "{}", "~", "\\ ", "{ }", "\\phantom{}"}


def lint(latex: str) -> list[str]:
    """Structural defects that no compiler complains about.

    Both come from real failures: an empty cell is what is left behind when
    an object on the board was dropped from the diagram, and an arrow whose
    target falls outside the grid means it was hung off whatever node
    happened to be nearby instead of its real source."""
    if "\\begin{tikzcd}" not in latex:
        return []
    grid, edges = parse_tikzcd(latex)
    problems = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell.strip() in _EMPTY_CELL and any(
                    e["from"] == cell or e["to"] == cell for e in edges):
                problems.append(
                    f"row {r + 1}, column {c + 1} is an empty node with "
                    f"arrows attached — an object was probably dropped.")
            elif cell.strip() in _EMPTY_CELL and (r or c):
                problems.append(
                    f"row {r + 1}, column {c + 1} is empty. If it is padding "
                    f"that is fine; if an object belongs there, it is missing.")
    for e in edges:
        if e["off_grid"]:
            label = f' labelled "{e["label"]}"' if e["label"] else ""
            problems.append(
                f"the arrow{label} out of {e['from'] or 'an empty cell'} "
                f"points off the edge of the diagram — check it starts and "
                f"ends where the board has it.")
    return problems


def compile_snippet(latex: str, workdir: Path | None = None,
                    preamble: str = "") -> DiagramResult:
    """Compile one diagram on its own and rasterise it.

    Isolation is the point: an error here is attributable to this snippet and
    to nothing else, which is what makes the repair loop tractable."""
    latex = strip_fences(latex)
    if not latex:
        return DiagramResult(False, latex, note="Empty diagram.")

    # Not a TemporaryDirectory: the render outlives this call (the verifier
    # looks at it next), so cleanup is the caller's business.
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="dgm-"))
    workdir.mkdir(parents=True, exist_ok=True)

    src = workdir / "diagram.tex"
    src.write_text(WRAPPER % {"preamble": preamble, "body": latex})
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "diagram.tex"],
            cwd=workdir, capture_output=True, text=True,
            errors="replace", timeout=TIMEOUT)
    except FileNotFoundError:
        return DiagramResult(True, latex, note="No TeX installation; "
                                               "compile check skipped.")
    except subprocess.TimeoutExpired:
        return DiagramResult(False, latex,
                             note=f"pdflatex timed out after {TIMEOUT}s — "
                                  f"the diagram is probably looping.")
    log = workdir / "diagram.log"
    text = log.read_text(errors="replace") if log.exists() else proc.stdout
    pdf = workdir / "diagram.pdf"
    if proc.returncode != 0 or not pdf.exists():
        errors = _parse_errors(text)
        # Line numbers point into the wrapper; rebase them onto the snippet.
        offset = WRAPPER.split("%(body)s")[0].count("\n")
        for e in errors:
            if e.line is not None:
                e.line = max(1, e.line - offset)
        # A failure the parser cannot read must still say something. "pdflatex
        # failed." on its own is the worst possible answer: the agent is told
        # to fix a diagram and given nothing to fix, so it burns attempts
        # blind and then abandons the diagram. Hand back the log's own
        # complaint lines instead.
        return DiagramResult(False, latex, errors=errors,
                             note="" if errors else _log_tail(text))
    png = render_pdf(pdf, workdir / "diagram.png")
    return DiagramResult(True, latex, image=png,
                         note="" if png else "Compiled, but could not be "
                                             "rendered to an image.")


def render_pdf(pdf: Path, dest: Path, dpi: int = RENDER_DPI) -> Path | None:
    """PDF page 1 to PNG. PyMuPDF is already a dependency (fetch.py uses it
    for reference extraction), so this needs nothing new installed."""
    try:
        import fitz
    except ImportError:
        return None
    try:
        with fitz.open(pdf) as doc:
            if not doc.page_count:
                return None
            doc[0].get_pixmap(dpi=dpi).save(dest)
    except Exception:
        return None
    return dest if dest.exists() else None
