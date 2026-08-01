"""Equation numbers exist to be cited.

A plain \\begin{equation} follows what cites it, both ways: uncited displays
are starred, and one that a later lecture starts citing is numbered again on
the next assembly. \\label always stays — it is how a later lecture knows what
to \\cref, so stripping it from a starred display would make the equation
uncitable and freeze it that way for ever.

Only a cited label inside a starred *multi-line* display is left for a person:
those number per line via \\notag, so starring the whole environment is the
wrong instrument.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import equations as E

# --- what counts as a reference ---------------------------------------------
text = r"""
\cref{eq:a} \eqref{eq:b} \Cref{eq:c} \ref{eq:d} \autoref{eq:e}
\cref{eq:f,
  eq:g}
\labelcref{eq:h}
"""
got = E.referenced_labels(text)
assert got == {f"eq:{c}" for c in "abcdefgh"}, got
# A wrapped key list leaves whitespace on the second key; if that is not
# stripped the real label looks unreferenced and gets silently unnumbered.
assert "eq:g" in got and " eq:g" not in got
print(f"referenced_labels: {len(got)} label(s), wrapped key lists handled")

# --- unreferenced displays lose their number --------------------------------
src = r"""\begin{equation}
\label{eq:kept}
a = b
\end{equation}
\begin{equation}
\label{eq:dropped}
c = d
\end{equation}
\begin{equation}
e = f
\end{equation}
\begin{equation}
\label{eq:tagged}
g = h \tag{$\ast$}
\end{equation}
\begin{equation*}
i = j
\end{equation*}
"""
out, off, on = E.normalize_equation_numbering(src, {"eq:kept"})
assert (off, on) == (2, 0), (off, on)   # eq:dropped and the label-less one
assert r"\begin{equation}" + "\n" + r"\label{eq:kept}" in out
assert out.count(r"\begin{equation*}") == 3, out
assert r"\tag{$\ast$}" in out and out.count(r"\begin{equation}") == 2, \
    "an explicitly tagged display is deliberate and must keep its form"
# Every label survives: it is the only way a later lecture can cite this.
assert E.defined_labels(out) == E.defined_labels(src), \
    "unnumbering must not remove a label — that makes the display uncitable"
assert r"\label{eq:dropped}" in out
print(f"normalize_equation_numbering: {off} starred, all labels kept")

# Idempotent — the pass runs on every assembly, including repair rounds.
again, off2, on2 = E.normalize_equation_numbering(out, {"eq:kept"})
assert (off2, on2) == (0, 0) and again == out, "second run must be a no-op"
print("normalize_equation_numbering: idempotent")

# The round trip: lecture 9 starts citing something starred in lecture 3.
back, off3, on3 = E.normalize_equation_numbering(out, {"eq:kept", "eq:dropped"})
assert (off3, on3) == (0, 1), (off3, on3)
assert r"\begin{equation}" + "\n" + r"\label{eq:dropped}" in back, back
assert back.count(r"\begin{equation*}") == 2
# ...and back again if the citation is removed.
assert E.normalize_equation_numbering(back, {"eq:kept"})[0] == out
print("normalize_equation_numbering: renumbers when cited, both directions")

# --- referenced but unnumberable is reported, not fixed ---------------------
bad = r"""\begin{equation*}
\label{eq:starred}
x = y
\end{equation*}
\begin{align*}
\label{eq:aligned}
u &= v
\end{align*}
\begin{equation*}
\label{eq:has-tag}
p = q \tag{4.2}
\end{equation*}
\begin{align}
\label{eq:numbered}
s &= t
\end{align}
"""
referenced = {"eq:starred", "eq:aligned", "eq:has-tag", "eq:numbered"}
items = E.review_items(bad, referenced)
labels = sorted(i.label for i in items)
assert labels == ["eq:aligned"], \
    "only multi-line displays reach review — a cited equation* is unstarred"
assert all(i.kind == "unnumbered" for i in items)
assert "align*" in str(items[0])
# The cited equation* is fixed rather than reported, and the align* is not
# touched — starring per environment cannot express \notag per line.
fixed, off, on = E.normalize_equation_numbering(bad, referenced)
assert (off, on) == (0, 1), (off, on)
assert r"\begin{equation}" + "\n" + r"\label{eq:starred}" in fixed
assert r"\begin{align*}" in fixed, "multi-line displays are left alone"
print(f"review_items: {len(items)} reported; cited equation* auto-numbered")

# An unreferenced starred display is not anyone's problem.
assert E.review_items(bad, set()) == []
print("review_items: silent when nothing cites the display")

# --- dangling references ----------------------------------------------------
d = E.dangling_references(r"\cref{eq:here} \cref{eq:nowhere}",
                          {"eq:here"})
assert [i.label for i in d] == ["eq:nowhere"], d
assert "never defined" in str(d[0])
print("dangling_references: catches the ?? before it reaches the PDF")

# --- the pass is wired into assembly ----------------------------------------
src_txt = (ROOT / "build_course.py").read_text()
assert "normalize_equations(output_root, state, slugs)" in src_txt, \
    "write_document must run the pass, or nothing ever applies it"
fn = src_txt[src_txt.index("def normalize_equations"):]
fn = fn[:fn.index("def write_document")]
assert "referenced_labels(whole)" in fn, \
    "the referenced set must be gathered course-wide before any rewrite — " \
    "lecture 9 cites lecture 3"
print("normalize_equations: wired in, and course-wide")

print("\nALL OK")
