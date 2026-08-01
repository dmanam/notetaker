"""Things the assembler must enforce rather than ask the model for.

1. Every theorem environment the model declares shares the `theorem` counter,
   so the document has one numbering sequence and a \\cref is unambiguous.
2. \\theH<env> carries the section. Left alone it is the bare counter, so
   Theorem 1.1 and Theorem 2.1 both anchor at "theorem.1", hyperref drops the
   duplicate, and every link to either lands on whichever came first. The real
   course logged 463 duplicate destinations before this.
3. Re-declaring an environment that already exists is a hard error; both ways
   of getting there are decidable here.
4. Overfull boxes and un-bookmarkable titles are collected from the log so the
   final repair round can fix them.

All four are mechanically checkable, so none of them is left to a prompt.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_course as B
import latex_check as L

# --- 1. everything lands on the shared counter ------------------------------
got = B.normalize_theorem_decls([
    r"\declaretheorem[style=plain,name=Claim]{myclaim}",       # no numbering
    r"\declaretheorem[style=definition,numberwithin=section]{constr}",
    r"\declaretheorem[style=plain,numberlike=equation]{fact}",
    r"\declaretheorem[style=remark,unnumbered]{aside}",        # has no counter
    r"\declaretheorem[numberwithin=section,style=plain]{theorem}",  # the base
    r"\newtheorem{conj}{Conjecture}[section]",
])
assert "sibling=theorem" in got[0] and "name=Claim" in got[0], got[0]
assert "numberwithin" not in got[1] and "sibling=theorem" in got[1], got[1]
assert "numberlike" not in got[2] and "sibling=theorem" in got[2], got[2]
assert got[3] == r"\declaretheorem[style=remark,unnumbered]{aside}", \
    "an unnumbered environment has no counter and cannot be a sibling"
assert got[4] == r"\declaretheorem[numberwithin=section,style=plain]{theorem}", \
    "the base counter must keep numberwithin=section"
assert got[5] == r"\newtheorem{conj}[theorem]{Conjecture}", got[5]
print("normalize_theorem_decls: shared counter forced, unnumbered/base spared")

# --- 2. the anchors ---------------------------------------------------------
late = B.normalize_theorem_decls([r"\declaretheorem[style=plain]{myclaim}",
                                  r"\newtheorem{conj}{Conjecture}"])
block = B.theorem_anchor_block(late)
assert r"\renewcommand{\theHtheorem}{\theHsection.\arabic{theorem}}" in block
assert r"\renewcommand{\theHmyclaim}{\theHtheorem}" in block, block
assert r"\theHconj" not in block, \
    "hyperref hooks \\newtheorem itself — those need no redefinition"
assert "providecommand" not in block, \
    "\\theH<env> already exists; \\providecommand is noise"
for builtin in ("lemma", "proposition", "corollary", "definition", "remark"):
    assert f"\\renewcommand{{\\theH{builtin}}}{{\\theHtheorem}}" in block, builtin
print(f"theorem_anchor_block: {block.count('renewcommand')} anchors, "
      f"section included, \\newtheorem excluded")

# Order matters: thmtools must have declared the environments already.
pre, _ = B.course_preamble("T", {"preamble_additions": [
    r"\declaretheorem[style=plain]{myclaim}"]}, False)
assert pre.index(r"\declaretheorem[style=plain,sibling=theorem]{myclaim}") \
    < pre.index(r"\renewcommand{\theHtheorem}"), \
    "the \\theH block must come after every \\declaretheorem"
# ...and hyperref must already be loaded when \theH is renewed.
assert pre.index("{hyperref}") < pre.index(r"\renewcommand{\theHtheorem}")
print("course_preamble: \\theH block sits after the declarations and hyperref")

# --- 3. no environment is declared twice ------------------------------------
kept = B.drop_duplicate_theorems([
    r"\declaretheorem[sibling=theorem,style=plain]{lemma}",   # already built in
    r"\declaretheorem[sibling=theorem,style=plain]{novel}",
    r"\declaretheorem[sibling=theorem,style=remark]{novel}",  # dup of the above
    r"\crefname{novel}{Novel}{Novels}",                       # not a declaration
])
assert kept == [r"\declaretheorem[sibling=theorem,style=plain]{novel}",
                r"\crefname{novel}{Novel}{Novels}"], kept
names = B.declared_names(B.PREAMBLE_TEMPLATE)
assert "theorem" in names and "lemma" in names
pre2, _ = B.course_preamble("T", {"preamble_additions": [
    r"\declaretheorem[sibling=theorem,style=plain]{lemma}"]}, False)
assert len(re.findall(r"\\declaretheorem[^\n]*\{lemma\}", pre2)) == 1, \
    "a re-declared built-in must not reach the document"
print("drop_duplicate_theorems: built-ins and repeats removed, other lines kept")

# --- 4. presentation warnings ----------------------------------------------
log = r"""
Overfull \hbox (31.10075pt too wide) in paragraph at lines 1111--1115
Overfull \hbox (0.13063pt too wide) in paragraph at lines 2612--2615
Overfull \hbox (75.67139pt too wide) detected at line 1582
Overfull \hbox (31.10075pt too wide) in paragraph at lines 1111--1115
Underfull \hbox (badness 10000) in paragraph at lines 9--10
Package hyperref Warning: Token not allowed in a PDF string (Unicode):
(hyperref)                removing `math shift' on input line 4105.
"""
w = L.collect_warnings(log)
lines = sorted(x.line for x in w)
assert lines == [1111, 1582, 4105], lines
assert not any("0.13" in x.message for x in w), \
    "a 0.13pt overflow is invisible; chasing it rewrites good prose"
assert sum(1 for x in w if x.line == 1111) == 1, "latexmk repeats each pass"
assert not any("Underfull" in x.message for x in w), \
    "underfull boxes are a judgement call, not a defect"
assert any("texorpdfstring" in x.message for x in w)
assert any("margin" in x.message for x in w)
print(f"collect_warnings: {len(w)} real item(s); tiny, repeated and "
      f"underfull ones dropped")

# The polish prompt has to name the two fixes it is asking for.
for probe in ("texorpdfstring", "multline", "Overfull", "bookmark"):
    assert probe in B.POLISH_INSTRUCTION, probe
assert "sloppy" in B.POLISH_INSTRUCTION and "whole file" in B.POLISH_INSTRUCTION, \
    "must warn against the blunt \\sloppy fix"
src = (ROOT / "build_course.py").read_text()
assert re.search(r"polish\s*=\s*not errors", src), \
    "warnings must only be worked on once the document compiles"
print("POLISH_INSTRUCTION: covers both fixes; correctness runs first")

print("\nALL OK")
