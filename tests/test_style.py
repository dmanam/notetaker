"""Style extraction: parsing, the source check, and the render comparison.

The verification is the part worth testing. A rewrite that renders
differently from the original is a style sample that quietly says something
else, which is worse than having no sample — so it has to be caught by
compiling both, not by asking the model whether it did a good job.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import style_extract as S

root = Path(tempfile.mkdtemp(prefix="style-"))

SOURCE = r"""\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}
\newcommand{\cH}{\mathcal{H}}
\newcommand{\Zhat}{\widehat{\mathbb{Z}}}
\newtheorem{thm}{Theorem}
\begin{document}
\section{Preface}
These notes were written for a course given in the spring. Conventions: all
rings are commutative.

\section{The real material}
Let $\cH$ be a Hilbert space. We record the following.
\begin{thm}
Every $\Zhat$-module is complete.
\end{thm}
\end{document}
"""
src = root / "notes.tex"
src.write_text(SOURCE)

# --- splitting the document -------------------------------------------------
preamble, body = S.split_document(SOURCE)
assert r"\newcommand{\cH}" in preamble, "the macros must land in the preamble"
assert r"\section{Preface}" in body and r"\newcommand" not in body
assert S.split_document("no document here")[0] == ""
print("preamble split: macros on one side, prose on the other")

# --- parsing the extractor's output -----------------------------------------
OUT = """%%% PACKAGES: mathrsfs, stmaryrd
%%% PASSAGE
%%% ORIGINAL-START
Let $\\cH$ be a Hilbert space. We record the following.
%%% ORIGINAL-END
%%% REWRITTEN-START
Let $\\mathcal{H}$ be a Hilbert space. We record the following.
%%% REWRITTEN-END
"""
passages, pkgs = S.parse_output(OUT)
assert len(passages) == 1 and pkgs == ["mathrsfs", "stmaryrd"], (passages, pkgs)
assert passages[0].original.startswith("Let $\\cH$")
assert passages[0].rewritten.startswith("Let $\\mathcal{H}$")
assert S.parse_output("")[0] == []
print(f"parsed 1 passage and {len(pkgs)} requested package(s)")

# --- the original must really be in the source ------------------------------
# Without this the extractor could paraphrase the source into the ORIGINAL
# block, and the comparison would check its invention against itself.
real = S.Passage("Let $\\cH$ be a Hilbert space.", "irrelevant")
fake = S.Passage("Let $\\cH$ be a Banach space.", "irrelevant")
kept = S.check_originals([real, fake], SOURCE)
assert kept == [real], "a fabricated original must be dropped"
assert "not in the source" in fake.note
# whitespace differences must not count as fabrication
rewrapped = S.Passage("Let $\\cH$ be a\n   Hilbert   space.", "irrelevant")
assert S.check_originals([rewrapped], SOURCE) == [rewrapped]
print("fabricated originals dropped; rewrapped ones survive")

# --- the render comparison --------------------------------------------------
faithful = S.Passage(
    r"Let $\cH$ be a Hilbert space. Every $\Zhat$-module is complete.",
    r"Let $\mathcal{H}$ be a Hilbert space. "
    r"Every $\widehat{\mathbb{Z}}$-module is complete.")
S.verify(faithful, preamble, [], root / "ok")
assert faithful.verified, f"a faithful expansion must pass: {faithful.note} " \
                          f"({faithful.match:.2f})"
print(f"faithful macro expansion verified at {faithful.match:.0%}")

# The failure that matters: a rewrite that says something else. It compiles,
# it looks like the original, and only the typeset comparison catches it.
altered = S.Passage(
    r"Let $\cH$ be a Hilbert space. Every $\Zhat$-module is complete.",
    r"Let $\mathcal{H}$ be a Banach space. "
    r"Every $\widehat{\mathbb{Z}}$-module is separable.")
S.verify(altered, preamble, [], root / "altered")
assert not altered.verified, "a rewrite that changes the words must fail"
assert "renders differently" in altered.note
print(f"altered rewrite rejected at {altered.match:.0%}: {altered.note}")

broken = S.Passage(r"Let $\cH$ be a Hilbert space.",
                   r"Let $\mathcal{H$ be a Hilbert space.")   # unbalanced
S.verify(broken, preamble, [], root / "broken")
assert not broken.verified and "does not compile" in broken.note

# A rewrite still carrying the author's private macro cannot compile against
# the portable preamble — which is exactly what we want to find out.
private = S.Passage(r"Let $\cH$ be a Hilbert space.",
                    r"Let $\cH$ be a Hilbert space.")
S.verify(private, preamble, [], root / "private")
assert not private.verified, "an unexpanded private macro must not pass"
print("uncompilable and unexpanded rewrites both rejected")

# --- similarity is insensitive to line breaking, sensitive to words ---------
assert S.similarity("one two three", "one   two\nthree") == 1.0
assert S.similarity("one two three", "one two four") < S.MIN_MATCH
assert S.similarity("", "") == 1.0 and S.similarity("a", "") == 0.0
print("similarity ignores whitespace, not words")

# --- the cache round-trips and invalidates on edit --------------------------
import json
cache = S.cache_path(src, root)
cache.write_text(json.dumps({
    "source": str(src), "source_bytes": src.stat().st_size, "packages": [],
    "passages": [{"rewritten": "kept", "verified": True},
                 {"rewritten": "dropped", "verified": False}]}))
assert S.load(src, root) == ["kept"], "only verified passages are served"
src.write_text(SOURCE + "\n% edited\n")
assert S.load(src, root) is None, "an edited source must invalidate the cache"
print("cache serves only verified passages and notices an edit")

# --- the prompt and the parser must agree on the marker form ----------------
# They once did not: the prompt went through %-formatting, which silently
# turned every %%%% into %%, so the extractor was told one form and the
# parser looked for another and found nothing.
prompt = S.EXTRACT_PROMPT.replace("__N__", "5")
assert "__N__" not in prompt and "choose 5 passages" in prompt
for marker in ("%%% ORIGINAL-START", "%%% ORIGINAL-END",
               "%%% REWRITTEN-START", "%%% REWRITTEN-END", "%%% PACKAGES:"):
    assert marker in prompt, marker
round_trip = "\n".join(["%%% PASSAGE", "%%% ORIGINAL-START", "alpha",
                        "%%% ORIGINAL-END", "%%% REWRITTEN-START", "beta",
                        "%%% REWRITTEN-END"])
got, _ = S.parse_output(round_trip)
assert len(got) == 1 and got[0].original == "alpha" and got[0].rewritten == "beta"
print("the prompt's marker form is exactly what the parser reads")

print("\nALL OK")
