"""The two note-takers share their prompt, and the refactor changed no words.

build_course.py and generate_notes.py used to write the same rules out twice
and the copies drifted: the course prompt grew the fidelity, diagram and
display rules and banned \\ref, while the single-lecture prompt still asked
for \\ref and had never heard of tikz. The shared text now lives in
instructions.py, so the check that matters is that both prompts really do
carry it — a constant that only one of them imports is the same bug again.

The other half is the bibliography. The model is told never to write biblatex
machinery, which is only honest if something else writes it: the course
assembles its preamble, and a single lecture gets it attached afterwards.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import instructions as I
import bibliography as BIB
import build_course as B
import generate_notes as G

# --- every shared block reaches both prompts ---------------------------------
SHARED = ["ASR_INSTRUCTION", "FIDELITY_INSTRUCTION", "CROSSREF_RULE",
          "FRAMES_RULE", "CLARIFY_RULE", "MACRO_BRACING_RULE",
          "ASK_USER_RULE", "TODO_RULE", "DISPLAY_RULES", "DISFLUENCY_RULE",
          "HOUSE_STYLE_INSTRUCTION"]
for name in SHARED:
    block = getattr(I, name)
    for driver, prompt in (("course", B.SYSTEM_PROMPT),
                           ("single-lecture", G.SYSTEM_PROMPT)):
        assert block in prompt, f"{name} missing from the {driver} prompt"
print(f"{len(SHARED)} shared blocks present in both prompts")

# The two parameterised blocks: same judgement, one clause differs.
assert I.diagram_rules(board_tools=True) in B.SYSTEM_PROMPT
assert I.diagram_rules(board_tools=False) in G.SYSTEM_PROMPT
assert "crop_board" not in G.SYSTEM_PROMPT and "add_to_preamble" not in \
    G.SYSTEM_PROMPT, "the single-lecture driver has neither tool"
assert I.cite_rule(shared=True) in B.SYSTEM_PROMPT
assert I.cite_rule(shared=False) in G.SYSTEM_PROMPT
for flag in (True, False):
    for tail in ("Draw diagrams the lecturer did not draw",
                 "not an idealised one"):
        assert tail in I.diagram_rules(flag), (flag, tail)
    assert "the bibliography is assembled automatically" in I.cite_rule(flag)
print("diagram and citation rules differ only in their tool clause")

# --- the rules the drift produced -------------------------------------------
# The single-lecture prompt asked for \ref for years after the course prompt
# banned it, and never mentioned drawing anything.
assert "Never use \\ref or \\hyperref" in G.SYSTEM_PROMPT
assert "\\begin{tikzcd}" in G.SYSTEM_PROMPT
assert "cite_reference" in G.SYSTEM_PROMPT
assert "cleveref" in G.SYSTEM_PROMPT, \
    "a prompt that mandates \\cref must say to load the package"
for pkg in ("tikz", "tikz-cd", "todonotes"):
    assert pkg in G.SYSTEM_PROMPT, pkg
print("the single-lecture prompt now bans \\ref and loads what it needs")

# --- attaching a bibliography to a standalone document ----------------------
root = Path(tempfile.mkdtemp(prefix="instr-"))
tex = root / "notes.tex"
bib = root / BIB.BIB_FILENAME
DOC = ("\\documentclass{article}\n\\usepackage{hyperref}\n"
       "\\begin{document}\nSee \\cite{key1}.\n\\end{document}\n")
tex.write_text(DOC)

# Nothing cited: nothing to wire, and the file is left exactly as it was.
assert BIB.attach_to_document(tex, bib) is False and tex.read_text() == DOC
bib.write_text("@article{key1, title = {A paper}, year = {2019}}\n")

assert BIB.attach_to_document(tex, bib) is True
out = tex.read_text()
assert "\\addbibresource{references.bib}" in out
assert out.index("\\usepackage{hyperref}") < out.index("biblatex"), \
    "biblatex must load after hyperref"
assert out.index("biblatex") < out.index("\\begin{document}")
assert out.index("\\printbibliography") < out.index("\\end{document}")
print("attach_to_document: biblatex after hyperref, print before the end")

# Idempotent — check_and_fix calls this before every compile round.
after = tex.read_text()
assert BIB.attach_to_document(tex, bib) is False and tex.read_text() == after
assert after.count("\\printbibliography") == 1
print("attach_to_document: idempotent")

# A hand-written bibliography is left alone: two lists would print, and the
# \cite keys resolve against only one of them.
hand = root / "hand.tex"
HAND = ("\\documentclass{article}\n\\begin{document}\n\\cite{key1}\n"
        "\\begin{thebibliography}{9}\\bibitem{key1} A paper.\n"
        "\\end{thebibliography}\n\\end{document}\n")
hand.write_text(HAND)
assert BIB.attach_to_document(hand, bib) is False and hand.read_text() == HAND
# And a body-only fragment has nowhere to put a preamble.
frag = root / "body.tex"
frag.write_text("\\section{One}\n\\cite{key1}\n")
assert BIB.attach_to_document(frag, bib) is False
print("attach_to_document: declines a hand-written list and a body fragment")

# --- the tool is actually offered to the single-lecture agent ----------------
# The prompt telling the model to call cite_reference is worthless if the
# context never sets bib_file, which is what gates the handler.
src = (ROOT / "generate_notes.py").read_text()
assert src.count("bib_file=output_path.parent / BIB_FILENAME") == 1 and \
    "bib_file=bib_file" in src, \
    "both the write and the follow-up context must carry a bib_file"
assert "attach_to_document(output_path" in src, \
    "nothing else writes the biblatex lines into a standalone document"
print("generate_notes: bib_file set on both contexts, attachment wired in")

print("\nALL OK")
