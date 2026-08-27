"""Every paragraph carries the moment it starts in the video, in the margin.

The mark is \\ts{hh:mm:ss}, and what has to be true of it is positional: it
belongs in the left margin, on the baseline of the line it marks, in
monospace, without moving the text. So this compiles the real course preamble
and measures the result in the PDF rather than checking that the source
contains a macro call.

Two placements have to work and they are not the same problem. At the start of
a paragraph TeX is still in vertical mode, where marginnote sets the note half
a line high; after a theorem head the mark is mid-line, where anything
relative to the current point (\\llap, say) lands on top of the text instead
of in the margin. Both are checked below, along with a \\marginpar's failure
mode — notes that drift onto a neighbouring paragraph — which is why this is
a marginnote and not a marginpar.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_course as B
import generate_notes as G
import instructions as I
import timestamps as T

# --- the rule reaches both note-takers, and the macro reaches the document ---
assert I.TIMESTAMP_RULE in B.SYSTEM_PROMPT, "the course writer is not told"
assert I.TIMESTAMP_RULE in G.SYSTEM_PROMPT, "the single-lecture writer is not"
preamble, _ = B.course_preamble("A Course", {}, False)
assert "\\newcommand{\\ts}" in preamble and "marginnote" in preamble
# The single-lecture prompt used to say the opposite in as many words.
assert "do not include them in the notes themselves" not in G.SYSTEM_PROMPT
# A verifier that rewrites a paragraph must not take its mark with it.
assert "Preserve every \\ts" in B.VERIFY_PROMPT
print("both prompts carry the rule; the course preamble defines the macro")

assert T.marks(r"\ts{00:01:02}Text \ts {01:00:00}More") == \
    ["00:01:02", "01:00:00"]
assert T.marks(r"no marks here") == []
print("marks() reads the stamps back out of a body")

# --- a model-written definition must never reach the preamble ---------------
# \ts is defined by the preamble above; a second \newcommand for it is not a
# style problem but a build failure ("command \ts already defined"), taking
# the whole course down over a macro the model was told not to write.
for line in (r"\newcommand{\ts}[1]{\textbf{#1}}", r"\renewcommand*{\ts}{x}",
             r"\def\ts#1{#1}", r"\providecommand{\tsfont}{\sffamily}"):
    assert T.defines_reserved(line), line
    assert T.drop_reserved(["\\usepackage{xcolor}", line]) == \
        ["\\usepackage{xcolor}"], line
# A macro that merely starts with the same letters is somebody else's.
for line in (r"\newcommand{\tsx}{y}", r"\newcommand{\tsfonts}{y}",
             r"\DeclareMathOperator{\Tors}{Tors}"):
    assert not T.defines_reserved(line), line
assert B.course_preamble("T", {"preamble_additions":
                               [r"\newcommand{\ts}[1]{BAD}"]}, False)[0].count(
    "\\newcommand{\\ts}") == 1, "the model's definition reached the preamble"
print("a model-written \\ts is dropped before it can break the build")

root = Path(tempfile.mkdtemp())

# --- attaching the macro to a model-written standalone document -------------
doc = root / "notes.tex"
DOC = ("\\documentclass{article}\n\\usepackage{amsmath}\n"
       "\\begin{document}\n\\ts{00:00:01}Hello.\n\\end{document}\n")
doc.write_text(DOC)
assert T.attach_macro(doc) is True
text = doc.read_text()
assert text.index("\\newcommand{\\ts}") < text.index("\\begin{document}")
assert T.attach_macro(doc) is False and doc.read_text() == text
print("attach_macro: defines the macro once, and only once")

# A model that writes its own definition anyway would clash with the one
# attached here, so its version is dropped rather than added to.
own = root / "own.tex"
own.write_text(DOC.replace("\\usepackage{amsmath}\n",
                           "\\usepackage{amsmath}\n\\newcommand{\\ts}[1]{[#1]}\n"))
assert T.attach_macro(own) is True
assert own.read_text().count("\\newcommand{\\ts}") == 1
assert "[#1]" not in own.read_text()
# Nothing to do for a document that never marked anything, or a fragment.
plain = root / "plain.tex"
plain.write_text(DOC.replace("\\ts{00:00:01}", ""))
assert T.attach_macro(plain) is False
frag = root / "frag.tex"
frag.write_text("\\section{One}\n\\ts{00:00:01}Body only.\n")
assert T.attach_macro(frag) is False
print("attach_macro: replaces a model's own version, skips what it cannot fix")

# --- and now what it actually looks like on the page ------------------------
BODY = r"""
\section{Measured}
\ts{00:00:01}
At a paragraph start, with descenders gjpqy. Text text text text
text text text text text text text text text text text text text text text.

\ts{00:00:02}
Without descenders: an oxen ate. Text text text text text text
text text text text text text text text text text text text text text text.

A control paragraph with no mark: the note-taker wrote it rather than the
lecturer saying it, so nothing in the margin claims otherwise. Text text text
text text text text text text text text text text text text text text text.

\begin{theorem}\label{thm:1:a}
\ts{00:01:01}
After a theorem head, which is set on the same line as the statement.
\end{theorem}

\begin{definition}
\ts{00:02:02}
\label{def:1:a}
A definition, marked before its label rather than after it.
\end{definition}

\begin{remark}
\ts{00:03:03}
A remark, whose style sets the head differently again.
\end{remark}

\begin{proof}
\ts{00:04:04}
A proof. Text text text text text text text text text text text text text.
\end{proof}

\ts{00:05:05}
Before a display. Text text text text text text text text text.
\[ x + y = z \]
\ts{00:06:06}
And after one, with descenders: gjpqy.

\ts{00:07:07}
\textbf{A run-in heading.} Then the rest of the paragraph, text
text text text text text text text text text text text text text text text.

\ts{00:08:08}
Short.

\ts{00:08:30}
Another short one, close behind the last.

\ts{00:09:09}
With inline math $\frac{a}{b}$ raising the line. Text text text.
"""
# The marks go on a line of their own. Checked here because the fixture below
# is what the placement measurement runs on: if someone tucks a mark back onto
# the text line, the measurement stops being about the layout we ask for.
for ln in BODY.splitlines():
    if "\\ts{" in ln:
        assert ln.strip().startswith("\\ts{") and ln.strip().endswith("}"), ln

if shutil.which("latexmk") is None:
    print("(no latexmk on PATH — skipping the placement measurement)")
    sys.exit(0)

tex = root / "course.tex"
tex.write_text(preamble + BODY + "\n" + B.CLOSING + "\n")
out = root / "build"
proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                       "-outdir=" + str(out), tex.name],
                      cwd=root, capture_output=True, text=True,
                      errors="replace", timeout=900)
assert proc.returncode == 0, proc.stdout[-3000:]
log = (out / "course.log").read_text(errors="replace")
# A marginpar that will not fit where it belongs is moved down the page and
# says so. marginnote never floats, so this must never appear.
assert "Marginpar on page" not in log, "a mark drifted off its own line"

import fitz
# The course preamble puts a title page and a contents page first, so the
# body is not on page one.
lines = []
for pno, page in enumerate(fitz.open(out / "course.pdf")):
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            span = ln["spans"][0]
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if txt:
                lines.append({"page": pno,
                              "base": round(span["origin"][1], 2),
                              "x": round(span["origin"][0], 2),
                              "font": span["font"], "color": span["color"],
                              "text": txt})
stamps = [l for l in lines if re.fullmatch(r"\d\d:\d\d:\d\d", l["text"])]
body = [l for l in lines if l not in stamps]
assert len(stamps) == 12, f"{len(stamps)} marks reached the page, expected 12"

TEXT_LEFT = 72.0            # geometry margin=1in, at 72 PostScript points
def rgb(value):
    return (value >> 16) & 255, (value >> 8) & 255, value & 255

for st in stamps:
    assert st["font"].startswith("CMTT"), (st["text"], st["font"])
    # Gray, so a mark on every paragraph stays quiet next to the text. Not
    # black (which is what an unset \color leaves it), and not so pale it
    # cannot be read: one per paragraph is a lot of marks either way.
    red, green, blue = rgb(st["color"])
    assert red == green == blue, (st["text"], rgb(st["color"]))
    assert 80 < red < 180, (st["text"], rgb(st["color"]))
    # In the margin: clear of the text block, and clear of the paper edge.
    assert 5 < st["x"] < TEXT_LEFT - 5, (st["text"], st["x"])
    # On the baseline of the line it marks — not the line above or below it.
    same = [l for l in body if l["page"] == st["page"]
            and abs(l["base"] - st["base"]) < 0.01]
    assert same, f"{st['text']} sits on no line of text (x={st['x']})"
print(f"{len(stamps)} marks: gray {rgb(stamps[0]['color'])} monospace, in "
      f"the left margin, on the baseline of the line each one marks")

# The mark must not push the text around: the stamped paragraphs and the
# unstamped control have to set identically.
def spacing(prefix):
    i = next(i for i, l in enumerate(body) if l["text"].startswith(prefix))
    assert body[i + 1]["page"] == body[i]["page"], prefix
    return round(body[i + 1]["base"] - body[i]["base"], 2)
control_line = [l for l in body if l["text"].startswith("A control")][0]
assert rgb(control_line["color"]) == (0, 0, 0), \
    "the prose picked up the mark's colour"
# A paragraph with no spoken origin carries no time: the margin says which
# paragraphs are the lecture's and which are the note-taker's, and a mark on
# this one would be a claim that the lecturer said it.
assert not [st for st in stamps if st["page"] == control_line["page"]
            and abs(st["base"] - control_line["base"]) < 0.01], \
    "the note-taker's own paragraph was marked as if the lecture said it"
control = spacing("A control paragraph")
for prefix in ("At a paragraph start", "Without descenders"):
    assert spacing(prefix) == control, (prefix, spacing(prefix), control)
print(f"line spacing unchanged by the marks ({control}pt, marked and not)")

# --- and why the rule says a newline, not a blank line ----------------------
# A blank line after the mark ends the paragraph, so the mark lands on an
# empty one and the text it was meant to mark starts a line further down with
# nothing beside it. The prompt warns about this; here is the warning being
# true.
stranded = root / "stranded.tex"
stranded.write_text(
    "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n"
    + T.TIMESTAMP_PREAMBLE + "\\begin{document}\n"
    "\\ts{00:00:01}\n\nStranded: the mark is a paragraph of its own. Text "
    "text text text text text text text text text text text.\n"
    "\\end{document}\n")
sout = root / "strandedbuild"
proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                       "-outdir=" + str(sout), stranded.name],
                      cwd=root, capture_output=True, text=True,
                      errors="replace", timeout=900)
assert proc.returncode == 0, proc.stdout[-3000:]
bad = []
for block in fitz.open(sout / "stranded.pdf")[0].get_text("dict")["blocks"]:
    for ln in block.get("lines", []):
        txt = "".join(sp["text"] for sp in ln["spans"]).strip()
        if txt:
            bad.append((round(ln["spans"][0]["origin"][1], 2), txt))
mark = [b for b in bad if b[1].startswith("00:00:01")][0]
text_line = [b for b in bad if b[1].startswith("Stranded")][0]
assert mark[0] != text_line[0], \
    "a blank line after the mark is harmless after all — the prompt can stop "\
    "warning about it"
print(f"a blank line after a mark strands it {text_line[0] - mark[0]:.2f}pt "
      f"from its text, which is why the rule asks for a newline")

# --- the two margins do not fight -------------------------------------------
# \todo puts its notes in a margin too. obeyDraft hides them in an ordinary
# build, but a draft build shows them, and then both kinds of margin material
# are on the page at once: the timestamps have to stay left and the todos
# right. That is what the grouped \reversemarginpar in the macro buys, and
# it is invisible until someone builds a draft, so it is checked here.
draft = root / "draft.tex"
draft.write_text(
    "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n"
    "\\usepackage[colorinlistoftodos]{todonotes}\n" + T.TIMESTAMP_PREAMBLE +
    "\\begin{document}\n\\ts{00:00:01}A paragraph with a "
    "\\todo{RIGHT} note in it. Text text text text text text text "
    "text text text text text.\n\\end{document}\n")
dout = root / "draftbuild"
proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                       "-outdir=" + str(dout), draft.name],
                      cwd=root, capture_output=True, text=True,
                      errors="replace", timeout=900)
assert proc.returncode == 0, proc.stdout[-3000:]
found = {}
for block in fitz.open(dout / "draft.pdf")[0].get_text("dict")["blocks"]:
    for ln in block.get("lines", []):
        txt = "".join(sp["text"] for sp in ln["spans"]).strip()
        where = round(ln["spans"][0]["origin"][0], 2)
        if txt.startswith("00:00:01"):
            found["stamp"] = where
        elif txt.startswith("RIGHT"):
            found["todo"] = where
assert found.get("stamp", 999) < TEXT_LEFT, found
assert found.get("todo", 0) > 400, found
print(f"timestamps left ({found['stamp']}), todos right ({found['todo']})")

shutil.rmtree(root, ignore_errors=True)
print("\nALL OK")
