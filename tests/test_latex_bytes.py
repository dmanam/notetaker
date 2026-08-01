"""TeX output that is not valid UTF-8 must not take the build down.

pdflatex writes raw 8-bit bytes to stdout — font names, ^^-escaped characters,
a stray byte echoed back from the source. `subprocess.run(text=True)` decodes
strictly, so the UnicodeDecodeError is raised *inside* the call, before any
log parsing can be tolerant about it. That is how a full 24-lecture run died at
the assembly step with every section already written.

No TeX installation is needed: a stub `pdflatex`/`latexmk` on PATH that emits
the offending byte exercises exactly the decode path that broke.
"""
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

bindir = Path(tempfile.mkdtemp(prefix="texstub-"))
work = Path(tempfile.mkdtemp(prefix="texwork-"))

# Emits a raw 0xf2 — the byte that actually crashed the run — then fails, so
# the error path (which reads stdout when no .log exists) is the one taken.
STUB = ("#!/bin/sh\n"
        "printf 'Font \\362 not loadable\\n'\n"
        "printf '! Undefined control sequence.\\n'\n"
        "exit 1\n")

for name in ("pdflatex", "latexmk"):
    p = bindir / name
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ['PATH']}"

import latex_check          # noqa: E402  (must follow the PATH change)
import diagrams             # noqa: E402

tex = work / "course.tex"
tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}\n")

errors = latex_check.check_latex(tex)
assert errors is not None, "the stub is on PATH, so this is not the no-TeX case"
assert any("Undefined control sequence" in e.message for e in errors), \
    f"the real error must survive the undecodable byte: {[e.message for e in errors]}"
print(f"check_latex: survived a raw 0xf2 and still parsed {len(errors)} error(s)")

# The diagram gate shells out to pdflatex the same way and broke the same way.
res = diagrams.compile_snippet("\\begin{tikzcd} A \\end{tikzcd}", work / "d", "")
assert res.ok is False, "the stub exits 1, so this must be a failure result"
# Parsed errors go in .errors; .note is the fallback for a failure the parser
# could not read at all, so it is empty exactly when .errors is populated.
assert any("Undefined control sequence" in e.message for e in res.errors), \
    f"errors={[e.message for e in res.errors]} note={res.note!r}"
print("compile_snippet: survived the same byte and reported the real error")

# And the byte itself must not appear raw in anything handed to the model.
for blob in ([e.describe() for e in errors]
             + [e.describe() for e in res.errors] + [res.note]):
    blob.encode("utf-8")    # raises if a lone surrogate or bad byte slipped in
print("every message is encodable UTF-8")

print("\nALL OK")
