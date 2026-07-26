"""
latex_check.py — best-effort compile check for generated LaTeX.

Prefers latexmk (which runs biber and repeats passes as needed, so citations
and cross-references resolve); falls back to a single pdflatex pass. Silently
skips when no TeX installation is available. Undefined references are
warnings, not errors — only hard errors ("! ..." lines) and undefined
citations are reported.

Errors come back as LatexError records carrying the line number in the
compiled file, so the caller can attribute each one to the section that
produced it and hand it back to the model to fix (see build_course.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Log boilerplate that follows a LaTeX error and carries no information.
_NOISE = re.compile(
    r"^\s*(See the LaTeX manual|Type\s+H\s+<return>|\.\.\.\s*$|\s*$)")

# Errors that are only ever a consequence of an earlier one — reporting them
# alongside the real cause is noise, and they are not independently fixable.
_CONSEQUENCE = ("Emergency stop.", "==> Fatal error occurred",
                "Job aborted, no legal \\end found")


@dataclass
class LatexError:
    message: str
    line: int | None = None       # line number in the compiled .tex file
    detail: str = ""              # the log excerpt (offending source shown)
    citations: list[str] = field(default_factory=list)  # undefined cite keys

    def __str__(self) -> str:
        return (f"{self.message} (line {self.line})" if self.line
                else self.message)

    def describe(self) -> str:
        """Full rendering for a repair prompt."""
        out = str(self)
        if self.detail:
            out += "\n" + "\n".join("    " + ln
                                    for ln in self.detail.splitlines())
        return out


def _parse_errors(text: str) -> list[LatexError]:
    lines = text.splitlines()
    errors: list[LatexError] = []
    for i, ln in enumerate(lines):
        if not ln.startswith("! "):
            continue
        message = ln[2:].strip()
        lineno = None
        detail: list[str] = []
        for j in range(i + 1, min(i + 15, len(lines))):
            nxt = lines[j]
            if nxt.startswith("! "):
                break
            m = re.match(r"^l\.(\d+)(.*)$", nxt)
            if m:
                lineno = int(m.group(1))
                detail.append(nxt)
                if j + 1 < len(lines) and lines[j + 1].strip():
                    detail.append(lines[j + 1])
                break
            if not _NOISE.match(nxt):
                detail.append(nxt)
        errors.append(LatexError(message, lineno, "\n".join(detail)))

    causes = [e for e in errors
              if not e.message.startswith(_CONSEQUENCE)]
    return causes or errors


def tokens_of(err: LatexError) -> list[str]:
    """Identifiers an error is about — a missing package/file, an undefined
    macro or environment. Used to find the source that caused it when the
    log gives no usable line number."""
    text = err.message + "\n" + err.detail
    found: list[str] = []
    for m in re.finditer(r"[`'\"]([^`'\"\s]+)['\"]|\\([a-zA-Z@]{2,})"
                         r"|Environment ([a-zA-Z@*]+) undefined", text):
        tok = m.group(1) or m.group(2) or m.group(3)
        if not tok:
            continue
        tok = re.sub(r"\.(sty|cls|def|tex|bib)$", "", tok).strip("{}.,;:")
        if len(tok) > 2 and tok not in found:
            found.append(tok)
    return found


def _undefined_citations(log_text: str) -> LatexError | None:
    # "Citation 'key' on page 1 undefined on input line 5."
    missing = sorted(set(
        re.findall(r"Citation '([^']+)'[^\n]*undefined", log_text)))
    if not missing:
        return None
    shown = ", ".join(missing[:8])
    more = f" … (+{len(missing) - 8})" if len(missing) > 8 else ""
    return LatexError(f"undefined citation(s): {shown}{more}",
                      citations=missing)


def check_latex(tex_path: Path) -> list[LatexError] | None:
    """Compile-check tex_path. Returns error records (empty list = clean),
    or None when no LaTeX toolchain is available."""
    tex_path = Path(tex_path).resolve()
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None and pdflatex is None:
        return None

    with tempfile.TemporaryDirectory(prefix="notetaker-tex-") as td:
        if latexmk:
            # -pdf drives pdflatex + biber/bibtex and reruns until stable.
            cmd = [latexmk, "-pdf", "-interaction=nonstopmode",
                   "-outdir=" + td, tex_path.name]
        else:
            cmd = [pdflatex, "-interaction=nonstopmode", "-draftmode",
                   "-output-directory", td, tex_path.name]
        try:
            proc = subprocess.run(cmd, cwd=tex_path.parent,
                                  capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return [LatexError("LaTeX compilation timed out after 900s")]

        # The .log is more complete than stdout (latexmk buffers/filters).
        log = Path(td) / (tex_path.stem + ".log")
        log_text = log.read_text(errors="replace") if log.exists() else ""
        errors = _parse_errors(log_text or proc.stdout)
        if proc.returncode != 0 and not errors:
            errors.append(LatexError(f"LaTeX exited with code {proc.returncode}"))

        # Surface unresolved citations (a missing .bib entry is a real
        # problem, but LaTeX only warns about it) — but only once the
        # document compiles: a hard error aborts the run before biber, so
        # every citation then looks undefined. Fix the errors first and the
        # next pass reports the citations that are genuinely missing.
        if not errors:
            undefined = _undefined_citations(log_text)
            if undefined:
                errors.append(undefined)
    return errors


def print_errors(tex_path: Path, errors: list[LatexError],
                 limit: int = 10) -> None:
    print(f"\nWARNING: {Path(tex_path).name} does not compile cleanly "
          f"({len(errors)} error(s)):")
    for e in errors[:limit]:
        print(f"  ! {e}")
    if len(errors) > limit:
        print(f"  … and {len(errors) - limit} more")
