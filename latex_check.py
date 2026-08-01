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


# TeX reports an error two ways, and which one you get depends on a flag.
# Plain: "! Undefined control sequence."  With -file-line-error:
# "./diagram.tex:7: Undefined control sequence." — no leading "! " at all.
# Reading only the first form is how the diagram compile-repair loop went
# blind: its pdflatex call passes -file-line-error, so every error parsed to
# nothing and the agent was told "pdflatex failed." with no detail, burned its
# attempts and wrote prose instead of the diagram. The filename is required to
# be a TeX source so an ordinary log line carrying a colon and a number cannot
# masquerade as an error.
_FILE_LINE = re.compile(r"^(?:\./)?(\S+\.(?:tex|sty|cls|ltx|def)):(\d+):\s*(.+)$")


def _error_head(line: str) -> tuple[str, int | None] | None:
    """(message, line number) if this log line starts an error, else None."""
    if line.startswith("! "):
        return line[2:].strip(), None
    m = _FILE_LINE.match(line)
    if m:
        return m.group(3).strip(), int(m.group(2))
    return None


def _parse_errors(text: str) -> list[LatexError]:
    lines = text.splitlines()
    errors: list[LatexError] = []
    for i, ln in enumerate(lines):
        head = _error_head(ln)
        if head is None:
            continue
        message, lineno = head
        detail: list[str] = []
        for j in range(i + 1, min(i + 15, len(lines))):
            nxt = lines[j]
            if _error_head(nxt) is not None:
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
    # Both spellings in one log would double-report the same error.
    seen, unique = set(), []
    for e in errors:
        key = (e.message, e.line)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    errors = unique

    causes = [e for e in errors
              if not e.message.startswith(_CONSEQUENCE)]
    return causes or errors


# Boxes and bookmarks: things that compile perfectly and still come out wrong
# in the PDF — text running into the margin, and a section title whose maths
# hyperref cannot put in a bookmark.
_OVERFULL = re.compile(
    r"^Overfull \\hbox \(([\d.]+)pt too wide\)"
    r"(?:.*?at lines (\d+)--\d+|.*?at line (\d+))")
_PDFSTRING = re.compile(r"removing `([^']*)' on input line (\d+)")

# An overfull box under a couple of points is invisible on the page. Reporting
# it spends a repair round — and a model asked to fix an imperceptible overflow
# will rewrite good prose to chase it.
MIN_OVERFULL_PT = 2.0


def collect_warnings(log_text: str,
                     min_overfull_pt: float = MIN_OVERFULL_PT
                     ) -> list[LatexError]:
    """Problems that do not stop the compile but are visible in the PDF.

    Returned as LatexError records so they travel through the same
    attribute-to-a-section-and-hand-back machinery as real errors.
    """
    out: list[LatexError] = []
    for ln in log_text.splitlines():
        m = _OVERFULL.match(ln)
        if m:
            pt = float(m.group(1))
            if pt < min_overfull_pt:
                continue
            line = int(m.group(2) or m.group(3))
            out.append(LatexError(
                f"overfull \\hbox ({pt:.1f}pt too wide) — text runs into the "
                f"margin", line))
            continue
        m = _PDFSTRING.search(ln)
        if m:
            out.append(LatexError(
                f"hyperref cannot put {m.group(1)!r} in a PDF bookmark — the "
                f"section title needs \\texorpdfstring", int(m.group(2))))
    # The same overfull box is reported once per compile pass by latexmk.
    seen, unique = set(), []
    for w in out:
        key = (w.message, w.line)
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique


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
    return compile_document(tex_path)[0]


def compile_document(tex_path: Path
                     ) -> tuple[list[LatexError] | None, list[LatexError]]:
    """(errors, warnings) from one compile — the warnings come out of the same
    log, so asking for both costs nothing extra."""
    tex_path = Path(tex_path).resolve()
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None and pdflatex is None:
        return None, []

    with tempfile.TemporaryDirectory(prefix="notetaker-tex-") as td:
        if latexmk:
            # -pdf drives pdflatex + biber/bibtex and reruns until stable.
            cmd = [latexmk, "-pdf", "-interaction=nonstopmode",
                   "-outdir=" + td, tex_path.name]
        else:
            cmd = [pdflatex, "-interaction=nonstopmode", "-draftmode",
                   "-output-directory", td, tex_path.name]
        try:
            # errors="replace": TeX writes raw 8-bit bytes to stdout (font
            # names, ^^-escaped chars from a stray byte in the source), and
            # strict UTF-8 decoding raises *inside* subprocess.run — so the
            # crash lands before any log parsing, taking the whole build down
            # after every lecture is already written. The .log is read with
            # the same tolerance below for the same reason.
            proc = subprocess.run(cmd, cwd=tex_path.parent,
                                  capture_output=True, text=True,
                                  errors="replace", timeout=900)
        except subprocess.TimeoutExpired:
            return [LatexError("LaTeX compilation timed out after 900s")], []

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
        warnings = collect_warnings(log_text)
    return errors, warnings


def print_warnings(tex_path: Path, warnings: list[LatexError],
                   limit: int = 10) -> None:
    print(f"\n{Path(tex_path).name} compiles, with {len(warnings)} "
          f"presentation issue(s):")
    for w in warnings[:limit]:
        print(f"  · {w}")
    if len(warnings) > limit:
        print(f"  … and {len(warnings) - limit} more")


def print_errors(tex_path: Path, errors: list[LatexError],
                 limit: int = 10) -> None:
    print(f"\nWARNING: {Path(tex_path).name} does not compile cleanly "
          f"({len(errors)} error(s)):")
    for e in errors[:limit]:
        print(f"  ! {e}")
    if len(errors) > limit:
        print(f"  … and {len(errors) - limit} more")
