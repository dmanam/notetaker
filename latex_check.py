"""
latex_check.py — best-effort compile check for generated LaTeX.

Uses pdflatex when it is on PATH (draft mode, scratch output dir); silently
skips when no TeX installation is available. Undefined cross-references are
warnings, not errors — only hard errors ("! ..." lines) are reported.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def check_latex(tex_path: Path) -> list[str] | None:
    """Compile-check tex_path. Returns error messages (empty list = clean),
    or None when pdflatex is not available."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return None
    tex_path = Path(tex_path).resolve()
    with tempfile.TemporaryDirectory(prefix="notetaker-tex-") as td:
        try:
            proc = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-draftmode",
                 "-output-directory", td, tex_path.name],
                cwd=tex_path.parent, capture_output=True, text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return ["pdflatex timed out after 300s"]
    errors = [m.group(1).strip()
              for m in re.finditer(r"^! (.+)$", proc.stdout, re.MULTILINE)]
    if proc.returncode != 0 and not errors:
        errors.append(f"pdflatex exited with code {proc.returncode}")
    return errors


def report_latex_check(tex_path: Path) -> None:
    errors = check_latex(tex_path)
    if errors is None:
        print("(pdflatex not found on PATH — skipping compile check)")
    elif errors:
        print(f"\nWARNING: {Path(tex_path).name} does not compile cleanly "
              f"({len(errors)} error(s)):")
        for e in errors[:10]:
            print(f"  ! {e}")
        if len(errors) > 10:
            print(f"  … and {len(errors) - 10} more")
        print("  A follow-up revision run (--answer) can be used to fix these.")
    else:
        print(f"Compile check OK: {Path(tex_path).name}")
