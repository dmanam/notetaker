"""equations.py — numbering that matches how the notes actually cite.

A display gets a number so it can be referred to. An unreferenced number is
noise: it invites the reader to look for the citation that never comes, and on
a 200-page course it pushes every later number along, so the one equation a
reader is hunting for is never where a half-remembered "(3.14)" says it is.

So a plain \\begin{equation} follows what cites it, in both directions: cited
displays are numbered, uncited ones are starred, and an equation that lecture 9
starts citing next week is numbered again on the next assembly. \\label always
stays put — it is how a later lecture knows what to \\cref, and a label that
existed only while the equation happened to be numbered could never be cited
into existence.

The only thing left for a person is a cited label inside a starred *multi-line*
display (align, gather, multline). Those number per line via \\notag, so
starring the environment as a whole is the wrong instrument and which line the
reference meant is a judgement call.

"Referenced" is decided across the whole course, never one section at a time —
a lemma in lecture 3 is routinely cited from lecture 9, and a per-section
answer would unnumber exactly the equations that carry the course.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# \ref, \cref, \Cref, \eqref, \labelcref, \autoref, \pageref, \vref — and the
# comma-separated key lists cleveref takes.
# labelc before label: the alternation is ordered, and \labelcref must not be
# read as \label + "cref".
_REF = re.compile(
    r"\\(?:eq|labelc|label|c|C|auto|page|v|name)?ref\*?\s*\{([^{}]*)\}")
_LABEL = re.compile(r"\\label\s*\{([^{}]*)\}")
# \tag and \tag*, but not \notag.
_TAG = re.compile(r"\\tag\*?\s*\{")

# One display environment, starred or not, with the star matched on both ends
# so \begin{equation} never pairs with \end{equation*}.
_EQUATION = re.compile(
    r"\\begin\{equation(\*?)\}(.*?)\\end\{equation\1\}", re.DOTALL)

# Multi-line displays. Their numbering is per line (\notag, \nonumber), so a
# whole-environment star is the wrong instrument and they are only reported.
_MULTILINE = re.compile(
    r"\\begin\{(align|gather|multline|flalign|eqnarray)(\*?)\}(.*?)"
    r"\\end\{\1\2\}", re.DOTALL)


@dataclass
class ReviewItem:
    """A numbering problem that needs a person, not a rewrite."""
    label: str
    kind: str          # "unnumbered" | "dangling"
    context: str       # the environment or a snippet, for the report

    def __str__(self) -> str:
        if self.kind == "dangling":
            return (f"{self.label}: referenced but never defined "
                    f"({self.context})")
        return (f"{self.label}: referenced, but sits in an unnumbered "
                f"{self.context} with no \\tag — the reference cannot resolve "
                f"to a number")


def referenced_labels(text: str) -> set[str]:
    """Every label the text points at, from any of the \\ref spellings.

    Key lists are split on commas and stripped: a \\cref{a,\\n  b} wrapped
    across a line leaves whitespace on the second key, and treating " b" as a
    distinct label would make the real one look unreferenced.
    """
    out: set[str] = set()
    for m in _REF.finditer(text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                out.add(key)
    return out


def defined_labels(text: str) -> set[str]:
    return {m.group(1).strip() for m in _LABEL.finditer(text)}


def normalize_equation_numbering(text: str,
                                 referenced: set[str]) -> tuple[str, int, int]:
    """Number every cited display and unnumber every uncited one.

    Returns (text, unnumbered, numbered).

    Both directions, because the course is written a lecture at a time: an
    equation nothing cites yet becomes equation*, and if lecture 9 later
    cites it, the next assembly numbers it again. That is why the \\label
    stays on an unnumbered display — it is how a later lecture knows what to
    \\cref in the first place, and a label that only exists while the
    equation happens to be numbered could never be cited into existence.

    A display carrying \\tag is left exactly as it is in either direction:
    the tag is a number chosen by hand, so it is deliberate whether or not
    anything cites it.
    """
    unnumbered = numbered = 0

    def rewrite(m: re.Match) -> str:
        star, body = m.group(1), m.group(2)
        if _TAG.search(body):
            return m.group(0)
        cited = bool({lab.strip() for lab in _LABEL.findall(body)} & referenced)
        want = "" if cited else "*"
        if want == star:
            return m.group(0)
        nonlocal unnumbered, numbered
        if want:
            unnumbered += 1
        else:
            numbered += 1
        return f"\\begin{{equation{want}}}{body}\\end{{equation{want}}}"

    return _EQUATION.sub(rewrite, text), unnumbered, numbered


def review_items(text: str, referenced: set[str]) -> list[ReviewItem]:
    """Cited labels in a display this pass cannot number for you.

    Only the multi-line environments reach here. A starred `equation` is
    handled automatically — normalize_equation_numbering unstars it the
    moment something cites it — but align, gather and multline number per
    line, with \\notag deciding which lines get one. Starring or unstarring
    the whole environment is the wrong instrument, and choosing which line
    the reference meant is the reviewer's call.
    """
    items: list[ReviewItem] = []
    for m in _MULTILINE.finditer(text):
        if not m.group(2):
            continue                   # numbered: nothing to answer for
        body = m.group(3)
        if _TAG.search(body):
            continue                   # explicitly tagged, which is allowed
        for label in _LABEL.findall(body):
            label = label.strip()
            if label in referenced:
                items.append(
                    ReviewItem(label, "unnumbered", f"{m.group(1)}*"))
    return items


def dangling_references(text: str, defined: set[str]) -> list[ReviewItem]:
    """Labels the text cites that nothing defines — a "??" in the PDF."""
    return [ReviewItem(label, "dangling", "no \\label anywhere in the course")
            for label in sorted(referenced_labels(text) - defined)]
