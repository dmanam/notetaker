"""
style_extract.py — turn a whole document into a few representative style
samples that stand on their own.

`--style-exemplar` used to take the first few thousand characters of the file
after \\begin{document}. That is the worst part to take: the opening of a set
of lecture notes is a preface, an overview, a list of conventions — prose
that reads nothing like the body, which is where the register actually
lives. And it arrives full of the author's private macros, so the model
reading it sees \\cH, \\Spec, \\dsq and has to guess.

So a model reads the whole document, picks passages from across it, and
rewrites each to stand alone: private macros expanded, packages limited to
the ones the notes already load. The rewrite is then CHECKED, not trusted —
both versions are compiled and the text of the two PDFs compared. A rewrite
that renders differently is discarded, because a style sample that quietly
says something else is worse than no sample.

The check is deliberately independent of the model that did the rewriting:
it compares typeset output, which is the thing that actually has to match.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# A book-length set of notes runs to ~500k characters, and clipping the tail
# costs exactly the late passages the extractor is told to prefer — the whole
# point is to sample past the introduction. 600k is ~150k tokens, which the
# context holds comfortably; the cap is only here so a pathological input
# cannot blow up the prompt.
MAX_SOURCE = 600_000        # of a source document to show the extractor
PASSAGES = 5
MIN_MATCH = 0.97            # rendered-text agreement for a rewrite to pass

ORIGINAL = re.compile(r"%%%\s*ORIGINAL-START\s*\n(.*?)\n%%%\s*ORIGINAL-END",
                      re.DOTALL)
REWRITTEN = re.compile(r"%%%\s*REWRITTEN-START\s*\n(.*?)\n%%%\s*REWRITTEN-END",
                       re.DOTALL)
EXTRA_PACKAGES = re.compile(r"%%%\s*PACKAGES:\s*(.*)")


EXTRACT_PROMPT = r"""\
You are preparing STYLE SAMPLES from a set of mathematical notes. They will
be shown to a model writing other notes, so that it can match this author's
register. Nothing about the subject matter matters — only how the author
writes.

Read the whole document, then choose __N__ passages that are representative of
the writing in the BODY of it. Some guidance on choosing:

- Take them from different places, well spread through the document. Do not
  take the opening: a preface, an overview, a list of conventions and an
  introduction are the least representative pages in any set of notes.
- Prefer passages that show the author doing ordinary work — stating and
  proving something, setting up a definition and using it, explaining why a
  construction is the right one. That is where register lives.
- Each should be self-contained enough to read cold: a few hundred words,
  ideally a complete environment or two plus the prose around them.
- Between them they should show the range: how much is spelled out, how
  proofs are laid out, how formal the prose is, how displays are used.

Then REWRITE each passage so it stands on its own:

- Expand every macro the author defined. A reader of the sample has no
  preamble, and \cH or \Zh tells them nothing. Write \mathcal{H},
  \widehat{\mathbb{Z}}, and so on, in full.
- Use only these packages, which the reader already has: amsmath, amsthm,
  amssymb, thmtools, mathtools, enumitem, tikz-cd and tikz. Theorem
  environments available: theorem, lemma, proposition, corollary,
  definition, example, exercise, remark, notation. Map the author's
  environments onto these (their `thm` becomes theorem, their `dfn` becomes
  definition, and so on).
- If a passage genuinely cannot be rendered without another package, you may
  ask for it — but prefer choosing a different passage.
- LEAVE CROSS-REFERENCES ALONE. \cref, \ref, \eqref and \label stay exactly
  as the author wrote them. Pointing at something outside the passage will
  typeset as ?? — that is expected, it is the same in the original, and how
  an author cross-references is part of the style worth showing. Do not turn
  them into prose: the rewrite is compared against the original by what it
  typesets, and replacing a reference with "the previous lemma" is a change
  to the rendered text that will fail the comparison.
- Change NOTHING else. Same words, same sentences, same mathematics, same
  structure. This is a transcription into portable LaTeX, not an edit: the
  point of the sample is destroyed if you improve the prose.

Write the output file as a sequence of blocks, exactly in this form:

%%% PASSAGE
%%% ORIGINAL-START
<the passage copied VERBATIM from the source, byte for byte>
%%% ORIGINAL-END
%%% REWRITTEN-START
<your portable rewrite>
%%% REWRITTEN-END

The ORIGINAL must be an exact copy of a contiguous stretch of the source —
it is checked against the file, and it is what your rewrite is compared
against. If you need extra packages, add one line before the first passage:

%%% PACKAGES: mathrsfs, stmaryrd

Nothing else in the file.
"""


DEFAULT_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amsthm,amssymb,mathtools}
\usepackage{thmtools}
\usepackage{enumitem}
\usepackage{tikz-cd}
\usetikzlibrary{arrows.meta,decorations.pathmorphing,positioning,calc,patterns}
%(extra)s
%% So a passage keeps the author's \cref/\ref rather than having them
%% paraphrased away: a reference out of the passage's scope typesets as ??
%% on both sides, which compares equal, whereas prose substituted for it
%% does not.
\usepackage[hidelinks]{hyperref}
\usepackage[nameinlink,noabbrev]{cleveref}
\declaretheorem[numberwithin=section,style=plain]{theorem}
\declaretheorem[sibling=theorem,style=plain]{lemma}
\declaretheorem[sibling=theorem,style=plain]{proposition}
\declaretheorem[sibling=theorem,style=plain]{corollary}
\declaretheorem[sibling=theorem,style=definition]{definition}
\declaretheorem[sibling=theorem,style=definition]{example}
\declaretheorem[sibling=theorem,style=definition]{exercise}
\declaretheorem[sibling=theorem,style=remark]{remark}
\declaretheorem[sibling=theorem,style=remark]{notation}
"""


@dataclass
class Passage:
    original: str
    rewritten: str
    verified: bool = False
    note: str = ""
    match: float = 0.0

    def to_dict(self) -> dict:
        return {"rewritten": self.rewritten, "verified": self.verified,
                "match": round(self.match, 4), "note": self.note}


def split_document(text: str) -> tuple[str, str]:
    """(preamble, body). The preamble is needed to compile the ORIGINAL of a
    passage, since that is what its macros are defined in."""
    m = re.search(r"\\begin\{document\}", text)
    if not m:
        return "", text
    return text[:m.start()], text[m.end():]


def parse_output(text: str) -> tuple[list, list]:
    """(passages, extra packages) from the extractor's file."""
    originals = ORIGINAL.findall(text or "")
    rewrites = REWRITTEN.findall(text or "")
    pkgs = []
    m = EXTRA_PACKAGES.search(text or "")
    if m:
        pkgs = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return ([Passage(o.strip(), r.strip())
             for o, r in zip(originals, rewrites)], pkgs)


# ---------------------------------------------------------------------------
# Verification: compile both, compare what the reader would see
# ---------------------------------------------------------------------------

def _render_text(body: str, preamble: str, workdir: Path) -> str | None:
    """Typeset a fragment and return the text of the resulting page.

    Comparing rendered text rather than source is the point: two spellings of
    the same mathematics are supposed to differ in source and agree here."""
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "sample.tex"
    src.write_text(f"{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        "-halt-on-error", "sample.tex"],
                       cwd=workdir, capture_output=True, timeout=180)
    pdf = workdir / "sample.pdf"
    if r.returncode != 0 or not pdf.exists():
        return None
    try:
        import fitz
    except ImportError:
        return None
    try:
        with fitz.open(pdf) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return None


def _normalise(text: str) -> str:
    """Whitespace and line breaking are not differences we care about: the
    same paragraph set in a different measure breaks in different places."""
    return re.sub(r"\s+", " ", text or "").strip()


def similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    a, b = _normalise(a), _normalise(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def verify(passage: Passage, source_preamble: str, extra_packages: list,
           workdir: Path) -> Passage:
    """Compile the original against its own preamble and the rewrite against
    the portable one, and compare what each renders to."""
    extra = "\n".join(f"\\usepackage{{{p}}}" for p in extra_packages)
    portable = DEFAULT_PREAMBLE % {"extra": extra}

    want = _render_text(passage.original, source_preamble, workdir / "orig")
    if want is None:
        passage.note = ("the original does not compile on its own, so the "
                        "rewrite cannot be checked against it")
        return passage
    got = _render_text(passage.rewritten, portable, workdir / "new")
    if got is None:
        passage.note = "the rewrite does not compile"
        return passage

    passage.match = similarity(want, got)
    passage.verified = passage.match >= MIN_MATCH
    if not passage.verified:
        passage.note = (f"renders differently from the original "
                        f"({passage.match:.0%} agreement)")
    return passage


def check_originals(passages: list, source: str) -> list:
    """Drop passages whose "original" is not actually in the document.

    Without this the extractor could paraphrase the source into the ORIGINAL
    block and the comparison would be against its own invention rather than
    against the author."""
    flat = _normalise(source)
    kept = []
    for p in passages:
        if _normalise(p.original) in flat:
            kept.append(p)
        else:
            p.note = "the quoted original is not in the source document"
    return kept


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CACHE_SUFFIX = ".style.json"


def cache_path(source: Path, cache_dir: Path | None = None) -> Path:
    source = Path(source)
    if cache_dir is None:
        return source.with_name(source.name + CACHE_SUFFIX)
    return Path(cache_dir) / (source.stem + CACHE_SUFFIX)


def load(source: Path, cache_dir: Path | None = None) -> list | None:
    path = cache_path(source, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("source_bytes") != Path(source).stat().st_size:
        return None                 # the document changed; extract again
    return [p["rewritten"] for p in data.get("passages", [])
            if p.get("verified")]


def extract(source: Path, cache_dir: Path | None = None, *,
            backend: str = "subscription", model: str | None = None,
            passages: int = PASSAGES, force: bool = False,
            log_dir: Path | None = None) -> list:
    """Representative, portable, verified style samples from `source`."""
    source = Path(source).resolve()
    if not force:
        cached = load(source, cache_dir)
        if cached is not None:
            print(f"  style: {len(cached)} verified passage(s) cached "
                  f"for {source.name}")
            return cached

    text = source.read_text(errors="replace")
    preamble, _body = split_document(text)
    clipped = text[:MAX_SOURCE]
    if len(text) > MAX_SOURCE:
        print(f"  style: {source.name} is {len(text)/1000:.0f}k chars; "
              f"showing the extractor the first {MAX_SOURCE/1000:.0f}k")

    from claude_backend import run_agent
    from notes_tools import NotesToolContext

    work = Path(cache_dir or source.parent) / f"style-{source.stem}"
    work.mkdir(parents=True, exist_ok=True)
    ctx = NotesToolContext(refs_dir=work / "refs", read_roots=[work.resolve()])
    out_file = work / "passages.tex"
    print(f"\n=== Extracting style samples from {source.name} ===", flush=True)
    run_agent(
        system_prompt=EXTRACT_PROMPT.replace("__N__", str(passages)),
        user_text=(f"The document is below.\n\n"
                   f"<document name=\"{source.name}\">\n{clipped}\n</document>"),
        ctx=ctx,
        output_file=out_file,
        backend=backend,
        model=model,
        role="style-extract",
        log_dir=log_dir,
    )
    found, pkgs = parse_output(out_file.read_text(errors="replace")
                              if out_file.exists() else "")
    if pkgs:
        print(f"  style: extractor asked for extra packages: {', '.join(pkgs)}")
    kept = check_originals(found, text)
    dropped = len(found) - len(kept)
    for n, p in enumerate(kept, 1):
        verify(p, preamble, pkgs, work / f"check-{n:02d}")

    good = [p for p in kept if p.verified]
    print(f"  style: {len(found)} passage(s) proposed, "
          f"{dropped} not found in the source, "
          f"{len(kept) - len(good)} failed verification, "
          f"{len(good)} kept")
    for p in kept:
        if not p.verified:
            print(f"    dropped: {p.note}")

    cache_path(source, cache_dir).write_text(json.dumps({
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "packages": pkgs,
        "passages": [p.to_dict() for p in found],
    }, indent=2))
    return [p.rewritten for p in good]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--passages", type=int, default=PASSAGES)
    ap.add_argument("--backend", default="subscription")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    got = extract(a.source, a.cache_dir, backend=a.backend,
                  passages=a.passages, force=a.force)
    for n, p in enumerate(got, 1):
        print(f"\n----- passage {n} -----\n{p[:600]}")


if __name__ == "__main__":
    main()
