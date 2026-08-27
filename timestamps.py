"""
timestamps.py — where each paragraph starts in the video, in the margin.

These notes are read next to the recording, and the question a reader asks
most often is "where does this come from?". So every paragraph and every
theorem-like environment carries \\ts{hh:mm:ss}: the moment the lecturer
began that material, set in the left margin in gray monospace, quiet
enough to read past and there when it is wanted.

The mark is a marginnote rather than a \\marginpar. Marginpars float apart to
keep clear of each other, which is right for a handful of notes and wrong for
one per paragraph: they slide onto neighbouring paragraphs, and a timestamp
against the wrong paragraph is worse than none. marginnote does not move.

Three details in the macro are each load-bearing:

  \\leavevmode      At the start of a paragraph TeX is still in vertical mode,
                   and marginnote there sets the note half a line above the
                   text it marks. Entering horizontal mode first puts it on
                   the baseline, which is also where it lands when the mark
                   follows a theorem head.
  \\reversemarginpar Set inside the note's own group, so this note goes to the
                   left while \\todo keeps the right margin. Set globally, the
                   two would pile into one margin.
  \\ignorespaces    The mark is followed by the text it marks, and without
                   this the space after it would open the paragraph.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: What the model writes, and what a reader sees in the margin.
MARK_RE = re.compile(r"\\ts\s*\{([^{}]*)\}")

#: Defined here rather than in the prompt, so a document cannot be produced
#: with the marks in it and no way to set them.
TIMESTAMP_PREAMBLE = r"""%% Timestamps in the left margin: \ts{hh:mm:ss} marks where a paragraph or
%% an environment starts in the video (see timestamps.py for why marginnote).
\usepackage{marginnote}
%% xcolor may already be loaded (todonotes pulls it in); asking for it with no
%% options cannot clash with an earlier load that had some.
\usepackage{xcolor}
\newcommand{\tsfont}{\normalfont\ttfamily\footnotesize\color{black!55}}
%% Each lecture's video, keyed by its section number, plus a fallback for a
%% document that is one lecture. Both are written by the assembler, not by the
%% note-taker, which writes nothing but \ts{hh:mm:ss}.
\newcommand{\tsvidany}{}
\newcommand{\tsvid}[2]{\expandafter\gdef\csname tsvid#1\endcsname{#2}}
\newcommand{\tsvidall}[1]{\renewcommand{\tsvidany}{#1}}
\def\tsparse#1:#2:#3\tsend{\number\numexpr#1*3600+#2*60+#3\relax}
\newcommand{\ts}[1]{\leavevmode{%
  \reversemarginpar
  \edef\tsid{\ifcsname tsvid\arabic{section}\endcsname%
    \csname tsvid\arabic{section}\endcsname\else\tsvidany\fi}%
  \ifx\tsid\empty
    \marginnote{\tsfont #1}%
  \else
    \edef\tsurl{https://youtu.be/\tsid?t=\tsparse#1\tsend}%
    \hypersetup{pdfborder={0 0 0}}%
    \marginnote{\href{\tsurl}{\tsfont #1}}%
  \fi}\ignorespaces}
"""

#: Macros the preamble above owns. A second definition of either is a build
#: error ("command already defined"), so a model that writes one anyway must
#: not be able to put it in the file.
RESERVED = (r"\ts", r"\tsfont", r"\tsvid", r"\tsvidall", r"\tsvidany",
            r"\tsparse")

_NAMES = "|".join(re.escape(m) for m in sorted(RESERVED, key=len,
                                                reverse=True))
_RESERVED_DEF = re.compile(
    r"\\(?:new|renew|provide)command\s*\*?\s*\{?\s*(?:" + _NAMES + r")"
    r"(?![a-zA-Z])"
    r"|\\def\s*(?:" + _NAMES + r")(?![a-zA-Z])")


def marks(body: str) -> list[str]:
    """Every timestamp marked in a body, in the order they appear."""
    return [m.group(1).strip() for m in MARK_RE.finditer(body)]


def defines_reserved(line: str) -> bool:
    """Does this line define one of the macros the preamble above owns?"""
    return bool(_RESERVED_DEF.search(line))


def drop_reserved(lines: list[str]) -> list[str]:
    """Preamble additions with any redefinition of the timestamp macros gone.

    Enforced here rather than asked for in the prompt because it is
    mechanically decidable: the model is told \\ts is already defined, and a
    model that defines it anyway would otherwise break the whole build.
    """
    return [line for line in lines if not defines_reserved(line)]


def attach_macro(tex_file: Path, video: str | None = None) -> bool:
    """Define \\ts in a model-written standalone document that uses it.

    The single-lecture document's preamble is written by the model, which is
    told to write the marks and not the machinery for them — the same split
    as the bibliography, and for the same reason: a rule enforced in code
    holds, and a rule left to the prompt mostly holds. Returns True if the
    document was changed. Idempotent, because the fix rounds re-run this on a
    file it already edited.
    """
    tex_file = Path(tex_file)
    block = TIMESTAMP_PREAMBLE + video_default(video)
    text = tex_file.read_text()
    if not MARK_RE.search(text):
        return False                     # nothing marked, nothing to define
    if r"\begin{document}" not in text:
        return False                     # a fragment has no preamble to add to
    at = text.index(r"\begin{document}")
    head, body = text[:at], text[at:]
    # Lift our own block out before purging, so that re-running this cannot
    # keep dropping and re-adding it: what is rebuilt below is byte-identical
    # to what was there, and an unchanged file is not rewritten.
    head = head.replace(block, "")
    kept = [ln for ln in head.splitlines(keepends=True)
            if not defines_reserved(ln)]
    new_text = "".join(kept).rstrip() + "\n\n" + block + "\n" + body
    if new_text == text:
        return False
    tex_file.write_text(new_text)
    return True


#: YouTube ids are eleven characters of this alphabet, underscores and
#: hyphens included — both occur, and both survive \edef into the URL.
_VIDEO_ID = r"[A-Za-z0-9_-]{11}"
_WATCH = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#\s]*&)?v=|embed/|shorts/"
    r"|live/|v/))(" + _VIDEO_ID + r")")


def youtube_id(url: str | None) -> str | None:
    """The video id in a YouTube URL, or None for anything else."""
    m = _WATCH.search(str(url or "").strip())
    return m.group(1) if m else None


def read_video_id(lecture_dir: Path) -> str | None:
    """The YouTube id a lecture was downloaded from, from its info.json.

    None for a lecture that came from a local file or a plain URL: there is
    nothing to link to, and its marks stay unlinked rather than pointing at
    some other lecture's video.
    """
    path = Path(lecture_dir) / "info.json"
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if info.get("source_type") != "youtube":
        return None
    return youtube_id(info.get("webpage_url")) or youtube_id(info.get("source"))


def video_table(videos: dict[int, str]) -> str:
    """\\tsvid lines registering each lecture's video.

    Keyed by section number, which is the lecture number because each lecture
    is a \\section — so the marks in a lecture link into that lecture's video
    without anything in the body saying which video it is.
    """
    return "".join(f"\\tsvid{{{n}}}{{{v}}}\n"
                   for n, v in sorted((videos or {}).items()) if v)


def video_default(video: str | None) -> str:
    """The same, for a document that is a single lecture and has no lecture
    numbering to key on."""
    return f"\\tsvidall{{{video}}}\n" if video else ""
