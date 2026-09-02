#!/usr/bin/env python3
"""audit_run.py — what a course run actually produced, per lecture.

Written because the printed log cannot be trusted on its own. The
unread-board warning does not say which pass skipped the stills, a writer
that skipped boards and a checker that opened the seven it needed look
identical in the console, and "done." says nothing about whether a diagram
reached the notes. All of that is recoverable from output/logs/*.jsonl and
the section files, so recover it rather than eyeballing the log.

    python audit_run.py [--output-dir output]

Columns:
  cue        times the lecture refers to something drawn on the board
  diag       tikzcd blocks in the finished section
  prov       diagrams carrying a "% board N" provenance comment
  eq         numbered equations
  todo       unresolved \\todo markers
  ts         margin timestamps — a section with prose and none of them is a
             writer that ignored the rule, so it is flagged
  W          boards the WRITING pass opened, out of the lecture's total
  V          boards the CHECKING pass opened (targeted; a low number is fine)
  ?          open questions queued for --answer
A writer that did not open every board is the one row worth acting on, so it
is flagged rather than left to be read off a ratio.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bibliography import inline_entries
from timestamps import marks


def boards_total(lecture_dir: Path) -> int:
    path = lecture_dir / "boards" / "boards.json"
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return len(json.load(f).get("boards", []))
    except (OSError, ValueError):
        return 0


def boards_opened(log_dir: Path, slug: str, role: str) -> int | None:
    """How many stills that pass opened, from its own log record.

    None means "that pass has no finished log" and 0 means "it opened none" —
    two different facts that must not share a symbol. The log only carries a
    boards_read field when the set is non-empty, so a closed log without one
    is a genuine zero, and conflating it with missing data is how a pass that
    looked at nothing gets read as a pass that was never run.

    The record clips long lists, so the integers present are a lower bound;
    the "+N more" marker carries the rest."""
    found = None
    for path in sorted(log_dir.glob(f"*{role}-{slug}*.jsonl")):
        closed, seen = False, None
        try:
            for line in open(path):
                row = json.loads(line)
                if "boards_read" in row:
                    got = row["boards_read"]
                    ints = [x for x in got if isinstance(x, int)]
                    extra = re.search(r"\+(\d+) more", str(got))
                    seen = len(ints) + (int(extra.group(1)) if extra else 0)
                # The closing record is the one carrying the run summary.
                if "tool_counts" in row or "finished" in row or "usage" in row:
                    closed = True
        except (OSError, ValueError):
            continue
        if seen is not None:
            found = max(found or 0, seen)
        elif closed:
            found = max(found or 0, 0)
    return found


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


# Language that means a diagram was on the board. "Let me draw the diagram",
# "this diagram commutes", "a diagram chase" — the lecturer is pointing at
# something drawn, and a lecture full of these whose notes contain no tikzcd
# has lost it. Deliberately narrow: "triangle" is excluded because a
# distinguished triangle is not a drawing, and it is the most common word
# that would otherwise look like one.
_CUES = re.compile(
    r"\bdraw(?:ing|s|n)?\b|\b(?:this|the|that|following|a)\s+diagram\b"
    r"|\bdiagram\s+chase\b|\bpicture\b|\bsketch\b"
    r"|\b(?:this|the)\s+square\b", re.I)


# Pointing the reader at something only the pipeline can see: the transcript,
# a numbered board, a still. The reader has the notes and the video, so "board
# 7 shows" tells them about a file they do not have.
_ARTIFACT = re.compile(
    r"\btranscripts?\b|\bboards?\s*#?\s*\d+"
    r"|\bboard\s+(?:image|still|photo)s?\b|\bvideo\s+frames?\b", re.I)


def _strip_todos(text: str) -> str:
    """The text with \todo{...} taken out, braces matched."""
    out, i = [], 0
    for m in re.finditer(r"\\todo\s*(?:\[[^\]]*\])?\s*\{", text):
        if m.start() < i:
            continue
        depth, j = 0, m.end() - 1
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[i:m.start()])
        i = j + 1
    out.append(text[i:])
    return "".join(out)


def artifact_mentions(text: str) -> int:
    """How often the prose points at the working materials.

    Comments and \todo notes are stripped first: both are addressed to
    whoever is running this, who does have the transcript and the boards, and
    naming a board in them is the useful thing to do. The rule is about the
    document a reader sees.
    """
    return len(_ARTIFACT.findall(_strip_todos(re.sub(r"(?<!\\)%.*", "", text))))


def diagram_cues(lecture_dir: Path) -> int:
    """How often the lecture refers to something drawn on the board."""
    path = lecture_dir / "transcript.json"
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            segments = json.load(f).get("segments", [])
    except (OSError, ValueError):
        return 0
    return len(_CUES.findall(" ".join(s.get("text", "") for s in segments)))


def open_questions(section: Path) -> int:
    path = section.with_name(section.name + ".questions.json")
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return 0
    items = data if isinstance(data, list) else data.get("questions", [])
    return sum(1 for q in items if isinstance(q, dict) and not q.get("answer"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    a = ap.parse_args()
    root, log_dir = a.output_dir, a.output_dir / "logs"
    state_path = root / "course_state.json"
    if not state_path.exists():
        raise SystemExit(f"no state at {state_path}")
    with open(state_path) as f:
        state = json.load(f)
    sections = state.get("sections", {})
    order = sorted(sections, key=lambda s: sections[s]["lecture_num"])

    print(f"{'#':>3} {'lecture':40s} {'kB':>5} {'cue':>4} {'diag':>4} "
          f"{'prov':>4} {'eq':>3} {'todo':>4} {'ts':>4} {'W':>7} {'V':>4} "
          f"{'?':>2}")
    flagged, dropped, unmarked, leaking, byhand = [], [], [], [], []
    totals = dict(diag=0, prov=0, todo=0, q=0, cue=0, ts=0)
    for slug in order:
        d = root / slug
        section = d / "section.tex"
        text = section.read_text(errors="replace") if section.exists() else \
            sections[slug].get("body", "")
        total = boards_total(d)
        w = boards_opened(log_dir, slug, "write")
        v = boards_opened(log_dir, slug, "verify")
        diag = count(r"\\begin\{tikzcd\}", text)
        prov = count(r"%\s*board\s+\d+", text)
        todo = count(r"\\todo\{", text)
        ts = len(marks(text))
        q = open_questions(section)
        cue = diagram_cues(d)
        totals["cue"] += cue
        # The failure this column exists for: the lecturer kept pointing at
        # the board and none of it reached the notes.
        if cue >= 3 and diag == 0:
            dropped.append((slug, cue))
        totals["diag"] += diag; totals["prov"] += prov
        totals["todo"] += todo; totals["q"] += q; totals["ts"] += ts
        # Every paragraph should carry one. A section with prose and none at
        # all is not a lecture without paragraphs, it is the rule ignored.
        if ts == 0 and len(text) > 1000:
            unmarked.append(slug)
        leaks = artifact_mentions(text)
        if leaks:
            leaking.append((slug, leaks))
        # References the writer wrote out itself. The checking pass turns
        # these back into real citations, so any left here survived it.
        hand = inline_entries(text)
        if hand:
            byhand.append((slug, len(hand)))
        short = w is not None and total and w < total
        if short:
            flagged.append((slug, w, total))
        print(f"{sections[slug]['lecture_num']:>3} {slug[:40]:40s} "
              f"{len(text)/1000:5.0f} {cue:>4} {diag:>4}"
              f"{'!' if (cue >= 3 and diag == 0) else ' '}{prov:>4} "
              f"{count(r'\\begin\{equation\}', text):>3} {todo:>4} "
              f"{ts:>4}{'!' if (ts == 0 and len(text) > 1000) else ' '}"
              f"{(str(w) + '/' + str(total)):>7}{'!' if short else ' '} "
              f"{(v if v is not None else '-'):>4} {q:>2}")

    if byhand:
        print("\nSections with references written by hand — these never "
              f"reached the bibliography, so nothing \\cite{{}}s them:")
        for slug, n in byhand:
            print(f"  {slug}: {n} place(s)")
    if leaking:
        print("\nSections whose prose points at the transcript or a numbered "
              "board — the reader has neither:")
        for slug, n in leaking:
            print(f"  {slug}: {n} mention(s)")
    if unmarked:
        print("\nSections with prose and no margin timestamps — nothing in "
              "them points back at the video:")
        for slug in unmarked:
            print(f"  {slug}")
    print(f"\n{len(order)} lecture(s): {totals['diag']} diagram(s), "
          f"{totals['prov']} board-attributed, {totals['todo']} \\todo, "
          f"{totals['ts']} margin timestamp(s), "
          f"{totals['q']} open question(s)")
    if dropped:
        print("\nLectures that point at the board repeatedly and draw "
              "nothing — the diagrams were lost:")
        for slug, cue in dropped:
            print(f"  {slug}: {cue} cue(s), 0 diagrams")
    if flagged:
        print("\nWriters that did not open every board — anything written only "
              "on those stills came from audio:")
        for slug, w, total in flagged:
            print(f"  {slug}: {w} of {total}")
    else:
        print("Every writing pass opened every board still.")


if __name__ == "__main__":
    main()
