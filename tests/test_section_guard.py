"""A written section must be a lecture, not the agent's narration.

A section was cached as a few hundred bytes ending "I'll wait for the
subagents to complete before continuing." The agent had dispatched board
readers, treated the calls as asynchronous and ended its turn; the pipeline saw
a written file, cached it, and moved on. Nothing warned, and the course lost a
chapter — the worst kind of failure, because a cached stub looks identical to a
real lecture on the next run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_course import SectionNotWritten, looks_like_section

# The actual bytes that were stored, shortened.
NARRATION = ("I'll start by reading the boards. Given there are 42, I'll "
             "delegate careful transcription to subagents in parallel "
             "batches.I'll read all nine board images in order.I've launched "
             "five parallel board-reading agents. I'll wait for the subagents "
             "to complete before continuing.")
ok, why = looks_like_section(NARRATION)
assert not ok and "narration" in why, why
print(f"the real failure is caught: {why[:60]}…")

for bad, expect in (
        ("", "empty"),
        ("   \n  ", "empty"),
        ("Let me first read the transcript and then write the notes.", "narration"),
        ("Some prose with no heading at all, " + "x" * 9000, "\\section"),
        ("\\section{Lecture 1: Short}\nOne line.", "characters"),
):
    ok, why = looks_like_section(bad)
    assert not ok, f"should have been rejected: {bad[:40]!r}"
    assert expect in why, f"{expect!r} not in {why!r}"
print("empty, narration, heading-less and stub sections all rejected")

# Long enough, has a heading, but no mathematics — a plausible-looking stub.
prose = "\\section{Lecture 1: Test}\n" + ("Ordinary prose. " * 400)
ok, why = looks_like_section(prose)
assert not ok and "environment" in why, why
print(f"a long heading-plus-prose stub is still rejected: {why[:50]}…")

# A real section passes.
real = ("\\section{Lecture 1: Quotient rings}\n\\label{lec:1}\n"
        + "\\begin{definition}\\label{def:1:a}A ring.\\end{definition}\n"
        + "\\begin{theorem}\\label{thm:1:b}It works.\\end{theorem}\n"
        + "\\begin{equation}\\label{eq:1:c}x=y\\end{equation}\n"
        + ("Genuine mathematical prose about the construction. " * 120))
ok, why = looks_like_section(real)
assert ok, f"a real section must pass: {why}"
print("a real section passes")

# The exception exists and carries the reason, so the caller can skip caching.
try:
    raise SectionNotWritten("lec-7: it begins as narration")
except SectionNotWritten as exc:
    assert "lec-7" in str(exc)
print("SectionNotWritten carries which lecture and why")

print("\nALL OK")
