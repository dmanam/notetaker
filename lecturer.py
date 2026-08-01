"""Who gave each lecture, and how the notes should name them.

Published notes write "Whitlock defines" or "as Ostrand remarked" — surname
alone, the way one professional refers to another. They do not write "Dana
says", "the speaker claims", or "our lecturer". Getting that right needs a
name, and nothing in the pipeline has one: the transcript usually never says
it, and the uploader is an institution ("Northfield Institute for
Mathematical Sciences"), not a person. So the name is asked once per lecture,
with a guess offered as the default, and remembered in the course state.

The guess is a single cheap model call covering the whole series, not a regex
per title. Titles here look like "Dana Whitlock - 1/24 Advanced Topology", and
the hard part is not finding two capitalised words but deciding which pair is
a person: "Spectral Sequences | Marek Ostrand" has two candidates and only one
lecturer. Seeing every title at once settles most cases — a phrase common to
all of them is the series, and a name in only some of them is a speaker the
series alternates with — and that is exactly what a per-title pattern cannot
see.

A wrong name is worse than no name, because it gets printed in the notes as an
attribution. So the guesser is told to answer "unknown" when the evidence does
not point at a person, and the fallback everywhere is the phrase "the
lecturer", which is never wrong, only anonymous.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

UNKNOWN = "the lecturer"

# Opening words of each transcript shown to the guesser. Enough to catch an
# introduction ("it's a pleasure to welcome...") or a self-introduction, short
# enough that the whole series fits in one cheap call.
HEAD_CHARS = 700


# ---------------------------------------------------------------------------
# Prompt fragments
# ---------------------------------------------------------------------------

# Appended to the writing and checking system prompts. The rule is general;
# the name itself arrives per lecture, from lecturer_note below.
ATTRIBUTION_INSTRUCTION = """
Naming the lecturer: the task says who is lecturing. Where the notes refer to
them, use the surname alone — "Whitlock defines", "as Ostrand pointed out",
"Halloway's argument" — which is how a professional refers to another
professional in writing. Not the first name, not "the speaker", not "our
lecturer", and not "the lecturer" when you have been given a name; if the task
says no name is on record, "the lecturer" is then the correct phrase. Treat the
surname as its bearer does: particles stay attached ("van der Waerden", "de
Jong"), and where the family name comes first it is still the family name you
use. This governs how to refer to them, not how often — attribute where the
mathematics needs attributing, and otherwise write in the notes' own voice
("we define", "the idea is"), which is what these notes are."""


def lecturer_note(name: str | None) -> str:
    """The per-lecture block naming the lecturer for the model."""
    name = (name or "").strip()
    if not name or name == UNKNOWN:
        return ("The lecturer's name is not on record for this lecture. Where "
                "the notes need to name them, write \"the lecturer\" — do not "
                "guess a name and do not infer one from the transcript.\n\n")
    return (f"The lecturer is {name}; refer to them by surname alone.\n\n")


GUESS_PROMPT = """You identify who delivered each lecture in a recorded series.

You are given one entry per lecture: its number, the title of the recording,
who uploaded it, and the opening words of the transcript. Write the output file
as one line per lecture, in the order given:

  <number>: <the lecturer's full name>

or, where you are not confident:

  <number>: unknown

Nothing else in the file — no preamble, no explanation, no commentary on a line
of its own.

The answer is the person who stood up and gave that lecture. Not the channel or
institution that posted the recording (an uploader is almost never the
lecturer). Not whoever introduces them. Not a mathematician whose work is being
discussed. Not the name of the series.

How to read the evidence:
- A phrase appearing in every title is the name of the series or the course,
  not a person, however much it reads like a name.
- A name appearing in only some titles, where the others name someone else, is
  a lecturer: series alternate between speakers, and you should then answer
  per lecture rather than forcing one name onto all of them.
- "Speaker Name - Topic", "Topic | Speaker Name", "Topic (Speaker Name)" and
  "Speaker Name: Topic" are all common. Which side is a person's name settles
  it; the position does not.
- The transcript opening may hold an introduction ("it is a pleasure to
  welcome...", "today we will hear from...") or a self-introduction ("my name
  is..."). Either names the lecturer. Someone merely thanked is not
  necessarily the one speaking.
- Where the title names a person and the transcript opening names a different
  one, prefer the title unless the opening is plainly an introduction of the
  speaker.

Answer "unknown" unless the evidence actually points at a person. A wrong name
is worse than no name: it will be printed in the notes as an attribution.
Inventing a plausible-sounding name from the subject matter is the specific
failure to avoid."""


# ---------------------------------------------------------------------------
# What the guesser is shown
# ---------------------------------------------------------------------------

def lecture_meta(lecture_dir: Path) -> dict:
    """Title and uploader for a lecture directory, from whichever file has it."""
    for name in ("transcript.json", "info.json"):
        path = Path(lecture_dir) / name
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        meta = data.get("metadata", data) if isinstance(data, dict) else {}
        if meta.get("title"):
            return meta
    return {}


def transcript_head(lecture_dir: Path, chars: int = HEAD_CHARS) -> str:
    """The first words spoken, where an introduction would be."""
    path = Path(lecture_dir) / "transcript.json"
    if not path.exists():
        return ""
    try:
        with open(path) as f:
            segments = json.load(f).get("segments", [])
    except (OSError, ValueError):
        return ""
    text = " ".join(s.get("text", "").strip() for s in segments[:60])
    return " ".join(text.split())[:chars]


def describe(lecture_dirs: list[Path]) -> str:
    """The evidence block: one entry per lecture, numbered as asked for."""
    parts = []
    for n, d in enumerate(lecture_dirs, 1):
        meta = lecture_meta(d)
        rows = [f"{n}. title: {meta.get('title') or Path(d).name}"]
        if meta.get("uploader"):
            rows.append(f"   uploaded by: {meta['uploader']}")
        head = transcript_head(d)
        if head:
            rows.append(f"   transcript opens: \"{head}\"")
        parts.append("\n".join(rows))
    return "\n\n".join(parts)


def parse_guesses(text: str, count: int) -> dict[int, str]:
    """{lecture number: name} from the guesser's reply.

    Out-of-range numbers and "unknown" are dropped rather than stored: an
    absent guess and a guess of "the lecturer" mean different things at the
    prompt (no suggestion versus a suggestion of anonymity), and only the
    former is honest about what the model actually said."""
    out: dict[int, str] = {}
    for line in (text or "").splitlines():
        m = re.match(r"\s*(\d{1,3})\s*[:.)]\s*(.+?)\s*$", line)
        if not m:
            continue
        num, name = int(m.group(1)), m.group(2).strip()
        if not 1 <= num <= count:
            continue
        name = name.strip("*_`\"'")
        if name.lower().strip(".") in ("unknown", "unclear", "none", "n/a",
                                      "not sure", "the lecturer"):
            continue
        # A name, not a sentence about one.
        if len(name) > 60 or len(name.split()) > 5:
            continue
        out[num] = name
    return out


def guess(lecture_dirs: list[Path], *, backend: str = "subscription",
          model: str | None = None, frame_model: str | None = None,
          work_dir: Path | None = None,
          log_dir: Path | None = None) -> dict[str, str]:
    """{slug: guessed name} for the whole series, in one cheap model call.

    Every lecture is shown even when only some need answering: the
    cross-lecture comparison is where most of the signal is."""
    if not lecture_dirs:
        return {}
    from claude_backend import (API_FRAME_MODEL, CODEX_FRAME_MODEL,
                                SUBSCRIPTION_FRAME_MODEL, run_agent)
    from notes_tools import NotesToolContext

    cheap = frame_model or {"subscription": SUBSCRIPTION_FRAME_MODEL,
                            "codex": CODEX_FRAME_MODEL}.get(backend,
                                                            API_FRAME_MODEL)
    work = Path(work_dir or Path(lecture_dirs[0]).parent / "lecturers")
    work.mkdir(parents=True, exist_ok=True)
    out_file = work / "guesses.txt"
    ctx = NotesToolContext(refs_dir=work / "refs", read_roots=[work.resolve()])
    try:
        run_agent(
            system_prompt=GUESS_PROMPT,
            user_text=(f"{len(lecture_dirs)} lecture(s) of one series:\n\n"
                       f"{describe(lecture_dirs)}"),
            ctx=ctx,
            output_file=out_file,
            backend=backend,
            model=cheap,
            role="lecturer-guess",
            log_dir=log_dir,
        )
    except Exception as exc:
        print(f"  (could not guess lecturer names: {exc})")
        return {}
    text = out_file.read_text(errors="replace") if out_file.exists() else ""
    by_num = parse_guesses(text, len(lecture_dirs))
    return {Path(d).name: by_num[n]
            for n, d in enumerate(lecture_dirs, 1) if n in by_num}


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------

def ask(lecture_dirs: list[Path], guesses: dict[str, str], *,
        input_fn=None, interactive: bool | None = None) -> dict[str, str]:
    """Put the question to the user, once per lecture, in one sitting.

    Enter accepts the suggestion; "?" records that the name is not known. With
    no terminal the suggestions are taken silently and printed, so unattended
    runs still work — the failure mode is an anonymous attribution, not a
    wrong one."""
    if not lecture_dirs:
        return {}
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    answers: dict[str, str] = {}

    if not interactive:
        for d in lecture_dirs:
            answers[Path(d).name] = guesses.get(Path(d).name, UNKNOWN)
        named = sum(1 for v in answers.values() if v != UNKNOWN)
        print(f"  (no terminal — taking the suggested name for "
              f"{named}/{len(answers)} lecture(s); the rest stay "
              f"\"{UNKNOWN}\". Use --lecturer to set one for all.)")
        for d in lecture_dirs:
            print(f"    {Path(d).name}: {answers[Path(d).name]}")
        return answers

    ask_input = input_fn or input
    print(f"\n=== Who is lecturing? ({len(lecture_dirs)} lecture(s)) ===")
    print("The notes refer to the lecturer by surname, as published notes do.")
    print("Enter accepts the suggestion; \"?\" if you do not know "
          "(the notes then say \"the lecturer\").")
    for n, d in enumerate(lecture_dirs, 1):
        slug = Path(d).name
        title = lecture_meta(d).get("title") or slug
        suggestion = guesses.get(slug, "")
        print(f"\n[{n}/{len(lecture_dirs)}] {title}")
        prompt = (f"    lecturer [{suggestion}]: " if suggestion
                  else f"    lecturer (blank = \"{UNKNOWN}\"): ")
        try:
            reply = ask_input(prompt).strip()
        except EOFError:
            print("    (input closed — taking the suggestions for the rest)")
            for rest in lecture_dirs[n - 1:]:
                answers.setdefault(Path(rest).name,
                                   guesses.get(Path(rest).name, UNKNOWN))
            break
        if reply in ("?", "-"):
            answers[slug] = UNKNOWN
        elif reply:
            answers[slug] = reply
        else:
            answers[slug] = suggestion or UNKNOWN
    return answers


def resolve(lecture_dirs: list[Path], state: dict, *,
            forced: str | None = None, ask_for: list[Path] | None = None,
            backend: str = "subscription", model: str | None = None,
            frame_model: str | None = None, work_dir: Path | None = None,
            log_dir: Path | None = None,
            input_fn=None, interactive: bool | None = None) -> dict[str, str]:
    """Settle the lecturer for every lecture, and record it in `state`.

    `forced` (--lecturer) applies to all of them without asking, which is the
    common case: most series have one speaker. Otherwise the question is put
    only for `ask_for` — the lectures about to be written — since a name
    cannot change notes that already exist, and stored answers are never
    re-asked."""
    names: dict[str, str] = dict(state.setdefault("lecturers", {}))
    slugs = [Path(d).name for d in lecture_dirs]

    if forced:
        for s in slugs:
            names[s] = forced.strip()
        state["lecturers"] = names
        print(f"Lecturer: {forced.strip()} (all {len(slugs)} lecture(s)).")
        return names

    candidates = [d for d in (ask_for if ask_for is not None else lecture_dirs)
                  if Path(d).name not in names]
    if candidates:
        guesses = guess(lecture_dirs, backend=backend, model=model,
                        frame_model=frame_model, work_dir=work_dir,
                        log_dir=log_dir)
        names.update(ask(candidates, guesses, input_fn=input_fn,
                         interactive=interactive))
        state["lecturers"] = names
    return names


def main() -> None:
    """Print what would be suggested for a course directory, and answer nothing.

    The guesser is the one part of this that a unit test cannot check, so it
    is worth being able to look at its output on a real series before a run
    that will attribute mathematics to whatever it says."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", type=Path,
                    help="Course output directory (one subdirectory per "
                         "lecture, each with a transcript.json)")
    ap.add_argument("--backend", default="subscription")
    ap.add_argument("--frame-model", default=None,
                    help="Model that does the guessing (default: the "
                         "backend's cheap one)")
    a = ap.parse_args()
    dirs = sorted(d for d in a.output_dir.iterdir()
                  if (d / "transcript.json").exists())
    if not dirs:
        raise SystemExit(f"no lectures with a transcript under {a.output_dir}")
    got = guess(dirs, backend=a.backend, frame_model=a.frame_model,
                work_dir=a.output_dir / "lecturers")
    for n, d in enumerate(dirs, 1):
        title = lecture_meta(d).get("title") or d.name
        print(f"{n:3d}. {got.get(d.name, '(' + UNKNOWN + ')'):22s} {title}")
    print(f"\nnamed {len(got)}/{len(dirs)}; the rest would default to "
          f"\"{UNKNOWN}\"")


if __name__ == "__main__":
    main()
