"""Resolving who lectured: guess parsing, the question, and what is stored.

The property worth protecting is that a name is never invented. A guess the
model was not confident about, a line of commentary, a number for a lecture
that does not exist — each must end up as "the lecturer" rather than as an
attribution in the notes. Nothing here calls a model.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lecturer as L

root = Path(tempfile.mkdtemp(prefix="lecturer-"))


def make(slug: str, title: str, uploader: str = "", opening: str = "") -> Path:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(json.dumps({
        "metadata": {"title": title, "uploader": uploader},
        "segments": [{"start": 0, "end": 5, "text": opening}] if opening else [],
    }))
    return d


dirs = [
    make("whitlock-1", "Dana Whitlock - 1/24 Advanced Topology",
         "Northfield Institute"),
    make("ostrand-2", "Marek Ostrand - 2/24 Advanced Topology",
         "Northfield Institute"),
    make("mystery-3", "Advanced Topology 3", "Northfield Institute",
         opening="Thanks. So last time we defined the completed "
                 "tensor product."),
]

# --- the per-lecture block ---------------------------------------------------
assert "Dana Whitlock" in L.lecturer_note("Dana Whitlock")
assert "surname" in L.lecturer_note("Dana Whitlock")
for blank in (None, "", "   ", L.UNKNOWN):
    note = L.lecturer_note(blank)
    assert "not on record" in note and "do not guess" in note, blank
print("lecturer_note: names a name, and says so when there is none")

# --- what the guesser is shown -----------------------------------------------
shown = L.describe(dirs)
assert "1. title: Dana Whitlock - 1/24 Advanced Topology" in shown
assert "3. title: Advanced Topology 3" in shown
assert "uploaded by: Northfield Institute" in shown
assert "completed tensor product" in shown, "the transcript opening is evidence"
assert L.transcript_head(dirs[0]) == "", "no segments, no opening"
print("describe: numbered titles, uploader, and the opening words")

# --- parsing the guesses ----------------------------------------------------
reply = """1: Dana Whitlock
2: Marek Ostrand
3: unknown
"""
assert L.parse_guesses(reply, 3) == {1: "Dana Whitlock", 2: "Marek Ostrand"}, \
    "an 'unknown' must be absent, not stored as a name"

# Everything a model might add around the answers, and everything it might
# get wrong about the numbering.
noisy = """Here are the lecturers:
1. **Dana Whitlock**
2) Marek Ostrand
4: Someone Else
0: Nobody
3: I could not determine this from the evidence provided, sorry
"""
got = L.parse_guesses(noisy, 3)
assert got == {1: "Dana Whitlock", 2: "Marek Ostrand"}, got
for word in ("unknown", "Unknown.", "n/a", "none", "the lecturer", "not sure"):
    assert L.parse_guesses(f"1: {word}", 1) == {}, word
assert L.parse_guesses("", 3) == {} and L.parse_guesses(None, 3) == {}
print("parse_guesses: drops hedges, prose, and numbers out of range")

# --- the question -----------------------------------------------------------
guesses = {"whitlock-1": "Dana Whitlock", "ostrand-2": "Marek Ostrand"}

# Enter accepts the suggestion; "?" declines to name anyone; typed text wins.
replies = iter(["", "Mara Ostrand", "?"])
answers = L.ask(dirs, guesses, input_fn=lambda _: next(replies),
                interactive=True)
assert answers == {"whitlock-1": "Dana Whitlock",
                   "ostrand-2": "Mara Ostrand",
                   "mystery-3": L.UNKNOWN}, answers

# A lecture with no suggestion and a blank answer stays anonymous.
assert L.ask([dirs[2]], {}, input_fn=lambda _: "", interactive=True) == \
    {"mystery-3": L.UNKNOWN}

# The prompt has to show the default, or Enter means nothing.
seen = []
L.ask(dirs[:1], guesses, input_fn=lambda p: seen.append(p) or "",
      interactive=True)
assert "Dana Whitlock" in seen[0], seen

# Input closing part-way through must not lose the remaining lectures.
def die(_):
    raise EOFError
answers = L.ask(dirs, guesses, input_fn=die, interactive=True)
assert answers == {"whitlock-1": "Dana Whitlock",
                   "ostrand-2": "Marek Ostrand",
                   "mystery-3": L.UNKNOWN}, answers
print("ask: Enter takes the guess, '?' declines, EOF falls back to guesses")

# No terminal: take the guesses silently. Unattended runs are the reason the
# fallback is a phrase and not a name.
answers = L.ask(dirs, guesses, interactive=False)
assert answers["mystery-3"] == L.UNKNOWN
assert answers["whitlock-1"] == "Dana Whitlock"
print("ask: with no terminal the guesses stand, unguessed stay anonymous")

# --- resolve ----------------------------------------------------------------
# --lecturer applies to everything and asks nothing (no model call: a guess()
# that ran here would be a bug, so make it fatal).
def explode(*a, **k):
    raise AssertionError("guess() must not run when --lecturer is given")
real_guess, L.guess = L.guess, explode
state = {}
names = L.resolve(dirs, state, forced="  Dana Whitlock  ")
assert names == {d.name: "Dana Whitlock" for d in dirs}, names
assert state["lecturers"] == names, "the answer must be recorded in state"

# Stored answers are never re-asked, and neither are lectures outside ask_for.
state = {"lecturers": {"whitlock-1": "Dana Whitlock"}}
asked = []
L.guess = lambda dirs_, **k: {"ostrand-2": "Marek Ostrand"}
names = L.resolve(dirs, state, ask_for=dirs[:2],
                  input_fn=lambda p: asked.append(p) or "", interactive=True)
assert len(asked) == 1, f"only the unanswered lecture should be asked: {asked}"
assert "Marek Ostrand" in asked[0]
assert names == {"whitlock-1": "Dana Whitlock", "ostrand-2": "Marek Ostrand"}
assert "mystery-3" not in names, "a lecture not being written is not asked about"

# Nothing to ask means nothing to run.
L.guess = explode
state = {"lecturers": {d.name: "Dana Whitlock" for d in dirs}}
assert L.resolve(dirs, state, ask_for=dirs) == state["lecturers"]
assert L.resolve([], state) == state["lecturers"]
L.guess = real_guess
print("resolve: --lecturer covers all; stored and unwritten lectures unasked")

# --- the instruction reaches both drivers -----------------------------------
import build_course as B
import generate_notes as G
for name, prompt in (("write", B.SYSTEM_PROMPT), ("verify", B.VERIFY_PROMPT),
                     ("single-lecture", G.SYSTEM_PROMPT)):
    assert L.ATTRIBUTION_INSTRUCTION in prompt, name
# Surname alone, and the anonymous phrase spelled the same way everywhere.
assert "surname alone" in L.ATTRIBUTION_INSTRUCTION
assert f'"{L.UNKNOWN}"' in L.ATTRIBUTION_INSTRUCTION
print("the attribution rule is in the writing, checking and single-lecture "
      "prompts")

# Register travels with attribution — both are appended to the writing prompts
# rather than written into them. (The rest of the shared text, including house
# typography, is checked in test_instructions.py.)
import notes_tools as N
for name, prompt in (("write", B.SYSTEM_PROMPT),
                     ("single-lecture", G.SYSTEM_PROMPT)):
    assert N.REGISTER_INSTRUCTION in prompt, name
print("the register instruction is in both writing prompts")

print("\nALL OK")
