"""A macro whose body ends in a script must be braced at the definition.

\\Gm defined as \\mathbb{G}_{m} breaks at every call site that adds its own
script (\\Gm_{A} → "Double subscript"). Left alone, each lecture that hits it
braces its own call sites while the definition stays broken for every other
one. Two prompt changes address that: a standing rule in the write prompt, and
a targeted note handed to the repair pass when the error actually appears.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_course as B
from latex_check import LatexError

# --- the standing rule reaches the agent that writes the preamble -----------
prompt = B.SYSTEM_PROMPT
assert "add_to_preamble" in prompt
for probe in ("ends in a superscript or subscript",
              r"\newcommand{\Gm}{{\mathbb{G}_{m}}}",
              "Double subscript"):
    assert probe in prompt, f"missing from SYSTEM_PROMPT: {probe!r}"
# The rule must show the braced form winning, not just mention braces.
good = prompt.index(r"\newcommand{\Gm}{{\mathbb{G}_{m}}}")
bad = prompt.index(r"\newcommand{\Gm}{\mathbb{G}_{m}}")
assert good < bad, "the correct form must be shown before the counterexample"
print("SYSTEM_PROMPT: states the rule with both forms, correct one first")

# --- the repair note fires on exactly this error, and not otherwise ---------
assert B.double_script_note([]) == ""
assert B.double_script_note(
    [LatexError("Undefined control sequence.", 5)]) == "", \
    "an unrelated error must not drag in advice about macro definitions"

for msg in ("Double subscript.", "Double superscript."):
    note = B.double_script_note([LatexError(msg, 12)])
    assert note, msg
    assert "renewcommand" in note.lower(), note
    assert "add_to_preamble" in note, note
    # The whole point is to redirect away from the reported line.
    assert "not the call site" in note, note
print("double_script_note: fires on both spellings, points at the definition")

# Mixed batches still get it — the double-script error is usually one of many.
mixed = [LatexError("Missing $ inserted.", 3), LatexError("Double subscript.", 9)]
assert B.double_script_note(mixed), "must fire when bundled with other errors"
print("double_script_note: fires when bundled with unrelated errors")

# --- the note is actually reachable from the repair prompt ------------------
src = (ROOT / "build_course.py").read_text()
assert re.search(r"script_note\s*=\s*double_script_note\(errors\)", src), \
    "double_script_note must be wired into _fix_section"
assert "{script_note}" in src, "the note must be interpolated into user_text"
print("_fix_section: computes the note and interpolates it")

print("\nALL OK")
