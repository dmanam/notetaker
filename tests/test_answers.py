"""Follow-up runs: answers are applied once, and \\todo markers are answerable.

Two defects, both reported from a real course:

1. An answer given *while a lecture was being written* was never marked
   applied — build_course called mark_answers_applied only after a revision
   pass. So every later --answer/--answer-all announced "N answer(s) from an
   earlier run were never applied" and re-delivered them to the model, on
   every run, for ever.

2. --answer-all counted \\todo markers when picking lectures but never showed
   them to the user; they were only ever swept by the model. There was no way
   to answer a \\todo however many there were.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build_course as B
import claude_backend as CB
import notes_tools as NT

# --- \todo extraction survives braces inside the marker ---------------------
body = r"""\section{One}
\todo{simple one}
Text with \todo[inline]{nested {\mathbb Z} braces and \text{more}} inline.
\todo{spanning
two lines}
Not a marker: \todonotes or \todos{x}
"""
items = B.todo_items(body)
assert items == ["simple one",
                 r"nested {\mathbb Z} braces and \text{more}",
                 "spanning two lines"], items
# The brace-counting matters: a non-greedy regex stops at the first inner "}".
assert r"{\mathbb Z}" in items[1], "inner braces must not truncate the body"
print(f"todo_items: extracted {len(items)} marker(s), inner braces intact")

assert B.todo_items(r"\section{No todos}") == []
# count_todos and todo_items must agree, or the survey and the prompt diverge.
assert CB.count_todos(body) == len(items), (CB.count_todos(body), len(items))
print("todo_items agrees with count_todos")

# --- answering a \todo is optional and non-interactive runs are unchanged ---
asked = []


def fake_input(prompt, should_abort=None):
    asked.append(prompt)
    return {1: "", 2: "the answer", 3: ""}[len(asked)]


real_input, B.ask_user_input = B.ask_user_input, fake_input
try:
    block = B.ask_todo_answers(["first", "second", "third"], 4)
finally:
    B.ask_user_input = real_input
assert len(asked) == 3, "every marker must be offered"
assert block is not None and "the answer" in block, block
assert "first" not in block and "third" not in block, \
    "skipped markers must not be presented to the model as answered"
print("ask_todo_answers: only answered markers are passed on")

# No terminal available -> ask_user_input returns "" -> nothing collected, and
# the run proceeds exactly as it did before this feature existed.
B.ask_user_input = lambda prompt, should_abort=None: ""
try:
    assert B.ask_todo_answers(["a", "b"], 1) is None
finally:
    B.ask_user_input = real_input
assert B.ask_todo_answers([], 1) is None
print("ask_todo_answers: silent when nothing is answered or no terminal")

# --- an applied answer is not re-delivered ----------------------------------
work = Path(tempfile.mkdtemp(prefix="answers-"))
section = work / "section.tex"
section.write_text("\\section{One}\n")
qf = CB.questions_file_for(section)
qf.write_text(json.dumps({"question_seq": 2, "questions": [
    {"id": 1, "kind": "clarify", "text": "garbled", "context": "", "guess": "G",
     "answer": "", "deferred": False, "delivered": True}]}))

ctx = NT.NotesToolContext(refs_dir=work / "refs", read_roots=[work])
CB.load_saved_questions(ctx, section)
# answer == "" means "the user accepted the guess" — a real answer, not a
# missing one. Before the fix this was reported as unapplied on every run.
assert not NT.is_open(ctx.questions[0]), "an accepted guess is not open"
unapplied = [q for q in ctx.questions
             if q.get("answer") is not None and not q.get("applied")]
assert len(unapplied) == 1, "precondition: starts out unapplied"

CB.mark_answers_applied(ctx, section)
again = NT.NotesToolContext(refs_dir=work / "refs", read_roots=[work])
CB.load_saved_questions(again, section)
assert again.questions[0].get("applied") is True, again.questions[0]
assert CB.collect_followup_answers(again, section) is None, \
    "an applied answer must not be re-delivered to the model"
assert CB.open_question_count(section) == 0
print("mark_answers_applied: an accepted guess is applied once, not re-sent")

# --- and the write pass is what records it ----------------------------------
src = (ROOT / "build_course.py").read_text()
write_pass = src[src.index("def generate_section"):src.index("# Persistent state")]
assert "mark_answers_applied" in write_pass, \
    "generate_section must mark answers applied — this is the reported bug"
print("generate_section: marks answers applied after the write pass")

print("\nALL OK")
