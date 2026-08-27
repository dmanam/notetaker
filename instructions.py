r"""instructions.py — the parts of the system prompt both note-takers share.

There are two drivers. build_course.py writes one section of a running course;
generate_notes.py writes a single lecture as a standalone document. They differ
in real ways — one has a fixed preamble and neighbouring lectures to cite, the
other writes its own \documentclass — but most of what the prompt says is about
neither: how to treat an ASR transcript, what not to invent, when to draw a
diagram, how to punctuate.

Those parts used to be written out twice, and the copies drifted. The course
prompt grew the fidelity rules, the diagram rules, the display-mode rules, the
macro-bracing trap and the citation tool; the single-lecture prompt kept its
original short list and still told the model to use \ref, which the course
prompt explicitly forbids. Nobody decided that — a rule was added to whichever
file was open at the time. So the shared text lives here and is composed into
both, and a rule added here reaches both by construction.

What stays in the drivers is what is genuinely theirs: body-only output versus a
whole document, lecture-numbered labels, the preamble tool, the environments
that already exist.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# What the transcript is
# ---------------------------------------------------------------------------

ASR_INSTRUCTION = r"""The transcript was produced by automatic speech recognition and may contain errors:
misheared words, mangled technical terms, or nonsensical phrases where the speaker
said something the recogniser could not handle. Treat the transcript as a rough guide,
not a verbatim record. If a passage does not make mathematical sense, it is likely a
transcription error — use the clarify_transcript tool rather than reproducing the
garbled text."""

# ---------------------------------------------------------------------------
# Not writing more than the lecture supports
# ---------------------------------------------------------------------------

FIDELITY_INSTRUCTION = r"""Fidelity. Notes like these fail in characteristic ways, and all of them come
from writing more than the lecture supports:
- Material you add that the lecturer did not say — a justification, an
  "equivalently", a slicker proof, a historical attribution, an illustrative
  example — is where errors concentrate. Add it only where you are certain,
  and never present your own reasoning as the lecturer's. An equivalence is a
  mathematical claim: if the lecturer did not state it, either verify it
  properly or leave it out. If your own gloss genuinely helps, mark it as
  yours ("Editorially: ...") so a reader can weigh it separately.
- Preserve the lecturer's confidence. "I think", "morally speaking", "I
  forgot", "I don't know", "this might be wrong", "I'm not sure how you'd
  define it" are content, not disfluency — keep them. Never convert a hedge
  into an assertion, and never state as settled something the lecturer
  flagged as open, conjectural, or half-remembered.
- A correction supersedes what it corrects. Lecturers correct themselves and
  audiences correct them, sometimes many minutes later. Write what the
  lecture concluded, not what was first said: do not restate a claim that was
  retracted, and do not reuse an example that was refuted as though it still
  supported the point. You have the whole transcript — when a passage sounds
  hesitant or draws a question, read ahead before writing it up.
- Never attribute to the lecturer a reference they did not give. If they only
  gestured at one ("I think there's a paper by X"), what you cite has to be
  something they could have meant, which usually means it was already public
  at the lecture date given in the task — but not always. Lecturers point at
  work that is not out yet, their own and other people's ("this will be in
  our next paper", "Y has a proof of that"), and there a later preprint is
  the right citation precisely because it is the work they described. What
  is never allowed is filling a vague gesture with a paper you found that
  merely fits, whenever it appeared. If the identification is your inference
  rather than their reference, cite it as your own pointer ("see also"), and
  where the lecture flagged the work as forthcoming, say so.
- A \todo does not license a false statement. Flagging a missing reference
  while asserting the claim is backwards: assert only the part you are sure
  of, and put the uncertainty inside the \todo."""

# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------

# Both documents load cleveref, so both get the same rule. \ref is banned
# rather than merely discouraged: it prints a bare number, and a reader who
# meets "by 2.3" has to go and find out what kind of thing 2.3 is.
TIMESTAMP_RULE = r"""- Mark where the material starts in the recording. Begin every paragraph
  that came from the lecture with \ts{hh:mm:ss}, and put the same mark
  immediately after \begin{theorem}, \begin{definition}, \begin{proof} and
  every other environment of that kind. It sets the time in the left margin,
  so a reader who wants to hear a passage can find it in the video.
  Write the mark on a line of its own — a newline before it and a newline
  after it, and never a blank line after it, which would end the paragraph
  and strand the mark on an empty one:

      \ts{00:12:34}
      The paragraph begins here, on the next line.

      \begin{definition}\label{def:3:perfectoid}
      \ts{00:14:02}
      A definition, marked inside the environment.
      \end{definition}

  The time is the [hh:mm:ss] on the transcript line where the lecturer STARTS
  that material — not where they finish it, not where you happened to confirm
  it on a board, and never a time you estimated. Copy the transcript's own
  mark. Where a paragraph gathers a point made over several minutes, mark
  where the point begins. Where you write something up out of the order it
  was said, the mark follows the lecture and not your paragraph order, so the
  times in the margin may go backwards — that is correct, and better than a
  tidy sequence that is wrong.
  A paragraph with no spoken origin gets NO mark. If you wrote it yourself —
  a sentence joining two topics, a summary of your own, background the
  lecturer neither said nor put on the board — leave it unmarked. The margin
  is then a record of where the notes come from, and an unmarked paragraph is
  yours rather than the lecture's. That is a statement about provenance, not
  a way out of finding a time: material that did come from the lecture has
  one, and marking it is not optional.
  One mark per paragraph or environment: not per sentence, not on a \section
  heading, not inside a display, and not on the continuation of a paragraph
  you have already marked. A list belongs to the paragraph that introduces
  it — mark that paragraph, not each item.
  \ts is already defined and needs no package. Never define it, redefine it,
  or add anything to the preamble for it."""


CROSSREF_RULE = r"""- Use \cref{label} for ALL cross-references (mid-sentence) and \Cref{label}
  at the start of a sentence. cleveref automatically produces the correct
  type name and number, e.g. "Theorem 2.3", "Definition 1.4", "Lecture 2".
  Never use \ref or \hyperref for cross-references."""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

FRAMES_RULE = r"""- Whenever the transcript mentions something drawn, written, or shown
  visually, consult the video frames (using the frame tools or subagent
  available to you) so you can transcribe the mathematics accurately."""

CLARIFY_RULE = r"""- Use the clarify_transcript tool when a word or phrase in the transcript seems
  garbled, misheared, or mathematically nonsensical — provide the exact garbled
  text, the surrounding context, and your best guess. Do not reproduce garbled
  text in the notes."""

_CITE_HEAD = r"""- Cite sources with the cite_reference tool: give it an arXiv ID, DOI, or
  URL and it returns a key for \cite{key}, adding the entry to """
_CITE_SHARED = "the course's\n  shared bibliography"
_CITE_OWN = "this lecture's\n  bibliography"
_CITE_TAIL = r""" (safe to call again for the same source). For arXiv
  IDs and DOIs the metadata is fetched for you; for anything else (lecture
  notes, a book, a web page) also pass title, author, and year — look them
  up in the document itself if you must, since an entry without an author
  cannot get a proper [Sch19]-style citation label. Cite papers and books
  the lecturer names, and references you consulted for a definition or
  notation. Before finishing, audit the draft for every identifiable paper,
  book, theorem attribution, and external source you used; call
  cite_reference for each one and put the returned \cite{key} in the notes.
  A prose-only author/title mention or a hand-written \footnote is not a
  substitute for \cite. If a source is too vague to identify reliably,
  preserve the lecturer's attribution without inventing bibliographic
  metadata. Never write bibliography entries, \bibitem, or
  \printbibliography yourself — the bibliography is assembled automatically."""


def cite_rule(shared: bool) -> str:
    """Collect sources with the tool rather than by hand.

    shared distinguishes a course's running bibliography, which every lecture
    adds to and reads from, from one lecture's own. The last sentence holds
    either way: the model never writes the biblatex machinery, because
    build_course assembles it and generate_notes attaches it afterwards
    (bibliography.attach_to_document).
    """
    return (_CITE_HEAD + (_CITE_SHARED if shared else _CITE_OWN) + _CITE_TAIL)


ASK_USER_RULE = r"""- Use the ask_user tool whenever you are uncertain how to typeset a specific
  symbol or notation — for example, a symbol that requires a niche package,
  non-standard blackboard bold, or field-specific convention you are not
  confident about. Ask instead of silently guessing — then continue
  provisionally with your best rendering (marked with \todo) until the
  answer arrives."""

# Ends mid-sentence on purpose: each driver appends its own note about where
# todonotes comes from (already loaded, or to be added to the preamble).
TODO_RULE = r"""- Use \todo{...} inline to flag any location where you are uncertain about
  mathematical content rather than typesetting: for example, a formula you
  could only partially read from a frame, a logical step that seems incomplete,
  or a passage where your best-effort reconstruction may be wrong. Prefer
  \todo{} over silently guessing; it lets the human reviewer find and fix
  uncertain spots in the compiled PDF."""

# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

# Indented as a continuation, so it can sit under the preamble bullet where a
# preamble tool exists and under its own lead-in where the model writes the
# preamble itself.
MACRO_BRACING_RULE = r"""  When a macro's expansion ends in a superscript or subscript, wrap the whole
  body in braces: \newcommand{\Gm}{{\mathbb{G}_{m}}}, not
  \newcommand{\Gm}{\mathbb{G}_{m}}; \newcommand{\ur}[1]{{#1^{\triangleright}}},
  not \newcommand{\ur}[1]{#1^{\triangleright}}. Unbraced, the first call site
  that attaches its own script — \Gm_{A}, \ur A' —
  is a "Double subscript"/"Double superscript" error, and the error surfaces
  wherever it happens to be used rather than at the definition."""

# ---------------------------------------------------------------------------
# What goes on the page
# ---------------------------------------------------------------------------

_DIAGRAM_HEAD = r"""- Draw what was drawn. A diagram the lecturer put on the board is part of the
  mathematics, not decoration, and prose is a poor substitute for it: render
  commutative diagrams with tikz-cd (\begin{tikzcd}), and anything else
  informative that was drawn — a picture of a space, a filtration, a covering,
  a sketch that carries an idea — with tikz."""

# Where the packages come from, and whether there is anything to crop.
_DIAGRAM_TOOLS = r""" Both are already loaded; do not
  add them via add_to_preamble. Crop the board to the diagram before you read
  it, and compile-check what you write; the instructions below say how.
  """
_DIAGRAM_NO_TOOLS = r""" Load both in the preamble
  (\usepackage{tikz} and \usepackage{tikz-cd}).
  """

_DIAGRAM_TAIL = r"""Reproduce the lecturer's diagram, not an idealised one, and do not invent
  arrows, objects or labels you cannot see. A purely decorative drawing (an
  underline, a box round a word) is not worth drawing.
- Draw diagrams the lecturer did not draw, wherever one would make the
  mathematics clearer. Whether something was drawn on the slate is an
  accident of the lecture; whether it reads better as a diagram is a question
  about the notes. A square that commutes, a span or cospan, a lifting
  problem, a factorisation, a short exact sequence, a tower of maps — all of
  these are clearer as a diagram than as a sentence with arrows in it, even
  when the lecturer said them aloud and wrote nothing. The mathematics must
  still be exactly what the lecture asserts: composing a diagram is a
  decision about presentation, never a licence to add a map the lecture does
  not claim."""


def diagram_rules(board_tools: bool) -> str:
    """Draw what was drawn, and draw what wasn't.

    board_tools says whether this driver has crop_board/check_diagram and a
    fixed preamble with tikz in it. Only that one clause differs; the
    judgement — reproduce what is there, invent nothing — is the same either
    way, which is why it is not worth two copies.
    """
    return (_DIAGRAM_HEAD
            + (_DIAGRAM_TOOLS if board_tools else _DIAGRAM_NO_TOOLS)
            + _DIAGRAM_TAIL)


DISPLAY_RULES = r"""- Use display mode freely, for emphasis and for structure. A definition worth
  stating, an equation the argument turns on, a condition being checked —
  put it on its own line. Reserve inline mathematics for things that read as
  part of a sentence.
- The point of both is that unbroken prose is hard to read, and these are
  notes people will read at the pace of the mathematics rather than the pace
  of English. Displayed formulas and diagrams give the eye somewhere to
  land, mark what matters, and let a reader find a result again later. That
  is how mathematical lecture notes are conventionally written, and it is
  what your reader expects; a page that is a wall of text is harder to use
  regardless of how good the sentences are."""

DISFLUENCY_RULE = (
    "- Clean up speech disfluencies but preserve the mathematical content "
    "faithfully.")

# ---------------------------------------------------------------------------
# House typography — appended after the rule list, alongside register and
# attribution, because it is about the page rather than the mathematics.
# ---------------------------------------------------------------------------

HOUSE_STYLE_INSTRUCTION = """
Dashes: punctuate a parenthetical break with an en dash, "--", never an em
dash, "---". Both are correct English; these notes set the en dash, and what
looks wrong is mixing the two on one page. The same "--" is the dash in a
range ("pages 10--12", "Lemmas 2--4") and in a double-barrelled name
(Eilenberg--Mac Lane, Cauchy--Schwarz); a hyphen, "-", stays for compound
words."""
