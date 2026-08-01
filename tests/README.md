# tests/

Run them directly; there is no runner and no framework:

```sh
for t in tests/test_*.py; do python "$t" || echo "FAILED $t"; done
```

Each file is a script of assertions with a printed line per property checked.
They need ffmpeg, pdflatex and numpy on `PATH` (`nix develop` supplies all
three if you use the flake), and
they build their own fixtures in a temp directory — none of them touch
`output/`.

These live in the repo on purpose. They used to sit in a scratch directory,
which was cleared between sessions and took a dozen suites with it; anything
worth writing twice belongs under version control the first time.

No test calls a model. Where a feature depends on one — reading a board,
extracting a style passage, guessing who lectured — what is tested is the
parsing, the gating and the fallback around it, which is where the bugs have
actually been. The model's own output is checked by running the thing on real
input and looking (`python lecturer.py output`, `python style_extract.py FILE`).

Most assertions here encode a specific failure that actually happened rather
than a property someone thought of in advance — the comments say which. That
is the useful part when one of them breaks: the comment tells you what the
code got wrong last time, which is usually what it is about to get wrong
again.
