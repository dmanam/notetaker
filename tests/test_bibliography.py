"""Entries go into references.bib with their redundant links taken out.

biblatex already prints an eprint and a DOI as links. arxiv.org/bibtex hands
back an entry whose url is the abs page for its own eprint, and Crossref one
whose url is dx.doi.org/<its own DOI>, so an untouched entry renders the same
address twice — once as arXiv:2102.13459 or as the DOI, and once again in
full underneath it. What is checked here is that the duplicate goes and a url
that points somewhere else stays, because the second kind is the only link
some entries have.

The removal is textual, so the other half of the job is that what is left is
still a parseable entry in both layouts we are handed: arXiv writes one field
per line, Crossref writes the whole entry on one.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import bibliography as B

ARXIV = """@misc{bhatt2014proetale,
      title={The pro-etale topology for schemes},
      author={Bhatt, Bhargav and Scholze, Peter},
      year={2014},
      eprint={1309.1198},
      archivePrefix={arXiv},
      primaryClass={math.AG},
      url={https://arxiv.org/abs/1309.1198}, 
}
"""
CROSSREF = ("@article{Dold_1958, title={Quasifaserungen}, volume={67}, "
            "ISSN={0003-486X}, url={http://dx.doi.org/10.2307/1970005}, "
            "DOI={10.2307/1970005}, number={2}, journal={The Annals of "
            "Mathematics}, author={Dold, Albrecht and Thom, Rene}, "
            "year={1958}, month=Mar, pages={239} }\n")

# --- the two redundant shapes ------------------------------------------------
a = B.strip_redundant_url(ARXIV)
assert "url=" not in a and "url =" not in a, a
for kept in ("eprint={1309.1198}", "archivePrefix={arXiv}",
             "primaryClass={math.AG}", "title={The pro-etale topology"):
    assert kept in a, kept
assert B.entry_key(a) == "bhatt2014proetale"
# The field had a line to itself and the line goes with it: no blank line, no
# trailing spaces left behind where it stood.
assert "\n\n" not in a and " \n" not in a, repr(a)

c = B.strip_redundant_url(CROSSREF)
assert "url=" not in c and "ISSN={0003-486X}, DOI={10.2307/1970005}" in c, c
assert B.entry_key(c) == "Dold_1958" and c.rstrip().endswith("}")
print("redundant url removed from both the arXiv and the Crossref layout")

# Crossref lowercases the DOI it puts in the DOI field and leaves the url it
# built from that DOI in the original case. DOIs are case-insensitive, so
# comparing them literally would keep every url of this shape.
mixed = CROSSREF.replace("url={http://dx.doi.org/10.2307/1970005}",
                         "url={http://dx.doi.org/10.1007/BF02684313}") \
                .replace("DOI={10.2307/1970005}", "DOI={10.1007/bf02684313}")
assert "url=" not in B.strip_redundant_url(mixed), mixed
# Either resolver host, either scheme.
alt = CROSSREF.replace("url={http://dx.doi.org/10.2307/1970005}",
                       "url={https://doi.org/10.2307/1970005}")
assert "url=" not in B.strip_redundant_url(alt)
for form in ("https://arxiv.org/abs/1309.1198v2",   # a version of its eprint
             "http://arxiv.org/pdf/1309.1198",      # the pdf, same paper
             "https://doi.org/10.48550/arXiv.1309.1198"):  # arXiv's own DOI
    swapped = ARXIV.replace("https://arxiv.org/abs/1309.1198", form)
    assert "url=" not in B.strip_redundant_url(swapped), form
print("every spelling of the same link is recognised (version, pdf, resolver)")

# --- links that are not redundant stay --------------------------------------
# A published version, an author's copy, a project page: the entry would lose
# its only pointer to where the thing actually is.
for url in ("https://www.cambridge.org/core/product/1234",
            "https://people.mpim-bonn.mpg.de/scholze/Condensed.pdf",
            "https://arxiv.org/abs/2102.13459"):        # a different paper
    for entry in (ARXIV, CROSSREF):
        swapped = entry.replace("https://arxiv.org/abs/1309.1198", url) \
                       .replace("http://dx.doi.org/10.2307/1970005", url)
        assert "url" in B.strip_redundant_url(swapped), (url, entry[:20])
# An @online entry has no eprint and no DOI — the url is all it has.
online = B._online_entry("https://example.org/notes.pdf", "Some notes")
assert B.strip_redundant_url(online) == online
print("a url pointing anywhere else is left alone")

# An eprint with no archivePrefix is not necessarily an arXiv eprint, and a
# url that is not a bare resolver link is not the DOI's link.
noprefix = ARXIV.replace("      archivePrefix={arXiv},\n", "")
assert "url=" in B.strip_redundant_url(noprefix)
deep = CROSSREF.replace("http://dx.doi.org/10.2307/1970005",
                        "http://dx.doi.org/10.2307/1970005/full-text.pdf")
assert "url=" in B.strip_redundant_url(deep)
print("no eprint type and no bare resolver link: nothing is removed")

# --- a doi field holding the resolver URL ------------------------------------
# arXiv writes doi={https://doi.org/10.1017/fmp.2021.4}, and biblatex prefixes
# the doi field with the resolver itself: the address prints twice over,
# behind a link that resolves to nothing.
url_doi = ARXIV.replace("      eprint={1309.1198},\n",
                        "      eprint={1309.1198},\n"
                        "      doi={https://doi.org/10.1017/fmp.2021.4},\n")
fixed = B.normalize_doi(url_doi)
assert "doi={10.1017/fmp.2021.4}" in fixed, fixed
assert "doi.org" not in fixed.replace("https://arxiv.org/abs/1309.1198", "")
# Crossref writes the bare DOI already, and rewriting it would be churn.
assert B.normalize_doi(CROSSREF) == CROSSREF
# The field name is spelled both ways and neither is wrong.
assert "DOI = {10.2307/1970005}" in B.normalize_doi(
    CROSSREF.replace("DOI={10.2307/1970005}",
                     "DOI = {https://dx.doi.org/10.2307/1970005}"))
# An entry can need both fixes at once, which is what tidy_entry is for.
both = B.tidy_entry(url_doi)
assert "doi={10.1017/fmp.2021.4}" in both and "url=" not in both, both
print("a doi field holding a resolver URL is reduced to the DOI")

# --- urldate goes with the url ----------------------------------------------
dated = ARXIV.replace("      url={https://arxiv.org/abs/1309.1198}, \n",
                      "      url={https://arxiv.org/abs/1309.1198},\n"
                      "      urldate = {2026-01-01},\n")
assert "urldate" not in B.strip_redundant_url(dated), \
    "a date saying when a url was seen means nothing without the url"
print("urldate is dropped with the url it dated")

# --- bibliography the model wrote itself ------------------------------------
# The failure this catches: the notes come back with the references written
# out by hand — a \bibitem list, an entry pasted in, an arXiv number sitting
# in a sentence — instead of registered with cite_reference. It compiles, it
# looks like a bibliography, and none of it is in references.bib, so no later
# lecture can cite the same paper and no \cite key resolves to it.
for text, want in [
        ("\\begin{thebibliography}{9}\n\\bibitem{BS} Bhatt, Scholze.", 2),
        ("@article{foo2019, title={Bar}, author={Baz}}", 1),
        ("\\bibliography{refs}", 1),
        ("\\section*{References}", 1),
        ("\\subsection{Bibliography}", 1),
        ("As shown in Bhatt--Scholze, arXiv:1309.1198, the site is nice.", 1),
        ("\\footnote{P. Scholze, \\emph{Condensed}, https://doi.org/10.1007/x}", 1),
        # What the notes are supposed to look like instead.
        ("See \\cite{bhatt2014} for the details.", 0),
        ("\\printbibliography[heading=bibintoc]", 0),
        # A comment is not the document, and the word alone is not a citation.
        ("% see https://arxiv.org/abs/1309.1198 for the source", 0),
        ("The lecturer posted it on the arXiv last year.", 0)]:
    got = B.inline_entries(text)
    assert len(got) == want, (text[:40], got)
# The preamble legitimately carries an arxiv.org URL — it is where the
# formatting of arXiv eprints is defined — so the scan starts at the body.
whole = ("\\documentclass{article}\n" + (B.BIB_PREAMBLE % "references.bib")
         + "\n\\begin{document}\n\\cite{x}\n\\printbibliography\n"
         "\\end{document}\n")
assert B.inline_entries(whole) == [], B.inline_entries(whole)
# Line numbers are counted in the whole file, not in the body the scan
# actually walks — they are what the checker is given to go and look at.
marked = whole.replace("\\cite{x}", "\\bibitem{y} A paper.")
want_line = next(i for i, ln in enumerate(marked.splitlines(), 1)
                 if "bibitem" in ln)
found = B.inline_entries(marked)
assert len(found) == 1 and found[0].startswith(f"line {want_line}:"), \
    (want_line, found)
print("inline_entries: finds hand-written references, spares comments and "
      "the preamble")

root = Path(tempfile.mkdtemp())

# --- over a file, and over a file twice -------------------------------------
bib = root / "references.bib"
bib.write_text(f"% source: arxiv:1309.1198\n{url_doi}\n"
               f"% source: doi:10.2307/1970005\n{CROSSREF}\n"
               f"% source: https://example.org/notes.pdf\n{online}\n")
assert B.tidy_bibliography(bib) == 2
text = bib.read_text()
assert B.list_keys(bib) == ["bhatt2014proetale", "Dold_1958",
                            B.entry_key(online)]
# The % source: markers are how a later run finds the key it already assigned;
# rewriting entries around them must not disturb them.
for marker in ("% source: arxiv:1309.1198", "% source: doi:10.2307/1970005",
               "% source: https://example.org/notes.pdf"):
    assert marker in text, marker
assert B.tidy_bibliography(bib) == 0 and bib.read_text() == text
assert B.tidy_bibliography(root / "nothing.bib") == 0
print("prune over a file: markers kept, keys kept, second pass a no-op")

# --- what is left still parses, and still compiles --------------------------
entries = {e["key"]: e for e in B.list_entries(bib)}
assert entries["Dold_1958"]["year"] == "1958"
assert entries["bhatt2014proetale"]["author"] == "Bhatt, Bhargav and Scholze, Peter"
# Braces inside a field must not be mistaken for the end of it, in the reader
# and in the remover alike.
braced = ("@book{n1968, author = {N{\\\"o}beling, Georg}, "
          "title = {Topologie der Vereine}, DOI={10.1007/BF01361153}, "
          "url = {https://doi.org/10.1007/BF01361153}}\n")
pruned = B.strip_redundant_url(braced)
assert "url" not in pruned and "N{\\\"o}beling" in pruned, pruned
assert B._field(pruned, "author") == "N{\\\"o}beling, Georg"
print("nested braces survive both the reader and the remover")

if shutil.which("latexmk"):
    # The removal is textual, so the question a compile answers and nothing
    # else does is whether biber still reads the entry — and whether the link
    # that justified removing the url is really printed in its place.
    import subprocess
    doc = root / "doc.tex"
    doc.write_text(
        "\\documentclass{article}\n\\usepackage{hyperref}\n"
        + (B.BIB_PREAMBLE % "references.bib")
        + "\n\\begin{document}\n\\cite{bhatt2014proetale}\\cite{Dold_1958}\n"
        + B.BIB_PRINT + "\n\\end{document}\n")
    out = root / "build"
    proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                           "-outdir=" + str(out), doc.name],
                          cwd=root, capture_output=True, text=True,
                          errors="replace", timeout=900)
    log = (out / "doc.log").read_text(errors="replace")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "Citation" not in log or "undefined" not in log, \
        "biber could not read an entry the pruning rewrote"
    bbl = (out / "doc.bbl").read_text(errors="replace")
    assert "1309.1198" in bbl and "10.2307/1970005" in bbl, \
        "the eprint and the DOI are what the url was redundant against"
    assert "\\field{url}" not in bbl and "verb{url}" not in bbl, \
        "a url that reached the bibliography was not removed after all"
    print("biber reads the pruned entries; eprint and DOI still print")
else:
    print("(no latexmk on PATH — skipping the compile check)")

shutil.rmtree(root, ignore_errors=True)
print("\nALL OK")
