"""
bibliography.py — A running BibTeX file for the course.

Entries are appended to <output-dir>/references.bib as the models cite
things, each preceded by a `% source: …` marker so the same paper is never
added twice (and so a later run can look up the key it already assigned).

Sources:
  arXiv IDs/URLs — arxiv.org/bibtex/<id> (proper entry with a real key)
  DOIs           — doi.org content negotiation (application/x-bibtex)
  anything else  — a generated @online entry

An entry's url is dropped when it only repeats a link the entry already
carries — the abs page for its own eprint, the resolver link for its own DOI
— because biblatex prints those from the eprint and doi fields anyway and the
address would otherwise appear twice. A doi field holding the resolver URL
rather than the DOI is reduced to the DOI, for the same reason: biblatex puts
the resolver back.
"""

from __future__ import annotations

import re
import urllib.request
from datetime import date
from pathlib import Path

from fetch import ARXIV_ID_RE, arxiv_id_of

DOI_RE = r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"

BIB_FILENAME = "references.bib"
BIB_PREAMBLE = r"""
%% biblatex loads after hyperref; the running bibliography
\RequirePackage[
  style=alphabetic,
  maxbibnames=99,
  minbibnames=99,
  maxalphanames=99,
  minalphanames=99,
  useprefix=true,
]{biblatex}
%% improve arXiv citation formatting
\DeclareFieldFormat{eprint:arXiv}{%%
  \href{https://arxiv.org/abs/#1}{{\tt arXiv:\allowbreak#1\iffieldundef{eprintclass}{}{\discretionary{}{}{\,}[\thefield{eprintclass}]}}}%%
}
\addbibresource{%s}
"""

BIB_PRINT = "\\printbibliography[heading=bibintoc]"


def _get(url: str, accept: str | None = None) -> str:
    headers = {"User-Agent": "notetaker/1.0 (academic note-taking tool)"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def doi_of(url_or_id: str) -> str | None:
    m = re.search(DOI_RE, url_or_id.strip())
    return m.group(0).rstrip(".") if m else None


def source_key(url_or_id: str) -> str:
    """Canonical identity of a citable source (matches fetch.py's caching)."""
    aid = arxiv_id_of(url_or_id)
    if aid:
        return f"arxiv:{aid}"
    doi = doi_of(url_or_id)
    if doi:
        return f"doi:{doi}"
    return url_or_id.strip()


def entry_key(entry: str) -> str | None:
    m = re.search(r"@\w+\s*\{\s*([^,\s]+)\s*,", entry)
    return m.group(1) if m else None


def _slug_key(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    words = [w for w in re.split(r"[\s_-]+", s) if w][:4]
    return "-".join(words)[:48] or "reference"


STOPWORDS = {"a", "an", "the", "on", "of", "in", "for", "and", "to", "lectures",
             "notes", "introduction"}


def _title_words(title: str) -> list[str]:
    return [w for w in re.split(r"[\s_-]+", re.sub(r"[^\w\s-]", "", title))
            if w and w.lower() not in STOPWORDS]


def _last_name(author: str) -> str:
    """Surname of the first author ('Ostrand, Marek' / 'Marek Ostrand' /
    'A and B' all give the first author's surname)."""
    first = re.split(r"\s+and\s+|;", author.strip())[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1] if parts else first


def _fallback_label(title: str) -> str:
    """Last-resort alphabetic tag, used ONLY when the author is unknown —
    otherwise biblatex builds the label from author+year itself."""
    tag = "".join(w[0].upper() for w in _title_words(title)[:3])
    return tag or "Web"


def _online_entry(url: str, title: str | None, author: str | None = None,
                  year: str | int | None = None) -> str:
    title = title or url
    if author:
        words = _title_words(title)
        key = (f"{_last_name(author).lower()}{year or ''}"
               f"{words[0].lower() if words else ''}")
        key = re.sub(r"[^\w]", "", key) or _slug_key(title)
    else:
        key = _slug_key(title)

    fields = [f"      title = {{{title}}},"]
    if author:
        fields.append(f"      author = {{{author}}},")
    else:
        # No author to build an alphabetic label from.
        fields.append(f"      label = {{{_fallback_label(title)}}},")
    if year:
        fields.append(f"      year = {{{year}}},")
    fields.append(f"      url = {{{url}}},")
    fields.append(f"      urldate = {{{date.today().isoformat()}}},")
    return "@online{" + key + ",\n" + "\n".join(fields) + "\n}\n"


def fetch_bibtex(url_or_id: str, title: str | None = None,
                 author: str | None = None,
                 year: str | int | None = None) -> str:
    """Return a BibTeX entry for a source (never raises — falls back to a
    generated @online entry built from the supplied author/title/year)."""
    aid = arxiv_id_of(url_or_id)
    if aid:
        try:
            entry = _get(f"https://arxiv.org/bibtex/{aid}").strip()
            if entry.startswith("@"):
                return entry + "\n"
        except Exception:
            pass
        return _online_entry(f"https://arxiv.org/abs/{aid}",
                             title or f"arXiv:{aid}", author, year)

    doi = doi_of(url_or_id)
    if doi:
        try:
            entry = _get(f"https://doi.org/{doi}",
                         accept="application/x-bibtex").strip()
            if entry.startswith("@"):
                return entry + "\n"
        except Exception:
            pass
        return _online_entry(f"https://doi.org/{doi}", title or doi,
                             author, year)

    url = url_or_id.strip()
    if not url.startswith(("http://", "https://")):
        url = ""
    return _online_entry(url or url_or_id.strip(), title, author, year)


REPLACEMENT = "�"


def sanitize_entry(entry: str) -> tuple[str, bool]:
    """Make a fetched entry safe to compile. Returns (entry, was_mojibake).

    Publisher metadata is sometimes already corrupt at the source — Crossref
    serves Nöbeling as 'N\\ufffdbeling', replacement character and all — and a
    U+FFFD in the .bib is a hard LaTeX error ("Unicode character ... not set
    up for use with LaTeX") that takes the whole document down. We cannot
    recover the intended letter, so substitute a character that compiles and
    tell the user, rather than silently shipping a build-breaking file."""
    if REPLACEMENT not in entry:
        return entry, False
    return entry.replace(REPLACEMENT, "?"), True


def _ascii_key(key: str) -> str:
    """Cite keys should be plain ASCII: they are typed by hand into \\cite{}
    and passed between biber and LaTeX."""
    cleaned = re.sub(r"[^\w:-]", "", key, flags=re.ASCII)
    return cleaned or "reference"


def existing_key(bib_file: Path, url_or_id: str) -> str | None:
    """The cite key already assigned to this source, if any."""
    if not bib_file.exists():
        return None
    marker = f"% source: {source_key(url_or_id)}\n"
    text = bib_file.read_text()
    idx = text.find(marker)
    if idx == -1:
        return None
    return entry_key(text[idx + len(marker):])


def _unique_key(text: str, key: str) -> str:
    if not re.search(rf"@\w+\s*\{{\s*{re.escape(key)}\s*,", text):
        return key
    n = 2
    while re.search(rf"@\w+\s*\{{\s*{re.escape(key)}{n}\s*,", text):
        n += 1
    return f"{key}{n}"


def cite(bib_file: Path, url_or_id: str, title: str | None = None,
         author: str | None = None,
         year: str | int | None = None) -> tuple[str, bool]:
    """Ensure the source is in the bibliography. Returns (cite_key, added).

    author/year are used only when the source has no fetchable metadata
    (i.e. it is not an arXiv paper or a DOI) — supply them so biblatex can
    build a proper alphabetic label instead of falling back to a title tag.
    """
    key = existing_key(bib_file, url_or_id)
    if key:
        return key, False

    entry = fetch_bibtex(url_or_id, title, author, year)
    entry, mojibake = sanitize_entry(entry)
    entry = tidy_entry(entry)
    key = entry_key(entry) or _slug_key(title or url_or_id)
    ascii_key = _ascii_key(key)
    if ascii_key != key:
        entry = entry.replace(key, ascii_key, 1)
        key = ascii_key
    bib_file.parent.mkdir(parents=True, exist_ok=True)
    text = bib_file.read_text() if bib_file.exists() else ""
    unique = _unique_key(text, key)
    if unique != key:
        entry = entry.replace(key, unique, 1)
        key = unique

    note = ""
    if mojibake:
        note = (f"% FIXME: the publisher's own metadata for this entry is "
                f"corrupt (it contained U+FFFD); '?' stands in for character(s)"
                f" that could not be recovered. Correct by hand.\n")
        print(f"  Warning: upstream metadata for {url_or_id} is corrupt — "
              f"unrecoverable character(s) replaced with '?' in "
              f"\\cite{{{key}}}; fix the entry by hand.")
    with open(bib_file, "a") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(f"\n% source: {source_key(url_or_id)}\n{note}{entry}")
    return key, True


# ---------------------------------------------------------------------------
# Reading and rewriting fields
# ---------------------------------------------------------------------------

def _field_span(entry: str, name: str) -> tuple[int, int, str] | None:
    """(start, end, value) of one BibTeX field, tolerating nested braces
    (author={N{\\"o}beling}) and unbraced values (year=1968). start is at the
    field name, end just past the closing brace/quote/token."""
    m = re.search(rf"\b{name}\s*=\s*", entry, re.I)
    if not m:
        return None
    i = m.end()
    if i >= len(entry):
        return None
    if entry[i] == "{":
        depth = 0
        for j in range(i, len(entry)):
            if entry[j] == "{":
                depth += 1
            elif entry[j] == "}":
                depth -= 1
                if depth == 0:
                    return m.start(), j + 1, entry[i + 1:j]
        return m.start(), len(entry), entry[i + 1:]
    if entry[i] == '"':
        j = entry.find('"', i + 1)
        if j == -1:
            return m.start(), len(entry), entry[i + 1:]
        return m.start(), j + 1, entry[i + 1:j]
    m2 = re.match(r"[^,}\s]+", entry[i:])
    return (m.start(), i + m2.end(), m2.group(0)) if m2 else (m.start(), i, "")


def _field(entry: str, name: str) -> str:
    span = _field_span(entry, name)
    return span[2] if span else ""


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


def _drop_field(entry: str, name: str) -> str:
    """Remove a field and its separator, in both the layouts we are handed:
    arXiv's one-field-per-line and Crossref's everything-on-one-line."""
    span = _field_span(entry, name)
    if not span:
        return entry
    start, end, _ = span
    end += re.match(r"[ \t]*,?[ \t]*", entry[end:]).end()
    head = entry[:start]
    if re.search(r"\n[ \t]*$", head):
        # The field had a line to itself; take the line with it.
        head = re.sub(r"[ \t]*$", "", head)
        if entry[end:end + 1] == "\n":
            end += 1
    return head + entry[end:]


# ---------------------------------------------------------------------------
# Redundant links
# ---------------------------------------------------------------------------

def _norm_url(url: str) -> str:
    u = _clean(url).replace("\\", "")
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.rstrip("/")


def _doi_url_target(url: str) -> str | None:
    """The DOI a URL resolves, if the URL is nothing but a DOI resolver link."""
    m = re.fullmatch(rf"(?:dx\.)?doi\.org/({DOI_RE})", _norm_url(url), re.I)
    return m.group(1) if m else None


def _arxiv_url_id(url: str) -> str | None:
    """The eprint a URL points at, if the URL is nothing but an arXiv link."""
    u = _norm_url(url)
    m = re.fullmatch(rf"arxiv\.org/(?:abs|pdf|html|e-print)/({ARXIV_ID_RE})"
                     rf"(?:\.pdf)?", u, re.I)
    if m:
        return m.group(1)
    doi = _doi_url_target(u) or ""
    m = re.fullmatch(rf"10\.48550/arxiv\.({ARXIV_ID_RE})", doi, re.I)
    return m.group(1) if m else None


def _bare_eprint(eprint: str) -> str:
    return re.sub(r"v\d+$", "", eprint.strip()).lower()


def _url_is_redundant(entry: str, url: str) -> bool:
    eprint = _clean(_field(entry, "eprint"))
    archive = (_clean(_field(entry, "archivePrefix"))
               or _clean(_field(entry, "eprinttype")))
    if eprint and archive.lower() == "arxiv":
        aid = _arxiv_url_id(url)
        if aid and _bare_eprint(aid) == _bare_eprint(eprint):
            return True
    doi = doi_of(_clean(_field(entry, "doi")))
    target = _doi_url_target(url)
    # DOIs are case-insensitive, and Crossref lowercases the one it puts in
    # the DOI field while leaving the url it built from it alone.
    return bool(doi and target and target.lower() == doi.lower())


def strip_redundant_url(entry: str) -> str:
    """Drop a url field that only repeats a link the entry already carries.

    biblatex renders the eprint and the DOI as links in their own right, so
    an entry whose url is the abs page for its own eprint prints the same
    address twice — once as arXiv:2102.13459 and once in full underneath.
    That is exactly what arxiv.org/bibtex hands back, and Crossref does the
    same with dx.doi.org. A url pointing anywhere else (a published version,
    an author's page) is real information and stays.
    """
    span = _field_span(entry, "url")
    if not span or not _url_is_redundant(entry, span[2]):
        return entry
    # urldate says when the url was seen; without a url it says nothing.
    return _drop_field(_drop_field(entry, "url"), "urldate")


def normalize_doi(entry: str) -> str:
    """Reduce a doi field to the bare DOI.

    arXiv's BibTeX puts the resolver URL in the field
    (doi={https://doi.org/10.1017/fmp.2021.4}), and biblatex prefixes the doi
    field with the resolver itself — so the entry prints the address twice
    over, behind a link (https://doi.org/https://doi.org/...) that resolves to
    nothing.
    """
    span = _field_span(entry, "doi")
    if not span:
        return entry
    start, end, value = span
    doi = doi_of(_clean(value))
    if not doi or _clean(value) == doi:
        return entry
    # The field name is written both ways (doi= from arXiv, DOI= from
    # Crossref) and biber does not care, so keep whichever it already is.
    head = re.match(r"\bdoi\s*=\s*", entry[start:], re.I).group(0)
    return entry[:start] + head + "{" + doi + "}" + entry[end:]


def tidy_entry(entry: str) -> str:
    """Every fix applied to an entry on its way into the bibliography."""
    return normalize_doi(strip_redundant_url(entry))


def _entry_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each @entry in a .bib file, by brace matching."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"@\w+\s*\{", text):
        if spans and m.start() < spans[-1][1]:
            continue
        depth = 0
        for j in range(m.end() - 1, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((m.start(), j + 1))
                    break
    return spans


def tidy_bibliography(bib_file: Path) -> int:
    """tidy_entry over a .bib already on disk; returns the number of
    entries changed. New entries are cleaned as they are added — this is for
    the ones an earlier run wrote. Idempotent."""
    if not bib_file.exists():
        return 0
    text = bib_file.read_text()
    out, changed = text, 0
    for start, end in reversed(_entry_spans(text)):
        entry = text[start:end]
        pruned = tidy_entry(entry)
        if pruned != entry:
            out = out[:start] + pruned + out[end:]
            changed += 1
    if changed:
        bib_file.write_text(out)
    return changed


def list_entries(bib_file: Path) -> list[dict]:
    """Every entry as {key, author, title, year} — so an agent can be shown
    what is already cited instead of re-deriving identifiers it already has."""
    if not bib_file.exists():
        return []
    text = bib_file.read_text()
    out = []
    for chunk in re.split(r"\n(?=@)", text):
        chunk = chunk.strip()
        if not chunk.startswith("@"):
            continue
        key = entry_key(chunk)
        if not key:
            continue
        out.append({
            "key": key,
            "author": _clean(_field(chunk, "author")),
            "title": _clean(_field(chunk, "title")),
            "year": _clean(_field(chunk, "year")),
        })
    return out


def has_entries(bib_file: Path) -> bool:
    return bib_file.exists() and "@" in bib_file.read_text()


def list_keys(bib_file: Path) -> list[str]:
    if not bib_file.exists():
        return []
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_file.read_text())


#: Bibliography written by hand into the notes instead of registered with
#: cite_reference. Each pattern is something that only ever appears when the
#: model has decided to be its own bibliography: a \bibitem list, an entry
#: pasted straight out of a search result, a References section the assembler
#: is already printing, or an arXiv number left in the prose where a \cite
#: key belongs.
_INLINE = [
    (re.compile(r"\\begin\{thebibliography\}|\\bibitem\b"),
     "a hand-written bibliography"),
    (re.compile(r"^[ \t]*@\w+\s*\{", re.M),
     "a BibTeX entry pasted into the notes"),
    (re.compile(r"\\bibliography\s*\{"),
     "a \\bibliography command (this course uses biblatex)"),
    (re.compile(r"\\(?:sub)*section\*?\s*\{\s*(?:references|bibliography|"
                r"works cited)\s*\}", re.I),
     "a references section written by hand"),
    (re.compile(r"arxiv:\s*\d{4}\.\d{4,5}|arxiv\.org/(?:abs|pdf)/"
                r"|doi\.org/10\.|\\doi\s*\{", re.I),
     "an arXiv id or DOI in the text, where a \\cite key belongs"),
]


def inline_entries(text: str) -> list[str]:
    """Bibliography the model wrote itself, one finding per line found.

    The bibliography is assembled from cite_reference calls, so anything of
    this shape in the notes is a reference that never reached the .bib: it
    renders as a second, inconsistent list of sources, it is invisible to
    every later lecture that might have cited the same paper, and the key
    that should point at it does not exist.

    Comments are stripped first, and everything above \begin{document} is
    skipped — the preamble legitimately contains an arxiv.org URL, because
    that is where the formatting of arXiv eprints is defined.
    """
    if r"\begin{document}" in text:
        offset = text[:text.index(r"\begin{document}")].count("\n")
        text = text[text.index(r"\begin{document}"):]
    else:
        offset = 0
    lines = [re.sub(r"(?<!\\)%.*", "", ln) for ln in text.splitlines()]
    body = "\n".join(lines)
    found: dict[int, str] = {}
    for pattern, what in _INLINE:
        for m in pattern.finditer(body):
            num = body[:m.start()].count("\n") + offset + 1
            excerpt = lines[num - offset - 1].strip()[:90]
            found.setdefault(num, f"line {num}: {what} — {excerpt}")
    return [found[n] for n in sorted(found)]


def attach_to_document(tex_file: Path, bib_file: Path) -> bool:
    """Wire a model-written standalone document up to the .bib it cited into.

    The course assembles its own preamble and can put biblatex where it
    belongs; a single lecture's document is written by the model, which is
    told (correctly) never to write bibliography machinery itself. Somebody
    still has to add it, and doing it here rather than in the prompt is the
    difference between a rule that holds and a rule that mostly holds.

    \\usepackage goes immediately before \\begin{document}: biblatex must load
    after hyperref, and that is the one position guaranteed to be after it
    wherever the model put it. Returns True if the document was changed —
    False when there is nothing cited, when it is already wired up, or when
    there is no \\begin{document} to work with (a body-only file).

    Idempotent, because the fix rounds re-run this on a file it already
    edited.
    """
    if not has_entries(bib_file):
        return False
    text = tex_file.read_text()
    if r"\addbibresource" in text and r"\printbibliography" in text:
        return False
    if r"\begin{document}" not in text or r"\end{document}" not in text:
        return False
    if r"\begin{thebibliography}" in text:
        # A hand-written bibliography and biblatex would print two lists, and
        # the \cite keys resolve against only one of them.
        print("  Warning: the notes contain a hand-written thebibliography; "
              f"leaving {tex_file.name} alone. The entries collected in "
              f"{bib_file.name} are not wired in.")
        return False

    if r"\addbibresource" not in text:
        text = text.replace(
            "\\begin{document}",
            (BIB_PREAMBLE % bib_file.name) + "\n\n\\begin{document}", 1)
    if r"\printbibliography" not in text:
        idx = text.rindex("\\end{document}")
        text = text[:idx] + BIB_PRINT + "\n\n" + text[idx:]
    tex_file.write_text(text)
    return True
