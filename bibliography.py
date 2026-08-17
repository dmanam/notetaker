"""
bibliography.py — A running BibTeX file for the course.

Entries are appended to <output-dir>/references.bib as the models cite
things, each preceded by a `% source: …` marker so the same paper is never
added twice (and so a later run can look up the key it already assigned).

Sources:
  arXiv IDs/URLs — arxiv.org/bibtex/<id> (proper entry with a real key)
  DOIs           — doi.org content negotiation (application/x-bibtex)
  anything else  — a generated @online entry
"""

from __future__ import annotations

import re
import urllib.request
from datetime import date
from pathlib import Path

from fetch import arxiv_id_of

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


def _field(entry: str, name: str) -> str:
    """One BibTeX field value, tolerating nested braces (author={N{\\"o}beling})
    and unbraced values (year=1968)."""
    m = re.search(rf"\b{name}\s*=\s*", entry, re.I)
    if not m:
        return ""
    i = m.end()
    if i >= len(entry):
        return ""
    if entry[i] == "{":
        depth, start = 0, i + 1
        for j in range(i, len(entry)):
            if entry[j] == "{":
                depth += 1
            elif entry[j] == "}":
                depth -= 1
                if depth == 0:
                    return entry[start:j]
        return entry[start:]
    if entry[i] == '"':
        j = entry.find('"', i + 1)
        return entry[i + 1:j] if j != -1 else entry[i + 1:]
    m2 = re.match(r"[^,}\s]+", entry[i:])
    return m2.group(0) if m2 else ""


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


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
