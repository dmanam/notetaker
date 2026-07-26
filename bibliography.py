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
    """Surname of the first author ('Scholze, Peter' / 'Peter Scholze' /
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
    key = entry_key(entry) or _slug_key(title or url_or_id)
    bib_file.parent.mkdir(parents=True, exist_ok=True)
    text = bib_file.read_text() if bib_file.exists() else ""
    unique = _unique_key(text, key)
    if unique != key:
        entry = entry.replace(key, unique, 1)
        key = unique

    with open(bib_file, "a") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(f"\n% source: {source_key(url_or_id)}\n{entry}")
    return key, True


def has_entries(bib_file: Path) -> bool:
    return bib_file.exists() and "@" in bib_file.read_text()


def list_keys(bib_file: Path) -> list[str]:
    if not bib_file.exists():
        return []
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_file.read_text())
