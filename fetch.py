"""
fetch.py — Retrieve and cache web references (arXiv papers, PDFs, HTML pages).

Supports:
  - Bare arXiv IDs:          2310.12345, 2310.12345v2, math/0601462
  - arXiv-prefixed IDs:      arxiv:2310.12345   (case-insensitive)
  - arXiv URLs:              https://arxiv.org/{abs,pdf,html,e-print}/<id>
  - PDF URLs:                https://example.com/paper.pdf
  - General web pages:       https://example.com/page.html

For arXiv papers, the extracted text prefers the TeX source (the paper's
actual macros and notation), falling back to the HTML rendering, then the
PDF. All artifacts are cached as real files — the unpacked source tree and,
always, the rendered PDF (which shows the resolved theorem/equation
numbering) — so agents can open them directly with their own file tools.

PDF text extraction prefers PyMuPDF (much better output than pypdf,
especially for layout) and falls back to pypdf.
"""

import gzip
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Trim very long documents so they don't flood the context window.
MAX_CHARS = 150_000

# New-style (2007+) and old-style (math/0601462) arXiv identifiers.
ARXIV_ID_RE = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "li", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = [ln.strip() for ln in text.splitlines()]
        # Collapse runs of blank lines to a single blank line
        out, prev_blank = [], False
        for ln in lines:
            blank = (ln == "")
            if blank and prev_blank:
                continue
            out.append(ln)
            prev_blank = blank
        return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Identifier handling
# ---------------------------------------------------------------------------

def arxiv_id_of(url_or_id: str) -> str | None:
    """Extract an arXiv identifier from an ID, arxiv:-prefixed ID, or any
    arXiv URL form; None if this isn't an arXiv reference."""
    s = url_or_id.strip()
    m = re.fullmatch(rf"arxiv:({ARXIV_ID_RE})", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if re.fullmatch(ARXIV_ID_RE, s):
        return s
    m = re.search(rf"arxiv\.org/(?:abs|pdf|html|e-print)/({ARXIV_ID_RE})", s)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _fetch_raw(url: str) -> tuple[bytes, str]:
    """Return (body_bytes, content_type)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "notetaker/1.0 (academic note-taking tool)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_type = resp.headers.get("Content-Type", "")
        return resp.read(), content_type


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _clean_tex_title(t: str) -> str:
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)
    t = re.sub(r"[{}~$]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _unpack_tex_source(raw: bytes, dest: Path) -> tuple[str, str, str | None]:
    """Unpack an arXiv e-print (gzipped tarball or single gzipped file) into
    dest as real files, and return (concatenated_tex_text, title,
    main_file_relpath). Returns ("", "", None) when the submission has no
    usable TeX source (e.g. PDF-only)."""
    if raw[:4] == b"%PDF":
        return "", "", None

    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            dest.mkdir(parents=True, exist_ok=True)
            tar.extractall(dest, filter="data")
    except tarfile.TarError:
        # Not a tarball: single gzipped file, or already-plain text.
        try:
            data = gzip.decompress(raw)
        except OSError:
            data = raw
        if data[:4] == b"%PDF":
            return "", "", None
        text = data.decode("utf-8", errors="replace")
        if not ("\\documentclass" in text or "\\begin{document}" in text
                or "\\section" in text):
            return "", "", None
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "main.tex").write_text(text)

    tex_files: list[tuple[str, str]] = []
    for f in sorted(dest.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".tex", ".bbl"):
            tex_files.append((str(f.relative_to(dest)),
                              f.read_text(errors="replace")))
    if not tex_files:
        return "", "", None

    # Main file (the one with \documentclass) first, rest alphabetically.
    tex_files.sort(key=lambda kv: ("\\documentclass" not in kv[1], kv[0]))
    main_name = (tex_files[0][0]
                 if "\\documentclass" in tex_files[0][1] else None)
    text = "\n\n".join(f"%% ===== {name} =====\n{content}"
                       for name, content in tex_files)

    title = ""
    m = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{(.+)", text)
    if m:
        title = _clean_tex_title(m.group(1).splitlines()[0].rstrip("}"))
    return text, title, main_name


def _extract_pdf_text(raw: bytes) -> tuple[str, str]:
    """Return (text, title). Prefers PyMuPDF; falls back to pypdf."""
    try:
        import pymupdf
        doc = pymupdf.open(stream=raw, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        title = ((doc.metadata or {}).get("title") or "").strip()
        if text.strip():
            return text, title
    except ImportError:
        pass

    if not HAS_PYPDF:
        raise RuntimeError(
            "Neither pymupdf nor pypdf is installed; cannot read PDFs. "
            "Enter the nix devshell or: pip install pymupdf"
        )
    reader = pypdf.PdfReader(io.BytesIO(raw))
    title = ""
    if reader.metadata and reader.metadata.title:
        title = reader.metadata.title.strip()
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), title


def _extract_html_text(raw: bytes) -> tuple[str, str]:
    """Return (text, title)."""
    html = raw.decode("utf-8", errors="replace")
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text(), title


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_reference(url_or_id: str, refs_dir: Path) -> dict:
    """
    Fetch a web reference and cache it — and all its artifacts, as real
    files — under refs_dir/<hash>/:
      source/       unpacked TeX source tree (arXiv, when available)
      paper.pdf     the rendered PDF (always fetched for arXiv — it shows the
                    resolved theorem/equation numbering; also kept for direct
                    PDF URLs)
      paper.html    the HTML rendering, when it was used
      text.txt      extracted text (what goes into context)

    Returns a dict with keys:
      url           URL the primary text came from
      original      the original url_or_id argument
      title         document title (may be empty)
      format        "tex" | "html" | "pdf" — primary text format
      local_path    cached text file, relative to refs_dir.parent
      assets        {"source_dir", "main_tex", "pdf", "html"} — file paths
                    relative to refs_dir.parent (entries may be None)
      text          extracted text (may be truncated)

    Raises urllib.error.URLError or RuntimeError on failure.
    Re-uses cached content if available.
    """
    refs_dir.mkdir(parents=True, exist_ok=True)
    aid = arxiv_id_of(url_or_id)
    cache_key = f"arxiv:{aid}" if aid else url_or_id.strip()
    base = refs_dir.parent

    ref_dir = refs_dir / hashlib.sha1(cache_key.encode()).hexdigest()[:14]
    text_path = ref_dir / "text.txt"
    meta_path = ref_dir / "meta.json"

    if meta_path.exists() and text_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        assets = meta.setdefault("assets", {})
        # Caches written before outlines existed get one on first reuse.
        if not assets.get("outline"):
            outline = write_outline(ref_dir, assets, text_path)
            if outline:
                assets["outline"] = str(outline.relative_to(base))
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        rel_outline = assets.get("outline")
        # Clip on the way out: text.txt holds the full document now, and a
        # cache hit must not put 450k of it into the prompt.
        meta["text"] = clip_for_prompt(
            text_path.read_text(), text_path,
            base / rel_outline if rel_outline else None)
        return meta

    ref_dir.mkdir(parents=True, exist_ok=True)
    assets: dict = {"source_dir": None, "main_tex": None,
                    "pdf": None, "html": None}

    def rel(p: Path) -> str:
        return str(p.relative_to(base))

    if aid:
        text, title, fmt = "", "", "pdf"
        url = f"https://arxiv.org/abs/{aid}"

        # 1. TeX source — the paper's actual macros and notation.
        try:
            raw, _ = _fetch_raw(f"https://arxiv.org/e-print/{aid}")
            text, title, main_name = _unpack_tex_source(raw, ref_dir / "source")
            if text:
                fmt = "tex"
                url = f"https://arxiv.org/e-print/{aid}"
                assets["source_dir"] = rel(ref_dir / "source")
                if main_name:
                    assets["main_tex"] = rel(ref_dir / "source" / main_name)
        except (urllib.error.URLError, OSError):
            pass

        # 2. HTML rendering — only needed when there is no TeX text.
        if not text:
            try:
                raw, _ = _fetch_raw(f"https://arxiv.org/html/{aid}")
                html_text, html_title = _extract_html_text(raw)
                if len(html_text) > 500:  # a real rendering, not an error page
                    (ref_dir / "paper.html").write_bytes(raw)
                    assets["html"] = rel(ref_dir / "paper.html")
                    text, title, fmt = html_text, html_title, "html"
                    url = f"https://arxiv.org/html/{aid}"
            except (urllib.error.URLError, OSError):
                pass

        # 3. PDF — always cached: it is the typeset form with the resolved
        # theorem/equation numbering, and the fallback when TeX/HTML are
        # garbled or absent.
        try:
            raw, _ = _fetch_raw(f"https://arxiv.org/pdf/{aid}")
            (ref_dir / "paper.pdf").write_bytes(raw)
            assets["pdf"] = rel(ref_dir / "paper.pdf")
            if not text:
                text, title = _extract_pdf_text(raw)
                url = f"https://arxiv.org/pdf/{aid}"
        except (urllib.error.URLError, OSError):
            if not text:
                raise
    else:
        url = url_or_id.strip()
        raw, content_type = _fetch_raw(url)
        is_pdf = ("application/pdf" in content_type
                  or url.lower().split("?")[0].endswith(".pdf"))
        if is_pdf:
            (ref_dir / "paper.pdf").write_bytes(raw)
            assets["pdf"] = rel(ref_dir / "paper.pdf")
            text, title = _extract_pdf_text(raw)
            fmt = "pdf"
        else:
            (ref_dir / "paper.html").write_bytes(raw)
            assets["html"] = rel(ref_dir / "paper.html")
            text, title = _extract_html_text(raw)
            fmt = "html"

    if not title:
        title = cache_key

    # Persist the FULL extraction, then clip only the copy that goes into the
    # prompt. Truncating before writing used to throw the tail away entirely,
    # so "read the rest from the cached file" was not actually possible.
    text_path.write_text(text)
    outline = write_outline(ref_dir, assets, text_path)
    if outline:
        assets["outline"] = rel(outline)
    text = clip_for_prompt(text, text_path, outline)
    meta = {
        "url": url,
        "original": url_or_id,
        "title": title,
        "format": fmt,
        "local_path": str(text_path.relative_to(base)),
        "assets": assets,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    meta["text"] = text
    return meta


def clip_for_prompt(text: str, path: Path, outline: Path | None = None) -> str:
    """Cut a document down to what fits in a prompt, telling the model exactly
    how to get the rest. A bare '[truncated]' marker is a dead end: it says
    content is missing but not how much, where the full copy is, or what to do
    — so the model retries the same read or fills the gap by guessing."""
    if len(text) <= MAX_CHARS:
        return text
    shown, total = MAX_CHARS, len(text)
    note = [
        f"[Only the first {shown:,} of {total:,} characters are shown here "
        f"({100 * shown // total}%).",
        f"The COMPLETE text is on disk at: {path}",
        "To see the rest, do not re-fetch — read that file directly in ranges "
        "with your file-reading tool (offset/limit), or search it with "
        "search_document / grep for the term you need.",
    ]
    if outline is not None:
        note.append(f"An outline with line numbers is at: {outline} — use it "
                    f"to jump straight to the relevant part.")
    return text[:shown] + "\n\n" + "\n".join(note) + "]"


_TEX_HEADING = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection)\*?\s*\{", re.M)
_TEX_ENV = re.compile(
    r"\\begin\{(theorem|proposition|lemma|corollary|definition|remark|"
    # greedy to the last ] on the line: titles contain brackets themselves,
    # as in \begin{theorem}[Freeness of $\Z[S]$]
    r"example|conjecture)\}[ \t]*(\[[^\n]*\])?")


def _balanced(text: str, open_idx: int) -> str:
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return " ".join(text[open_idx + 1:j].split())
    return ""


def write_outline(ref_dir: Path, assets: dict, text_path: Path) -> Path | None:
    """Write outline.txt: headings with line numbers for TeX sources, the
    bookmark table for PDFs. Large papers are otherwise navigated by reading
    linearly until the read is cut off."""
    lines: list[str] = []

    src = assets.get("source_dir")
    if src:
        src_dir = ref_dir.parent.parent / src if not (ref_dir / src).exists() \
            else ref_dir / src
        src_dir = src_dir if src_dir.exists() else ref_dir / "source"
        for tex in sorted(src_dir.rglob("*.tex")) if src_dir.exists() else []:
            try:
                body = tex.read_text(errors="replace")
            except OSError:
                continue
            found = []
            for m in _TEX_HEADING.finditer(body):
                title = _balanced(body, m.end() - 1)
                if title:
                    found.append((body.count("\n", 0, m.start()) + 1,
                                  m.group(1), title))
            for m in _TEX_ENV.finditer(body):
                label = (m.group(2) or "").strip("[]")
                found.append((body.count("\n", 0, m.start()) + 1,
                              m.group(1), label))
            if found:
                lines.append(f"== {tex.name} ({len(body):,} chars) ==")
                for lineno, kind, title in sorted(found):
                    lines.append(f"  line {lineno:>6}  {kind:<13} {title}"
                                 .rstrip())
                lines.append("")

    pdf = assets.get("pdf")
    if pdf:
        pdf_path = ref_dir / Path(pdf).name
        if pdf_path.exists():
            try:
                import fitz
                with fitz.open(pdf_path) as doc:
                    toc = doc.get_toc()
                    if toc:
                        lines.append(f"== {pdf_path.name} "
                                     f"({doc.page_count} pages) ==")
                        for level, title, page in toc:
                            lines.append(f"  page {page:>4}  "
                                         f"{'  ' * (level - 1)}{title}")
                        lines.append("")
            except Exception:
                pass

    if not lines:
        return None
    out = ref_dir / "outline.txt"
    out.write_text(
        f"Outline of the cached copies of this reference.\n"
        f"Full extracted text: {text_path} ({text_path.stat().st_size:,} bytes)"
        f"\nUse the line numbers to read just the part you need.\n\n"
        + "\n".join(lines))
    return out


def describe_assets(meta: dict, base: Path) -> str:
    """One block listing a reference's locally cached files, for the model to
    open with its own file tools."""
    assets = meta.get("assets") or {}
    lines = []
    if assets.get("source_dir"):
        main = (f" (main file: {Path(assets['main_tex']).name})"
                if assets.get("main_tex") else "")
        lines.append(f"  TeX source: {base / assets['source_dir']}{main}")
    if assets.get("pdf"):
        lines.append(f"  Rendered PDF: {base / assets['pdf']}  <- the typeset "
                     f"paper with resolved theorem/equation numbers; use it "
                     f"when you need the numbering as cited, or when the "
                     f"extracted text looks garbled")
    if assets.get("html"):
        lines.append(f"  HTML: {base / assets['html']}")
    if assets.get("outline"):
        lines.append(f"  Outline: {base / assets['outline']}  <- headings and "
                     f"theorem environments with line numbers; read this "
                     f"first for a long document and jump to the part you "
                     f"need instead of reading from the top")
    if meta.get("local_path"):
        full = base / meta["local_path"]
        try:
            size = full.stat().st_size
        except OSError:
            size = 0
        if size:
            lines.append(f"  Full extracted text: {full} ({size:,} bytes)"
                         + ("  <- longer than fits in one read; use the "
                            "outline, or search_document, or read it in "
                            "ranges" if size > MAX_CHARS else ""))
    if not lines:
        return ""
    return ("Cached locally (open with your file tools / view_pdf_page):\n"
            + "\n".join(lines) + "\n")


def load_cached_reference(meta: dict, output_root: Path) -> dict:
    """Re-attach .text to a metadata dict loaded from state. The cached file
    now holds the full extraction, so clip it the same way a fresh fetch
    does — otherwise a 450k source would land whole in the prompt."""
    if "text" not in meta:
        p = output_root / meta["local_path"]
        if p.exists():
            outline = meta.get("assets", {}).get("outline")
            meta["text"] = clip_for_prompt(
                p.read_text(), p,
                output_root / outline if outline else None)
        else:
            meta["text"] = ""
    return meta
