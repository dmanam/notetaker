#!/usr/bin/env python3
"""
notes_mcp_server.py — Stdio MCP server exposing the note-taking tools to the
Codex CLI. Spawned automatically by the 'codex' backend in claude_backend.py:

  codex exec -c mcp_servers.notes.command=... -c mcp_servers.notes.args=[...]

Tool state (recorded corrections, preamble additions, frame count) is
persisted to the context's state_file so the parent process can merge it back
after the run. stdout is the MCP protocol channel — all diagnostics go to
stderr, and user prompts go through /dev/tty (see notes_tools.ask_user_input).
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

from notes_tools import NotesToolContext, ToolResult, build_handlers, build_tools


def _text(result: ToolResult) -> str:
    if result.is_error:
        raise RuntimeError(str(result.content))
    assert isinstance(result.content, str)
    return result.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True,
                        help="Path to the serialized NotesToolContext JSON")
    args = parser.parse_args()

    ctx = NotesToolContext.load(Path(args.context))
    # Codex views local files natively — deliver binary results (PDF pages)
    # as files rather than inline images.
    ctx.files_mode = True
    handlers = build_handlers(ctx)
    specs = {t["name"]: t for t in build_tools(ctx)}
    server = FastMCP("notes")

    @server.tool(name="fetch_document",
                 description=specs["fetch_document"]["description"])
    def fetch_document(url_or_id: str) -> str:
        return _text(handlers["fetch_document"]({"url_or_id": url_or_id}))

    @server.tool(name="clarify_transcript",
                 description=specs["clarify_transcript"]["description"])
    def clarify_transcript(transcript_text: str, context: str,
                           timestamp: str = "", guess: str = "") -> str:
        return _text(handlers["clarify_transcript"]({
            "transcript_text": transcript_text,
            "context": context,
            "timestamp": timestamp,
            "guess": guess,
        }))

    @server.tool(name="ask_user", description=specs["ask_user"]["description"])
    def ask_user(question: str, timestamp: str = "",
                 provisional: str = "") -> str:
        return _text(handlers["ask_user"]({"question": question,
                                           "timestamp": timestamp,
                                           "provisional": provisional}))

    @server.tool(name="get_user_answers",
                 description=specs["get_user_answers"]["description"])
    def get_user_answers() -> str:
        return _text(handlers["get_user_answers"]({}))

    @server.tool(name="view_pdf_page",
                 description=specs["view_pdf_page"]["description"])
    def view_pdf_page(path: str, page: int) -> str:
        # files_mode: the handler saves a PNG and returns its path.
        return _text(handlers["view_pdf_page"]({"path": path, "page": page}))

    if "cite_reference" in handlers:
        @server.tool(name="cite_reference",
                     description=specs["cite_reference"]["description"])
        def cite_reference(url_or_id: str, title: str = "",
                           author: str = "", year: str = "") -> str:
            return _text(handlers["cite_reference"](
                {"url_or_id": url_or_id, "title": title,
                 "author": author, "year": year}))

    if "add_to_preamble" in handlers:
        @server.tool(name="add_to_preamble",
                     description=specs["add_to_preamble"]["description"])
        def add_to_preamble(latex: str) -> str:
            return _text(handlers["add_to_preamble"]({"latex": latex}))

    if "get_frame" in handlers:
        if ctx.frames_dir:
            # File mode (codex): the handler saves a JPEG and returns its path.
            @server.tool(name="get_frame",
                         description=specs["get_frame"]["description"])
            def get_frame(timestamp: float | str) -> str:
                return _text(handlers["get_frame"]({"timestamp": timestamp}))
        else:
            @server.tool(name="get_frame",
                         description=specs["get_frame"]["description"])
            def get_frame(timestamp: float | str) -> Image:
                result = handlers["get_frame"]({"timestamp": timestamp})
                if result.is_error:
                    raise RuntimeError(str(result.content))
                b64 = result.content[0]["source"]["data"]
                return Image(data=base64.b64decode(b64), format="jpeg")

    server.run()  # stdio transport


if __name__ == "__main__":
    main()
