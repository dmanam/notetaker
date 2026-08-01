"""
agent_log.py — a record of what each agent actually did, under output/logs/.

Two files per run of the pipeline:

  output/logs/index.jsonl              one summary line per agent invocation
  output/logs/<ts>-<role>-<slug>.jsonl the full event trace for that one agent

The index answers "which agents ran, what did they cost, which ones flailed";
the per-agent trace answers "what exactly did the verifier do to lecture 12".
Both are JSON Lines, so they can be read with a text editor or aggregated with
a couple of lines of Python.

Logging must never break a run: every write is best-effort, and a failure to
log is reported once and then ignored.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

INDEX_FILENAME = "index.jsonl"
# Long strings (prompts, tool results, agent replies) are clipped in the trace
# — enough to see what happened, not enough to make the log unreadable.
CLIP = 4000
_BOARD_FILE = re.compile(r"/boards/board-0*(\d+)\.[a-z]+$")


def _clip(value, limit: int = CLIP):
    if isinstance(value, str):
        return (value if len(value) <= limit
                else value[:limit] + f"… [+{len(value) - limit} chars]")
    if isinstance(value, dict):
        return {k: _clip(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        clipped = [_clip(v, limit) for v in list(value)[:20]]
        if len(value) > 20:
            clipped.append(f"… [+{len(value) - 20} more]")
        return clipped
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _clip(str(value), limit)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")[:60] or "none"


class AgentLog:
    """One agent invocation. Use via start_log(); close() writes the index."""

    def __init__(self, path: Path, index_path: Path, meta: dict):
        self.path = path
        self.index_path = index_path
        self.meta = meta
        self.t0 = time.time()
        self.tool_counts: dict[str, int] = {}
        self.boards_seen: set = set()
        self.n_events = 0
        self._broken = False
        self.event("start", **meta)

    # -- writing ---------------------------------------------------------

    def _write(self, target: Path, record: dict) -> None:
        if self._broken:
            return
        try:
            with open(target, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:                      # disk full, bad path, …
            self._broken = True
            print(f"(agent logging disabled: {exc})", flush=True)

    def event(self, kind: str, **fields) -> None:
        self.n_events += 1
        self._write(self.path, {
            "t": datetime.now().isoformat(timespec="seconds"),
            "dt": round(time.time() - self.t0, 2),
            "kind": kind,
            **{k: _clip(v) for k, v in fields.items()},
        })

    def tool(self, name: str, tool_input=None, result=None,
             seconds: float | None = None, is_error: bool = False) -> None:
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        # Which board stills the agent actually opened. It is told to read
        # them all and will sometimes skip one anyway, which is worth
        # knowing when a section turns out to be wrong about notation.
        if isinstance(tool_input, dict):
            m = _BOARD_FILE.search(str(tool_input.get("file_path", "")))
            if m:
                self.boards_seen.add(int(m.group(1)))
        self.event("tool", name=name, input=tool_input, result=result,
                   seconds=seconds, error=is_error or None)

    def close(self, **summary) -> dict:
        record = {
            **self.meta,
            "trace": self.path.name,
            "seconds": round(time.time() - self.t0, 1),
            "events": self.n_events,
            "tools": dict(sorted(self.tool_counts.items(),
                                 key=lambda kv: -kv[1])),
            "tool_calls": sum(self.tool_counts.values()),
            **({"boards_read": sorted(self.boards_seen)}
               if self.boards_seen else {}),
            **{k: _clip(v, 600) for k, v in summary.items()},
        }
        self.event("end", **{k: v for k, v in record.items()
                             if k not in self.meta})
        self._write(self.index_path, record)
        return record


class NullLog:
    """Stand-in when logging is off, so callers need no conditionals."""

    def event(self, *a, **k): pass
    def tool(self, *a, **k): pass
    def close(self, **k): return {}


def start_log(log_dir: Path | None, *, role: str, lecture: str | None,
              **meta) -> AgentLog | NullLog:
    if not log_dir:
        return NullLog()
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{_slug(role)}-{_slug(lecture or 'course')}.jsonl"
        path = log_dir / name
        n = 2
        while path.exists():                        # same second, same role
            path = log_dir / name.replace(".jsonl", f"-{n}.jsonl")
            n += 1
        return AgentLog(path, log_dir / INDEX_FILENAME,
                        {"role": role, "lecture": lecture, "pid": os.getpid(),
                         **{k: _clip(v, 600) for k, v in meta.items()}})
    except OSError as exc:
        print(f"(agent logging disabled: {exc})", flush=True)
        return NullLog()


# -- reading back ---------------------------------------------------------

def read_index(log_dir: Path) -> list[dict]:
    path = Path(log_dir) / INDEX_FILENAME
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def summarize(log_dir: Path, limit: int = 40) -> str:
    """Human-readable digest of the most recent agent runs."""
    rows = read_index(log_dir)
    if not rows:
        return f"No agent logs in {log_dir}."
    lines = [f"{len(rows)} agent run(s) logged in {log_dir}",
             f"{'role':<14}{'lecture':<26}{'sec':>7}{'tools':>7}  cost"]
    for r in rows[-limit:]:
        lines.append(
            f"{str(r.get('role'))[:13]:<14}"
            f"{str(r.get('lecture') or '-')[:25]:<26}"
            f"{r.get('seconds', 0):>7.0f}"
            f"{r.get('tool_calls', 0):>7}"
            f"  {r.get('cost', '')}")
    by_role: dict[str, list] = {}
    for r in rows:
        by_role.setdefault(str(r.get("role")), []).append(r)
    lines.append("\nby role:")
    for role, rs in sorted(by_role.items()):
        secs = sum(r.get("seconds", 0) for r in rs)
        tools = sum(r.get("tool_calls", 0) for r in rs)
        lines.append(f"  {role:<14} {len(rs):>4} run(s)  "
                     f"{secs / 60:>7.1f} min  {tools:>5} tool calls")
    return "\n".join(lines)
