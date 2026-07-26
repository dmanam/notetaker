"""
usage.py — Token-usage tracking with API-equivalent cost.

Sources differ per backend:
  api           — exact per-response usage from the Messages API, priced with
                  the table below
  subscription  — usage and API-equivalent cost as computed by Claude Code
                  itself (reported even under subscription auth)
  codex         — total token count parsed from codex's output footer; no
                  public price table for the GPT models, so tokens only
"""

from __future__ import annotations

from dataclasses import dataclass


# $ per million tokens: (input, output). Cache reads bill at 0.1x the input
# rate; cache writes (5-minute TTL) at 1.25x.
ANTHROPIC_PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4": (5.00, 25.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "haiku": (1.00, 5.00),
    "opus": (5.00, 25.00),
    "sonnet": (3.00, 15.00),
}


def anthropic_price(model: str) -> tuple[float, float] | None:
    for prefix, price in ANTHROPIC_PRICES.items():
        if model.startswith(prefix):
            return price
    return None


@dataclass
class Usage:
    input_tokens: int = 0          # uncached input
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None  # None = unknown (no price table)
    note: str = ""

    def add_anthropic_response(self, model: str, usage) -> None:
        """Accumulate a Messages API response.usage object."""
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cread = getattr(usage, "cache_read_input_tokens", 0) or 0
        cwrite = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cread
        self.cache_write_tokens += cwrite
        price = anthropic_price(model)
        if price:
            pin, pout = price
            cost = (inp * pin + out * pout
                    + cread * pin * 0.1 + cwrite * pin * 1.25) / 1e6
            self.cost_usd = (self.cost_usd or 0.0) + cost

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd
        if other.note and other.note not in self.note:
            self.note = "; ".join(x for x in (self.note, other.note) if x)

    def any(self) -> bool:
        return bool(self.input_tokens or self.output_tokens
                    or self.cache_read_tokens or self.cache_write_tokens
                    or self.cost_usd or self.note)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": self.cost_usd,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Usage":
        return cls(
            input_tokens=d.get("input_tokens", 0) or 0,
            output_tokens=d.get("output_tokens", 0) or 0,
            cache_read_tokens=d.get("cache_read_tokens", 0) or 0,
            cache_write_tokens=d.get("cache_write_tokens", 0) or 0,
            cost_usd=d.get("cost_usd"),
            note=d.get("note", "") or "",
        )


def usage_from_claude_code(cc_usage: dict | None,
                           total_cost_usd: float | None) -> Usage:
    """Build a Usage from Claude Code's ResultMessage fields."""
    cc_usage = cc_usage or {}
    return Usage(
        input_tokens=cc_usage.get("input_tokens", 0) or 0,
        output_tokens=cc_usage.get("output_tokens", 0) or 0,
        cache_read_tokens=cc_usage.get("cache_read_input_tokens", 0) or 0,
        cache_write_tokens=cc_usage.get("cache_creation_input_tokens", 0) or 0,
        cost_usd=total_cost_usd,
        note="cost as computed by Claude Code",
    )


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def format_usage(u: Usage) -> str:
    counters = (u.input_tokens or u.output_tokens
                or u.cache_read_tokens or u.cache_write_tokens)
    if not counters:
        return u.note or "no usage recorded"
    total_in = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens
    s = (f"{_fmt_tokens(total_in)} in "
         f"({_fmt_tokens(u.cache_read_tokens)} cache-read) / "
         f"{_fmt_tokens(u.output_tokens)} out")
    if u.cost_usd is not None:
        s += f" ≈ ${u.cost_usd:.2f} API-equivalent"
    if u.note:
        s += f" ({u.note})"
    return s
