"""Bounded Markdown reports and exact-copy validation (ADR-358)."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from tldw_chatbook.Utils.log_sanitizer import sanitize_string

MAX_REPORT_BYTES = 60_000
MAX_AUTHORED_BYTES = 8_000
MAX_REPORT_ITEMS = 500
SECTION_BUDGETS = {"logs": 24_000, "trace": 12_000, "tools": 8_000, "exceptions": 4_000}


@dataclass(frozen=True)
class BugReportDraft:
    description: str = field(repr=False)
    steps: str = field(default="", repr=False)
    expected: str = field(default="", repr=False)
    actual: str = field(default="", repr=False)


@dataclass(frozen=True)
class BugReportOptions:
    include_logs: bool = True
    include_trace: bool = True
    include_tool_content: bool = False


@dataclass(frozen=True)
class BugReportItem:
    source: Literal["logs", "exceptions", "trace", "tools"]
    text: str = field(repr=False)
    event_key: str | None = field(default=None, repr=False)
    observed_at: str | None = None


@dataclass(frozen=True)
class BugReportEvidence:
    environment: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    logs: tuple[BugReportItem, ...] = field(default=(), repr=False)
    exceptions: tuple[BugReportItem, ...] = field(default=(), repr=False)
    trace: tuple[BugReportItem, ...] = field(default=(), repr=False)
    tools: tuple[BugReportItem, ...] = field(default=(), repr=False)
    inventory: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class BugReportCopyCheck:
    state: Literal["ready", "review_required", "oversize", "unavailable"]
    text: str = field(repr=False)
    byte_count: int
    message: str


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _fenced(text: str) -> str:
    runs = re.findall(r"`+|~+", text)
    fences = [
        character
        * max(3, 1 + max((len(run) for run in runs if run[0] == character), default=0))
        for character in ("`", "~")
    ]
    fence = min(fences, key=len)
    return f"{fence}text\n{text}\n{fence}\n"


def _shorten(text: str, budget: int) -> str:
    marker = "\n[entry truncated to report budget]"
    available = max(0, budget - _size(marker))
    prefix = text.encode("utf-8")[:available].decode("utf-8", errors="ignore")
    # Redaction already ran on the complete source; avoid partial identifier tokens.
    if prefix != text and prefix and not prefix[-1].isspace():
        prefix = re.sub(r"\S+$", "", prefix)
    return prefix + marker


def _section(title: str, texts: tuple[str, ...], budget: int, empty: str) -> str:
    heading = f"## {title}\n\n"
    if not texts:
        return heading + empty + "\n"
    kept: list[str] = []
    for text in reversed(texts[-MAX_REPORT_ITEMS:]):
        try:
            safe = sanitize_string(text)
        except Exception:  # A failed evidence field never becomes a raw fallback.
            safe = "[evidence unavailable: sanitization failed]"
        candidate = "\n\n".join(reversed([*kept, safe]))
        if _size(heading + _fenced(candidate)) + 160 > budget:
            if not kept:
                safe = _shorten(safe, max(0, budget - _size(heading) - 512))
                while _size(heading + _fenced(safe)) + 160 > budget and safe:
                    safe = _shorten(safe, max(0, _size(safe) // 2))
                kept.append(safe)
            break
        kept.append(safe)
    omitted = len(texts) - len(kept)
    notice = (
        f"[{omitted} older entries omitted from this excerpt]\n\n" if omitted else ""
    )
    return heading + notice + _fenced("\n\n".join(reversed(kept)))


def render_bug_report(
    draft: BugReportDraft, evidence: BugReportEvidence, options: BugReportOptions
) -> str:
    """Render a sanitized report without discarding user-authored text.

    Raises:
        ValueError: Description is empty or authored text exceeds its budget.
    """
    if not draft.description.strip():
        raise ValueError("Description is required.")
    fields = (
        ("Description", draft.description),
        ("Reproduction steps", draft.steps),
        ("Expected behavior", draft.expected),
        ("Actual behavior", draft.actual),
    )
    if sum(_size(text) for _, text in fields) > MAX_AUTHORED_BYTES:
        raise ValueError(
            "Description and related fields must fit within 8,000 UTF-8 bytes."
        )
    try:
        authored = "\n".join(
            f"## {label}\n\n{_fenced(sanitize_string(text))}"
            for label, text in fields
            if text.strip()
        )
    except Exception:
        raise ValueError(
            "Description sanitization is unavailable; keep the draft and retry."
        ) from None
    if _size(authored) > MAX_AUTHORED_BYTES:
        raise ValueError(
            "Description and formatting must fit within 8,000 UTF-8 bytes."
        )
    environment = tuple(f"{name}: {value}" for name, value in evidence.environment)
    inventory = (
        "Application-wide session logs; not isolated to the selected conversation."
        if options.include_logs
        else "Application-wide session logs excluded.",
        "Recognized credentials and PII are masked. Review before sharing; private text can remain.",
        "Console trace included when available."
        if options.include_trace
        else "Console trace excluded.",
    )
    tools_enabled = options.include_trace and options.include_tool_content
    inventory += (
        "Tool inputs/results: opted in; may quote files, documents, and other conversations."
        if tools_enabled
        else "Tool inputs/results: excluded.",
    )
    parts = [
        "# Chatbook bug report v1\n",
        _section("Environment", environment, 1_600, "Environment facts unavailable."),
        _section(
            "Evidence inventory",
            evidence.inventory,
            1_600,
            "Source details unavailable.",
        )
        + "\n".join(inventory)
        + "\n",
        authored,
    ]
    if options.include_logs:
        parts += [
            _section(
                "Exceptions",
                tuple(item.text for item in evidence.exceptions),
                4_000,
                "No exception traceback was captured in the selected log excerpt.",
            ),
            _section(
                "Logs",
                tuple(item.text for item in evidence.logs),
                24_000,
                "Session logs unavailable or empty.",
            ),
        ]
    if options.include_trace:
        parts.append(
            _section(
                "Console trace",
                tuple(item.text for item in evidence.trace),
                12_000,
                "Console trace unavailable for the selected context.",
            )
        )
    if tools_enabled:
        parts.append(
            _section(
                "Redacted tool inputs/results",
                tuple(item.text for item in evidence.tools),
                8_000,
                "Governed tool content unavailable for this excerpt.",
            )
        )
    report = "\n".join(parts)
    if _size(report) > MAX_REPORT_BYTES:
        raise ValueError(
            "Report formatting exceeds the 60,000-byte budget; reduce the description."
        )
    return report


def prepare_bug_report_copy(text: str) -> BugReportCopyCheck:
    """Validate final edited text; request re-review when sanitization changes it."""
    try:
        byte_count = _size(text)
        if byte_count > MAX_REPORT_BYTES:
            return BugReportCopyCheck(
                "oversize",
                text,
                byte_count,
                f"Report is {byte_count:,} bytes; limit is 60,000. Shorten it before copying.",
            )
        safe = sanitize_string(text)
        safe_count = _size(safe)
    except Exception:
        return BugReportCopyCheck(
            "unavailable",
            text,
            0,
            "Report sanitization is unavailable. Your draft is retained.",
        )
    if safe_count > MAX_REPORT_BYTES:
        return BugReportCopyCheck(
            "oversize",
            safe,
            safe_count,
            f"Report is {safe_count:,} bytes after redaction; limit is 60,000.",
        )
    if safe != text:
        return BugReportCopyCheck(
            "review_required",
            safe,
            safe_count,
            "Recognized sensitive text was masked. Review it, then press Copy report again.",
        )
    return BugReportCopyCheck("ready", safe, safe_count, "Report ready to copy.")
