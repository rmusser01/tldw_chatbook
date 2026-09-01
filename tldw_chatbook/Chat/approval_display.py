"""Shared display helpers for Console tool-approval surfaces (ADR-090).

The argument-summarization helpers below moved here VERBATIM from
``tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card`` (task 5 of the
permission-request-context-summaries plan) so the advisory context/summary
lines can reuse one clipping/label vocabulary alongside the existing
argument budgets. The card keeps backwards-compatible aliases for its
historical underscore names.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from tldw_chatbook.MCP.redaction import redact_mapping

_ARGS_SUMMARY_LIMIT = 80

#: TASK-695: per-VALUE budget inside the summary above. Without it a single
#: bulk argument (a `write_file` body, a pasted document) consumes the whole
#: line and every other argument -- including the destination the decision
#: turns on -- is clipped away. Sized so a typical path survives intact
#: while a payload is obviously abbreviated.
_ARGS_VALUE_LIMIT = 34

#: Floor for a shared value budget: below this a value is all ellipsis and
#: tells the reader nothing, so it is better to overflow the line cap (which
#: clips the tail) than to render every argument as noise.
_ARGS_MIN_VALUE_LIMIT = 10

#: ADR-090: display cap for one advisory line (tail-biased).
RATIONALE_DISPLAY_CAP = 240
CONTEXT_LABEL = "Model context:"
SUMMARY_LABEL = "Summary:"

#: ADR-090 (Qodo review #7): single named cap for tool-description capture
#: at the three pending-row producers (MCP, local, builtin) and the
#: summarizer prompt -- one constant so the egress bound cannot drift
#: between tool owners.
TOOL_DESCRIPTION_CAPTURE_CAP = 300

#: TASK-695: argument names that say WHERE a call acts. Matched as whole
#: tokens (see `_is_destination_key`), so `profile` is not a file and
#: `urinal` is not a URL.
_DESTINATION_TOKENS: frozenset[str] = frozenset(
    {
        "path", "paths", "filepath", "file", "files", "filename",
        "dir", "dirs", "directory", "folder",
        "dest", "destination", "target", "output", "out",
        "src", "source", "input",
        "url", "uri", "endpoint", "host", "hostname",
        "cmd", "command", "script",
    }
)


def _snake_case(key: Any) -> str:
    """Return ``key`` lowercased with camelCase split into ``_`` tokens.

    Takes ``Any``, not ``str``: these keys come straight from model output,
    where a malformed payload can carry a non-string key. `re.sub` raises
    TypeError on one, which used to take down the whole approval row -- an
    approval the user cannot answer, blocking the run until the auto-deny
    fires. Coerced here, at the boundary.

    Args:
        key: One argument name from a tool call, of any type.

    Returns:
        The key as a lowercase, ``_``-separated string.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()


def _is_destination_key(key: Any) -> bool:
    """Whether ``key`` names WHERE a call acts, rather than what it carries.

    TASK-695: these are the arguments an approval decision actually turns
    on -- the file being written, the URL being fetched, the command being
    run. They are hoisted ahead of bulk payloads so a large ``content`` can
    never push the destination off the end of the summary.

    Args:
        key: One argument name from the model's tool call. Any type -- see
            `_snake_case` for why this is not narrowed to ``str``.

    Returns:
        True when the name looks like a destination/target.
    """
    # Matched on TOKENS, not substrings: `profile` contains "file" and
    # `urinal` contains "uri", and a false positive reorders the line so the
    # real destination lands later in a budget-limited summary -- the exact
    # failure this hoisting exists to prevent.
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])", _snake_case(key))
        if token
    }
    return bool(tokens & _DESTINATION_TOKENS)


def summarize_arguments(arguments: Mapping[str, Any] | None) -> str:
    """Return ONE payload as a compact, ``markup=False``-safe summary.

    TASK-695: the summary used to be one ``json.dumps`` blob clipped at
    ``_ARGS_SUMMARY_LIMIT``, and ``json.dumps`` preserves the model's key
    order -- so a ``write_file`` emitting ``content`` before ``file_path``
    showed 67 characters of file body and truncated the destination out of
    view. The card asked "may I write this?" without showing where.

    Two changes fix that without raising the global cap (which would only
    move the cliff): destination-like keys are rendered FIRST, and every
    value gets its own budget so one bulk payload cannot consume the line.

    Secret-looking values (``api_key``, ``token``, ``password``, ...) are
    redacted before rendering -- redaction parity with every other MCP
    display/log boundary (see ``tldw_chatbook.MCP.redaction``'s module
    docstring); the approval card is the one place a raw secret argument
    was still reaching the screen unredacted. Redaction runs BEFORE the
    reordering and clipping below, so neither can expose a secret.
    """
    try:
        redacted = redact_mapping(dict(arguments or {}))
    except Exception:  # noqa: BLE001 -- a bad arg must never crash rendering
        return str(arguments or {})[: _ARGS_SUMMARY_LIMIT]
    if not redacted:
        return "{}"

    # Destinations first, each group keeping the model's own order so a call
    # with several paths still reads in the order it was made.
    ordered = sorted(redacted.items(), key=lambda kv: not _is_destination_key(kv[0]))

    def _render(value: Any, budget: int) -> str:
        try:
            text = json.dumps(value, default=str, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            text = json.dumps(str(value))
        if len(text) > budget:
            return text[: max(1, budget - 1)] + "…"
        return text

    # Destinations are rendered first and at the full per-value budget; what
    # they leave is split evenly among the remaining arguments. Without the
    # split, the second bulk argument still starves everything after it --
    # a fixed per-value cap only moves the cliff along by one key.
    destinations = [kv for kv in ordered if _is_destination_key(kv[0])]
    payloads = [kv for kv in ordered if not _is_destination_key(kv[0])]
    overhead = sum(len(json.dumps(str(key))) + 2 for key, _ in ordered) + 2
    spent = sum(len(_render(v, _ARGS_VALUE_LIMIT)) for _, v in destinations)
    share = _ARGS_VALUE_LIMIT
    if payloads:
        remaining = _ARGS_SUMMARY_LIMIT - overhead - spent
        share = max(_ARGS_MIN_VALUE_LIMIT, min(_ARGS_VALUE_LIMIT, remaining // len(payloads)))

    parts = [
        f"{json.dumps(str(key))}:"
        f"{_render(value, _ARGS_VALUE_LIMIT if _is_destination_key(key) else share)}"
        for key, value in ordered
    ]
    text = "{" + ",".join(parts) + "}"
    if len(text) > _ARGS_SUMMARY_LIMIT:
        return text[: _ARGS_SUMMARY_LIMIT - 1] + "…"
    return text


def summarize_row_arguments(entry: Mapping[str, Any]) -> str:
    """Return the summary for one COLLAPSED row -- every call's arguments.

    TASK-1845: a row that says "x3" must show all three targets or the count
    is concealing the decision, so each grouped call's payload is rendered on
    its own line and capped independently (one long payload cannot push the
    others off screen). Redaction applies to every line, not just the first.

    Takes the collapsed ENTRY, not an arguments mapping. The two shapes were
    once distinguished by sniffing for an ``all_arguments`` key inside the
    arguments themselves, which both mis-fires on a tool that genuinely has
    an argument by that name and -- as shipped -- silently did nothing,
    because the render site passed the first call's arguments and the branch
    never ran.
    """
    sets = entry.get("all_arguments")
    if not sets:
        # Not a collapsed entry (or a row with no arguments at all): fall
        # back to the single payload so a caller can never render nothing.
        return summarize_arguments(entry.get("arguments"))
    rendered = [summarize_arguments(payload) for payload in sets]
    # De-duplicate identical payloads while preserving order: N identical
    # calls are one decision with one target, and repeating it N times
    # would bury a genuinely different target further down.
    seen: set[str] = set()
    unique = [r for r in rendered if not (r in seen or seen.add(r))]
    return "\n".join(unique)


def format_context_line(text: object, cap: int = RATIONALE_DISPLAY_CAP) -> str:
    """Tail-biased display clip for one advisory context/summary line.

    Args:
        text: Raw advisory text (model rationale or summarizer output).
        cap: Maximum rendered length including the ellipsis.

    Returns:
        The clipped line, or "" for blank/absent input.
    """
    from tldw_chatbook.Agents.agent_models import normalize_rationale

    return normalize_rationale(text, cap=cap)
