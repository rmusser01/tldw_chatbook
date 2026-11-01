"""Pure native approval-summary policy shared by terminal and phone reads."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.MCP.redaction import redact_mapping

APPROVAL_OPTIONS = (
    ("Once", "approve_once"),
    ("This session", "approve_session"),
    ("Always · these args", "allow_matching"),
    ("Always", "always_allow"),
    ("Deny", "deny"),
)


def approval_options(requested: object) -> list[tuple[str, str]]:
    """Apply the native default and supported-option filter for ordinary tools."""
    if not isinstance(requested, (list, tuple)) or not requested:
        return list(APPROVAL_OPTIONS)
    wanted = set(requested)
    narrowed = [pair for pair in APPROVAL_OPTIONS if pair[1] in wanted]
    return narrowed or list(APPROVAL_OPTIONS)


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

#: TASK-695: argument names that say WHERE a call acts. Matched as whole
#: tokens (see `_is_destination_key`), so `profile` is not a file and
#: `urinal` is not a URL.
_DESTINATION_TOKENS: frozenset[str] = frozenset(
    {
        "path",
        "paths",
        "filepath",
        "file",
        "files",
        "filename",
        "dir",
        "dirs",
        "directory",
        "folder",
        "dest",
        "destination",
        "target",
        "output",
        "out",
        "src",
        "source",
        "input",
        "url",
        "uri",
        "endpoint",
        "host",
        "hostname",
        "cmd",
        "command",
        "script",
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


def format_redacted_arguments(redacted: Mapping[str, Any]) -> str:
    """Format already-redacted native review arguments, destinations first."""
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
        share = max(
            _ARGS_MIN_VALUE_LIMIT, min(_ARGS_VALUE_LIMIT, remaining // len(payloads))
        )

    parts = [
        f"{json.dumps(str(key))}:"
        f"{_render(value, _ARGS_VALUE_LIMIT if _is_destination_key(key) else share)}"
        for key, value in ordered
    ]
    text = "{" + ",".join(parts) + "}"
    if len(text) > _ARGS_SUMMARY_LIMIT:
        return text[: _ARGS_SUMMARY_LIMIT - 1] + "…"
    return text


def remote_argument_summary(arguments: Mapping[str, Any] | None) -> str:
    """Fail closed on malformed values; phone review never uses the raw fallback."""
    if arguments is not None and not isinstance(arguments, Mapping):
        raise ValueError("Malformed approval arguments.")
    redacted = redact_mapping(dict(arguments or {}))
    json.dumps(redacted, allow_nan=False)
    return format_redacted_arguments(redacted)
