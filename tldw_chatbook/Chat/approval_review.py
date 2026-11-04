"""Pure native approval-summary policy shared by terminal and phone reads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.MCP.redaction import redact_mapping

from .approval_constants import (  # noqa: F401 - compatibility exports
    APPROVAL_OPTIONS,
    approval_options,
)
from .approval_formatting import (  # noqa: F401 - existing compatibility exports
    _ARGS_MIN_VALUE_LIMIT,
    _ARGS_SUMMARY_LIMIT,
    _ARGS_VALUE_LIMIT,
    _DESTINATION_TOKENS,
    _is_destination_key,
    _snake_case,
    format_redacted_arguments,
)


def remote_argument_summary(arguments: Mapping[str, Any] | None) -> str:
    """Fail closed on malformed values; phone review never uses the raw fallback."""
    if arguments is not None and not isinstance(arguments, Mapping):
        raise ValueError("Malformed approval arguments.")
    redacted = redact_mapping(dict(arguments or {}))
    json.dumps(redacted, allow_nan=False)
    return format_redacted_arguments(redacted)
