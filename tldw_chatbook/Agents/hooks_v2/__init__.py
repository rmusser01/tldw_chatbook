"""Explicit v2 hook definitions, matching and result validation."""

from .matching import UnsupportedEventField, expand_input, matches_handler
from .models import HookEvent, HookHandler, HookResult
from .validation import (
    decode_native_handlers,
    decode_result,
    handler_phase,
    parse_event,
    parse_handlers,
    parse_native_handlers,
    parse_result,
)

__all__ = [
    "HookEvent",
    "HookHandler",
    "HookResult",
    "UnsupportedEventField",
    "decode_native_handlers",
    "decode_result",
    "expand_input",
    "handler_phase",
    "matches_handler",
    "parse_event",
    "parse_handlers",
    "parse_native_handlers",
    "parse_result",
]
