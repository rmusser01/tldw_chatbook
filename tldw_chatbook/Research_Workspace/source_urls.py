"""Shared URL admission for Research Workspace source intake."""

from __future__ import annotations

import unicodedata

from tldw_chatbook.Utils.input_validation import validate_url


def validate_research_source_url(value: object) -> bool:
    """Return whether ``value`` is a bounded ingest-supported web URL."""

    if not isinstance(value, str) or any(
        unicodedata.category(character) in {"Cc", "Cf"} for character in value
    ):
        return False
    return validate_url(value)
