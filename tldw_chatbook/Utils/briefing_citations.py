"""The literal item-marker convention shared by briefing generation and stored provenance."""

import re
from collections.abc import Iterator
from unicodedata import decimal

_CITATION = re.compile(r"\[item (\d+)\]", re.IGNORECASE)


def iter_citation_ids(body_markdown: str) -> Iterator[int | None]:
    """Yield distinct markers in order, with a placeholder for each unrepresentable ID."""
    seen = set()
    for match in _CITATION.finditer(body_markdown or ""):
        digits = match.group(1)
        if not digits.isascii():
            digits = "".join(str(decimal(char)) for char in digits)
        digits = digits.lstrip("0") or "0"
        if digits not in seen:
            seen.add(digits)
            number = int(digits) if len(digits) <= 19 else None
            yield number if number is not None and number < 2**63 else None


def extract_citation_ids(body_markdown: str) -> list[int]:
    """Return the ordered item IDs cited by a briefing body."""
    return [item for item in iter_citation_ids(body_markdown) if item is not None]
