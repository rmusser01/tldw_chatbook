"""The literal item-marker convention shared by briefing generation and stored provenance."""

import re
from collections.abc import Iterator

_CITATION = re.compile(r"\[item (\d+)\]", re.IGNORECASE)


def iter_citation_ids(body_markdown: str) -> Iterator[int]:
    """Yield distinct bracketed item IDs in their first-seen order."""
    seen = set()
    for match in _CITATION.finditer(body_markdown or ""):
        item_id = int(match.group(1))
        if item_id not in seen:
            seen.add(item_id)
            yield item_id


def extract_citation_ids(body_markdown: str) -> list[int]:
    """Return the ordered item IDs cited by a briefing body."""
    return list(iter_citation_ids(body_markdown))
