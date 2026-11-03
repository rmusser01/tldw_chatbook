"""Immutable subscription briefing provenance and its URL snapshot codec."""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


def _sanitize_provenance_url(value: object) -> str | None:
    """Strip credentials, query, and fragment from one HTTP(S) snapshot URL."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if parsed.scheme.casefold() not in {"http", "https"} or not hostname:
            return None
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme.casefold(), netloc, parsed.path, "", ""))
    except ValueError:
        return None


@dataclass(frozen=True)
class BriefingProvenanceRow:
    """One selected item's immutable snapshot at briefing publication."""

    item_id: int
    selection_position: int
    citation_position: int | None
    featured: bool
    cited: bool
    item_title: str | None
    item_url: str | None
    item_published_date: str | None
    item_created_at: str | None
    item_effective_date: str | None
    source_id: int | None
    source_name: str | None
    source_type: str | None
    source_url: str | None
