"""Pure Library rail preference contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tldw_chatbook.Library.library_content_evidence import (
    LIBRARY_CONTENT_SOURCES,
    LibraryContentEvidence,
)
from tldw_chatbook.Utils.Utils import coerce_bool_flag

LIBRARY_RAIL_SECTION_IDS = (
    "browse",
    "artifacts",
    "create",
    "study",
    "ingest",
    "details",
)


_LIBRARY_CONTENT_SOURCE_COUNT = len(LIBRARY_CONTENT_SOURCES)


class LibraryLifecycle(str, Enum):
    """Persisted Library progressive-disclosure lifecycle."""

    UNKNOWN = "unknown"
    STARTER = "starter"
    EXPANDED = "expanded"
    GRADUATED = "graduated"


@dataclass(frozen=True)
class LibraryRailPreferences:
    """Persisted open/collapsed preferences for Library rail sections."""

    browse_open: bool = True
    artifacts_open: bool = True
    create_open: bool = True
    # F-017: the Study handoff rows live in their own section (they open
    # the Study destination; they create nothing).
    study_open: bool = True
    ingest_open: bool = True
    details_open: bool = False


def coerce_library_rail_preferences(raw: Any) -> LibraryRailPreferences:
    """Normalize stored Library rail preferences.

    Args:
        raw: Dict-like stored value from ``library.rail_state``.

    Returns:
        Preferences with invalid or missing fields replaced by defaults.
    """
    defaults = LibraryRailPreferences()
    if not isinstance(raw, dict):
        return defaults
    return LibraryRailPreferences(
        browse_open=coerce_bool_flag(raw.get("browse_open"), defaults.browse_open),
        artifacts_open=coerce_bool_flag(raw.get("artifacts_open"), defaults.artifacts_open),
        create_open=coerce_bool_flag(raw.get("create_open"), defaults.create_open),
        study_open=coerce_bool_flag(raw.get("study_open"), defaults.study_open),
        ingest_open=coerce_bool_flag(raw.get("ingest_open"), defaults.ingest_open),
        details_open=coerce_bool_flag(raw.get("details_open"), defaults.details_open),
    )


def coerce_library_lifecycle(
    raw: Any,
    *,
    is_new_profile: bool,
) -> LibraryLifecycle:
    """Normalize a stored Library lifecycle value.

    Args:
        raw: Stored lifecycle value, or ``None`` when absent.
        is_new_profile: Whether the profile was created in the current run.

    Returns:
        The stored lifecycle, or the safe default for absent/corrupt storage.

    An absent value defaults to EXPANDED so a returning profile does not flash
    the starter rail; the screen demotes an unstored EXPANDED once the
    seven-source evidence settles all-EMPTY (task-32349).
    """
    if raw is None:
        if is_new_profile:
            return LibraryLifecycle.UNKNOWN
        return LibraryLifecycle.EXPANDED
    try:
        return LibraryLifecycle(raw)
    except (TypeError, ValueError):
        return LibraryLifecycle.EXPANDED


def serialize_library_lifecycle(lifecycle: LibraryLifecycle) -> str:
    """Serialize a Library lifecycle to its stable string value."""
    return lifecycle.value


def _validated_evidence(
    evidence: Sequence[LibraryContentEvidence],
) -> tuple[LibraryContentEvidence, ...]:
    values = tuple(evidence)
    if len(values) != _LIBRARY_CONTENT_SOURCE_COUNT:
        raise ValueError("Library lifecycle evidence requires exactly seven sources")
    if not all(isinstance(value, LibraryContentEvidence) for value in values):
        raise TypeError("evidence values must be LibraryContentEvidence")
    return values


def aggregate_library_lifecycle(
    lifecycle: LibraryLifecycle,
    evidence: Sequence[LibraryContentEvidence],
) -> LibraryLifecycle:
    """Apply one authoritative seven-source evidence snapshot to a lifecycle."""
    values = _validated_evidence(evidence)
    if LibraryContentEvidence.HAS_USER_CONTENT in values:
        return LibraryLifecycle.GRADUATED
    if lifecycle is LibraryLifecycle.UNKNOWN and all(
        value is LibraryContentEvidence.EMPTY for value in values
    ):
        return LibraryLifecycle.STARTER
    return lifecycle


def explore_library_lifecycle(lifecycle: LibraryLifecycle) -> LibraryLifecycle:
    """Expand the Library after an explicit Explore action."""
    if lifecycle in (LibraryLifecycle.UNKNOWN, LibraryLifecycle.STARTER):
        return LibraryLifecycle.EXPANDED
    return lifecycle


def return_library_lifecycle_to_starter(
    lifecycle: LibraryLifecycle,
    evidence: Sequence[LibraryContentEvidence],
) -> LibraryLifecycle:
    """Return an expanded Library to Starter when all sources are empty."""
    values = _validated_evidence(evidence)
    if lifecycle is LibraryLifecycle.EXPANDED and all(
        value is LibraryContentEvidence.EMPTY for value in values
    ):
        return LibraryLifecycle.STARTER
    return lifecycle


def serialize_library_rail_preferences(
    preferences: LibraryRailPreferences,
) -> dict[str, bool]:
    """Serialize Library rail preferences to the persistence shape.

    Args:
        preferences: Rail open/collapsed state to persist.

    Returns:
        Dict of boolean ``*_open`` flags suitable for ``library.rail_state``.
    """
    return {
        "browse_open": bool(preferences.browse_open),
        "artifacts_open": bool(preferences.artifacts_open),
        "create_open": bool(preferences.create_open),
        "study_open": bool(preferences.study_open),
        "ingest_open": bool(preferences.ingest_open),
        "details_open": bool(preferences.details_open),
    }
