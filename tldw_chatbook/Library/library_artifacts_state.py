"""Immutable, namespaced read contracts for Library's artifact inventory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

ArtifactSource = Literal["chatbook", "live_report", "kept_report"]
ArtifactView = Literal["all", "chatbooks", "reports"]
ArtifactSort = Literal["newest", "title"]
ReadDirection = Literal["after", "before"]
ArtifactOrderKey = tuple[int | str, ArtifactSource, int]
ARTIFACT_PAGE_SIZE = 20
MISSING_TIMESTAMP_ORDER = 9223372036854775807


@dataclass(frozen=True)
class ArtifactKey:
    source: ArtifactSource
    native_id: int

    def __post_init__(self) -> None:
        if self.source not in ("chatbook", "live_report", "kept_report"):
            raise ValueError("Unknown artifact source")
        if type(self.native_id) is not int or self.native_id <= 0:
            raise ValueError("Artifact ID must be a positive integer")


@dataclass(frozen=True)
class ArtifactScope:
    view: ArtifactView = "reports"
    query: str = ""
    sort: ArtifactSort = "newest"
    kept_only: bool = False

    def __post_init__(self) -> None:
        if self.view not in ("all", "chatbooks", "reports"):
            raise ValueError("Unknown artifact view")
        if self.sort not in ("newest", "title"):
            raise ValueError("Unknown artifact sort")
        if not isinstance(self.query, str) or type(self.kept_only) is not bool:
            raise ValueError("Invalid artifact filter")
        object.__setattr__(self, "query", self.query.strip())


@dataclass(frozen=True)
class ArtifactSummary:
    key: ArtifactKey
    order_key: ArtifactOrderKey
    title: str
    source_label: str
    copy_label: str
    status: str
    type_label: str
    revision: str
    created_at: str = ""


@dataclass(frozen=True)
class ArtifactSourceWindow:
    items: tuple[ArtifactSummary, ...]
    total: int
    before_boundary: int
    equal_boundary: int


@dataclass(frozen=True)
class ArtifactPage:
    scope: ArtifactScope
    items: tuple[ArtifactSummary, ...]
    total: int
    start: int


@dataclass(frozen=True)
class ArtifactDetail:
    key: ArtifactKey
    revision: str
    body: str
    truncated: bool
    can_keep: bool
    can_export: bool
    can_play: bool
    can_share: bool
    source_available: bool
    details: tuple[tuple[str, str], ...]
    source_conversation_id: str | None = None
    source_message_id: str | None = None


def validate_artifact_window(
    scope: ArtifactScope,
    boundary: ArtifactOrderKey | None,
    direction: ReadDirection,
    limit: int,
    inclusive: bool = False,
) -> None:
    """Validate the shared owner boundary before executing source SQL."""
    if not isinstance(scope, ArtifactScope):
        raise TypeError("Expected ArtifactScope")
    if type(limit) is not int or not 1 <= limit <= ARTIFACT_PAGE_SIZE:
        raise ValueError("Artifact limit must be an integer in 1..20")
    if direction not in ("after", "before") or type(inclusive) is not bool:
        raise ValueError("Invalid artifact window direction or inclusion")
    if boundary is not None:
        if not isinstance(boundary, tuple) or len(boundary) != 3:
            raise ValueError("Invalid artifact boundary")
        ArtifactKey(boundary[1], boundary[2])
        expected = int if scope.sort == "newest" else str
        if type(boundary[0]) is not expected:
            raise ValueError("Boundary must match the requested sort")


def report_summary(row: dict, source: ArtifactSource) -> ArtifactSummary:
    """Project an owner's body-free metadata row into its immutable summary."""
    key = ArtifactKey(source, row["id"])
    revision = hashlib.sha256(
        json.dumps(
            {name: value for name, value in row.items() if name != "order_value"},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return ArtifactSummary(
        key=key,
        order_key=(row["order_value"], source, key.native_id),
        title=row["title"],
        source_label=row["title"],
        copy_label="Live" if source == "live_report" else "Kept",
        status=row["status"],
        type_label="Report",
        revision=revision,
        created_at=str(
            row.get("created_at")
            or row.get("original_created_at")
            or row.get("kept_at")
            or ""
        ),
    )


def chatbook_actions(*, is_saved_response: bool, usable_zip: bool) -> frozenset[str]:
    """Derive actions from a usable export, never from the Chatbook label alone.

    Saved responses and pack records both remain previewable/manageable. A
    saved-response label does not imply an export; only an existing usable ZIP
    enables sharing. Source navigation is resolved separately against its owner.
    """
    actions = {"preview", "manage_packs"}
    if usable_zip:
        actions.add("share")
    return frozenset(actions)
