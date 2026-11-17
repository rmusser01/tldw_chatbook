"""Finite session-surface values bound to one adopted read-only publication."""

from __future__ import annotations

from dataclasses import dataclass

from .console_chat_models import ConsoleRunMarker
from .console_execution_projection_view import ConsoleProjectionView


def _entry_ids(entries: object, session_ids: set[str]) -> None:
    if type(entries) is not tuple:
        raise TypeError("Console session-surface entries must be tuples")
    seen = set()
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("Console session-surface entry is invalid")
        session_id = entry[0]
        if (
            type(session_id) is not str
            or session_id not in session_ids
            or session_id in seen
        ):
            raise ValueError("Console session-surface identity is invalid")
        seen.add(session_id)
    if seen != session_ids:
        raise ValueError("Console session-surface entries do not cover its rows")


@dataclass(frozen=True, slots=True)
class ConsoleSessionSurfaceProjection:
    """Original observations only; None remains distinct from NONE or zero."""

    view: ConsoleProjectionView
    streaming_session_id: str | None
    run_markers: tuple[tuple[str, ConsoleRunMarker], ...] | None
    queue_counts: tuple[tuple[str, int], ...] | None

    def __post_init__(self) -> None:
        if type(self.view) is not ConsoleProjectionView:
            raise TypeError("Original Console projection view is required")
        session_ids = {row.id for row in self.view.session_rows}
        if self.streaming_session_id is not None and (
            type(self.streaming_session_id) is not str
            or self.streaming_session_id not in session_ids
        ):
            raise ValueError("Console streaming session is unavailable")
        if self.run_markers is not None:
            _entry_ids(self.run_markers, session_ids)
            if any(type(marker) is not ConsoleRunMarker for _, marker in self.run_markers):
                raise TypeError("Original Console run markers are required")
        if self.queue_counts is not None:
            _entry_ids(self.queue_counts, session_ids)
            if any(type(count) is not int or count < 0 for _, count in self.queue_counts):
                raise ValueError("Console queue counts must be nonnegative integers")


def encode_session_surface_projection(value: ConsoleSessionSurfaceProjection) -> dict:
    """Return the three plain JSON fields, without serializing the view.

    Args:
        value: Validated original observations bound to their adopted view.

    Returns:
        Detached JSON objects and arrays preserving absent observations.

    Raises:
        TypeError: The value is not an original surface projection.
    """
    if type(value) is not ConsoleSessionSurfaceProjection:
        raise TypeError("Original Console session-surface projection is required")
    return {
        "streaming_session_id": value.streaming_session_id,
        "run_markers": (
            None if value.run_markers is None else [
                {"session_id": session_id, "marker": marker.value}
                for session_id, marker in value.run_markers
            ]
        ),
        "queue_counts": (
            None if value.queue_counts is None else [
                {"session_id": session_id, "count": count}
                for session_id, count in value.queue_counts
            ]
        ),
    }


def decode_session_surface_projection(
    view: ConsoleProjectionView, value: dict
) -> ConsoleSessionSurfaceProjection:
    """Decode fixed plain JSON values against the exact adopted view.

    Args:
        view: Original read-only view to retain, without copying or rebuilding it.
        value: Plain JSON record, already detached from any frozen wire payload.

    Returns:
        A validated frozen surface projection with original enum leaves.

    Raises:
        TypeError: A record, collection or marker has a nonexact type.
        ValueError: Fields, identities, coverage, marker or count are invalid.
    """
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("Plain Console session-surface data is required")
    if set(value) != {"streaming_session_id", "run_markers", "queue_counts"}:
        raise ValueError("Console session-surface fields are invalid")
    markers = None
    if value["run_markers"] is not None:
        if type(value["run_markers"]) is not list:
            raise TypeError("Console run markers must be a plain array")
        markers = []
        for entry in value["run_markers"]:
            if (
                type(entry) is not dict
                or any(type(key) is not str for key in entry)
                or set(entry) != {"session_id", "marker"}
                or type(entry["marker"]) is not str
            ):
                raise TypeError("Console run-marker record is invalid")
            markers.append((entry["session_id"], ConsoleRunMarker(entry["marker"])))
        markers = tuple(markers)
    counts = None
    if value["queue_counts"] is not None:
        if type(value["queue_counts"]) is not list:
            raise TypeError("Console queue counts must be a plain array")
        counts = []
        for entry in value["queue_counts"]:
            if (
                type(entry) is not dict
                or any(type(key) is not str for key in entry)
                or set(entry) != {"session_id", "count"}
            ):
                raise TypeError("Console queue-count record is invalid")
            counts.append((entry["session_id"], entry["count"]))
        counts = tuple(counts)
    return ConsoleSessionSurfaceProjection(
        view, value["streaming_session_id"], markers, counts
    )
