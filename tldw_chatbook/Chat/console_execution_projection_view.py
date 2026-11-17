"""Finite read-only Console values; no local store or execution authority."""

from __future__ import annotations

from dataclasses import dataclass

from .console_chat_controller import ConsolePendingDecisionProjection
from .console_chat_models import (
    ConsoleChatMessage,
    ConsoleRunState,
    ConsoleRunStatus,
    copy_console_message,
)
from .console_control_dispatch import ConsoleProjection
from .console_execution_projection_values import (
    _SESSION,
    ConsoleSessionView,
    _record,
)


class ConsoleProjectionUnavailable(LookupError):
    """The publication contains no transcript frame for the requested session."""


def _count(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Console projection revision or count is invalid")


@dataclass(frozen=True, slots=True)
class ConsoleProjectionView:
    """One published set of inert rows and explicitly supplied transcript frames.

    Epoch, visit, physical retirement and adoption belong to the real client.
    Message records are decoded render values; methods never read native state.
    """

    source_revision: int
    session_rows: tuple[ConsoleSessionView, ...]
    active_session_id: str | None
    fingerprints: tuple[tuple[str, tuple[int, int]], ...]
    projections: tuple[ConsoleProjection, ...]

    def __post_init__(self) -> None:
        _count(self.source_revision)
        if any(
            type(values) is not tuple
            for values in (self.session_rows, self.fingerprints, self.projections)
        ):
            raise TypeError("Console projection collections must be original tuples")
        rows = {}
        for row in self.session_rows:
            # Reuse the existing finite row schema; never traverse a session graph.
            _record(row, ConsoleSessionView, _SESSION, False)
            if not row.id or not row.incarnation_id or row.id in rows:
                raise ValueError("Console projection session identity is invalid")
            rows[row.id] = row
        if self.active_session_id is not None and (
            type(self.active_session_id) is not str
            or self.active_session_id not in rows
        ):
            raise ValueError("Console projection active session is unavailable")

        fingerprint_ids = set()
        for entry in self.fingerprints:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("Console projection fingerprint entry is invalid")
            session_id, fingerprint = entry
            if (
                type(session_id) is not str
                or session_id not in rows
                or session_id in fingerprint_ids
            ):
                raise ValueError("Console projection fingerprint identity is invalid")
            if type(fingerprint) is not tuple or len(fingerprint) != 2:
                raise TypeError("Console projection fingerprint is invalid")
            for value in fingerprint:
                _count(value)
            fingerprint_ids.add(session_id)
        if fingerprint_ids != rows.keys():
            raise ValueError("Console projection fingerprints do not cover its rows")

        frame_ids = set()
        for frame in self.projections:
            if type(frame) is not ConsoleProjection:
                raise TypeError("Original Console projection frame is required")
            _count(frame.source_revision)
            _count(frame.display_revision)
            if (
                type(frame.session_id) is not str
                or frame.session_id not in rows
                or frame.session_id in frame_ids
                or frame.source_revision != self.source_revision
                or type(frame.incarnation_id) is not str
                or frame.incarnation_id != rows[frame.session_id].incarnation_id
            ):
                raise ValueError("Console projection frame identity changed")
            if type(frame.messages) is not tuple or any(
                type(message) is not ConsoleChatMessage for message in frame.messages
            ):
                raise TypeError("Original decoded Console messages are required")
            if (
                type(frame.run_state) is not ConsoleRunState
                or type(frame.run_state.status) is not ConsoleRunStatus
                or type(frame.run_state.visible_copy) is not str
            ):
                raise TypeError("Original decoded Console run state is required")
            if frame.decision is not None and (
                type(frame.decision) is not ConsolePendingDecisionProjection
                or frame.decision.session_id != frame.session_id
            ):
                raise ValueError("Console projection decision session changed")
            frame_ids.add(frame.session_id)

    def sessions(self) -> list[ConsoleSessionView]:
        """Return original frozen rows in their supplied creation order."""
        return list(self.session_rows)

    def session(self, session_id: str) -> ConsoleSessionView:
        """Return the exact frozen row, raising KeyError when it is absent."""
        for row in self.session_rows:
            if row.id == session_id:
                return row
        raise KeyError(session_id)

    def _projection_for_session(self, session_id: str) -> ConsoleProjection:
        for frame in self.projections:
            if frame.session_id == session_id:
                return frame
        raise ConsoleProjectionUnavailable(session_id)

    def messages_for_session(self, session_id: str) -> list[ConsoleChatMessage]:
        """Return original shallow rendering copies from the supplied frame.

        Raises:
            ConsoleProjectionUnavailable: No frame was supplied for this session.
        """
        frame = self._projection_for_session(session_id)
        return [copy_console_message(message) for message in frame.messages]

    def session_fingerprint(self, session_id: str) -> tuple[int, int]:
        """Return the supplied count/last-visible-length pair, or raise KeyError.

        Equal-length edits and non-final message changes may be invisible. This
        original cheap hint is neither complete snapshot nor source authority.
        """
        for original_id, fingerprint in self.fingerprints:
            if original_id == session_id:
                return fingerprint
        raise KeyError(session_id)

    def display_projection_revision(self, session_id: str) -> int:
        """Return the supplied frame revision, refusing an unavailable frame."""
        return self._projection_for_session(session_id).display_revision
