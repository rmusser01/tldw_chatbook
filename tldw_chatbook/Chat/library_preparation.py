"""Minimized durable disclosure for Library turn preparation outcomes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionWriter,
)


LIBRARY_PREPARATION_EVENT_KIND = "library_preparation"
LIBRARY_PREPARATION_MAX_BYTES = 1024
LIBRARY_PREPARATION_RESULT_COUNT_MAX = (1 << 31) - 1
LIBRARY_PREPARATION_SOURCE_TYPES = ("notes", "media", "conversations")

_EVENT_KEYS = (
    "version",
    "outcome",
    "attempt_id",
    "result_count",
    "source_types",
)
_DISCLOSURE_OUTCOMES = frozenset({"zero_matches", "bypassed"})
_NO_EVENT_OUTCOMES = frozenset(
    {
        "cancelled",
        "retrieval_failure",
        "persistence_failure",
        "destination_changed",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z", re.ASCII)


class LibraryPreparationValidationError(ValueError):
    """A preparation event or row violated the bounded v1 contract."""


@dataclass(frozen=True, slots=True)
class LibraryPreparationEvent:
    """The only durable outcomes produced by automatic preparation."""

    version: Literal[1]
    outcome: Literal["zero_matches", "bypassed"]
    attempt_id: str
    result_count: int
    source_types: tuple[Literal["notes", "media", "conversations"], ...]


@dataclass(frozen=True, slots=True)
class LibraryPreparationView:
    """One safe disclosure projected onto its active durable USER turn."""

    turn_id: str
    outcome: Literal["zero_matches", "bypassed"]
    result_count: int
    source_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LibraryPreparationContribution:
    """Persist one bounded event through the shared caller-owned writer."""

    event: LibraryPreparationEvent
    owner_message_key: Literal["user"] = "user"

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Persist one bounded sidecar without receiving transaction control."""
        if self.owner_message_key != "user":
            raise ValueError("Library preparation requires a USER message owner.")
        user_message_id = message_ids.get(self.owner_message_key)
        if not isinstance(user_message_id, str) or not user_message_id:
            raise ValueError("Library preparation requires a USER message owner.")
        payload_json = encode_library_preparation_event(self.event)
        sequence = writer.next_trajectory_sequence()
        writer.execute(
            "INSERT INTO message_trajectory_metadata("
            "message_id, conversation_id, turn_id, seq, event_kind, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_message_id,
                conversation_id,
                user_message_id,
                sequence,
                LIBRARY_PREPARATION_EVENT_KIND,
                payload_json,
            ),
        )


def library_preparation_event_for_outcome(
    outcome: str | None,
    *,
    attempt_id: str,
    result_count: int,
    source_types: tuple[str, ...],
) -> LibraryPreparationEvent | None:
    """Create a disclosure event, or no event for cancelled/failure outcomes."""
    if outcome is None or outcome in _NO_EVENT_OUTCOMES:
        return None
    if outcome not in _DISCLOSURE_OUTCOMES:
        raise LibraryPreparationValidationError(
            "Unsupported Library preparation outcome."
        )
    event = LibraryPreparationEvent(
        version=1,
        outcome=cast(Literal["zero_matches", "bypassed"], outcome),
        attempt_id=attempt_id,
        result_count=result_count,
        source_types=cast(
            tuple[Literal["notes", "media", "conversations"], ...],
            source_types,
        ),
    )
    _validate_event(event)
    return event


def encode_library_preparation_event(event: LibraryPreparationEvent) -> str:
    """Encode one exact v1 payload and enforce its 1024-byte storage cap."""
    _validate_event(event)
    payload = {
        "version": event.version,
        "outcome": event.outcome,
        "attempt_id": event.attempt_id,
        "result_count": event.result_count,
        "source_types": list(event.source_types),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > LIBRARY_PREPARATION_MAX_BYTES:
        raise LibraryPreparationValidationError(
            "Library preparation payload exceeds its byte limit."
        )
    return encoded


def decode_library_preparation_event(value: object) -> LibraryPreparationEvent:
    """Decode an exact bounded v1 payload, rejecting every additive field."""
    if type(value) is not str:
        raise LibraryPreparationValidationError("Invalid Library preparation payload.")
    try:
        payload_size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise LibraryPreparationValidationError(
            "Invalid Library preparation payload."
        ) from exc
    if payload_size > LIBRARY_PREPARATION_MAX_BYTES:
        raise LibraryPreparationValidationError("Invalid Library preparation payload.")

    def reject_constant(_value: str) -> None:
        raise LibraryPreparationValidationError(
            "Invalid Library preparation payload."
        )

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        decoded: dict[str, object] = {}
        for key, item in pairs:
            if key in decoded:
                raise LibraryPreparationValidationError(
                    "Invalid Library preparation payload."
                )
            decoded[key] = item
        return decoded

    try:
        data = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise LibraryPreparationValidationError(
            "Invalid Library preparation payload."
        ) from exc
    if type(data) is not dict or set(data) != set(_EVENT_KEYS):
        raise LibraryPreparationValidationError("Invalid Library preparation payload.")

    source_types = data["source_types"]
    if type(source_types) is not list:
        raise LibraryPreparationValidationError("Invalid Library preparation payload.")
    event = LibraryPreparationEvent(
        version=cast(Literal[1], data["version"]),
        outcome=cast(Literal["zero_matches", "bypassed"], data["outcome"]),
        attempt_id=cast(str, data["attempt_id"]),
        result_count=cast(int, data["result_count"]),
        source_types=cast(
            tuple[Literal["notes", "media", "conversations"], ...],
            tuple(source_types),
        ),
    )
    _validate_event(event)
    return event


def project_library_preparation(
    rows: Iterable[Any],
    active_turn_ids: Iterable[str],
) -> tuple[LibraryPreparationView, ...]:
    """Project exactly one valid preparation disclosure per active USER turn.

    Malformed, wrong-owner, or duplicate rows make that turn inert. Output order
    follows the caller's active lineage and is independent of row input order.
    """
    active_order = tuple(dict.fromkeys(active_turn_ids))
    active = frozenset(active_order)
    candidates: dict[str, tuple[int, LibraryPreparationEvent]] = {}
    corrupt: set[str] = set()

    for row in rows:
        if str(_field(row, "event_kind") or "") != LIBRARY_PREPARATION_EVENT_KIND:
            continue
        turn_id = _field(row, "turn_id")
        if type(turn_id) is not str or turn_id not in active:
            continue
        if turn_id in corrupt:
            continue
        message_id = _field(row, "message_id")
        sequence = _field(row, "seq")
        if message_id != turn_id or type(sequence) is not int or sequence < 1:
            corrupt.add(turn_id)
            continue
        try:
            event = decode_library_preparation_event(_field(row, "payload_json"))
        except LibraryPreparationValidationError:
            corrupt.add(turn_id)
            continue
        if turn_id in candidates:
            candidates.pop(turn_id)
            corrupt.add(turn_id)
            continue
        candidates[turn_id] = (sequence, event)

    views: list[LibraryPreparationView] = []
    for turn_id in active_order:
        candidate = candidates.get(turn_id)
        if turn_id in corrupt or candidate is None:
            continue
        _, event = candidate
        views.append(
            LibraryPreparationView(
                turn_id=turn_id,
                outcome=event.outcome,
                result_count=event.result_count,
                source_types=tuple(event.source_types),
            )
        )
    return tuple(views)


def _validate_event(event: LibraryPreparationEvent) -> None:
    """Validate the closed vocabulary, identifiers, counts, and categories."""
    if type(event.version) is not int or event.version != 1:
        raise LibraryPreparationValidationError("Invalid Library preparation event.")
    if type(event.outcome) is not str or event.outcome not in _DISCLOSURE_OUTCOMES:
        raise LibraryPreparationValidationError("Invalid Library preparation event.")
    if type(event.attempt_id) is not str or _IDENTIFIER_RE.fullmatch(event.attempt_id) is None:
        raise LibraryPreparationValidationError("Invalid Library preparation event.")
    if (
        type(event.result_count) is not int
        or event.result_count < 0
        or event.result_count > LIBRARY_PREPARATION_RESULT_COUNT_MAX
    ):
        raise LibraryPreparationValidationError("Invalid Library preparation event.")
    if type(event.source_types) is not tuple:
        raise LibraryPreparationValidationError("Invalid Library preparation event.")
    source_types = tuple(event.source_types)
    canonical = tuple(
        source_type
        for source_type in LIBRARY_PREPARATION_SOURCE_TYPES
        if source_type in source_types
    )
    if canonical != source_types:
        raise LibraryPreparationValidationError("Invalid Library preparation event.")


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read one row field from mappings, sqlite rows, or row-like objects."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    try:
        return obj[name]
    except Exception:  # noqa: BLE001 - row/object shapes intentionally vary
        return getattr(obj, name, default)
