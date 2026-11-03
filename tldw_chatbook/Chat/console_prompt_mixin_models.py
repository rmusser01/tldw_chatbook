"""Immutable operational mix-in values and lossless preference compatibility."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal
from uuid import UUID

from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LOCAL_COMPILED_LANE_LIMIT,
    LOCAL_DEFINITION_LIMIT,
    LOCAL_REQUEST_LIMIT,
    LiteralMixinDefinition,
)
from tldw_chatbook.Utils.input_validation import strict_json_loads

MixinPlacement = Literal["before", "after"]


class MixinPreferenceConflictError(ValueError):
    """The exact preference revision or understood format is unavailable."""


class MixinSnapshotUnavailableError(ValueError):
    """Exact operational snapshot ownership or content is unavailable."""


class MixinSourceChangedError(ValueError):
    """The source authority changed while resolving a selection."""


def _text(value: object) -> None:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("mixin_invalid_text")
    try:
        if len(value.encode("utf-8")) > LOCAL_REQUEST_LIMIT:
            raise ValueError("mixin_text_limit")
    except UnicodeError:
        raise ValueError("mixin_invalid_text") from None


def _integer(value: object, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value < (1 << 63):
        raise ValueError("mixin_invalid_integer")


def _json(value: object, limit: int = LOCAL_REQUEST_LIMIT) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(raw.encode("utf-8")) > limit:
            raise ValueError("mixin_json_limit")
        return raw
    except (TypeError, UnicodeError, RecursionError):
        raise ValueError("mixin_invalid_json") from None


@dataclass(frozen=True)
class MixinSourceRef:
    authority_key: str
    backend: Literal["local", "server"]
    record_uuid: str

    def __post_init__(self) -> None:
        _text(self.authority_key)
        if any(token in self.authority_key for token in ("://", "@", "?", "#")):
            raise ValueError("mixin_invalid_authority")
        if type(self.backend) is not str or self.backend not in ("local", "server"):
            raise ValueError("mixin_invalid_backend")
        try:
            if (
                type(self.record_uuid) is not str
                or str(UUID(self.record_uuid)) != self.record_uuid
            ):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError("mixin_invalid_uuid") from None


@dataclass(frozen=True)
class MixinSelection:
    source: MixinSourceRef
    enabled: bool
    name: str = field(repr=False)
    placement: MixinPlacement
    last_used_version: int | None = None

    def __post_init__(self) -> None:
        if type(self.source) is not MixinSourceRef or type(self.enabled) is not bool:
            raise ValueError("mixin_invalid_selection")
        _text(self.name)
        if type(self.placement) is not str or self.placement not in ("before", "after"):
            raise ValueError("mixin_invalid_placement")
        if self.last_used_version is not None:
            _integer(self.last_used_version, 1)


@dataclass(frozen=True)
class MixinPreferences:
    schema_version: int = 1
    revision: int = 0
    entries: tuple[MixinSelection, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("mixin_invalid_preference_schema")
        _integer(self.revision)
        if type(self.entries) is not tuple or any(
            type(item) is not MixinSelection for item in self.entries
        ):
            raise ValueError("mixin_invalid_entries")
        if len({item.source for item in self.entries}) != len(self.entries):
            raise ValueError("mixin_duplicate_reference")
        _json(asdict(self))


@dataclass(frozen=True)
class MixinSnapshot:
    source: MixinSourceRef
    record_version: int
    name: str = field(repr=False)
    definition: LiteralMixinDefinition = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.source) is not MixinSourceRef
            or type(self.definition) is not LiteralMixinDefinition
        ):
            raise ValueError("mixin_invalid_snapshot")
        _integer(self.record_version, 1)
        _text(self.name)
        if len(self.definition.body) > LOCAL_COMPILED_LANE_LIMIT:
            raise ValueError("mixin_body_limit")
        _json(asdict(self.definition), LOCAL_DEFINITION_LIMIT)


@dataclass(frozen=True)
class MixinAttemptSnapshot:
    items: tuple[MixinSnapshot, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or any(
            type(item) is not MixinSnapshot for item in self.items
        ):
            raise ValueError("mixin_invalid_items")
        if len({item.source for item in self.items}) != len(self.items):
            raise ValueError("mixin_duplicate_reference")


@dataclass(frozen=True)
class MixinAttemptWrite:
    attempt_id: str
    user_message_id: str
    assistant_message_id: str
    snapshot: MixinAttemptSnapshot = field(repr=False)
    retry_of_attempt_id: str | None = None
    replaces_attempt_id: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        for value in (self.attempt_id, self.user_message_id, self.assistant_message_id):
            _text(value)
        for predecessor in (self.retry_of_attempt_id, self.replaces_attempt_id):
            if predecessor is not None:
                _text(predecessor)
                if predecessor == self.attempt_id:
                    raise ValueError("mixin_retry_cycle")
        if (
            type(self.snapshot) is not MixinAttemptSnapshot
            or self.user_message_id == self.assistant_message_id
        ):
            raise ValueError("mixin_invalid_attempt")


@dataclass(frozen=True)
class MixinPreferenceRead:
    status: Literal["absent", "supported", "unsupported", "corrupt"]
    preferences: MixinPreferences | None = field(repr=False)
    raw: object = field(repr=False)


@dataclass(frozen=True)
class MixinReplayRead:
    status: Literal["legacy", "recorded", "unavailable"]
    snapshot: MixinAttemptSnapshot | None = field(repr=False)


@dataclass(frozen=True)
class MixinForkAttempt:
    """Historical receipt; a predecessor may lie outside the copied prefix."""

    attempt_id: str
    user_message_id: str | None
    assistant_message_id: str | None
    retry_of_attempt_id: str | None
    snapshot: MixinAttemptSnapshot = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.attempt_id)
        for anchor in (
            self.user_message_id,
            self.assistant_message_id,
            self.retry_of_attempt_id,
        ):
            if anchor is not None:
                _text(anchor)
        if type(self.snapshot) is not MixinAttemptSnapshot:
            raise ValueError("mixin_invalid_fork_snapshot")


@dataclass(frozen=True)
class MixinForkState:
    """Immutable history, with exact body-free markers keyed by message identity."""

    attempts: tuple[MixinForkAttempt, ...] = field(default=(), repr=False)
    markers: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if type(self.attempts) is not tuple or any(
            type(row) is not MixinForkAttempt for row in self.attempts
        ):
            raise ValueError("mixin_invalid_fork_attempts")
        ids = {row.attempt_id for row in self.attempts}
        if len(ids) != len(self.attempts) or type(self.markers) is not tuple:
            raise ValueError("mixin_invalid_fork_state")
        markers = set()
        for entry in self.markers:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("mixin_invalid_fork_marker")
            message_id, raw = entry
            if type(raw) is not str:
                raise ValueError("mixin_invalid_fork_marker")
            _text(message_id)
            value = strict_json_loads(raw)
            if (
                type(value) is not dict
                or set(value) - {"mixin_attempt_id", "mixin_snapshot_schema_version"}
                or message_id in markers
            ):
                raise ValueError("mixin_invalid_fork_marker")
            markers.add(message_id)
        by_id = {row.attempt_id: row for row in self.attempts}
        for row in self.attempts:
            seen = {row.attempt_id}
            predecessor = row.retry_of_attempt_id
            while predecessor is not None:
                if predecessor in seen or predecessor not in by_id:
                    raise ValueError("mixin_invalid_fork_lineage")
                seen.add(predecessor)
                predecessor = by_id[predecessor].retry_of_attempt_id

    def replay(self, message_id: str) -> MixinReplayRead:
        raw = dict(self.markers).get(message_id)
        if raw is None:
            return MixinReplayRead("unavailable", None)
        marker = strict_json_loads(raw)
        linked = [
            row for row in self.attempts if row.assistant_message_id == message_id
        ]
        if not marker:
            return MixinReplayRead("unavailable" if linked else "legacy", None)
        if (
            type(marker.get("mixin_snapshot_schema_version")) is not int
            or marker.get("mixin_snapshot_schema_version") != 1
        ):
            return MixinReplayRead("unavailable", None)
        selected = next(
            (row for row in linked if row.attempt_id == marker.get("mixin_attempt_id")),
            None,
        )
        return (
            MixinReplayRead("recorded", selected.snapshot)
            if selected is not None
            else MixinReplayRead("unavailable", None)
        )

    def remap(self, message_ids: Mapping[str, str]) -> MixinForkState:
        """Retain the selected prefix and its predecessors without inventing anchors."""
        from dataclasses import replace

        by_id = {row.attempt_id: row for row in self.attempts}
        selected = {
            row.attempt_id
            for row in self.attempts
            if row.assistant_message_id in message_ids
        }
        pending = list(selected)
        while pending:
            predecessor = by_id[pending.pop()].retry_of_attempt_id
            if predecessor is not None and predecessor not in selected:
                selected.add(predecessor)
                pending.append(predecessor)
        return MixinForkState(
            tuple(
                replace(
                    row,
                    user_message_id=message_ids.get(row.user_message_id),
                    assistant_message_id=message_ids.get(row.assistant_message_id),
                )
                for row in self.attempts
                if row.attempt_id in selected
            ),
            tuple(
                (message_ids[key], raw)
                for key, raw in self.markers
                if key in message_ids
            ),
        )


def decode_mixin_preferences(raw: object) -> MixinPreferenceRead:
    """Preserve unknown/corrupt values; only absence means empty selections."""
    if raw is None:
        return MixinPreferenceRead("absent", MixinPreferences(), raw)
    try:
        if type(raw) is str and len(raw.encode("utf-8")) > LOCAL_REQUEST_LIMIT:
            raise ValueError
        value = strict_json_loads(raw) if type(raw) is str else raw
        if type(raw) is not str:
            _json(raw)
        if type(value) is not dict or type(value.get("schema_version")) is not int:
            raise ValueError
        if value["schema_version"] != 1:
            return MixinPreferenceRead("unsupported", None, raw)
        if (
            set(value) != {"schema_version", "revision", "entries"}
            or type(value["entries"]) is not list
        ):
            raise ValueError
        entries = []
        for entry in value["entries"]:
            if type(entry) is not dict or set(entry) != {
                "source",
                "enabled",
                "name",
                "placement",
                "last_used_version",
            }:
                raise ValueError
            source = entry["source"]
            if type(source) is not dict or set(source) != {
                "authority_key",
                "backend",
                "record_uuid",
            }:
                raise ValueError
            entries.append(
                MixinSelection(**{**entry, "source": MixinSourceRef(**source)})
            )
        preferences = MixinPreferences(
            value["schema_version"], value["revision"], tuple(entries)
        )
        return MixinPreferenceRead("supported", preferences, raw)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return MixinPreferenceRead("corrupt", None, raw)


def encode_mixin_preferences(value: MixinPreferences) -> str:
    """Encode one validated content-free selection document."""
    if type(value) is not MixinPreferences:
        raise ValueError("mixin_invalid_preferences")
    return _json(asdict(value))
