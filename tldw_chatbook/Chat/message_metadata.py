"""Structured per-message metadata for Console transcript rows (task-2364).

Facts ABOUT a turn that are not part of what the turn says: which engine
produced it, whether it was cut off, and -- for a voice turn whose row is
created before its transcript exists -- what became of that transcript.

Why a frozen dataclass and not a dict: before this field existed, the
facts lived in UI copy. The realtime reseed builder stripped a visible
"⏹ interrupted" marker out of message content by string match, and an
input transcript that legitimately came back empty left its row empty
with nothing recording why. A free-form dict would have moved the
guessing from content parsing to key spelling; a closed dataclass with a
closed ``transcript_status`` vocabulary makes a wrong key or a typo'd
status fail where it is written.

Persisted as the LOCAL-ONLY ``messages.metadata_json`` column (schema
v31), mirroring ``provider_usage.ProviderUsage``/``usage_json``: it
describes what THIS device observed while producing the row, is never
part of a sync payload, and every device records its own.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

#: Closed vocabulary for ``MessageMetadata.transcript_status``.
#:
#: - ``""``      -- not a transcribed turn (every non-voice row).
#: - ``pending`` -- the row exists but its transcript has not arrived yet
#:   (a realtime user row is created at turn-commit so it cannot be
#:   ordered after the reply it prompted).
#: - ``final``   -- the transcript arrived and filled the row.
#: - ``empty``   -- the provider transcribed the turn and it contained no
#:   words. THE strand case: without this the row stays empty forever with
#:   nothing saying whether the user said nothing or the pipeline broke.
#: - ``failed``  -- a transcript existed but could not be written to the
#:   row (read/write failure, already logged).
TRANSCRIPT_STATUSES: frozenset[str] = frozenset(
    {"", "pending", "final", "empty", "failed"}
)


#: Closed vocabulary for the trusted source of a dynamically projected row.
TEMPLATE_KINDS: frozenset[str] = frozenset({"", "character_greeting"})


#: ``MessageMetadata.origin`` value for a row injected by the auto-wake
#: machinery (PR3a-2 Task 5): the SYSTEM-class transcript notice a finished
#: background sub-agent's completion delivery appends. Machine consumers
#: (exports, resume, any future "who wrote this row" logic) read THIS, not
#: the row's visible copy.
MESSAGE_ORIGIN_AGENT_WAKE = "agent_wake"

#: Closed vocabulary for ``MessageMetadata.origin``.
#:
#: - ``""``           -- an ordinary row (typed, streamed, or otherwise not
#:   machine-injected); every row written before this field existed.
#: - ``agent_wake``   -- a machine-injected auto-wake notice
#:   (:data:`MESSAGE_ORIGIN_AGENT_WAKE`). Never user input; a row carrying
#:   it must never be read as the user having said anything.
#:
#: Compatibility note (deliberate, local-only): ``from_json`` on an OLDER
#: build drops unknown keys, so a wake notice's origin marking is invisible
#: there -- the row degrades to a plain SYSTEM row. ``metadata_json`` is a
#: local-only column that never enters sync payloads, so the degradation is
#: confined to the device that downgraded; accepted rather than gated on a
#: schema bump.
MESSAGE_ORIGINS: frozenset[str] = frozenset({"", MESSAGE_ORIGIN_AGENT_WAKE})


CHARACTER_EMOTE_FALLBACK_REASONS: frozenset[str] = frozenset(
    {
        "",
        "no_active_pack",
        "state_unavailable",
        "asset_unavailable",
        "resolver_error",
        "parser_error",
        "heuristic_error",
        "stopped",
        "failed",
        "history_unavailable",
    }
)

_EMOTE_STATE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}\Z")
_EXPRESSION_KEY_RE = re.compile(
    r"(?:neutral|happy|excited|sad|angry|thinking|confused|surprised|"
    r"custom:[a-z0-9][a-z0-9_]{0,39})\Z"
)
_TOPIC_RE = re.compile(r"[a-z0-9]{1,40}\Z")
_CANVAS_CARD_STATUSES = frozenset({"updated", "temporary", "discarded", "failed"})
_CANVAS_ERROR_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class CanvasCardOriginMetadata:
    """Source-free assistant origin for one transcript Canvas card."""

    message_id: str
    run_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or not self.message_id
            or len(self.message_id.encode("utf-8")) > 256
        ):
            raise ValueError("Canvas card message identity is invalid")
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or len(self.run_id.encode("utf-8")) > 256
        ):
            raise ValueError("Canvas card run identity is invalid")


@dataclass(frozen=True, slots=True)
class CanvasCardMetadata:
    """Bounded metadata-only transcript projection for one Canvas revision."""

    canvas_id: str
    revision_id: str | None
    title: str
    sequence: int
    digest: str
    status: str
    origin: CanvasCardOriginMetadata
    reopenable: bool
    error_code: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.canvas_id, str)
            or not self.canvas_id
            or len(self.canvas_id.encode("utf-8")) > 256
        ):
            raise ValueError("Canvas card identity is invalid")
        if self.revision_id is not None and (
            not isinstance(self.revision_id, str)
            or not self.revision_id
            or len(self.revision_id.encode("utf-8")) > 256
        ):
            raise ValueError("Canvas card revision identity is invalid")
        if not isinstance(self.title, str) or len(self.title.encode("utf-8")) > 4096:
            raise ValueError("Canvas card title is too large")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("Canvas card sequence is invalid")
        if (
            not isinstance(self.digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None
        ):
            raise ValueError("Canvas card digest is invalid")
        if self.status not in _CANVAS_CARD_STATUSES:
            raise ValueError("Canvas card status is invalid")
        if not isinstance(self.origin, CanvasCardOriginMetadata):
            raise ValueError("Canvas card origin is invalid")
        if type(self.reopenable) is not bool:
            raise ValueError("Canvas card reopenable flag is invalid")
        if self.error_code is not None and _CANVAS_ERROR_RE.fullmatch(
            self.error_code
        ) is None:
            raise ValueError("Canvas card error code is invalid")
        if self.reopenable and (
            self.revision_id is None or self.status in {"discarded", "failed"}
        ):
            raise ValueError("Reopenable Canvas card requires a committed revision")


@dataclass(frozen=True, slots=True)
class CharacterEmoteEventMetadata:
    """One accepted safe emote state at a sanitized UTF-16 offset."""

    state: str
    at_char: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or _EMOTE_STATE_RE.fullmatch(self.state) is None:
            raise ValueError("state must be a normalized character emote slug")
        if isinstance(self.at_char, bool) or not isinstance(self.at_char, int) or self.at_char < 0:
            raise ValueError("at_char must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class CharacterEmoteMetadata:
    """Bounded local-only final expression facts for one assistant row."""

    sanitized_utf16_length: int
    mood_label: str | None = None
    mood_confidence: float | None = None
    mood_topic: str | None = None
    emote_events: tuple[CharacterEmoteEventMetadata, ...] = ()
    actor_kind: str = ""
    actor_id: int | None = None
    pack_id: int | None = None
    pack_version_id: int | None = None
    expression_key: str | None = None
    expression_id: int | None = None
    asset_id: int | None = None
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.sanitized_utf16_length, bool)
            or not isinstance(self.sanitized_utf16_length, int)
            or self.sanitized_utf16_length < 0
        ):
            raise ValueError("sanitized_utf16_length must be a nonnegative integer")
        if self.mood_label is not None and (
            not isinstance(self.mood_label, str)
            or _EMOTE_STATE_RE.fullmatch(self.mood_label) is None
        ):
            raise ValueError("mood_label must be a normalized character emote slug")
        if self.mood_confidence is not None and (
            isinstance(self.mood_confidence, bool)
            or not isinstance(self.mood_confidence, (int, float))
            or not math.isfinite(float(self.mood_confidence))
            or not 0.0 <= float(self.mood_confidence) <= 1.0
        ):
            raise ValueError("mood_confidence must be a finite value from zero to one")
        if self.mood_topic is not None and (
            not isinstance(self.mood_topic, str)
            or _TOPIC_RE.fullmatch(self.mood_topic) is None
        ):
            raise ValueError("mood_topic must be a bounded normalized topic")
        if not isinstance(self.emote_events, tuple) or len(self.emote_events) > 5:
            raise ValueError("emote_events must be a tuple of at most five events")
        previous_offset = -1
        for event in self.emote_events:
            if not isinstance(event, CharacterEmoteEventMetadata):
                raise ValueError("emote_events contains an invalid event")
            if event.at_char < previous_offset:
                raise ValueError("emote event offsets must be nondecreasing")
            if event.at_char > self.sanitized_utf16_length:
                raise ValueError("emote event offset exceeds sanitized text length")
            previous_offset = event.at_char
        if self.emote_events and self.mood_label != self.emote_events[-1].state:
            raise ValueError("mood_label must equal the final explicit emote state")
        if self.actor_kind not in {"", "character"}:
            raise ValueError("actor_kind must be blank or character")
        for name in (
            "actor_id",
            "pack_id",
            "pack_version_id",
            "expression_id",
            "asset_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive profile-local integer")
        if self.expression_key is not None and (
            not isinstance(self.expression_key, str)
            or _EXPRESSION_KEY_RE.fullmatch(self.expression_key) is None
        ):
            raise ValueError("expression_key must be a canonical bounded key")
        if self.fallback_reason not in CHARACTER_EMOTE_FALLBACK_REASONS:
            raise ValueError(
                "fallback_reason must be one of "
                f"{sorted(CHARACTER_EMOTE_FALLBACK_REASONS)}"
            )


@dataclass(frozen=True, slots=True)
class MessageMetadata:
    """Structured provenance/state facts about one transcript row.

    Attributes:
        engine: Which engine produced the row (e.g. ``"realtime"``), or
            ``""`` when unknown/not applicable.
        provider: Provider identifier the row was produced against, or
            ``""`` when unknown.
        model: Model identifier the row was produced against, or ``""``
            when unknown. For a transcribed USER row this is the
            transcription model, matching how usage is attributed.
        interrupted: True when the row's generation was cut off (realtime
            barge-in). The visible marker in the content is for the human
            reader; THIS is what machine consumers read.
        transcript_status: One of :data:`TRANSCRIPT_STATUSES`.
        template_kind: The closed kind of a trusted template source.
        template_source: The source text used by ``template_kind``.
        origin: One of :data:`MESSAGE_ORIGINS` -- ``"agent_wake"`` for a
            machine-injected auto-wake notice row, ``""`` otherwise.

    Raises:
        ValueError: If ``transcript_status`` or ``origin`` is outside its
            closed vocabulary. Refused at construction so a typo fails at
            the call site rather than silently never matching a reader.
    """

    engine: str = ""
    provider: str = ""
    model: str = ""
    interrupted: bool = False
    transcript_status: str = ""
    template_kind: str = ""
    template_source: str = ""
    origin: str = ""
    character_emote: CharacterEmoteMetadata | None = None
    canvas_cards: tuple[CanvasCardMetadata, ...] = ()

    def __post_init__(self) -> None:
        if self.transcript_status not in TRANSCRIPT_STATUSES:
            raise ValueError(
                "transcript_status must be one of "
                f"{sorted(TRANSCRIPT_STATUSES)}; got {self.transcript_status!r}"
            )
        if self.origin not in MESSAGE_ORIGINS:
            raise ValueError(
                "origin must be one of "
                f"{sorted(MESSAGE_ORIGINS)}; got {self.origin!r}"
            )
        if (
            not isinstance(self.canvas_cards, tuple)
            or len(self.canvas_cards) > 32
            or not all(isinstance(card, CanvasCardMetadata) for card in self.canvas_cards)
        ):
            raise ValueError("canvas_cards must contain at most 32 Canvas cards")
        if (
            not isinstance(self.template_kind, str)
            or self.template_kind not in TEMPLATE_KINDS
        ):
            raise ValueError(
                "template_kind must be one of "
                f"{sorted(TEMPLATE_KINDS)}; got {self.template_kind!r}"
            )
        if self.template_kind:
            if not isinstance(self.template_source, str) or not self.template_source.strip():
                raise ValueError(
                    "template_source must be a nonblank string when template_kind is set"
                )
        elif self.template_source:
            raise ValueError("template_source requires a recognized template_kind")

    @property
    def is_empty(self) -> bool:
        """Whether this carries no facts at all (every field defaulted).

        Returns:
            True when the instance is indistinguishable from "no metadata
            known" -- persistence skips writing it rather than storing a
            row of defaults.
        """
        return self == MessageMetadata()

    def to_json(self) -> str:
        """Serialize for the ``messages.metadata_json`` column.

        Returns:
            A stable (key-sorted) JSON object string.
        """
        return json.dumps(asdict(self), sort_keys=True)

    def remap_canvas_origins(self, message_ids: Mapping[str, str]) -> "MessageMetadata":
        """Return metadata with only Canvas card message origins remapped."""

        cards = tuple(
            replace(
                card,
                origin=replace(
                    card.origin,
                    message_id=message_ids.get(
                        card.origin.message_id, card.origin.message_id
                    ),
                ),
            )
            for card in self.canvas_cards
        )
        return replace(self, canvas_cards=cards)

    @classmethod
    def from_json(cls, raw: str | None) -> "MessageMetadata | None":
        """Rebuild from a stored payload, degrading instead of raising.

        This runs on the resume path against durable data that may predate
        the field, may have been written by a newer build, or may simply be
        corrupt -- none of which is worth failing a conversation load over.

        Args:
            raw: The stored ``metadata_json`` string, or ``None``.

        Returns:
            The decoded metadata, or ``None`` for a missing, non-object or
            unparseable payload. Unknown keys are dropped and an
            unrecognised ``transcript_status`` degrades to ``""`` rather
            than being passed through as if this build understood it.
        """
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        template_kind = _as_template_kind(data.get("template_kind"))
        template_source = _as_template_source(
            template_kind, data.get("template_source")
        )
        if template_kind and not template_source:
            # Provenance is a pair. Preserve unrelated metadata from a damaged
            # durable row, but do not restore a kind that has no usable source.
            template_kind = ""
        try:
            return cls(
                engine=_as_text(data.get("engine")),
                provider=_as_text(data.get("provider")),
                model=_as_text(data.get("model")),
                interrupted=_as_bool(data.get("interrupted")),
                transcript_status=_as_transcript_status(
                    data.get("transcript_status")
                ),
                template_kind=template_kind,
                template_source=template_source,
                origin=_as_origin(data.get("origin")),
                character_emote=_as_character_emote(data.get("character_emote")),
                canvas_cards=_as_canvas_cards(data.get("canvas_cards")),
            )
        except ValueError:
            # Direct construction remains strict. Stored data is an untrusted
            # compatibility boundary and must never prevent conversation load.
            return None


def _as_text(value: Any) -> str:
    return str(value) if value else ""


def _as_canvas_cards(value: Any) -> tuple[CanvasCardMetadata, ...]:
    if not isinstance(value, list) or len(value) > 32:
        return ()
    cards: list[CanvasCardMetadata] = []
    try:
        for raw in value:
            if not isinstance(raw, dict) or set(raw) != {
                "canvas_id",
                "revision_id",
                "title",
                "sequence",
                "digest",
                "status",
                "origin",
                "reopenable",
                "error_code",
            }:
                return ()
            origin = raw["origin"]
            if not isinstance(origin, dict) or set(origin) != {"message_id", "run_id"}:
                return ()
            cards.append(
                CanvasCardMetadata(
                    canvas_id=raw["canvas_id"],
                    revision_id=raw["revision_id"],
                    title=raw["title"],
                    sequence=raw["sequence"],
                    digest=raw["digest"],
                    status=raw["status"],
                    origin=CanvasCardOriginMetadata(
                        message_id=origin["message_id"], run_id=origin["run_id"]
                    ),
                    reopenable=raw["reopenable"],
                    error_code=raw["error_code"],
                )
            )
    except (KeyError, TypeError, ValueError):
        return ()
    return tuple(cards)


#: Payload spellings of a true boolean, lowercased. Anything not in here --
#: including the empty string and any unrecognised word -- reads as False.
_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "on"})


def _as_bool(value: Any) -> bool:
    """Coerce a stored payload value to a boolean without inverting it.

    Plain ``bool()`` is wrong here: every non-empty string is truthy, so a
    row whose flag was serialized as the STRING ``"false"`` -- a hand-edited
    payload, a foreign writer, a different serializer -- would restore as
    True and silently invert a durable fact on resume and in exports.

    Args:
        value: The raw value pulled out of the decoded JSON object.

    Returns:
        The boolean it denotes: a real bool as itself; an int/float by its
        own truthiness; a string by a closed token vocabulary
        (case-insensitive, whitespace-tolerant), where anything
        unrecognised is False; and every other type False, since a value
        this code cannot read is not evidence that the flag was set.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TOKENS
    return False


def _as_transcript_status(value: Any) -> str:
    status = _as_text(value)
    return status if status in TRANSCRIPT_STATUSES else ""


def _as_origin(value: Any) -> str:
    """Restore a stored origin, degrading an unrecognised one to ``""``.

    Same posture as ``_as_transcript_status``: a payload written by a NEWER
    build (a vocabulary this build does not know) must not be passed
    through as if this build understood it, and must not fail the load.
    """
    origin = _as_text(value)
    return origin if origin in MESSAGE_ORIGINS else ""


def _as_template_kind(value: Any) -> str:
    return "character_greeting" if value == "character_greeting" else ""


def _as_template_source(kind: Any, value: Any) -> str:
    if kind != "character_greeting" or not isinstance(value, str) or not value.strip():
        return ""
    return value


_CHARACTER_EMOTE_KEYS = frozenset(
    {
        "sanitized_utf16_length",
        "mood_label",
        "mood_confidence",
        "mood_topic",
        "emote_events",
        "actor_kind",
        "actor_id",
        "pack_id",
        "pack_version_id",
        "expression_key",
        "expression_id",
        "asset_id",
        "fallback_reason",
    }
)


def _as_character_emote(value: Any) -> CharacterEmoteMetadata | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - _CHARACTER_EMOTE_KEYS:
        return None
    events_value = value.get("emote_events", [])
    if not isinstance(events_value, list) or len(events_value) > 5:
        return None
    events: list[CharacterEmoteEventMetadata] = []
    try:
        for event in events_value:
            if not isinstance(event, dict) or set(event) != {"state", "at_char"}:
                return None
            events.append(
                CharacterEmoteEventMetadata(
                    state=event.get("state"),
                    at_char=event.get("at_char"),
                )
            )
        return CharacterEmoteMetadata(
            sanitized_utf16_length=value.get("sanitized_utf16_length"),
            mood_label=value.get("mood_label"),
            mood_confidence=value.get("mood_confidence"),
            mood_topic=value.get("mood_topic"),
            emote_events=tuple(events),
            actor_kind=value.get("actor_kind", ""),
            actor_id=value.get("actor_id"),
            pack_id=value.get("pack_id"),
            pack_version_id=value.get("pack_version_id"),
            expression_key=value.get("expression_key"),
            expression_id=value.get("expression_id"),
            asset_id=value.get("asset_id"),
            fallback_reason=value.get("fallback_reason", ""),
        )
    except (TypeError, ValueError):
        return None
