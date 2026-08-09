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
from dataclasses import asdict, dataclass
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

    Raises:
        ValueError: If ``transcript_status`` is outside the closed
            vocabulary. Refused at construction so a typo fails at the
            call site rather than silently never matching a reader.
    """

    engine: str = ""
    provider: str = ""
    model: str = ""
    interrupted: bool = False
    transcript_status: str = ""
    template_kind: str = ""
    template_source: str = ""

    def __post_init__(self) -> None:
        if self.transcript_status not in TRANSCRIPT_STATUSES:
            raise ValueError(
                "transcript_status must be one of "
                f"{sorted(TRANSCRIPT_STATUSES)}; got {self.transcript_status!r}"
            )
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
        return cls(
            engine=_as_text(data.get("engine")),
            provider=_as_text(data.get("provider")),
            model=_as_text(data.get("model")),
            interrupted=_as_bool(data.get("interrupted")),
            transcript_status=_as_transcript_status(data.get("transcript_status")),
            template_kind=_as_template_kind(data.get("template_kind")),
            template_source=_as_template_source(
                data.get("template_kind"), data.get("template_source")
            ),
        )


def _as_text(value: Any) -> str:
    return str(value) if value else ""


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


def _as_template_kind(value: Any) -> str:
    return "character_greeting" if value == "character_greeting" else ""


def _as_template_source(kind: Any, value: Any) -> str:
    if kind != "character_greeting" or not isinstance(value, str) or not value.strip():
        return ""
    return value
