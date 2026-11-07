"""The one-time citation-records answer for upgraded installs (TASK-33620.8.3).

Owner decision 2026-10-30. Citation records ship on (TASK-33620.8), but first
run writes the whole default config into a new profile's config.toml, so every
install created while the old default stood (from 2026-07-23) still carries
``[rag_citations] canonical_writes_enabled = false``. Those installs are asked
once, on the first reply sent with Library evidence, "Keep a record of which
sources replies cite?". The answer is kept in ``citation_records_choice.json``
beside config.toml, which is never rewritten for it.

Precedence (``canonical_writes_enabled_with_answer``):

* ``true`` or a missing key in config.toml: records on; the answer is ignored.
* exactly ``false`` (the value first run wrote before the default changed):
  a saved Keep turns records on; Don't keep, no answer or an unreadable answer
  leaves them off.
* anything else (``"false"``, ``0``, a malformed value or section): off, and
  never asked about.

Only the exact ``false`` can be the old shipped value, and only a profile
without citation records of its own is asked: a profile that has recorded
citations had records on, so a ``false`` it carries now was set on purpose.

The answer file is an ADR-126 settings-file owner (``citations.records_choice``);
every read and write goes through the raw participant boundary.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.Utils.timestamps import utc_now_iso

#: The answer file, beside the effective config.toml.
ANSWER_FILENAME = "citation_records_choice.json"
#: The raw participant route for the answer file (settings-file owner).
ROUTE = "citation_records_choice"
_SCHEMA_VERSION = 1


class CitationRecordsAnswer(str, Enum):
    """The user's answer to "Keep a record of which sources replies cite?"."""

    KEEP = "keep"
    OFF = "off"


class ConfiguredSwitch(str, Enum):
    """What config.toml's ``canonical_writes_enabled`` says, before any answer."""

    ON = "on"
    OLD_SHIPPED_OFF = "old_shipped_off"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class AnswerRead:
    """One read of the answer file.

    Attributes:
        answer: The saved answer, or ``None`` when there is none to use.
        readable: ``False`` when a file exists but cannot be read or parsed;
            such a profile is neither on by answer nor asked again.
    """

    answer: CitationRecordsAnswer | None
    readable: bool = True


def answer_file_path() -> Path:
    """Return the answer file beside the effective config.toml."""
    from tldw_chatbook.config import _get_effective_config_path

    return _get_effective_config_path().parent / ANSWER_FILENAME


def configured_switch() -> ConfiguredSwitch:
    """Classify config.toml's citation-records switch.

    Returns:
        ``ON`` for a true or missing value, ``OLD_SHIPPED_OFF`` for exactly
        ``false``, and ``OFF`` for anything else that reads as off.
    """
    from tldw_chatbook import config

    if config.get_rag_citation_canonical_writes_enabled():
        return ConfiguredSwitch.ON
    section = config.load_cli_config_and_ensure_existence().get("rag_citations")
    if isinstance(section, dict) and section.get("canonical_writes_enabled") is False:
        return ConfiguredSwitch.OLD_SHIPPED_OFF
    return ConfiguredSwitch.OFF


def effective_canonical_writes(
    switch: ConfiguredSwitch, answer: CitationRecordsAnswer | None
) -> bool:
    """Apply the precedence: a Keep outranks only the old shipped ``false``.

    Args:
        switch: What config.toml says.
        answer: The saved answer, if any.

    Returns:
        Whether canonical citation records are on.
    """
    if switch is ConfiguredSwitch.ON:
        return True
    return (
        switch is ConfiguredSwitch.OLD_SHIPPED_OFF
        and answer is CitationRecordsAnswer.KEEP
    )


def _parse_answer(text: str) -> CitationRecordsAnswer:
    payload = json.loads(text)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("citation_records_answer_unsupported")
    return CitationRecordsAnswer(payload.get("answer"))


def load_answer() -> AnswerRead:
    """Read the saved answer through the raw participant boundary.

    Returns:
        The answer; ``AnswerRead(None)`` when no file exists; and
        ``AnswerRead(None, readable=False)`` when one exists but cannot be
        read or parsed (or the boundary refused the read).
    """
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    try:
        with raw.scope(sys.modules[__name__], ROUTE) as operation:
            try:
                with raw.file(operation, raw.selected_target(operation), "r") as stream:
                    text = stream.read()
            except FileNotFoundError:
                return AnswerRead(None)
        return AnswerRead(_parse_answer(text))
    except Exception as exc:  # noqa: BLE001 - an unreadable answer is no answer
        logger.warning(
            "Citation records answer unreadable; exception_type={}",
            type(exc).__name__,
        )
        return AnswerRead(None, readable=False)


def save_answer(answer: CitationRecordsAnswer) -> None:
    """Save the answer atomically beside config.toml (never in it).

    Args:
        answer: The user's answer.

    Raises:
        Exception: The write failed; the previous answer file, if any, is kept.
    """
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    payload = json.dumps(
        {
            "schema_version": _SCHEMA_VERSION,
            "answer": CitationRecordsAnswer(answer).value,
            "answered_at": utc_now_iso(),
        },
        indent=2,
    )
    with raw.scope(sys.modules[__name__], ROUTE, writing=True) as operation:
        selected = raw.selected_target(operation)
        raw.mkdirs(operation)
        temporary = selected.with_suffix(selected.suffix + ".tmp")
        try:
            with raw.file(operation, temporary, "w") as stream:
                stream.write(payload)
            raw.replace(operation, temporary, selected)
        finally:
            raw.remove_temporary(operation, temporary)


def canonical_writes_enabled_with_answer() -> bool:
    """Return the citation-records switch with a saved answer applied.

    The answer file is read only when config.toml holds exactly ``false``.

    Returns:
        Whether canonical citation records are on.
    """
    switch = configured_switch()
    if switch is not ConfiguredSwitch.OLD_SHIPPED_OFF:
        return switch is ConfiguredSwitch.ON
    return effective_canonical_writes(switch, load_answer().answer)


def apply_answer(repository: Any, answer: CitationRecordsAnswer) -> bool:
    """Apply an answer to the live citation repository, under config.toml.

    Args:
        repository: The app's ``CitationTraceRepository``.
        answer: The answer just given.

    Returns:
        Whether citation records are now on.
    """
    enabled = effective_canonical_writes(configured_switch(), answer)
    setter = getattr(repository, "set_canonical_writes_enabled", None)
    if callable(setter):
        setter(enabled)
    question_for(repository).settle()
    return enabled


class CitationRecordsQuestion:
    """Whether one profile's question is waiting, and for which session.

    A question waits after a reply in a session went out with Library evidence
    and no record because records are off, on a profile that may be asked. It
    is taken (asked) at most once per process; an answer settles it for good.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settled = False
        self._eligible: bool | None = None
        self._pending_session_id: str | None = None

    def pending_for(self, session_id: str | None) -> bool:
        """Return whether the question waits for this session's reply."""
        with self._lock:
            return (
                not self._settled
                and session_id is not None
                and self._pending_session_id == session_id
            )

    def take_pending(self, session_id: str | None) -> bool:
        """Take the waiting question for this session; it is never asked twice.

        Args:
            session_id: The session whose reply just settled.

        Returns:
            True when the caller should ask now.
        """
        with self._lock:
            if (
                self._settled
                or session_id is None
                or self._pending_session_id != session_id
            ):
                return False
            self._settled = True
            self._pending_session_id = None
            return True

    def settle(self) -> None:
        """Answered, or found not to apply: never ask in this process again."""
        with self._lock:
            self._settled = True
            self._pending_session_id = None

    def _decide(self, eligible: bool, session_id: str) -> None:
        with self._lock:
            if self._settled:
                return
            self._eligible = eligible
            if eligible:
                self._pending_session_id = session_id
            else:
                self._settled = True

    def _known(self) -> bool | None:
        with self._lock:
            return False if self._settled else self._eligible


_QUESTIONS: "weakref.WeakKeyDictionary[Any, CitationRecordsQuestion]" = (
    weakref.WeakKeyDictionary()
)
_QUESTIONS_LOCK = threading.Lock()


def question_for(repository: Any) -> CitationRecordsQuestion:
    """Return the question state for one live citation repository (profile)."""
    with _QUESTIONS_LOCK:
        question = _QUESTIONS.get(repository)
        if question is None:
            question = CitationRecordsQuestion()
            _QUESTIONS[repository] = question
        return question


def _records_of_its_own(db: Any) -> bool:
    """Whether this profile has kept citation records (records were on)."""
    row = (
        db.get_connection()
        .execute(
            "SELECT EXISTS(SELECT 1 FROM rag_citation_traces"
            " WHERE origin IN ('local', 'legacy_inferred'))"
        )
        .fetchone()
    )
    return bool(row[0])


def _old_shipped_false_without_answer() -> bool:
    if configured_switch() is not ConfiguredSwitch.OLD_SHIPPED_OFF:
        return False
    read = load_answer()
    return read.readable and read.answer is None


async def note_unrecorded_library_evidence(
    repository: Any, session_id: str | None
) -> None:
    """Note a reply that went out with Library evidence and no record.

    Called off the UI loop's critical path by the Console capture when no
    citation record will cover the reply. Only the citation-records switch
    (``writes_disabled``) counts; a keyring problem is not a question to ask.
    Eligibility is decided once per process.

    Args:
        repository: The app's citation repository.
        session_id: The Console session the reply belongs to.
    """
    if session_id is None or repository is None:
        return
    try:  # Runs inside the send pipeline: an optional note never fails a send.
        if getattr(repository, "fingerprint_key_state", None) != "writes_disabled":
            return
        await _note_for_question(question_for(repository), repository, session_id)
    except Exception as exc:  # noqa: BLE001 - never ask on an unknown state
        logger.warning(
            "Citation records question not offered; exception_type={}",
            type(exc).__name__,
        )


async def _note_for_question(
    question: CitationRecordsQuestion, repository: Any, session_id: str
) -> None:
    known = question._known()
    if known is False:
        return
    if known is None:
        try:
            from .citation_key_first_use import run_citation_db_call

            eligible = await asyncio.to_thread(
                _old_shipped_false_without_answer
            ) and not await run_citation_db_call(
                getattr(repository, "db", None),
                _records_of_its_own,
                repository.db,
            )
        except Exception:
            question._decide(False, session_id)  # an unknown state never asks
            raise
        if eligible:
            logger.info("Citation records question waits for this reply")
        question._decide(eligible, session_id)
        return
    question._decide(True, session_id)


__all__ = [
    "ANSWER_FILENAME",
    "AnswerRead",
    "CitationRecordsAnswer",
    "CitationRecordsQuestion",
    "ConfiguredSwitch",
    "ROUTE",
    "answer_file_path",
    "apply_answer",
    "canonical_writes_enabled_with_answer",
    "configured_switch",
    "effective_canonical_writes",
    "load_answer",
    "note_unrecorded_library_evidence",
    "question_for",
    "save_answer",
]
