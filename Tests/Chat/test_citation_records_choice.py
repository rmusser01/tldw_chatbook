"""The saved citation-records answer and its precedence (TASK-33620.8.3).

Owner decision 2026-10-30: installs whose config.toml still carries the old
shipped ``[rag_citations] canonical_writes_enabled = false`` are asked once,
"Keep a record of which sources replies cite?". The answer is kept beside
config.toml, which is never rewritten for it, and outranks only that explicit
``false``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import citation_records_choice as choice
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_records_choice import (
    CitationRecordsAnswer,
    ConfiguredSwitch,
)
from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.bootstrap_profile

OFF = CitationProvenanceRuntimePolicy(canonical_writes_enabled=False)
MISSING = object()


def _config_with(monkeypatch, value) -> None:
    section = {} if value is MISSING else {"canonical_writes_enabled": value}
    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda *args, **kwargs: {"rag_citations": section},
    )


@pytest.fixture
def no_saved_answer():
    """The shared bootstrap profile keeps the answer file between tests."""
    path = choice.answer_file_path()
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "records.sqlite", client_id="records-q")
    yield database
    database.close_connection()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, ConfiguredSwitch.ON),
        (MISSING, ConfiguredSwitch.ON),
        ("true", ConfiguredSwitch.ON),
        (False, ConfiguredSwitch.OLD_SHIPPED_OFF),
        ("false", ConfiguredSwitch.OFF),
        (0, ConfiguredSwitch.OFF),
        ("invalid", ConfiguredSwitch.OFF),
    ],
)
def test_only_an_exact_false_reads_as_the_old_shipped_value(
    monkeypatch, value, expected
):
    _config_with(monkeypatch, value)

    assert choice.configured_switch() is expected


def test_a_malformed_section_is_off_and_not_the_old_shipped_value(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda *args, **kwargs: {"rag_citations": "not-a-table"},
    )

    assert choice.configured_switch() is ConfiguredSwitch.OFF


@pytest.mark.parametrize(
    ("value", "answer", "expected"),
    [
        (False, CitationRecordsAnswer.KEEP, True),
        (False, CitationRecordsAnswer.OFF, False),
        (False, None, False),
        (True, CitationRecordsAnswer.OFF, True),
        (MISSING, CitationRecordsAnswer.OFF, True),
        ("invalid", CitationRecordsAnswer.KEEP, False),
        ("false", CitationRecordsAnswer.KEEP, False),
    ],
)
def test_a_saved_keep_outranks_only_the_old_shipped_false(
    monkeypatch, no_saved_answer, value, answer, expected
):
    _config_with(monkeypatch, value)
    if answer is not None:
        choice.save_answer(answer)

    assert CitationProvenanceRuntimePolicy.from_config().canonical_writes_enabled is (
        expected
    )


def test_the_answer_file_round_trips_beside_config_toml(no_saved_answer):
    config_dir = Path(config_module._get_effective_config_path()).parent
    assert no_saved_answer == config_dir / "citation_records_choice.json"
    assert choice.load_answer() == choice.AnswerRead(None, readable=True)

    choice.save_answer(CitationRecordsAnswer.KEEP)
    assert choice.load_answer() == choice.AnswerRead(
        CitationRecordsAnswer.KEEP, readable=True
    )
    saved = json.loads(no_saved_answer.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["answer"] == "keep"
    datetime.fromisoformat(saved["answered_at"])

    choice.save_answer(CitationRecordsAnswer.OFF)
    assert choice.load_answer().answer is CitationRecordsAnswer.OFF
    assert not no_saved_answer.with_name(no_saved_answer.name + ".tmp").exists()


@pytest.mark.parametrize(
    "content", ["{not json", '{"answer": "maybe"}', "[]", '{"schema_version": 9}']
)
def test_an_unreadable_answer_is_neither_an_answer_nor_a_reason_to_ask(
    no_saved_answer, content
):
    choice.save_answer(CitationRecordsAnswer.KEEP)  # admit the file normally
    no_saved_answer.write_text(content, encoding="utf-8")

    assert choice.load_answer() == choice.AnswerRead(None, readable=False)


def _repository(db, *, enabled: bool = False) -> CitationTraceRepository:
    return CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=enabled),
        identity_context=None,
        fingerprint_codec=None,
    )


def _add_local_record(db: CharactersRAGDB) -> None:
    now = datetime.now(UTC).isoformat()
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            ) VALUES ('profile-1', 'trace-1', 1, 'request-1', 'generation-1',
                      'profile-1', 'local', 'sealed', 'complete', 'attempt-1',
                      'policy-1', '{}', 'active', ?, ?)
            """,
            (now, now),
        )


def test_question_waits_for_its_own_session_and_is_taken_once(
    monkeypatch, no_saved_answer, db
):
    _config_with(monkeypatch, False)
    repository = _repository(db)

    asyncio.run(choice.note_unrecorded_library_evidence(repository, "session-a"))
    question = choice.question_for(repository)

    assert question.pending_for("session-a")
    assert not question.take_pending("session-b")
    assert question.take_pending("session-a")
    assert not question.take_pending("session-a")
    asyncio.run(choice.note_unrecorded_library_evidence(repository, "session-a"))
    assert not question.pending_for("session-a")  # asked once per session


@pytest.mark.parametrize(
    "case",
    [
        "answered",
        "records_of_its_own",
        "on",
        "malformed",
        "key_unavailable",
        "no_session",
    ],
)
def test_question_is_never_raised_outside_the_old_shipped_false(
    monkeypatch, no_saved_answer, db, case
):
    _config_with(
        monkeypatch,
        {"on": True, "malformed": "invalid"}.get(case, False),
    )
    repository = _repository(db)
    if case == "answered":
        choice.save_answer(CitationRecordsAnswer.OFF)
    if case == "records_of_its_own":
        _add_local_record(db)
    if case == "key_unavailable":
        repository = _repository(db, enabled=True)
        repository._key_state = "keyring_unavailable"
    session_id = None if case == "no_session" else "session-a"

    asyncio.run(choice.note_unrecorded_library_evidence(repository, session_id))

    assert not choice.question_for(repository).pending_for("session-a")


def test_an_answer_applies_to_the_live_repository(monkeypatch, no_saved_answer, db):
    _config_with(monkeypatch, False)
    repository = _repository(db)
    assert repository.fingerprint_key_state == "writes_disabled"

    choice.apply_answer(repository, CitationRecordsAnswer.KEEP)
    assert repository.canonical_writes_enabled is True
    assert repository.fingerprint_key_state != "writes_disabled"

    choice.apply_answer(repository, CitationRecordsAnswer.OFF)
    assert repository.canonical_writes_enabled is False


def test_an_answer_never_overrides_config_toml_where_it_decides(
    monkeypatch, no_saved_answer, db
):
    _config_with(monkeypatch, True)
    repository = _repository(db, enabled=True)

    choice.apply_answer(repository, CitationRecordsAnswer.OFF)

    assert repository.canonical_writes_enabled is True
