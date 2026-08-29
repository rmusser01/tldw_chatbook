from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.chat_persistence_service import (
    ChatPersistenceService,
    _initial_metadata_object,
)
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    CONSOLE_GENERATION_SETTINGS_METADATA_KEY,
    CONSOLE_GENERATION_SETTINGS_VERSION,
    ConsoleGenerationSettingsReadStatus,
    ConsoleGenerationSettingsSnapshot,
    ConsoleGenerationSettingsWriteResult,
    ConsoleGenerationSettingsWriteStatus,
    merge_console_generation_settings,
    parse_console_generation_settings,
    snapshot_from_session_settings,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError


SAFE_FIELDS = {
    "provider",
    "model",
    "temperature",
    "top_p",
    "min_p",
    "top_k",
    "max_tokens",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "thinking_effort",
    "thinking_budget_tokens",
    "streaming",
}


def _snapshot(**changes: object) -> ConsoleGenerationSettingsSnapshot:
    baseline = ConsoleGenerationSettingsSnapshot(
        provider="openai",
        model="gpt-5",
        temperature=0.3,
        top_p=0.8,
        min_p=0.1,
        top_k=40,
        max_tokens=2048,
        seed=7,
        presence_penalty=0.2,
        frequency_penalty=-0.2,
        reasoning_effort="high",
        reasoning_summary="concise",
        verbosity="low",
        thinking_effort="medium",
        thinking_budget_tokens=4096,
        streaming=True,
    )
    return replace(baseline, **changes)


def _owned(
    snapshot: ConsoleGenerationSettingsSnapshot | None = None,
) -> dict[str, object]:
    value = snapshot or _snapshot()
    return {
        "version": CONSOLE_GENERATION_SETTINGS_VERSION,
        "provider": value.provider,
        "model": value.model,
        "temperature": value.temperature,
        "top_p": value.top_p,
        "min_p": value.min_p,
        "top_k": value.top_k,
        "max_tokens": value.max_tokens,
        "seed": value.seed,
        "presence_penalty": value.presence_penalty,
        "frequency_penalty": value.frequency_penalty,
        "reasoning_effort": value.reasoning_effort,
        "reasoning_summary": value.reasoning_summary,
        "verbosity": value.verbosity,
        "thinking_effort": value.thinking_effort,
        "thinking_budget_tokens": value.thinking_budget_tokens,
        "streaming": value.streaming,
    }


def _metadata(
    snapshot: ConsoleGenerationSettingsSnapshot | None = None,
) -> dict[str, object]:
    return {CONSOLE_GENERATION_SETTINGS_METADATA_KEY: _owned(snapshot)}


def test_codec_uses_exact_versioned_allowlist_and_never_serializes_other_settings() -> (
    None
):
    settings = ConsoleSessionSettings(
        provider="openai",
        model="gpt-5",
        base_url="https://secret-endpoint.example/v1",
        temperature=0.4,
        top_p=0.7,
        min_p=0.05,
        top_k=20,
        max_tokens=1024,
        seed=11,
        presence_penalty=0.1,
        frequency_penalty=-0.1,
        reasoning_effort="medium",
        reasoning_summary="auto",
        verbosity="high",
        thinking_effort="low",
        thinking_budget_tokens=2048,
        streaming=False,
        character_label="Private display name",
        system_prompt="Never persist me",
        source="user",
        pinned_prefill="Never persist me either",
    )

    merged = merge_console_generation_settings(
        {
            "sibling": {"keep": True},
            "credential_reference": "keyring://secret",
            "context_policy": {"compaction_mode": "summary"},
        },
        snapshot_from_session_settings(settings),
    )

    assert merged["sibling"] == {"keep": True}
    assert merged["credential_reference"] == "keyring://secret"
    assert merged["context_policy"] == {"compaction_mode": "summary"}
    owned = merged[CONSOLE_GENERATION_SETTINGS_METADATA_KEY]
    assert isinstance(owned, dict)
    assert set(owned) == {"version", *SAFE_FIELDS}
    assert owned["version"] == 1
    assert set(owned).isdisjoint(
        {
            "base_url",
            "api_key",
            "credentials",
            "credential_reference",
            "system_prompt",
            "source",
            "compaction_mode",
            "context_policy",
            "character_label",
            "character_name",
            "display_name",
            "pinned_prefill",
        }
    )


def test_parse_distinguishes_absent_valid_invalid_and_unsupported_version() -> None:
    absent = parse_console_generation_settings({"sibling": True})
    valid = parse_console_generation_settings(json.dumps(_metadata()))
    invalid = parse_console_generation_settings(
        {CONSOLE_GENERATION_SETTINGS_METADATA_KEY: {"version": 1}}
    )
    future_metadata = _metadata()
    future_metadata[CONSOLE_GENERATION_SETTINGS_METADATA_KEY]["version"] = 2  # type: ignore[index]
    future = parse_console_generation_settings(future_metadata)

    assert absent.status is ConsoleGenerationSettingsReadStatus.ABSENT
    assert absent.snapshot is None
    assert valid.status is ConsoleGenerationSettingsReadStatus.VALID
    assert valid.snapshot == _snapshot()
    assert invalid.status is ConsoleGenerationSettingsReadStatus.INVALID
    assert invalid.snapshot is None
    assert future.status is ConsoleGenerationSettingsReadStatus.UNSUPPORTED_VERSION
    assert future.snapshot is None


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param("{", id="malformed-json"),
        pytest.param("[]", id="array-json"),
        pytest.param('"scalar"', id="scalar-json"),
        pytest.param("null", id="null-json"),
        pytest.param('{"bad": NaN}', id="non-finite-json"),
        pytest.param(["not-an-object"], id="list-value"),
        pytest.param({1: "non-string-key"}, id="non-string-key"),
        pytest.param({"bad": float("inf")}, id="non-finite-mapping"),
    ],
)
def test_parse_rejects_malformed_or_non_object_metadata(metadata: object) -> None:
    result = parse_console_generation_settings(metadata)

    assert result.status is ConsoleGenerationSettingsReadStatus.INVALID
    assert result.snapshot is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unknown", "value"),
        ("version", True),
        ("provider", ""),
        ("provider", "p" * 129),
        ("model", ""),
        ("model", "m" * 257),
        ("temperature", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", -0.01),
        ("temperature", 2.01),
        ("top_p", -0.01),
        ("top_p", 1.01),
        ("min_p", -0.01),
        ("min_p", 1.01),
        ("top_k", True),
        ("top_k", 1.0),
        ("top_k", -1),
        ("max_tokens", 0),
        ("seed", -1),
        ("presence_penalty", -2.01),
        ("presence_penalty", 2.01),
        ("frequency_penalty", -2.01),
        ("frequency_penalty", 2.01),
        ("reasoning_effort", "extreme"),
        ("reasoning_summary", "everything"),
        ("verbosity", "verbose"),
        ("thinking_effort", "unlimited"),
        ("thinking_budget_tokens", 1023),
        ("streaming", 1),
    ],
)
def test_parse_rejects_unknown_or_invalid_owned_fields(
    field: str, value: object
) -> None:
    owned = _owned()
    owned[field] = value

    result = parse_console_generation_settings(
        {CONSOLE_GENERATION_SETTINGS_METADATA_KEY: owned}
    )

    assert result.status is ConsoleGenerationSettingsReadStatus.INVALID
    assert result.snapshot is None


def test_oversized_float_input_fails_closed_during_parse_and_projection() -> None:
    oversized = 10**400

    parsed = parse_console_generation_settings(
        _metadata(_snapshot(temperature=oversized))
    )

    assert parsed.status is ConsoleGenerationSettingsReadStatus.INVALID
    assert parsed.snapshot is None
    with pytest.raises(ValueError):
        snapshot_from_session_settings(
            ConsoleSessionSettings(provider="openai", temperature=oversized)
        )


def test_string_metadata_rejects_exponent_overflow_in_unrelated_sibling() -> None:
    metadata = json.dumps(_metadata())[:-1] + ', "sibling": 1e400}'

    parsed = parse_console_generation_settings(metadata)

    assert parsed.status is ConsoleGenerationSettingsReadStatus.INVALID
    assert parsed.snapshot is None


@pytest.mark.parametrize("model_length", [121, 256])
def test_model_ids_through_console_input_limit_are_durable(model_length: int) -> None:
    snapshot = _snapshot(model="m" * model_length)

    merged = merge_console_generation_settings({}, snapshot)

    assert parse_console_generation_settings(merged).snapshot == snapshot


def test_exact_integers_above_64_bit_are_durable() -> None:
    exact_integer = 2**80
    snapshot = _snapshot(
        top_k=exact_integer,
        max_tokens=exact_integer,
        seed=exact_integer,
        thinking_budget_tokens=exact_integer,
    )

    merged = merge_console_generation_settings({}, snapshot)

    assert parse_console_generation_settings(merged).snapshot == snapshot


def test_write_result_enforces_snapshot_status_invariants() -> None:
    snapshot = _snapshot()

    with pytest.raises(ValueError):
        ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.WRITTEN
        )
    for status in (
        ConsoleGenerationSettingsWriteStatus.INVALID,
        ConsoleGenerationSettingsWriteStatus.UNSUPPORTED_VERSION,
        ConsoleGenerationSettingsWriteStatus.MISSING,
    ):
        with pytest.raises(ValueError):
            ConsoleGenerationSettingsWriteResult(status, snapshot)

    assert (
        ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.WRITTEN, snapshot
        ).snapshot
        == snapshot
    )
    assert (
        ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.SUPERSEDED, snapshot
        ).snapshot
        == snapshot
    )
    assert (
        ConsoleGenerationSettingsWriteResult(
            ConsoleGenerationSettingsWriteStatus.SUPERSEDED
        ).snapshot
        is None
    )


def test_initial_metadata_object_preserves_strict_legacy_contract() -> None:
    assert _initial_metadata_object({"sibling": [1, True]}) == {"sibling": [1, True]}
    assert _initial_metadata_object('{"sibling": [1, true]}') == {"sibling": [1, True]}
    for invalid in (None, [], "null", {1: "value"}, '{"sibling": 1e400}'):
        with pytest.raises(ValueError):
            _initial_metadata_object(invalid)


def test_merge_refuses_malformed_invalid_and_future_owned_objects() -> None:
    invalid_owned = _metadata()
    invalid_owned[CONSOLE_GENERATION_SETTINGS_METADATA_KEY]["unexpected"] = True  # type: ignore[index]
    future_owned = _metadata()
    future_owned[CONSOLE_GENERATION_SETTINGS_METADATA_KEY]["version"] = 2  # type: ignore[index]

    with pytest.raises(ValueError):
        merge_console_generation_settings("{", _snapshot())
    with pytest.raises(ValueError):
        merge_console_generation_settings(invalid_owned, _snapshot())
    with pytest.raises(ValueError):
        merge_console_generation_settings(future_owned, _snapshot())


@pytest.fixture
def persistence(tmp_path):
    db = CharactersRAGDB(tmp_path / "generation-settings.sqlite", "generation-test")
    try:
        yield ChatPersistenceService(db), db
    finally:
        db.close_connection()


def test_persistence_writes_complete_snapshot_and_preserves_siblings(
    persistence,
) -> None:
    service, db = persistence
    conversation_id = service.create_conversation(
        conversation_title="Saved",
        metadata={"sibling": {"keep": True}},
    )
    desired = _snapshot(temperature=0.9, streaming=False)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=desired,
        expected_snapshot=None,
    )

    assert result.status is ConsoleGenerationSettingsWriteStatus.WRITTEN
    assert result.snapshot == desired
    record = db.get_conversation_by_id(conversation_id)
    assert json.loads(record["metadata"])["sibling"] == {"keep": True}
    assert (
        service.get_conversation_generation_settings(conversation_id).snapshot
        == desired
    )


def test_persistence_treats_missing_owned_object_as_expected_none(persistence) -> None:
    service, _db = persistence
    conversation_id = service.create_conversation(conversation_title="No metadata")

    before = service.get_conversation_generation_settings(conversation_id)
    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=_snapshot(),
        expected_snapshot=None,
    )

    assert before.status is ConsoleGenerationSettingsReadStatus.ABSENT
    assert result.status is ConsoleGenerationSettingsWriteStatus.WRITTEN


def test_persistence_refuses_oversized_float_without_mutation(persistence) -> None:
    service, db = persistence
    conversation_id = service.create_conversation(
        conversation_title="Oversized",
        metadata={"sibling": "keep"},
    )
    before = db.get_conversation_by_id(conversation_id)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=_snapshot(temperature=10**400),
        expected_snapshot=None,
    )

    after = db.get_conversation_by_id(conversation_id)
    assert result.status is ConsoleGenerationSettingsWriteStatus.INVALID
    assert after["version"] == before["version"]
    assert after["metadata"] == before["metadata"]


def test_persistence_refuses_exponent_overflow_sibling_without_mutation(
    persistence,
) -> None:
    service, db = persistence
    conversation_id = service.create_conversation(
        conversation_title="Exponent overflow",
        metadata=_metadata(_snapshot()),
    )
    current = db.get_conversation_by_id(conversation_id)
    overflow_metadata = json.dumps(_metadata(_snapshot()))[:-1] + (
        ', "sibling": 1e400}'
    )
    db.update_conversation(
        conversation_id,
        {"metadata": overflow_metadata},
        expected_version=current["version"],
    )
    before = db.get_conversation_by_id(conversation_id)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=_snapshot(temperature=1.0),
        expected_snapshot=_snapshot(),
    )

    after = db.get_conversation_by_id(conversation_id)
    assert result.status is ConsoleGenerationSettingsWriteStatus.INVALID
    assert result.snapshot is None
    assert after["version"] == before["version"]
    assert after["metadata"] == before["metadata"]


@pytest.mark.parametrize(
    ("owned", "expected_status"),
    [
        ({"version": 1}, ConsoleGenerationSettingsWriteStatus.INVALID),
        (
            {"version": 2, "future": True},
            ConsoleGenerationSettingsWriteStatus.UNSUPPORTED_VERSION,
        ),
    ],
)
def test_persistence_refuses_invalid_or_future_owned_data_without_mutation(
    persistence, owned, expected_status
) -> None:
    service, db = persistence
    conversation_id = service.create_conversation(
        conversation_title="Saved",
        metadata={CONSOLE_GENERATION_SETTINGS_METADATA_KEY: owned, "sibling": True},
    )
    before = db.get_conversation_by_id(conversation_id)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=_snapshot(temperature=1.1),
        expected_snapshot=None,
    )

    after = db.get_conversation_by_id(conversation_id)
    assert result.status is expected_status
    assert after["version"] == before["version"]
    assert after["metadata"] == before["metadata"]


def test_persistence_retries_sibling_only_conflict_with_fresh_merge(
    persistence, monkeypatch
) -> None:
    service, db = persistence
    baseline = _snapshot()
    desired = _snapshot(temperature=1.2)
    conversation_id = service.create_conversation(
        conversation_title="Saved",
        metadata={**_metadata(baseline), "sibling": "old"},
    )
    original_update = db.update_conversation
    first_call = True

    def race(conversation_id_arg, update_data, expected_version):
        nonlocal first_call
        if first_call:
            first_call = False
            current = db.get_conversation_by_id(conversation_id_arg)
            concurrent_metadata = json.loads(current["metadata"])
            concurrent_metadata["sibling"] = "fresh"
            original_update(
                conversation_id_arg,
                {"metadata": json.dumps(concurrent_metadata, sort_keys=True)},
                expected_version=current["version"],
            )
        return original_update(conversation_id_arg, update_data, expected_version)

    monkeypatch.setattr(db, "update_conversation", race)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=desired,
        expected_snapshot=baseline,
    )

    assert result.status is ConsoleGenerationSettingsWriteStatus.WRITTEN
    record = db.get_conversation_by_id(conversation_id)
    metadata = json.loads(record["metadata"])
    assert metadata["sibling"] == "fresh"
    assert parse_console_generation_settings(metadata).snapshot == desired


def test_persistence_refuses_owned_base_supersession(persistence, monkeypatch) -> None:
    service, db = persistence
    baseline = _snapshot()
    desired = _snapshot(temperature=1.2)
    external = _snapshot(temperature=1.8)
    conversation_id = service.create_conversation(
        conversation_title="Saved",
        metadata=_metadata(baseline),
    )
    original_update = db.update_conversation
    first_call = True

    def race(conversation_id_arg, update_data, expected_version):
        nonlocal first_call
        if first_call:
            first_call = False
            current = db.get_conversation_by_id(conversation_id_arg)
            original_update(
                conversation_id_arg,
                {"metadata": json.dumps(_metadata(external), sort_keys=True)},
                expected_version=current["version"],
            )
        return original_update(conversation_id_arg, update_data, expected_version)

    monkeypatch.setattr(db, "update_conversation", race)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=desired,
        expected_snapshot=baseline,
    )

    assert result.status is ConsoleGenerationSettingsWriteStatus.SUPERSEDED
    assert result.snapshot == external
    assert (
        service.get_conversation_generation_settings(conversation_id).snapshot
        == external
    )


@pytest.mark.parametrize(
    ("transition", "expected_status", "expected_snapshot_kind"),
    [
        ("deleted", ConsoleGenerationSettingsWriteStatus.MISSING, None),
        ("invalid", ConsoleGenerationSettingsWriteStatus.INVALID, None),
        (
            "future",
            ConsoleGenerationSettingsWriteStatus.UNSUPPORTED_VERSION,
            None,
        ),
        (
            "owned-removed",
            ConsoleGenerationSettingsWriteStatus.SUPERSEDED,
            None,
        ),
        (
            "owned-changed",
            ConsoleGenerationSettingsWriteStatus.SUPERSEDED,
            "external",
        ),
    ],
)
def test_persistence_classifies_fresh_state_after_final_conflict(
    persistence,
    monkeypatch,
    transition,
    expected_status,
    expected_snapshot_kind,
) -> None:
    service, db = persistence
    baseline = _snapshot()
    external = _snapshot(temperature=1.8)
    conversation_id = service.create_conversation(
        conversation_title="Saved",
        metadata=_metadata(baseline),
    )
    original_update = db.update_conversation
    attempt_number = 0

    def race(conversation_id_arg, update_data, expected_version):
        nonlocal attempt_number
        attempt_number += 1
        current = db.get_conversation_by_id(conversation_id_arg)
        if attempt_number == 1:
            metadata = json.loads(current["metadata"])
            metadata["first_sibling"] = True
            original_update(
                conversation_id_arg,
                {"metadata": json.dumps(metadata, sort_keys=True)},
                expected_version=current["version"],
            )
        elif transition == "deleted":
            db.soft_delete_conversation(
                conversation_id_arg,
                expected_version=current["version"],
            )
        else:
            metadata = json.loads(current["metadata"])
            if transition == "invalid":
                metadata[CONSOLE_GENERATION_SETTINGS_METADATA_KEY] = {"version": 1}
            elif transition == "future":
                metadata[CONSOLE_GENERATION_SETTINGS_METADATA_KEY] = {
                    "version": 2,
                    "future": True,
                }
            elif transition == "owned-removed":
                metadata.pop(CONSOLE_GENERATION_SETTINGS_METADATA_KEY)
            else:
                metadata.update(_metadata(external))
            original_update(
                conversation_id_arg,
                {"metadata": json.dumps(metadata, sort_keys=True)},
                expected_version=current["version"],
            )
        return original_update(conversation_id_arg, update_data, expected_version)

    monkeypatch.setattr(db, "update_conversation", race)

    result = service.update_conversation_generation_settings(
        conversation_id=conversation_id,
        snapshot=_snapshot(temperature=1.0),
        expected_snapshot=baseline,
    )

    assert result.status is expected_status
    assert attempt_number == 2
    if expected_snapshot_kind == "external":
        assert result.snapshot == external
    else:
        assert result.snapshot is None


def test_persistence_returns_missing_and_propagates_final_sibling_conflict(
    persistence, monkeypatch
) -> None:
    service, db = persistence
    missing = service.update_conversation_generation_settings(
        conversation_id="missing",
        snapshot=_snapshot(),
        expected_snapshot=None,
    )
    assert missing.status is ConsoleGenerationSettingsWriteStatus.MISSING

    baseline = _snapshot()
    conversation_id = service.create_conversation(
        conversation_title="Saved",
        metadata=_metadata(baseline),
    )
    original_update = db.update_conversation
    race_number = 0

    def always_race(conversation_id_arg, update_data, expected_version):
        nonlocal race_number
        current = db.get_conversation_by_id(conversation_id_arg)
        metadata = json.loads(current["metadata"])
        race_number += 1
        metadata[f"sibling_{race_number}"] = True
        original_update(
            conversation_id_arg,
            {"metadata": json.dumps(metadata, sort_keys=True)},
            expected_version=current["version"],
        )
        return original_update(conversation_id_arg, update_data, expected_version)

    monkeypatch.setattr(db, "update_conversation", always_race)

    with pytest.raises(ConflictError):
        service.update_conversation_generation_settings(
            conversation_id=conversation_id,
            snapshot=_snapshot(temperature=1.0),
            expected_snapshot=baseline,
        )
    assert race_number == 2
