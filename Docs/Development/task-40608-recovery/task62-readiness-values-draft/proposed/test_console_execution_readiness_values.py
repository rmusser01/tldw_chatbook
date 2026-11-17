"""Readiness values preserve original independent evidence without authority."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_execution_readiness_values import (
    decode_settings_readiness,
    encode_settings_readiness,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderTestEvidence,
)
from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
    canonical_connection_identity,
)


def _readiness():
    connection = canonical_connection_identity("custom", "https://example.test/v1")
    assert connection is not None
    identity = ProviderDraftIdentity(
        "custom", connection, "stored", 7, 3, "custom-ep:local"
    )
    endpoint_time = datetime(2026, 11, 17, 9, 12, tzinfo=timezone.utc)
    generation_time = datetime(
        2026, 11, 17, 3, 45, tzinfo=timezone(timedelta(hours=-5))
    )
    evidence = ProviderTestEvidence(
        identity,
        "model_listing_unavailable",
        (),
        "http_status",
        credential="authenticated",
        generation="succeeded",
        observed_at=endpoint_time,
        generation_observed_at=generation_time,
        generation_model="model-a",
    )
    return ConsoleSettingsReadiness(
        "Ready",
        "Generation succeeded; model listing unavailable.",
        True,
        operability="ready_to_send",
        provider_display_name="Custom endpoint",
        configuration="configured",
        credential="authenticated",
        credential_source="stored",
        endpoint="model_listing_unavailable",
        endpoint_category="http_status",
        model="unconfirmed",
        generation="succeeded",
        connection=identity,
        observed_at=endpoint_time,
        generation_observed_at=generation_time,
        evidence=evidence,
    )


def test_roundtrip_retains_independent_endpoint_and_generation_facts():
    original = _readiness()
    encoded = encode_settings_readiness(original)
    decoded = decode_settings_readiness(json.loads(json.dumps(encoded, allow_nan=False)))

    assert type(decoded) is ConsoleSettingsReadiness
    assert type(decoded.connection) is ProviderDraftIdentity
    assert type(decoded.evidence) is ProviderTestEvidence
    assert decoded == original
    assert decoded.evidence == original.evidence
    assert decoded.endpoint == "model_listing_unavailable"
    assert decoded.generation == "succeeded"
    assert decoded.evidence.generation_model == "model-a"
    assert decoded.observed_at.isoformat() == original.observed_at.isoformat()
    assert (
        decoded.generation_observed_at.isoformat()
        == original.generation_observed_at.isoformat()
    )
    assert decoded.evidence.observed_at == original.evidence.observed_at
    assert (
        decoded.evidence.generation_observed_at
        == original.evidence.generation_observed_at
    )
    assert decoded.evidence.identity == decoded.connection
    assert set(encoded["connection"]) == {
        "provider_key", "connection_identity", "credential_source",
        "credential_revision", "draft_generation", "custom_endpoint_id",
    }
    assert encoded["connection"]["connection_identity"] == list(
        original.connection.connection_identity
    )
    listed = replace(
        original,
        endpoint="reachable",
        endpoint_category=None,
        model="confirmed",
        evidence=replace(
            original.evidence,
            endpoint="reachable",
            category=None,
            model_ids=("model-a", "model-b"),
        ),
    )
    assert decode_settings_readiness(
        encode_settings_readiness(listed)
    ).evidence.model_ids == ("model-a", "model-b")


def test_unknown_blocked_and_independently_absent_observations_survive():
    unknown = ConsoleSettingsReadiness("Not ready", "Review settings.", False)
    decoded = decode_settings_readiness(encode_settings_readiness(unknown))
    assert decoded == unknown
    assert decoded.blocker == "readiness_unknown"
    assert decoded.connection is decoded.evidence is decoded.observed_at is None
    assert decoded.generation_observed_at is None

    original = _readiness()
    partial = replace(
        original,
        observed_at=None,
        evidence=replace(original.evidence, generation_observed_at=None),
    )
    decoded = decode_settings_readiness(encode_settings_readiness(partial))
    assert decoded.observed_at is None
    assert decoded.generation_observed_at == partial.generation_observed_at
    assert decoded.evidence.observed_at == partial.evidence.observed_at
    assert decoded.evidence.generation_observed_at is None

    # The original top-level constructor does not impose evidence's aware-time
    # rule or require connection/evidence for an independently supplied time.
    standalone = replace(unknown, observed_at=datetime(2026, 11, 17, 9, 12))
    assert decode_settings_readiness(encode_settings_readiness(standalone)) == standalone


def test_closed_shapes_reject_unknown_fields_objects_and_nonexact_records():
    encoded = encode_settings_readiness(_readiness())
    for payload in (object(), MappingProxyType(encoded), dict(encoded, api_key="private")):
        with pytest.raises(ConsoleProtocolError):
            decode_settings_readiness(payload)
    for field, value in (("label", object()), ("native_send_supported", 1)):
        with pytest.raises(ConsoleProtocolError):
            decode_settings_readiness(dict(encoded, **{field: value}))
    forged = replace(_readiness())
    object.__setattr__(forged, "evidence", object())
    with pytest.raises(ConsoleProtocolError):
        encode_settings_readiness(forged)


def test_original_identity_and_evidence_validation_cannot_be_bypassed():
    encoded = encode_settings_readiness(_readiness())
    for value in (True, float("nan"), -1):
        payload = deepcopy(encoded)
        payload["connection"]["credential_revision"] = value
        with pytest.raises(ConsoleProtocolError):
            decode_settings_readiness(payload)

    payload = deepcopy(encoded)
    payload["connection"]["connection_identity"][0] = "ollama"
    with pytest.raises(ConsoleProtocolError):
        decode_settings_readiness(payload)

    payload = deepcopy(encoded)
    payload["evidence"]["generation"] = "failed"
    with pytest.raises(ConsoleProtocolError):
        decode_settings_readiness(payload)
    payload = deepcopy(encoded)
    payload["evidence"]["observed_at"] = "2026-11-17T09:12:00"
    with pytest.raises(ConsoleProtocolError):
        decode_settings_readiness(payload)


def test_connection_identity_refuses_secret_bearing_endpoint_components():
    encoded = encode_settings_readiness(_readiness())
    for endpoint in (
        "https://user:private@example.test/v1",
        "https://example.test/v1?api_key=private",
        "https://example.test/v1#private",
    ):
        payload = deepcopy(encoded)
        payload["connection"]["connection_identity"][1] = endpoint
        with pytest.raises(ConsoleProtocolError) as caught:
            decode_settings_readiness(payload)
        assert "private" not in str(caught.value)
