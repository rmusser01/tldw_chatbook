"""Credential and generation facets for provider test evidence."""

from dataclasses import fields
from threading import Event, Thread

import pytest

from tldw_chatbook.Chat import provider_test_evidence as evidence_module
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
    ProviderReadinessSnapshot,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
)

ProviderGenerationProbeResult = getattr(
    evidence_module, "ProviderGenerationProbeResult", None
)


def _identity(
    *,
    credential_source: str = "stored",
    credential_revision: int = 2,
    draft_generation: int = 4,
    endpoint: str = "https://example.test/v1/chat/completions",
) -> ProviderDraftIdentity:
    return ProviderDraftIdentity(
        provider_key="custom",
        connection_identity=("custom", endpoint),
        credential_source=credential_source,
        credential_revision=credential_revision,
        draft_generation=draft_generation,
    )


def test_ready_configuration_can_retain_unverified_credential_evidence():
    snapshot = ProviderReadinessSnapshot(
        configuration="configured",
        credential="present_unverified",
        endpoint="not_tested",
        model="unconfirmed",
        generation="not_tested",
    )

    assert snapshot.credential == "present_unverified"
    assert snapshot.generation == "not_tested"


def test_successful_snapshot_rejects_missing_credential_evidence():
    with pytest.raises(ValueError):
        ProviderReadinessSnapshot(
            configuration="configured",
            endpoint="not_tested",
            model="unconfirmed",
            credential="missing",
            generation="succeeded",
        )


def test_successful_snapshot_authenticates_present_credential_evidence():
    snapshot = ProviderReadinessSnapshot(
        configuration="configured",
        endpoint="not_tested",
        model="unconfirmed",
        credential="present_unverified",
        generation="succeeded",
    )

    assert snapshot.credential == "authenticated"


def test_existing_positional_constructors_retain_their_field_meanings():
    identity = _identity(credential_source="none", credential_revision=0)

    snapshot = ProviderReadinessSnapshot(
        "configured", "unreachable", "unconfirmed", "timeout", None
    )
    evidence = ProviderTestEvidence(identity, "reachable", ("model-a",), None)

    assert snapshot.category == "timeout"
    assert snapshot.configuration_issue is None
    assert snapshot.credential == "not_required"
    assert snapshot.generation == "not_tested"
    assert evidence.category is None
    assert evidence.credential == "not_required"
    assert evidence.generation == "not_tested"
    assert evidence.generation_category is None


@pytest.mark.parametrize(
    "values",
    [
        {
            "configuration": "incomplete",
            "endpoint": "not_tested",
            "model": "missing",
            "credential": "authenticated",
        },
        {
            "configuration": "configured",
            "endpoint": "not_tested",
            "model": "unconfirmed",
            "credential": "available",
        },
        {
            "configuration": "configured",
            "endpoint": "not_tested",
            "model": "unconfirmed",
            "generation": "complete",
        },
    ],
)
def test_readiness_snapshot_rejects_invalid_or_contradictory_new_facets(values):
    with pytest.raises(ValueError):
        ProviderReadinessSnapshot(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"generation": "succeeded", "category": "timeout"},
        {"generation": "failed", "category": None},
        {"generation": "failed", "category": "secret-server-message"},
        {"generation": "pending", "category": None},
    ],
)
def test_generation_probe_result_rejects_incoherent_or_unbounded_values(values):
    with pytest.raises(ValueError):
        ProviderGenerationProbeResult(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"generation": "succeeded", "generation_category": "timeout"},
        {"generation": "failed", "generation_category": None},
        {
            "generation": "failed",
            "generation_category": "secret-provider-detail",
        },
        {"credential": "verified", "generation": "not_tested"},
    ],
)
def test_provider_evidence_rejects_invalid_generation_combinations(values):
    kwargs = {
        "identity": _identity(),
        "endpoint": "not_tested",
        "model_ids": (),
    }
    kwargs.update(values)

    with pytest.raises(ValueError):
        ProviderTestEvidence(**kwargs)


def test_generation_success_is_independent_of_endpoint_and_authenticates_present_key():
    store = ProviderTestEvidenceStore()
    identity = _identity()

    token = store.begin_generation(identity)
    assert store.evidence_for(identity) == ProviderTestEvidence(
        identity,
        "not_tested",
        (),
        credential="present_unverified",
        generation="testing",
    )

    assert store.settle_generation(
        token, ProviderGenerationProbeResult("succeeded", None)
    )
    assert store.evidence_for(identity) == ProviderTestEvidence(
        identity,
        "not_tested",
        (),
        credential="authenticated",
        generation="succeeded",
    )


def test_generation_success_keeps_keyless_provider_not_required():
    store = ProviderTestEvidenceStore()
    identity = _identity(credential_source="none", credential_revision=0)

    token = store.begin_generation(identity)
    assert store.settle_generation(token, ProviderGenerationProbeResult("succeeded"))

    assert store.evidence_for(identity).credential == "not_required"


def test_defaulted_credential_is_authenticated_by_direct_success_evidence():
    evidence = ProviderTestEvidence(
        _identity(),
        "not_tested",
        (),
        generation="succeeded",
    )

    assert evidence.credential == "authenticated"


def test_generation_failure_records_only_a_bounded_category():
    store = ProviderTestEvidenceStore()
    identity = _identity()

    token = store.begin_generation(identity)
    assert store.settle_generation(
        token, ProviderGenerationProbeResult("failed", "rate_limit")
    )

    assert store.evidence_for(identity).generation == "failed"
    assert store.evidence_for(identity).generation_category == "rate_limit"


def test_authentication_failure_downgrades_prior_success_for_same_identity():
    store = ProviderTestEvidenceStore()
    identity = _identity()

    success_token = store.begin_generation(identity)
    assert store.settle_generation(
        success_token, ProviderGenerationProbeResult("succeeded")
    )
    assert store.evidence_for(identity).credential == "authenticated"

    failure_token = store.begin_generation(identity)
    assert store.settle_generation(
        failure_token,
        ProviderGenerationProbeResult("failed", "authentication"),
    )

    evidence = store.evidence_for(identity)
    assert evidence.credential == "present_unverified"
    assert evidence.generation == "failed"
    assert evidence.generation_category == "authentication"


def test_generation_evidence_can_be_marked_changed_since_test():
    evidence = ProviderTestEvidence(
        _identity(),
        "not_tested",
        (),
        credential="present_unverified",
        generation="changed_since_test",
    )

    assert evidence.generation == "changed_since_test"


def test_endpoint_and_generation_tokens_are_not_interchangeable():
    store = ProviderTestEvidenceStore()
    identity = _identity()
    endpoint_token = store.begin(identity)
    generation_token = store.begin_generation(identity)

    assert not store.settle_generation(
        endpoint_token, ProviderGenerationProbeResult("succeeded")
    )
    assert not store.settle(
        generation_token, ProviderProbeResult("reachable", ("model-a",))
    )
    assert store.settle(
        endpoint_token, ProviderProbeResult("reachable", ("model-a",))
    )
    assert store.settle_generation(
        generation_token, ProviderGenerationProbeResult("succeeded")
    )


def test_generation_token_is_exact_identity_and_single_use():
    store = ProviderTestEvidenceStore()
    first = _identity()
    token = store.begin_generation(first)
    changed = _identity(
        endpoint="https://other.example.test/v1/chat/completions",
        draft_generation=5,
    )

    store.begin_generation(changed)

    assert not store.settle_generation(
        token, ProviderGenerationProbeResult("succeeded")
    )
    assert store.evidence_for(first) is None


def test_generation_begin_rejects_an_older_draft_generation():
    store = ProviderTestEvidenceStore()
    store.begin_generation(_identity(draft_generation=5))

    with pytest.raises(ValueError):
        store.begin_generation(_identity(draft_generation=4))


def test_replacement_token_wins_and_duplicate_settlement_is_rejected():
    store = ProviderTestEvidenceStore()
    identity = _identity()
    replaced = store.begin_generation(identity)
    current = store.begin_generation(identity)
    outcome = ProviderGenerationProbeResult("succeeded")

    assert not store.settle_generation(replaced, outcome)
    assert store.settle_generation(current, outcome)
    assert not store.settle_generation(current, outcome)


def test_endpoint_and_generation_settlements_merge_when_conversion_overlaps(
    monkeypatch,
):
    store = ProviderTestEvidenceStore()
    identity = _identity()
    endpoint_token = store.begin(identity)
    generation_token = store.begin_generation(identity)
    conversion_started = Event()
    continue_conversion = Event()
    original = evidence_module._evidence_from_exact_outcome

    def blocked_conversion(current_identity, current_outcome):
        conversion_started.set()
        assert continue_conversion.wait(timeout=2)
        return original(current_identity, current_outcome)

    monkeypatch.setattr(
        evidence_module,
        "_evidence_from_exact_outcome",
        blocked_conversion,
    )
    endpoint_settled: list[bool] = []
    thread = Thread(
        target=lambda: endpoint_settled.append(
            store.settle(
                endpoint_token,
                ProviderProbeResult("reachable", ("model-a",)),
            )
        )
    )
    thread.start()
    assert conversion_started.wait(timeout=2)
    try:
        assert store.settle_generation(
            generation_token, ProviderGenerationProbeResult("succeeded")
        )
    finally:
        continue_conversion.set()
    thread.join(timeout=2)

    assert endpoint_settled == [True]
    assert store.evidence_for(identity) == ProviderTestEvidence(
        identity,
        "reachable",
        ("model-a",),
        credential="authenticated",
        generation="succeeded",
    )


def test_generation_token_is_opaque_value_free_and_secret_free():
    store = ProviderTestEvidenceStore()
    token = store.begin_generation(_identity())

    assert repr(token) == "<ProviderGenerationTestToken>"
    assert not hasattr(token, "__dict__")
    assert "custom" not in repr(token)
    assert "example.test" not in repr(token)
    assert "identity" not in dir(token)


def test_extended_records_remain_frozen_and_slotted():
    identity = _identity()
    evidence = ProviderTestEvidence(identity, "reachable", ("model-a",))

    assert [field.name for field in fields(ProviderTestEvidence)] == [
        "identity",
        "endpoint",
        "model_ids",
        "category",
        "credential",
        "generation",
        "generation_category",
    ]
    assert not hasattr(evidence, "__dict__")
    with pytest.raises(AttributeError):
        evidence.generation = "failed"
