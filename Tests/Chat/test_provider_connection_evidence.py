"""TASK-33005.1: one process-memory owner of settled provider connection evidence.

Each surface (Chat settings, Settings, Console) keeps its own draft-scoped
``ProviderTestEvidenceStore``; settled facts are published to one owner
attached to the app, keyed by the connection (the draft identity without its
surface-local draft generation), and read back by every other surface.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.provider_readiness import ProviderReadiness
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderConnectionEvidence,
    ProviderDraftIdentity,
    ProviderGenerationProbeResult,
    ProviderProbeResult,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
    connection_credential_revision,
    provider_connection_evidence,
)

_LLAMA = ("llama_cpp", "http://127.0.0.1:9099")


def _identity(
    provider_key: str = "llama_cpp",
    connection_identity: tuple[str, str] = _LLAMA,
    *,
    credential_source: str = "none",
    credential_revision: int = 0,
    draft_generation: int = 1,
) -> ProviderDraftIdentity:
    return ProviderDraftIdentity(
        provider_key=provider_key,
        connection_identity=connection_identity,
        credential_source=credential_source,
        credential_revision=credential_revision,
        draft_generation=draft_generation,
    )


def _anthropic(draft_generation: int = 1) -> ProviderDraftIdentity:
    return _identity(
        "anthropic",
        ("anthropic", "https://api.anthropic.com/v1/chat/completions"),
        credential_source="stored",
        credential_revision=connection_credential_revision("sk-ant-test-key"),
        draft_generation=draft_generation,
    )


def _linked_store(app: object) -> ProviderTestEvidenceStore:
    return ProviderTestEvidenceStore(lambda: app)


def _settle(store, identity, result) -> bool:
    token = store.begin(identity)
    return store.settle(token, result)


REFUSED = ProviderProbeResult("unreachable", (), "connection_refused")
REACHABLE = ProviderProbeResult("reachable", ("model-a",))


def test_credential_revision_is_stable_secret_free_and_changes_with_the_key():
    first = connection_credential_revision("sk-test-one")
    assert first == connection_credential_revision("sk-test-one")
    assert first != connection_credential_revision("sk-test-two")
    assert connection_credential_revision(None) == 0
    assert connection_credential_revision("") == 0
    assert first > 0
    # It fits the identity's counter bound and never echoes the secret.
    assert first < 2**63
    assert "sk-test-one" not in repr(_identity(credential_revision=first))


def test_settlement_records_the_local_time_it_was_observed():
    store = ProviderTestEvidenceStore()
    identity = _identity()
    before = datetime.now().astimezone()
    assert _settle(store, identity, REACHABLE)
    observed = store.evidence_for(identity).observed_at
    assert observed is not None and observed.tzinfo is not None
    assert before <= observed <= datetime.now().astimezone()

    token = store.begin_generation(identity)
    assert store.settle_generation(token, ProviderGenerationProbeResult("succeeded"))
    assert store.evidence_for(identity).observed_at >= observed


def test_rebinding_an_earlier_observation_keeps_its_time():
    store = ProviderTestEvidenceStore()
    identity = _identity()
    earlier = datetime(2026, 10, 1, 9, 30).astimezone()
    rebound = ProviderTestEvidence(
        identity, "reachable", ("model-a",), observed_at=earlier
    )
    assert _settle(store, identity, rebound)
    assert store.evidence_for(identity).observed_at == earlier


def test_evidence_from_one_surface_reaches_another_for_the_same_connection():
    app = SimpleNamespace()
    modal_store = _linked_store(app)
    settings_store = _linked_store(app)
    tested = _identity(draft_generation=7)
    assert _settle(modal_store, tested, REFUSED)

    # Settings counts its own draft generations; the connection is the same.
    asked = _identity(draft_generation=2)
    evidence = settings_store.evidence_for(asked)
    assert evidence is not None
    assert evidence.identity == asked
    assert evidence.endpoint == "unreachable"
    assert evidence.category == "connection_refused"
    assert evidence.observed_at == modal_store.evidence_for(tested).observed_at
    # The Console reads the owner directly.
    owner = provider_connection_evidence(app)
    assert owner.evidence_for(asked) == evidence


def test_a_different_connection_never_receives_the_result():
    app = SimpleNamespace()
    store = _linked_store(app)
    assert _settle(store, _identity(), REFUSED)
    owner = provider_connection_evidence(app)

    other_provider = _anthropic()
    assert owner.evidence_for(other_provider) is None
    assert _linked_store(app).evidence_for(other_provider) is None

    other_endpoint = _identity(connection_identity=("llama_cpp", "http://127.0.0.1:8080"))
    other_key = _identity(credential_source="environment", credential_revision=5)
    for asked in (other_endpoint, other_key):
        # Draft stores only ever return exact connections.
        assert _linked_store(app).evidence_for(asked) is None


def test_changed_saved_endpoint_or_credential_reads_changed_since_test():
    app = SimpleNamespace()
    assert _settle(_linked_store(app), _identity(), REFUSED)
    owner = provider_connection_evidence(app)
    readiness = ProviderReadiness(
        provider="llama_cpp",
        provider_key="llama_cpp",
        requires_api_key=False,
        ready=True,
        api_key=None,
        api_key_source=None,
        env_var=None,
        reason="Ready",
        recovery=None,
    )
    for changed in (
        _identity(connection_identity=("llama_cpp", "http://127.0.0.1:8080")),
        _identity(credential_source="environment", credential_revision=5),
    ):
        stale = owner.evidence_for(changed)
        assert stale is not None and stale.identity != changed
        snapshot = readiness.snapshot(
            selected_model="model-a", evidence=stale, current_identity=changed
        )
        # Older evidence is marked, never reused as this connection's result.
        assert snapshot.endpoint == "changed_since_test"
        assert snapshot.category is None


def test_testing_one_provider_keeps_another_providers_result():
    app = SimpleNamespace()
    store = _linked_store(app)
    assert _settle(store, _identity(), REFUSED)
    anthropic = _anthropic()
    token = store.begin_generation(anthropic)
    assert store.settle_generation(token, ProviderGenerationProbeResult("succeeded"))

    owner = provider_connection_evidence(app)
    llama = owner.evidence_for(_identity(draft_generation=9))
    assert llama is not None and llama.endpoint == "unreachable"
    cloud = owner.evidence_for(_anthropic(draft_generation=9))
    assert cloud is not None and cloud.generation == "succeeded"
    assert cloud.credential == "authenticated"


def test_draft_edits_never_erase_shared_evidence():
    app = SimpleNamespace()
    settings_store = _linked_store(app)
    assert _settle(settings_store, _identity(), REFUSED)
    # Every semantic edit in Settings calls the no-argument invalidate().
    assert settings_store.invalidate()
    assert provider_connection_evidence(app).evidence_for(_identity()) is not None


def test_unsaved_draft_evidence_applies_only_to_that_draft_until_rebased():
    from tldw_chatbook.config import ConfigMutationResult

    app = SimpleNamespace()
    store = _linked_store(app)
    key = connection_credential_revision("sk-typed-key")
    tested = _identity(credential_source="draft", credential_revision=key)
    saved = _identity(
        credential_source="stored", credential_revision=key, draft_generation=2
    )
    assert _settle(store, tested, REACHABLE)
    owner = provider_connection_evidence(app)
    # The saved connection does not see an unsaved typed key's result.
    assert owner.evidence_for(saved).identity != saved

    lease = store.begin_save(tested)
    assert store.rebase_after_save(
        tested, saved, ConfigMutationResult(True, True, None), lease=lease
    )
    carried = owner.evidence_for(replace(saved, draft_generation=11))
    assert carried is not None and carried.endpoint == "reachable"
    assert carried.observed_at == store.evidence_for(saved).observed_at


def test_stale_and_duplicate_settlements_are_rejected_per_connection():
    app = SimpleNamespace()
    store = _linked_store(app)
    identity = _identity()
    replaced = store.begin(identity)
    current = store.begin(identity)
    owner = provider_connection_evidence(app)
    version = owner.version

    assert not store.settle(replaced, REACHABLE)
    assert owner.version == version
    assert store.settle(current, REFUSED)
    assert not store.settle(current, REACHABLE)
    assert owner.evidence_for(identity).endpoint == "unreachable"
    assert owner.version == version + 1


def test_an_older_observation_never_overwrites_a_newer_one():
    owner = ProviderConnectionEvidence()
    identity = _identity()
    newer = datetime.now().astimezone()
    owner.publish(ProviderTestEvidence(identity, "unreachable", (), "timeout", observed_at=newer))
    owner.publish(
        ProviderTestEvidence(
            identity, "reachable", ("model-a",), observed_at=newer - timedelta(minutes=5)
        )
    )
    assert owner.evidence_for(identity).endpoint == "unreachable"


def test_read_through_never_matches_an_identity_begin_would_refuse():
    """TASK-33001.3's invariant still holds with a shared owner behind a store."""
    app = SimpleNamespace()
    assert _settle(_linked_store(app), _identity(draft_generation=1), REFUSED)
    store = _linked_store(app)
    store.begin(_identity(draft_generation=5))
    assert store.evidence_for(_identity(draft_generation=1)) is None
    with pytest.raises(ValueError):
        store.begin(_identity(draft_generation=1))


def test_publishing_one_facet_keeps_the_other_connection_fact():
    app = SimpleNamespace()
    endpoint_surface = _linked_store(app)
    generation_surface = _linked_store(app)
    identity = _identity()
    assert _settle(endpoint_surface, identity, REFUSED)
    token = generation_surface.begin_generation(replace(identity, draft_generation=4))
    assert generation_surface.settle_generation(
        token, ProviderGenerationProbeResult("failed", "timeout")
    )
    merged = provider_connection_evidence(app).evidence_for(identity)
    assert merged.endpoint == "unreachable"
    assert merged.generation == "failed"


def test_concurrent_worker_settlements_for_two_connections_both_land():
    app = SimpleNamespace()
    owner = provider_connection_evidence(app)
    identities = [_identity(), _anthropic()]
    stores = [_linked_store(app) for _ in identities]
    tokens = [store.begin(identity) for store, identity in zip(stores, identities)]
    start = threading.Barrier(len(stores))
    results: list[bool] = []

    def settle(index: int) -> None:
        start.wait()
        for _ in range(200):
            owner.evidence_for(identities[1 - index])
        results.append(stores[index].settle(tokens[index], REFUSED))

    threads = [threading.Thread(target=settle, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True, True]
    for identity in identities:
        evidence = owner.evidence_for(replace(identity, draft_generation=99))
        assert evidence is not None and evidence.endpoint == "unreachable"
    assert owner.version == 2


def test_owner_is_app_attached_process_memory_only():
    app = SimpleNamespace()
    owner = provider_connection_evidence(app)
    assert provider_connection_evidence(app) is owner
    assert _settle(_linked_store(app), _identity(), REFUSED)
    # A restarted process is a new app object: every connection is untested.
    assert provider_connection_evidence(SimpleNamespace()).evidence_for(_identity()) is None
    # ADR-033: one narrow private attribute, not a root state object.
    assert [name for name in vars(app)] == ["_provider_connection_evidence"]


def test_store_without_an_active_app_stays_draft_only():
    from textual._context import NoActiveAppError

    def no_app() -> object:
        raise NoActiveAppError()

    store = ProviderTestEvidenceStore(no_app)
    assert _settle(store, _identity(), REFUSED)
    assert store.evidence_for(_identity()) is not None


@pytest.mark.parametrize("observed_at", ["09:30", 1700000000])
def test_observed_time_must_be_a_datetime(observed_at):
    with pytest.raises(ValueError):
        ProviderTestEvidence(_identity(), "reachable", (), observed_at=observed_at)
