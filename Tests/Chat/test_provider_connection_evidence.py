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
    # TASK-33005.3 review round 1 (rewritten on purpose): the generation fact
    # has its own time; the listing's time is never restamped by it.
    settled = store.evidence_for(identity)
    assert settled.observed_at == observed
    assert settled.generation_observed_at >= observed


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
        # Neither the owner nor any draft store hands over another connection.
        assert owner.evidence_for(asked) is None
        assert _linked_store(app).evidence_for(asked) is None


def test_a_discarded_draft_test_leaves_the_saved_connection_untested():
    """Review I-2: testing a draft endpoint, then discarding it, must not make
    the never-tested saved connection read 'changed since test'."""
    app = SimpleNamespace()
    settings_store = _linked_store(app)
    draft = _identity(connection_identity=("llama_cpp", "http://127.0.0.1:8080"))
    assert _settle(settings_store, draft, REACHABLE)
    assert settings_store.invalidate()  # Revert discards the draft.

    saved = _identity(draft_generation=3)
    assert provider_connection_evidence(app).evidence_for(saved) is None
    assert _linked_store(app).evidence_for(saved) is None


def test_changed_saved_endpoint_or_credential_never_reuses_older_evidence():
    """AC#4: a changed endpoint or credential is another connection. The owner
    never hands it the older record, and the surface still holding that
    record projects it as changed since test."""
    app = SimpleNamespace()
    store = _linked_store(app)
    assert _settle(store, _identity(), REFUSED)
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
        assert owner.evidence_for(changed) is None
        assert store.evidence_for(changed) is None
        snapshot = readiness.snapshot(
            selected_model="model-a",
            evidence=store.latest_evidence(),
            current_identity=changed,
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
    # The saved connection sees nothing of an unsaved typed key's result.
    assert owner.evidence_for(saved) is None

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


def test_an_earlier_begun_publication_never_overwrites_a_later_one():
    owner = ProviderConnectionEvidence()
    identity = _identity()
    assert owner.publish(
        ProviderTestEvidence(identity, "unreachable", (), "timeout"), order=2
    )
    assert not owner.publish(
        ProviderTestEvidence(identity, "reachable", ("model-a",)), order=1
    )
    assert owner.evidence_for(identity).endpoint == "unreachable"


def test_a_probe_begun_earlier_that_settles_last_never_overwrites_a_newer_one():
    """Review I-1: Settings 't' hangs on a dead server; Chat settings starts a
    probe after the server recovers and settles reachable first. The late
    timeout must not replace that fresher result in the shared owner."""
    app = SimpleNamespace()
    settings_store, chat_settings_store = _linked_store(app), _linked_store(app)
    hung = settings_store.begin(_identity(draft_generation=3))
    recovered = chat_settings_store.begin(_identity(draft_generation=8))
    assert chat_settings_store.settle(recovered, REACHABLE)
    late = ProviderProbeResult("unreachable", (), "timeout")
    # Settings still shows the result its own probe produced ...
    assert settings_store.settle(hung, late)
    assert settings_store.evidence_for(_identity(draft_generation=3)).endpoint == (
        "unreachable"
    )
    # ... but the shared owner keeps the probe that began last.
    owner = provider_connection_evidence(app)
    assert owner.evidence_for(_identity()).endpoint == "reachable"


def test_a_clock_stepping_backward_never_reorders_results(monkeypatch):
    from tldw_chatbook.Chat import provider_test_evidence as evidence_module

    noon = datetime(2026, 10, 1, 12, 0).astimezone()
    times = iter([noon, noon - timedelta(hours=1)])
    monkeypatch.setattr(evidence_module, "_local_now", lambda: next(times))
    app = SimpleNamespace()
    store = _linked_store(app)
    assert _settle(store, _identity(), REFUSED)
    assert _settle(store, _identity(), REACHABLE)  # The clock stepped back.
    later = provider_connection_evidence(app).evidence_for(_identity())
    assert later.endpoint == "reachable"
    assert later.observed_at == noon - timedelta(hours=1)


def test_rebinding_an_earlier_result_never_makes_it_newer():
    """A model-only edit rebinds Chat settings' earlier result to its next
    draft; the rebind is not a new observation, so the shared owner keeps
    the newer result another surface settled meanwhile."""
    app = SimpleNamespace()
    chat_settings_store, settings_store = _linked_store(app), _linked_store(app)
    first = _identity(draft_generation=1)
    assert _settle(chat_settings_store, first, REFUSED)
    earlier = chat_settings_store.evidence_for(first)
    assert _settle(settings_store, _identity(draft_generation=4), REACHABLE)
    owner = provider_connection_evidence(app)
    version = owner.version

    rebound = _identity(draft_generation=2)
    assert _settle(chat_settings_store, rebound, replace(earlier, identity=rebound))
    token = chat_settings_store.begin_generation(rebound)
    generation = ProviderTestEvidence(
        rebound, "not_tested", (), generation="failed", generation_category="timeout"
    )
    assert chat_settings_store.settle_generation(token, generation)

    assert owner.evidence_for(rebound).endpoint == "reachable"
    assert owner.evidence_for(rebound).generation == "not_tested"
    assert owner.version == version


def test_republishing_the_same_facts_at_the_same_time_is_not_a_change():
    owner = ProviderConnectionEvidence()
    seen = ProviderTestEvidence(
        _identity(), "reachable", ("model-a",), observed_at=datetime.now().astimezone()
    )
    assert owner.publish(seen, order=1)
    version = owner.version
    assert not owner.publish(seen, order=2)
    assert owner.version == version


def test_a_connection_that_sends_no_key_is_one_connection_whatever_names_it():
    """Review finding 5: Chat settings takes an explicit credential_source,
    Settings takes what resolves; they disagree only when no key resolves
    (e.g. credential_source = "stored" with no key). Both send nothing, so
    both are the same connection."""
    owner = ProviderConnectionEvidence()
    refused = ProviderTestEvidence(
        _identity(credential_source="stored"), "unreachable", (), "connection_refused"
    )
    assert owner.publish(refused, order=1)
    for source in ("none", "environment"):
        found = owner.evidence_for(_identity(credential_source=source))
        assert found is not None and found.endpoint == "unreachable"
    keyed = _identity(credential_source="stored", credential_revision=5)
    assert owner.evidence_for(keyed) is None
    # A later fact under another label merges into the same record.
    assert owner.publish(
        ProviderTestEvidence(
            _identity(credential_source="none"),
            "not_tested",
            (),
            generation="failed",
            generation_category="timeout",
        ),
        order=2,
    )
    merged = owner.evidence_for(_identity())
    assert (merged.endpoint, merged.generation) == ("unreachable", "failed")


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


@pytest.mark.parametrize(
    "observed_at",
    ["09:30", 1700000000, datetime(2026, 10, 1, 9, 30)],  # noqa: DTZ001 - naive on purpose
)
def test_observed_time_must_be_an_aware_datetime(observed_at):
    with pytest.raises(ValueError):
        ProviderTestEvidence(_identity(), "reachable", (), observed_at=observed_at)
