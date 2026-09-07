import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.widgets import Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import (
    _active_destination_screen,
    _static_text,
)
from Tests.UI.test_settings_configuration_hub import (
    StyledSettingsDestinationHarness,
    _click_scrolled_settings_button,
    _open_settings_category,
    _wait_for_settings_text,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderDraftIdentity,
    ProviderProbeResult,
    ProviderTestEvidence,
    ProviderTestEvidenceStore,
)
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbeOutcome,
)
from tldw_chatbook.UI.Screens.settings_screen import (
    SettingsScreen,
    overlay_provider_draft_config,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConnectionState,
)


def _semantic_identity(
    endpoint: str,
    *,
    provider_key: str = "custom",
    credential_source: str = "none",
    credential_revision: int = 0,
    draft_generation: int = 1,
) -> ProviderDraftIdentity:
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )

    connection_identity = canonical_connection_identity(provider_key, endpoint)
    assert connection_identity is not None
    return ProviderDraftIdentity(
        provider_key=provider_key,
        connection_identity=connection_identity,
        credential_source=credential_source,
        credential_revision=credential_revision,
        draft_generation=draft_generation,
    )


def _settled_store(identity: ProviderDraftIdentity) -> ProviderTestEvidenceStore:
    store = ProviderTestEvidenceStore()
    token = store.begin(identity)
    assert store.settle(
        token,
        ProviderTestEvidence(identity, "reachable", ("model-a", "model-b")),
    )
    return store


def _settle_identity(
    store: ProviderTestEvidenceStore, identity: ProviderDraftIdentity
) -> ProviderTestEvidence:
    evidence = ProviderTestEvidence(
        identity, "reachable", (f"model-{identity.draft_generation}",)
    )
    token = store.begin(identity)
    assert store.settle(token, evidence)
    return evidence


def _rebase_after_save(
    store: ProviderTestEvidenceStore,
    tested: ProviderDraftIdentity,
    saved: ProviderDraftIdentity,
    mutation: ConfigMutationResult,
) -> bool:
    lease = store.begin_save(tested)
    return store.rebase_after_save(tested, saved, mutation, lease=lease)


def test_equivalent_url_save_rebases_evidence_only_after_fully_applied():
    tested = _semantic_identity(
        "https://example.test/proxy/v1/models", draft_generation=4
    )
    saved = _semantic_identity(
        "https://example.test/proxy/v1/chat/completions", draft_generation=5
    )
    store = _settled_store(tested)

    partial = ConfigMutationResult(True, False, "cache_reload")
    assert not _rebase_after_save(store, tested, saved, partial)
    assert store.evidence_for(saved) is None
    assert store.evidence_for(tested) is None

    store = _settled_store(tested)
    applied = ConfigMutationResult(True, True, None)
    assert _rebase_after_save(store, tested, saved, applied)
    rebound = store.evidence_for(saved)
    assert rebound is not None
    assert rebound.identity == saved
    assert rebound.model_ids == ("model-a", "model-b")


def test_exact_draft_credential_save_rebases_to_stored_source():
    tested = _semantic_identity(
        "https://example.test/v1/models",
        credential_source="draft",
        credential_revision=8,
        draft_generation=4,
    )
    saved = _semantic_identity(
        "https://example.test/v1/chat/completions",
        credential_source="stored",
        credential_revision=8,
        draft_generation=5,
    )
    store = _settled_store(tested)

    assert _rebase_after_save(
        store, tested, saved, ConfigMutationResult(True, True, None)
    )
    rebound = store.evidence_for(saved)
    assert rebound is not None
    assert rebound.identity.credential_source == "stored"
    assert rebound.model_ids == ("model-a", "model-b")


@pytest.mark.parametrize(
    ("tested_source", "saved_source"),
    [
        ("stored", "environment"),
        ("stored", "none"),
        ("environment", "stored"),
        ("none", "stored"),
    ],
)
def test_other_credential_source_transitions_do_not_rebase(tested_source, saved_source):
    tested = _semantic_identity(
        "https://example.test/v1/models",
        credential_source=tested_source,
        credential_revision=8,
        draft_generation=4,
    )
    saved = _semantic_identity(
        "https://example.test/v1/chat/completions",
        credential_source=saved_source,
        credential_revision=8,
        draft_generation=5,
    )
    store = _settled_store(tested)

    assert not _rebase_after_save(
        store, tested, saved, ConfigMutationResult(True, True, None)
    )
    assert store.evidence_for(saved) is None


def test_draft_to_stored_revision_change_does_not_rebase():
    tested = _semantic_identity(
        "https://example.test/v1/models",
        credential_source="draft",
        credential_revision=8,
        draft_generation=4,
    )
    saved = _semantic_identity(
        "https://example.test/v1/chat/completions",
        credential_source="stored",
        credential_revision=9,
        draft_generation=5,
    )
    store = _settled_store(tested)

    assert not _rebase_after_save(
        store, tested, saved, ConfigMutationResult(True, True, None)
    )


@pytest.mark.parametrize(
    "changed",
    [
        _semantic_identity(
            "https://other.test/v1/chat/completions", draft_generation=2
        ),
        _semantic_identity(
            "https://example.test/v1/chat/completions",
            provider_key="openai",
            draft_generation=2,
        ),
        _semantic_identity(
            "https://example.test/v1/chat/completions",
            credential_source="draft",
            draft_generation=2,
        ),
        _semantic_identity(
            "https://example.test/v1/chat/completions",
            credential_revision=1,
            draft_generation=2,
        ),
    ],
)
def test_successful_save_does_not_rebase_changed_semantics(changed):
    tested = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    store = _settled_store(tested)

    assert not _rebase_after_save(
        store, tested, changed, ConfigMutationResult(True, True, None)
    )
    assert store.evidence_for(changed) is None
    assert store.evidence_for(tested) is None


def test_model_choice_from_returned_ids_does_not_invalidate_endpoint_evidence():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=7
    )
    store = _settled_store(identity)

    for selected_model in ("model-a", "model-b"):
        evidence = store.evidence_for(identity)
        assert evidence is not None
        assert selected_model in evidence.model_ids


def test_failed_save_does_not_rebase_evidence_to_saved_identity():
    tested = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    saved = _semantic_identity(
        "https://example.test/v1/models", draft_generation=2
    )
    store = _settled_store(tested)

    assert not _rebase_after_save(
        store,
        tested,
        saved,
        ConfigMutationResult(False, False, "before_replace"),
    )
    assert store.evidence_for(saved) is None
    assert store.evidence_for(tested) is None


def test_conflict_invalidates_even_when_mutation_claims_fully_applied():
    tested = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    saved = _semantic_identity(
        "https://example.test/v1/models", draft_generation=2
    )
    store = _settled_store(tested)

    assert not _rebase_after_save(
        store,
        tested,
        saved,
        ConfigMutationResult(True, True, None, conflict=True),
    )
    assert store.evidence_for(tested) is None
    assert store.evidence_for(saved) is None


def test_conflict_invalidates_active_test_token():
    tested = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    saved = _semantic_identity(
        "https://example.test/v1/models", draft_generation=2
    )
    store = ProviderTestEvidenceStore()
    token = store.begin(tested)
    lease = store.begin_save(tested)

    assert not store.rebase_after_save(
        tested,
        saved,
        ConfigMutationResult(True, True, None, conflict=True),
        lease=lease,
    )
    assert store.evidence_for(tested) is None
    assert not store.settle(
        token,
        ProviderTestEvidence(tested, "reachable", ("model-a",)),
    )


def test_late_partial_save_does_not_clear_newer_settled_evidence():
    first = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    second = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = ProviderTestEvidenceStore()
    _settle_identity(store, first)
    first_lease = store.begin_save(first)
    newer_evidence = _settle_identity(store, second)

    assert not store.rebase_after_save(
        first,
        first,
        ConfigMutationResult(False, False, "before_replace"),
        lease=first_lease,
    )
    assert store.evidence_for(second) == newer_evidence


@pytest.mark.parametrize(
    "mutation",
    [
        ConfigMutationResult(False, False, "before_replace", conflict=True),
        ConfigMutationResult(True, True, None, conflict=True),
    ],
    ids=["conflict", "conflict-fully-applied"],
)
def test_late_conflict_does_not_clear_newer_settled_evidence(mutation):
    first = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    second = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = ProviderTestEvidenceStore()
    _settle_identity(store, first)
    first_lease = store.begin_save(first)
    newer_evidence = _settle_identity(store, second)

    assert not store.rebase_after_save(first, first, mutation, lease=first_lease)
    assert store.evidence_for(second) == newer_evidence


@pytest.mark.parametrize(
    "mutation",
    [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(True, True, None, conflict=True),
    ],
    ids=["partial", "conflict"],
)
def test_stale_save_result_does_not_cancel_newer_active_test(mutation):
    first = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=1
    )
    second = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = ProviderTestEvidenceStore()
    _settle_identity(store, first)
    first_lease = store.begin_save(first)
    token = store.begin(second)

    assert not store.rebase_after_save(first, first, mutation, lease=first_lease)
    testing = store.evidence_for(second)
    assert testing is not None
    assert testing.endpoint == "testing"

    settled = ProviderTestEvidence(second, "reachable", ("model-2",))
    assert store.settle(token, settled)
    assert store.evidence_for(second) == settled


def test_successful_save_cannot_rebase_to_an_older_draft_generation():
    tested = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=3
    )
    older = _semantic_identity(
        "https://example.test/v1/models", draft_generation=2
    )
    store = _settled_store(tested)

    assert not _rebase_after_save(
        store, tested, older, ConfigMutationResult(True, True, None)
    )
    assert store.evidence_for(older) is None


def test_save_lease_is_immutable_value_free_and_secret_free():
    identity = _semantic_identity(
        "https://secret-host.test/v1/chat/completions", draft_generation=1
    )
    store = _settled_store(identity)

    lease = store.begin_save(identity)

    assert repr(lease) == "<ProviderEvidenceSaveLease>"
    assert not hasattr(lease, "__dict__")
    assert "custom" not in repr(lease)
    assert "secret-host" not in repr(lease)
    assert "identity" not in dir(lease)
    assert "epoch" not in dir(lease)
    with pytest.raises(AttributeError):
        lease.identity = identity


@pytest.mark.parametrize(
    "mutation",
    [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(True, True, None, conflict=True),
        ConfigMutationResult(True, True, None),
    ],
    ids=["partial", "conflict", "success"],
)
def test_same_identity_late_save_cannot_change_newer_settled_state(mutation):
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=3
    )
    store = _settled_store(identity)
    stale_lease = store.begin_save(identity)
    newer = _settle_identity(store, identity)

    assert not store.rebase_after_save(
        identity,
        identity,
        mutation,
        lease=stale_lease,
    )
    assert store.evidence_for(identity) == newer


@pytest.mark.parametrize(
    "mutation",
    [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(True, True, None, conflict=True),
        ConfigMutationResult(True, True, None),
    ],
    ids=["partial", "conflict", "success"],
)
def test_same_identity_late_save_cannot_cancel_newer_active_test(mutation):
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=3
    )
    store = _settled_store(identity)
    stale_lease = store.begin_save(identity)
    token = store.begin(identity)

    assert not store.rebase_after_save(
        identity,
        identity,
        mutation,
        lease=stale_lease,
    )
    settled = ProviderTestEvidence(identity, "reachable", ("model-new",))
    assert store.settle(token, settled)
    assert store.evidence_for(identity) == settled


def test_save_lease_is_single_use_after_successful_rebase():
    tested = _semantic_identity(
        "https://example.test/v1/models", draft_generation=2
    )
    saved = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=3
    )
    store = _settled_store(tested)
    lease = store.begin_save(tested)
    mutation = ConfigMutationResult(True, True, None)

    assert store.rebase_after_save(tested, saved, mutation, lease=lease)
    rebound = store.evidence_for(saved)
    assert not store.rebase_after_save(tested, saved, mutation, lease=lease)
    assert store.evidence_for(saved) == rebound


def test_rejected_save_callback_consumes_lease():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = _settled_store(identity)
    lease = store.begin_save(identity)

    assert not store.rebase_after_save(identity, identity, object(), lease=lease)
    assert not store.rebase_after_save(
        identity,
        identity,
        ConfigMutationResult(True, True, None),
        lease=lease,
    )


def test_parallel_save_lease_becomes_stale_after_first_rebase():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = _settled_store(identity)
    first = store.begin_save(identity)
    second = store.begin_save(identity)
    mutation = ConfigMutationResult(True, True, None)

    assert not store.rebase_after_save(identity, identity, mutation, lease=first)
    assert store.rebase_after_save(identity, identity, mutation, lease=second)


def test_mismatched_save_callback_cannot_consume_current_exact_lease():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    other = _semantic_identity(
        "https://other.test/v1/chat/completions", draft_generation=2
    )
    store = _settled_store(identity)
    lease = store.begin_save(identity)
    mutation = ConfigMutationResult(True, True, None)

    assert not store.rebase_after_save(other, other, mutation, lease=lease)
    assert store.rebase_after_save(identity, identity, mutation, lease=lease)


def test_begin_save_requires_exact_current_store_identity():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    other = _semantic_identity(
        "https://other.test/v1/chat/completions", draft_generation=2
    )
    store = ProviderTestEvidenceStore()

    assert store.begin_save(identity) is None
    store.begin(identity)
    assert store.begin_save(other) is None
    assert store.begin_save(identity) is not None


def test_cancel_save_consumes_only_current_lease_without_clearing_evidence():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = _settled_store(identity)
    lease = store.begin_save(identity)
    assert lease is not None

    assert store.cancel_save(lease)
    assert not store.cancel_save(lease)
    assert not store.rebase_after_save(
        identity,
        identity,
        ConfigMutationResult(True, True, None),
        lease=lease,
    )
    assert store.evidence_for(identity) is not None


def test_save_lease_storage_remains_single_and_bounded():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = _settled_store(identity)
    latest = None

    for _ in range(100):
        latest = store.begin_save(identity)

    assert latest is not None
    assert not hasattr(store, "_save_leases")
    assert store._save_lease is not None
    assert store._save_lease[0] is latest


def test_invalidated_save_lease_cannot_rebase_recreated_identical_evidence():
    identity = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    store = _settled_store(identity)
    lease = store.begin_save(identity)
    assert store.invalidate(identity)
    recreated = _settle_identity(store, identity)

    assert not store.rebase_after_save(
        identity,
        identity,
        ConfigMutationResult(True, True, None),
        lease=lease,
    )
    assert store.evidence_for(identity) == recreated


@pytest.mark.parametrize("state", ["changed-semantics", "testing"])
def test_current_fully_applied_save_advances_generation_without_preserved_evidence(
    state,
):
    tested = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=2
    )
    saved = _semantic_identity(
        "https://example.test/v1/chat/completions", draft_generation=8
    )
    store = ProviderTestEvidenceStore()
    token = None
    if state == "changed-semantics":
        _settle_identity(store, tested)
        saved = _semantic_identity(
            "https://other.test/v1/chat/completions", draft_generation=8
        )
    elif state == "testing":
        token = store.begin(tested)

    lease = store.begin_save(tested)
    assert not store.rebase_after_save(
        tested,
        saved,
        ConfigMutationResult(True, True, None),
        lease=lease,
    )
    if token is not None:
        assert not store.settle(
            token,
            ProviderTestEvidence(tested, "reachable", ("model-a",)),
        )
    with pytest.raises(ValueError):
        store.begin(_semantic_identity(
            "https://example.test/v1/chat/completions", draft_generation=7
        ))


def _base_config():
    return {
        "api_settings": {
            "llama_cpp": {"api_url": "http://localhost:8080/completion", "api_key": "fake-saved-key-not-real"},
            "openai": {"api_key": "fake-other-key-not-real"},
        }
    }


def test_overlay_endpoint_only_deep_copies_and_preserves_others():
    base = _base_config()
    merged = overlay_provider_draft_config(
        base,
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint="http://localhost:9099",
        draft_env_var=None,
        draft_api_key=None,
    )
    # draft endpoint overlaid
    assert merged["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:9099"
    # saved key + other provider preserved
    assert merged["api_settings"]["llama_cpp"]["api_key"] == "fake-saved-key-not-real"
    assert merged["api_settings"]["openai"]["api_key"] == "fake-other-key-not-real"
    # input not mutated
    assert base["api_settings"]["llama_cpp"]["api_url"] == "http://localhost:8080/completion"


def test_overlay_api_key_and_env_var():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var="MY_LLAMA_KEY",
        draft_api_key="fake-draft-key-not-real",
    )
    section = merged["api_settings"]["llama_cpp"]
    assert section["api_key"] == "fake-draft-key-not-real"
    assert section["api_key_env_var"] == "MY_LLAMA_KEY"


def test_overlay_api_key_clear_sets_empty():
    merged = overlay_provider_draft_config(
        _base_config(),
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key="",
    )
    assert merged["api_settings"]["llama_cpp"]["api_key"] == ""


def test_overlay_creates_missing_section():
    merged = overlay_provider_draft_config(
        {"api_settings": {}},
        provider_save_key="newprov",
        endpoint_key="api_base_url",
        draft_endpoint="http://x:1/v1",
        draft_env_var=None,
        draft_api_key=None,
    )
    assert merged["api_settings"]["newprov"]["api_base_url"] == "http://x:1/v1"


def test_overlay_no_fields_is_a_faithful_copy():
    base = _base_config()
    merged = overlay_provider_draft_config(
        base,
        provider_save_key="llama_cpp",
        endpoint_key="api_url",
        draft_endpoint=None,
        draft_env_var=None,
        draft_api_key=None,
    )
    assert merged == base
    assert merged is not base


def _bare_settings_screen(app_config):
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(app_config=app_config)
    return screen


def test_provider_source_ui_honors_persisted_explicit_keyless_decision():
    screen = _bare_settings_screen(
        {
            "api_settings": {
                "custom": {
                    "credential_source": "none",
                    "api_key": "saved-settings-ui-canary",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        }
    )
    screen._provider_draft = lambda: None

    assert screen._provider_current_credential_source("custom") == "none"


def test_findings_show_draft_endpoint_tagged():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:9099"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://localhost:9099", dirty={"endpoint"},
    )
    assert "http://localhost:9099 (draft)" in detail
    assert "8080" not in detail


def test_local_configuration_check_never_claims_live_verification():
    app_config = {"api_settings": {"openai": {"api_key": "fake-test-key"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("openai", app_config, environ={})

    detail, summary, passed = screen._build_provider_readiness_findings(
        "openai", "gpt-4o", readiness, draft_endpoint="", dirty=set()
    )

    assert passed is True
    assert detail.startswith("Configuration check |")
    assert "passed" not in summary.casefold()
    assert "verified" not in summary.casefold()
    assert "live generation has not been tested" in summary.casefold()


def test_exact_evidence_copy_keeps_listing_and_generation_independent():
    identity = _semantic_identity(
        "https://example.test/v1/models",
        provider_key="openai",
        credential_source="stored",
    )
    evidence = ProviderTestEvidence(
        identity,
        "reachable",
        ("gpt-4o",),
        credential="present_unverified",
        generation="not_tested",
    )

    copy = SettingsScreen._provider_exact_evidence_copy(evidence, "gpt-4o")

    assert "credential present, not verified" in copy
    assert "model listing reached" in copy
    assert "selected model confirmed" in copy
    assert "generation not tested" in copy
    assert "generation succeeded" not in copy


@pytest.mark.parametrize(
    ("category", "expected"),
    (
        ("connection_refused", "connection refused"),
        ("timeout", "timeout"),
        ("unauthorized", "unauthorized"),
    ),
)
def test_exact_evidence_copy_distinguishes_endpoint_failure_categories(
    category,
    expected,
):
    identity = _semantic_identity("https://example.test/v1/models")
    evidence = ProviderTestEvidence(
        identity,
        "unreachable",
        (),
        category=category,
    )

    copy = SettingsScreen._provider_exact_evidence_copy(evidence, "model-a")

    assert f"model listing failed ({expected})" in copy


def test_provider_edit_stale_copy_requires_a_new_configuration_check():
    copy = SettingsScreen._PROVIDER_TEST_STALE_COPY

    assert "changed since the last check" in copy
    assert "re-run Configuration check" in copy


def test_findings_relabel_draft_api_key_source_and_hide_value():
    app_config = {"api_settings": {"openai": {"api_key": "fake-draft-key-not-real"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    detail, summary, _passed = screen._build_provider_readiness_findings(
        "OpenAI", "gpt-4o", readiness,
        draft_endpoint="", dirty={"api_key"},
    )
    assert "api_key_source=draft api_key (unsaved)" in detail
    assert "fake-draft-key-not-real" not in detail and "fake-draft-key-not-real" not in summary


def test_findings_tag_draft_env_var_and_never_leak_value():
    # A custom-named credential env var whose NAME does not match the secret
    # pattern -- its raw value must still never be printed (task-483 folded in),
    # only presence via the ``<redacted>`` marker, plus the draft tag.
    app_config = {"api_settings": {"openai": {"api_key_env_var": "MY_CUSTOM_CRED"}}}
    screen = _bare_settings_screen(app_config)
    with patch.dict(os.environ, {"MY_CUSTOM_CRED": "env-secret-XYZ"}, clear=False):
        readiness = get_provider_readiness("OpenAI", app_config)
        detail, summary, _passed = screen._build_provider_readiness_findings(
            "OpenAI", "gpt-4o", readiness,
            draft_endpoint="", dirty={"credential_env_var"},
        )
    assert "(draft env var)" in detail
    assert "MY_CUSTOM_CRED=<redacted>" in detail
    assert "env-secret-XYZ" not in detail and "env-secret-XYZ" not in summary


def test_mask_url_userinfo_masks_password_in_endpoint():
    from tldw_chatbook.UI.Screens.settings_screen import _mask_url_userinfo

    assert _mask_url_userinfo("http://user:s3cret@host:8080/v1") == "http://user:***@host:8080/v1"
    assert _mask_url_userinfo("http://:s3cret@host/v1") == "http://***@host/v1"
    # password-less / non-URL inputs are unchanged
    assert _mask_url_userinfo("http://localhost:9099") == "http://localhost:9099"
    assert _mask_url_userinfo("") == ""
    # username-only userinfo (no password) is left as-is
    assert _mask_url_userinfo("http://user@host/v1") == "http://user@host/v1"
    # malformed/out-of-range port must not raise (uses .port property otherwise)
    assert _mask_url_userinfo("http://u:p@host:99999/v1") == "http://u:***@host:99999/v1"
    assert _mask_url_userinfo("http://u:p@host:notaport/v1") == "http://u:***@host:notaport/v1"
    # IPv6 host keeps its brackets while the password is masked
    assert (
        _mask_url_userinfo("http://u:p@[::1]:8080/v1") == "http://u:***@[::1]:8080/v1"
    )


def test_findings_mask_endpoint_userinfo_password():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://user:hunter2@host:9099/v1", dirty={"endpoint"},
    )
    assert "http://user:***@host:9099/v1 (draft)" in detail
    assert "hunter2" not in detail and "hunter2" not in summary


def test_findings_no_draft_has_no_tags():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp", "llama-3", readiness,
        draft_endpoint="http://localhost:8080", dirty=set(),
    )
    assert "(draft)" not in detail and "(unsaved)" not in detail
    assert "http://localhost:8080" in detail


def test_findings_avoid_ready_claim_when_blocked_on_missing_model():
    """TASK-366: a config-ready provider with no default model must not read
    'is ready' next to 'configuration=blocked' — the detail leads with one verdict
    consistent with the final status line, and still explains the block."""
    app_config = {"api_settings": {"openai": {"api_key": "placeholder-not-a-real-key"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    assert readiness.ready is True  # config-level readiness is fine...

    detail, _summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "", readiness,
        draft_endpoint="", dirty=set(),
    )

    assert passed is False
    assert "configuration=blocked" in detail
    assert "is ready" not in detail  # no contradictory ready claim
    assert "model" in detail.lower()  # verdict still explains the block


def test_findings_keep_configuration_only_verdict_when_passing():
    """A local pass describes configuration without claiming provider readiness."""
    app_config = {"api_settings": {"openai": {"api_key": "placeholder-not-a-real-key"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})

    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI", "gpt-4o", readiness,
        draft_endpoint="", dirty=set(),
    )

    assert passed is True
    assert "configuration=complete" in detail
    assert "is ready" not in detail
    assert "status=ready" not in detail
    assert "configured" in summary.lower()
    assert "live generation has not been tested" in summary.lower()


def test_mark_provider_test_result_stale_invalidates_prior_verdict():
    """TASK-366: editing a provider input must invalidate a prior Test result so
    a stale 'ready'/'blocked' verdict cannot linger while the form has changed.
    No-op when nothing has run or it is already stale."""
    screen = _bare_settings_screen({})
    screen._provider_test_result = (
        "Configuration check | llama.cpp configuration is complete | "
        "model=llama-3 | configuration=complete"
    )

    screen._mark_provider_test_result_stale()
    assert "re-run" in screen._provider_test_result.lower()

    # Idempotent: a second edit does not re-flag or accumulate.
    stale = screen._provider_test_result
    screen._mark_provider_test_result_stale()
    assert screen._provider_test_result == stale

    # No-op on the never-run sentinel.
    screen._provider_test_result = SettingsScreen._PROVIDER_TEST_NOT_RUN_COPY
    screen._mark_provider_test_result_stale()
    assert screen._provider_test_result == SettingsScreen._PROVIDER_TEST_NOT_RUN_COPY


def test_settings_converts_probe_outcome_to_exact_evidence_dto():
    outcome = SettingsEndpointProbeOutcome(
        state="reachable",
        summary="reachable (2 models)",
        model_ids=("model-a", "model-b"),
    )

    converted = SettingsScreen._provider_probe_result_from_outcome(outcome)

    assert type(converted) is ProviderProbeResult
    assert converted == ProviderProbeResult(
        endpoint="reachable",
        model_ids=("model-a", "model-b"),
    )


def test_settings_converts_tts_enum_probe_state_to_exact_chat_string():
    outcome = SettingsEndpointProbeOutcome(
        state=SpeechTTSConnectionState.UNREACHABLE,
        summary="unreachable: timeout",
        category="timeout",
    )

    converted = SettingsScreen._provider_probe_result_from_outcome(outcome)

    assert type(converted.endpoint) is str
    assert converted == ProviderProbeResult(
        endpoint="unreachable",
        model_ids=(),
        category="timeout",
    )


def test_model_edit_cancels_active_probe_token_but_not_settled_evidence():
    identity = _semantic_identity("https://example.test/v1/models")
    screen = _bare_settings_screen({})
    store = ProviderTestEvidenceStore()
    screen._provider_test_evidence_store = store
    screen._provider_current_draft_identity = lambda: identity
    screen._provider_test_result = "Provider test | endpoint probe: checking"
    screen._update_provider_test_result = lambda: None
    token = store.begin(identity)

    screen._update_provider_evidence_for_edit("model", "model-b")

    assert not store.settle(
        token,
        ProviderProbeResult(endpoint="reachable", model_ids=("model-a",)),
    )
    assert store.evidence_for(identity) is None
    assert "re-run" in screen._provider_test_result.lower()

    settled_token = store.begin(identity)
    assert store.settle(
        settled_token,
        ProviderProbeResult(
            endpoint="reachable",
            model_ids=("model-a", "model-b"),
        ),
    )
    screen._provider_test_result = "Provider test | endpoint reachable"
    screen._update_provider_evidence_for_edit("model", "model-b")
    assert store.evidence_for(identity) is not None
    assert "re-run" not in screen._provider_test_result.lower()


@pytest.mark.asyncio
async def test_probe_worker_cancellation_clears_exact_testing_state(monkeypatch):
    identity = _semantic_identity("https://example.test/v1/models")
    screen = _bare_settings_screen({})
    store = ProviderTestEvidenceStore()
    screen._provider_test_evidence_store = store
    screen._provider_test_result = "Provider test | endpoint probe: checking"
    screen._update_provider_test_result = lambda: None
    token = store.begin(identity)

    async def cancelled_probe(*_args, **_kwargs):
        raise asyncio.CancelledError("secret-cancel-detail")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        cancelled_probe,
    )

    with pytest.raises(asyncio.CancelledError):
        await SettingsScreen._provider_endpoint_probe_worker.__wrapped__(
            screen,
            "https://example.test/v1",
            "custom",
            "Provider test",
            "Provider test",
            identity,
            token,
        )

    assert store.evidence_for(identity) is None
    assert "checking" not in screen._provider_test_result.lower()
    assert "cancel" in screen._provider_test_result.lower()
    assert "secret-cancel-detail" not in screen._provider_test_result


@pytest.mark.asyncio
async def test_chat_settings_probe_worker_passes_explicit_chat_catalog_purpose(
    monkeypatch,
):
    screen = _bare_settings_screen({})
    screen._update_provider_test_result = lambda: None
    screen._apply_provider_endpoint_probe_outcome = lambda *_args, **_kwargs: None
    captured: dict[str, object] = {}

    async def capture_probe(base_url, **kwargs):
        captured["base_url"] = base_url
        captured.update(kwargs)
        return SettingsEndpointProbeOutcome(
            state="reachable",
            summary="reachable (1 model)",
            model_ids=("gpt-4o",),
        )

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        capture_probe,
    )

    await SettingsScreen._provider_endpoint_probe_worker.__wrapped__(
        screen,
        "https://example.test/v1/chat/completions",
        "openai",
        "Provider test",
        "Provider test passed",
    )

    assert captured == {
        "base_url": "https://example.test/v1/chat/completions",
        "provider": "openai",
        "purpose": "chat_catalog",
    }


@pytest.mark.asyncio
async def test_stale_probe_cancellation_does_not_clear_newer_testing_token(monkeypatch):
    older = _semantic_identity(
        "https://example.test/v1/models",
        draft_generation=1,
    )
    newer = _semantic_identity(
        "https://example.test/v1/models",
        draft_generation=2,
    )
    screen = _bare_settings_screen({})
    store = ProviderTestEvidenceStore()
    screen._provider_test_evidence_store = store
    screen._provider_test_result = "New provider test | endpoint probe: checking"
    screen._update_provider_test_result = lambda: None
    stale_token = store.begin(older)
    current_token = store.begin(newer)

    async def cancelled_probe(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        cancelled_probe,
    )

    with pytest.raises(asyncio.CancelledError):
        await SettingsScreen._provider_endpoint_probe_worker.__wrapped__(
            screen,
            "https://example.test/v1",
            "custom",
            "Old provider test",
            "Old provider test",
            older,
            stale_token,
        )

    evidence = store.evidence_for(newer)
    assert evidence is not None
    assert evidence.endpoint == "testing"
    assert screen._provider_test_result == (
        "New provider test | endpoint probe: checking"
    )
    assert store.settle(
        current_token,
        ProviderProbeResult(endpoint="reachable", model_ids=("model-a",)),
    )


def test_discovery_status_distinguishes_malformed_from_unsupported():
    """TASK-367: the model-discovery status surfaces DISTINCT copy for a
    malformed URL vs a valid-but-unsupported path, instead of collapsing both
    into the same generic /v1 message."""
    from types import SimpleNamespace

    screen = _bare_settings_screen({})
    malformed = SimpleNamespace(
        error=SimpleNamespace(
            kind="malformed_endpoint",
            message="This endpoint is not a valid URL.",
            recovery_hint="Enter a full http:// or https:// address.",
        )
    )
    unsupported = SimpleNamespace(
        error=SimpleNamespace(
            kind="unsupported_endpoint",
            message="This endpoint is not an OpenAI-compatible models endpoint.",
            recovery_hint="Configure an explicit /v1 or /v1/models endpoint.",
        )
    )

    malformed_status = screen._discovery_status_from_error(malformed)
    unsupported_status = screen._discovery_status_from_error(unsupported)

    assert "not a valid URL" in malformed_status
    assert "not an OpenAI-compatible" in unsupported_status
    assert malformed_status != unsupported_status


def test_provider_endpoint_url_validator_flags_malformed_only():
    """TASK-367: inline (blur) validation passes an empty or well-formed URL and
    fails a malformed one, e.g. a dropped scheme character."""
    from tldw_chatbook.UI.Screens.settings_screen import ProviderEndpointURLValidator

    validator = ProviderEndpointURLValidator()
    assert validator.validate("").is_valid
    assert validator.validate("http://127.0.0.1:9099/v1").is_valid
    assert not validator.validate("ttp://127.0.0.1:9099/v1").is_valid


def test_model_to_activate_after_save_prefers_first_saved_when_field_empty():
    """TASK-369: after saving discovered models, an empty Model field is filled
    with the first saved model (recognition over recall); a field the user
    already set is left untouched."""
    activate = SettingsScreen._model_to_activate_after_save
    assert activate("", ("gemma-4.gguf", "mistral-7b.gguf")) == "gemma-4.gguf"
    assert activate("   ", ("gemma-4.gguf",)) == "gemma-4.gguf"
    assert activate("already-chosen", ("gemma-4.gguf",)) == "already-chosen"
    assert activate("", ()) == ""
    assert activate("", ("", "  ", "real.gguf")) == "real.gguf"


@pytest.mark.asyncio
async def test_model_field_suggester_completes_discovered_ids():
    """TASK-369: the Model field offers discovered model ids for typeahead, so a
    prefix completes to the full gguf name."""
    from types import SimpleNamespace

    screen = _bare_settings_screen({})
    screen._model_discovery_models = (
        SimpleNamespace(model_id="gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf"),
        SimpleNamespace(model_id="mistral-7b-instruct.Q5_K_M.gguf"),
    )

    suggester = screen._model_field_suggester()
    assert suggester is not None
    assert (
        await suggester.get_suggestion("gemma")
        == "gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf"
    )

    # No discovered models -> nothing to suggest.
    screen._model_discovery_models = ()
    assert screen._model_field_suggester() is None


def test_discovery_row_labels_use_user_vocabulary_not_internal_jargon():
    """TASK-387: the model-discovery selection rows must read in plain language,
    not the internal ``runtime_discovered`` / ``capability=unknown`` enum dump."""
    screen = _bare_settings_screen({})
    screen._model_discovery_selected_model_ids = set()
    screen._model_discovery_models = (
        SimpleNamespace(
            model_id="gemma-4.gguf",
            source="runtime_discovered",
            capability_status="unknown",
            persisted=False,
        ),
        SimpleNamespace(
            model_id="mistral-7b.gguf",
            source="persisted_discovered",
            capability_status="known",
            persisted=True,
        ),
    )

    labels = [label for label, _id, _sel in screen._model_discovery_selection_options()]
    joined = " ".join(labels)

    # Model ids are still shown for recognition.
    assert "gemma-4.gguf" in joined
    assert "mistral-7b.gguf" in joined
    # Internal enum jargon is gone.
    assert "runtime_discovered" not in joined
    assert "persisted_discovered" not in joined
    assert "capability=" not in joined
    # Replaced by user-facing vocabulary.
    assert "discovered" in labels[0]
    assert "capabilities unknown" in labels[0]
    assert "capabilities known" in labels[1]


# --- Pilot tests: the clickable Test button path (AC#2/AC#3) + widget wiring ---
#
# These drive the real SettingsScreen through the harness
# Tests/UI/test_settings_configuration_hub.py uses (StyledSettingsDestinationHarness
# is required alongside _click_scrolled_settings_button -- every existing caller
# of that helper in the suite uses the styled harness so the detail-pane scroll
# geometry the click depends on is computed from real CSS).


def _provider_test_result_text(screen) -> str:
    return _static_text(screen.query_one("#settings-provider-test-result", Static))


async def _reachable_endpoint_probe(
    _base_url: str, **_kwargs: object
) -> SettingsEndpointProbeOutcome:
    return SettingsEndpointProbeOutcome(
        state="reachable",
        summary="reachable (1 model)",
        model_ids=("llama-3",),
    )


@pytest.mark.asyncio
async def test_test_provider_button_click_runs_the_check():
    """AC#2: clicking #settings-test-provider (not the 't' hotkey) runs the test."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        # Sanity: the test has not run yet (mount-time default copy only).
        assert _provider_test_result_text(screen) == "Configuration check has not run."

        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            _reachable_endpoint_probe,
        ):
            await _click_scrolled_settings_button(
                screen, pilot, "#settings-test-provider"
            )
            await _wait_for_settings_text(screen, pilot, "model listing reached")

        result_text = _provider_test_result_text(screen)
        assert "Configuration check" in result_text
        assert result_text != "Configuration check has not run."


@pytest.mark.asyncio
async def test_test_provider_button_runs_with_provider_input_focused():
    """AC#3: a real mouse click on the button still runs the check, starting
    from an Input-focused state.

    This proves the button is a working non-hotkey path: even when a text
    entry widget starts out focused, clicking ``#settings-test-provider``
    runs the readiness check and reads the current widget values.

    Note: by the time ``Button.Pressed`` dispatches, Textual has already
    moved keyboard focus onto the Button itself, so this test does not by
    itself pin the ``allow_text_entry_focus=True`` bypass in
    ``handle_test_provider`` -- see
    ``test_t_hotkey_does_not_run_test_while_input_focused`` below, which
    pins the actual rationale (the 't' hotkey no-ops while an input has
    focus, which is why a clickable button is needed).
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        model_input = screen.query_one("#settings-model-value", Input)
        model_input.focus()
        await pilot.pause()
        # Sanity: this is exactly the state that would make the 't' hotkey no-op.
        assert screen._settings_text_entry_has_focus() is True

        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            _reachable_endpoint_probe,
        ):
            await _click_scrolled_settings_button(
                screen, pilot, "#settings-test-provider"
            )
            await _wait_for_settings_text(screen, pilot, "model listing reached")

        assert "Configuration check" in _provider_test_result_text(screen)


@pytest.mark.asyncio
async def test_t_hotkey_does_not_run_test_while_input_focused():
    """AC#3 rationale: the 't' hotkey does not run the test while a text
    entry has focus -- this is why the clickable button is needed.

    Two things are pinned here:

    1. The observable behavior a real keypress produces: pressing 't' while
       the model Input is focused types "t" into the input rather than
       running the readiness check. (Textual's own Input widget consumes
       printable keys before the Screen's ``("t", "settings_test_category",
       ...)`` binding is even considered -- see
       ``Input.check_consume_key``/``Screen._binding_chain`` -- so this
       part alone would hold even if ``action_settings_test_category``'s
       internal guard were removed.)
    2. The actual guard: ``action_settings_test_category`` (the method the
       't' binding invokes, with no arguments -- i.e.
       ``allow_text_entry_focus=False``) is a no-op while
       ``_settings_text_entry_has_focus()`` is true. Calling it directly,
       the same way the binding dispatch would, is what makes this test
       fail if that guard is ever removed -- part 1 alone would not catch
       that regression, since Textual's own key consumption already
       prevents the keypress from reaching the binding either way.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        model_input = screen.query_one("#settings-model-value", Input)
        model_input.focus()
        await pilot.pause()
        # Sanity: this is exactly the state that would make the 't' hotkey no-op.
        assert screen._settings_text_entry_has_focus() is True

        before = _provider_test_result_text(screen)
        assert before == "Configuration check has not run."

        # 1. Real keypress: consumed by the focused Input, never reaches the
        # 't' binding at all.
        await pilot.press("t")
        await pilot.pause()
        assert _provider_test_result_text(screen) == before
        assert "t" in model_input.value

        # 2. Direct action-level check -- the same call Textual's binding
        # dispatch makes for the 't' hotkey (no arguments). This is the part
        # that actually exercises `_settings_text_entry_has_focus()`.
        screen.action_settings_test_category()
        await pilot.pause()
        assert _provider_test_result_text(screen) == before


@pytest.mark.asyncio
async def test_model_edit_during_probe_rejects_late_old_model_result(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://localhost:8080", "model": "model-a"}
    }
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_probe(*_args, **_kwargs):
        started.set()
        await release.wait()
        return SettingsEndpointProbeOutcome(
            state="reachable",
            summary="reachable (1 model)",
            model_ids=("model-a",),
        )

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        delayed_probe,
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.action_settings_test_category()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert "checking" in screen._provider_test_result

        model = screen.query_one("#settings-model-value", Input)
        model.value = "model-b"
        await pilot.pause()
        release.set()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        current_identity = screen._provider_current_draft_identity()
        assert current_identity is not None
        assert screen._provider_evidence_store().evidence_for(current_identity) is None
        assert "re-run" in screen._provider_test_result.lower()
        assert "endpoint reachable" not in screen._provider_test_result.lower()


@pytest.mark.asyncio
async def test_probe_worker_unexpected_exception_settles_bounded_failure(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://localhost:8080", "model": "model-a"}
    }

    async def failing_probe(*_args, **_kwargs):
        raise RuntimeError("secret-probe-detail")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        failing_probe,
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        identity = screen._provider_current_draft_identity()
        assert identity is not None

        screen.action_settings_test_category()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        result = screen._provider_test_result
        assert "checking" not in result.lower()
        assert "connection error" in result.lower()
        assert "secret-probe-detail" not in result
        evidence = screen._provider_evidence_store().evidence_for(identity)
        assert evidence is not None
        assert evidence.endpoint == "unreachable"
        assert evidence.category == "connection_error"


@pytest.mark.asyncio
async def test_test_provider_result_shows_draft_endpoint():
    """Wiring: a staged (unsaved) endpoint edit reaches the Test result.

    Exercises the widget-reading wrapper (``_provider_readiness_test_report``)
    that Task 2's unit tests (above) did not cover, by typing a draft endpoint
    into the real ``#settings-provider-endpoint-value`` input, firing its
    change handler (staging it dirty), then running the test via the button.
    The model is left unset so readiness never "passes" and no async endpoint
    probe worker starts -- keeping the assertion on the synchronously-set
    pre-probe detail line, which is where the draft tag is threaded through.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": ""}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://localhost:8080"},
        "openai": {"api_base_url": "https://api.openai.com/v1"},
    }
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://localhost:9099"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "Configuration check")

        detail = _provider_test_result_text(screen)
        assert "http://localhost:9099 (draft)" in detail
        assert "configuration=blocked" in detail
