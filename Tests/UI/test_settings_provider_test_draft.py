import asyncio
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
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
    saved = _semantic_identity("https://example.test/v1/models", draft_generation=2)
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
    saved = _semantic_identity("https://example.test/v1/models", draft_generation=2)
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
    saved = _semantic_identity("https://example.test/v1/models", draft_generation=2)
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
    older = _semantic_identity("https://example.test/v1/models", draft_generation=2)
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
    tested = _semantic_identity("https://example.test/v1/models", draft_generation=2)
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
        store.begin(
            _semantic_identity(
                "https://example.test/v1/chat/completions", draft_generation=7
            )
        )


def _base_config():
    return {
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://localhost:8080/completion",
                "api_key": "fake-saved-key-not-real",
            },
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
    assert (
        base["api_settings"]["llama_cpp"]["api_url"]
        == "http://localhost:8080/completion"
    )


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
    # No widgets or drafts: the probe worker's credential has nothing to read.
    screen._provider_current_draft_credential = lambda: None
    return screen


def _result_rows(detail: str) -> list[tuple[str, str]]:
    """TASK-33002.2: split a rendered Test result into (label, text) rows."""
    rows = []
    for line in detail.splitlines():
        label, _gap, text = line.partition("  ")
        rows.append((label, text.strip()))
    return rows


_READINESS_WORD = re.compile(
    r"Ready · not tested|Ready · (reachable|verified) \d\d:\d\d|Not ready · .+"
)


def _assert_labelled_rows(detail: str) -> dict[str, str]:
    """One fact per labelled row; no pipe dump, no key=value config spellings.

    TASK-33005.3 (rewritten on purpose): the spec §5 readiness word leads, in
    its own Readiness row, above the five fact rows.
    """
    rows = _result_rows(detail)
    assert rows[0][0] == "Readiness", detail
    assert _READINESS_WORD.fullmatch(rows[0][1]), detail
    assert sorted(label for label, _text in rows[1:]) == sorted(
        ("Config", "Key", "Endpoint", "Model", "Generation")
    ), detail
    assert " | " not in detail
    for spelling in ("model=", "api_key_source=", "configuration=", "api_key="):
        assert spelling not in detail, detail
    return dict(rows)


def test_settings_provider_label_names_a_custom_endpoint_escaped():
    """Qodo #2878 finding 3: Settings labels a ``custom-ep:`` id with its
    registry entry's name, escaped because that name is user text."""
    screen = _bare_settings_screen(
        {
            "custom_endpoints": {
                "gpu-box": {
                    "display_name": "GPU [box]",
                    "base_url": "http://127.0.0.1:9999/v1",
                    "family": "openai_compatible",
                }
            }
        }
    )

    assert screen._provider_display_name("custom-ep:gpu-box") == r"GPU \[box]"
    assert screen._provider_display_name("openai") == "OpenAI"


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


def test_provider_source_keeps_a_resolving_stored_key_over_a_populated_env_var(
    monkeypatch,
):
    """TASK-33001.13 (Qodo #2847): an untouched legacy section's stored key
    outranks the template's populated env var (ADR-012), so it reads "stored"."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-unit-canary-33001")
    screen = _bare_settings_screen(
        {
            "api_settings": {
                "openai": {
                    "api_key": "sk-stored-unit-canary-33001",
                    "api_key_env_var": "OPENAI_API_KEY",
                }
            }
        }
    )
    screen._provider_draft = lambda: None

    assert screen._provider_current_credential_source("openai") == "stored"


def test_findings_show_draft_endpoint_tagged():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:9099"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp",
        "llama-3",
        readiness,
        draft_endpoint="http://localhost:9099",
        dirty={"endpoint"},
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
    rows = _assert_labelled_rows(detail)
    assert _result_rows(detail)[1] == ("Config", "OpenAI is configured")
    # TASK-33002.2 AC#5: a cloud Test stays a local readiness check.
    assert rows["Key"] == "saved in config · present, not verified"
    assert rows["Generation"] == "not tested"
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

    app_config = {"api_settings": {"openai": {"api_key": "fake-test-key"}}}
    readiness = get_provider_readiness("openai", app_config, environ={})

    rows = dict(
        SettingsScreen._provider_test_rows(
            readiness,
            display_name="OpenAI",
            model="gpt-4o",
            endpoint="https://example.test/v1",
            evidence=evidence,
        )
    )

    assert rows["Key"] == "saved in config · present, not verified"
    assert rows["Endpoint"] == "https://example.test/v1 · model listing reached"
    assert rows["Model"] == "gpt-4o · listed by the server"
    assert rows["Generation"] == "not tested"
    # TASK-33005.3 (AC#4): a cloud listing that did not accept the key
    # proves nothing about it.
    assert rows["Readiness"] == "Ready · not tested"


def test_a_key_accepted_by_its_listing_reads_verified_and_never_generated():
    """TASK-33005.3 (AC#2/#5/#6): Settings words a listing-accepted cloud key
    'Ready · verified HH:MM' and keeps Generation 'not tested'."""
    from datetime import datetime

    identity = _semantic_identity(
        "https://example.test/v1/models",
        provider_key="openai",
        credential_source="stored",
        credential_revision=7,
    )
    evidence = ProviderTestEvidence(
        identity,
        "reachable",
        ("gpt-4o",),
        credential="listing_accepted",
        observed_at=datetime(2026, 10, 1, 14, 4).astimezone(),
    )
    app_config = {"api_settings": {"openai": {"api_key": "fake-test-key"}}}
    readiness = get_provider_readiness("openai", app_config, environ={})

    rows = dict(
        SettingsScreen._provider_test_rows(
            readiness,
            display_name="OpenAI",
            model="gpt-4o",
            endpoint="https://example.test/v1",
            evidence=evidence,
        )
    )

    assert rows["Readiness"] == "Ready · verified 14:04"
    # TASK-33005.4 (rewritten on purpose): the ADR-012 outcome phrase.
    assert rows["Key"] == "saved in config · key accepted (1 model listed)"
    assert rows["Generation"] == "not tested"


def test_a_paid_test_of_another_model_never_verifies_the_settings_rows():
    """Qodo #2958 finding 1: the Settings rows read the shared record, so a
    Chat settings paid test of model-a must not read model-b verified here
    either; the listing still reads "reachable"."""
    from datetime import datetime

    identity = _semantic_identity("http://127.0.0.1:9099/v1/models")
    evidence = ProviderTestEvidence(
        identity,
        "reachable",
        ("model-a", "model-b"),
        generation="succeeded",
        generation_model="model-a",
        observed_at=datetime(2026, 10, 1, 9, 0).astimezone(),
        generation_observed_at=datetime(2026, 10, 1, 9, 30).astimezone(),
    )
    readiness = get_provider_readiness("custom", {}, environ={})

    def rows(model: str) -> dict[str, str]:
        return dict(
            SettingsScreen._provider_test_rows(
                readiness,
                display_name="Custom",
                model=model,
                endpoint="http://127.0.0.1:9099/v1",
                evidence=evidence,
            )
        )

    assert rows("model-a")["Readiness"] == "Ready · verified 09:30"
    assert rows("model-a")["Generation"] == "succeeded"
    assert rows("model-b")["Readiness"] == "Ready · reachable 09:00"
    assert rows("model-b")["Generation"] == "not tested"


@pytest.mark.parametrize(
    ("listing", "category", "readiness_word", "key_text"),
    (
        ("unreachable", "unauthorized", "Not ready · key rejected", "key rejected"),
        # Qodo #2958 finding 5 (owner ruling, rewritten on purpose): a 403 key
        # check now records "listing unavailable", so neither row rejects the
        # key and nothing blocks.
        (
            "model_listing_unavailable",
            "http_status",
            "Ready · not tested",
            "present, not verified",
        ),
    ),
    ids=("401", "403"),
)
def test_a_rejected_key_reads_rejected_on_the_key_row_too(
    listing, category, readiness_word, key_text
):
    """TASK-33005 capture checkpoint (capture 08): Readiness said "Not ready ·
    key rejected" while Key said "present, not verified". One rejection, one
    word on both rows."""
    identity = _semantic_identity(
        "https://api.openai.com/v1/models",
        provider_key="openai",
        credential_source="stored",
        credential_revision=7,
    )
    evidence = ProviderTestEvidence(identity, listing, (), category=category)
    app_config = {"api_settings": {"openai": {"api_key": "fake-test-key"}}}
    readiness = get_provider_readiness("openai", app_config, environ={})

    rows = dict(
        SettingsScreen._provider_test_rows(
            readiness,
            display_name="OpenAI",
            model="gpt-4o",
            endpoint="",
            evidence=evidence,
        )
    )

    assert rows["Readiness"] == readiness_word
    assert rows["Key"] == f"saved in config · {key_text}"


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

    readiness = get_provider_readiness("custom", {}, environ={})

    rows = SettingsScreen._provider_test_rows(
        readiness,
        display_name="Custom",
        model="model-a",
        endpoint="https://example.test/v1",
        evidence=evidence,
    )

    # TASK-33002.2 AC#2: the failed listing leads the facts, instead of
    # "configured"; TASK-33005.3 put the readiness word above them.
    assert rows[0][0] == "Readiness"
    label, text = rows[1]
    assert label == "Endpoint"
    assert f"model listing failed ({expected})" in text
    assert dict(rows)["Config"] == "Custom is configured"


def test_cloud_endpoint_row_names_the_default_the_field_shows():
    """Captures flag 10: the Endpoint row said "provider default" while the
    empty Endpoint field showed https://api.openai.com/v1."""
    app_config = {"api_settings": {"openai": {"api_key": "fake-test-key"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("openai", app_config, environ={})

    rows = dict(
        SettingsScreen._provider_test_rows(
            readiness, display_name="OpenAI", model="gpt-4o", endpoint=""
        )
    )

    assert rows["Endpoint"] == (
        f"{screen._provider_endpoint_placeholder('openai')} (provider default)"
    )


def test_provider_edit_stale_copy_requires_a_new_configuration_check():
    copy = SettingsScreen._PROVIDER_TEST_STALE_COPY

    assert "changed since the last check" in copy
    assert "re-run Configuration check" in copy


def test_findings_relabel_draft_api_key_source_and_hide_value():
    app_config = {"api_settings": {"openai": {"api_key": "fake-draft-key-not-real"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    detail, summary, _passed = screen._build_provider_readiness_findings(
        "OpenAI",
        "gpt-4o",
        readiness,
        draft_endpoint="",
        dirty={"api_key"},
    )
    rows = _assert_labelled_rows(detail)
    assert rows["Key"] == "entered here, not saved yet · present, not verified"
    assert (
        "fake-draft-key-not-real" not in detail
        and "fake-draft-key-not-real" not in summary
    )


def test_findings_tag_draft_env_var_and_never_leak_value():
    # A custom-named credential env var whose NAME does not match the secret
    # pattern -- its raw value must still never be printed (task-483 folded in),
    # only presence via the ``<redacted>`` marker, plus the draft tag.
    app_config = {"api_settings": {"openai": {"api_key_env_var": "MY_CUSTOM_CRED"}}}
    screen = _bare_settings_screen(app_config)
    with patch.dict(os.environ, {"MY_CUSTOM_CRED": "env-secret-XYZ"}, clear=False):
        readiness = get_provider_readiness("OpenAI", app_config)
        detail, summary, _passed = screen._build_provider_readiness_findings(
            "OpenAI",
            "gpt-4o",
            readiness,
            draft_endpoint="",
            dirty={"credential_env_var"},
        )
    rows = _assert_labelled_rows(detail)
    assert rows["Key"] == (
        "from env var MY_CUSTOM_CRED (draft) · present, not verified"
    )
    assert "env-secret-XYZ" not in detail and "env-secret-XYZ" not in summary


@pytest.mark.parametrize(
    "missing_env",
    (True, False),
)
def test_findings_key_row_names_its_source_never_its_value(missing_env):
    """TASK-33002.2 AC#3: saved in config, from env var NAME, or missing."""
    environ = {} if missing_env else {"OPENAI_API_KEY": "sk-env-value-canary"}
    app_config = {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ=environ)
    detail, summary, _passed = screen._build_provider_readiness_findings(
        "OpenAI",
        "gpt-4o",
        readiness,
        draft_endpoint="",
        dirty=set(),
    )

    rows = _assert_labelled_rows(detail)
    if missing_env:
        # Round-1 I2: the Key row owns the blocker, so it leads with a
        # Settings-local next step; Config states only the verdict (spec §5:
        # each fact once) and no row spells a config table.
        assert _result_rows(detail)[1] == (  # [0] is Readiness (TASK-33005.3)
            "Key",
            "missing — enter one in the API key field or set OPENAI_API_KEY",
        )
        assert rows["Config"] == "OpenAI is not ready"
        assert "api_settings" not in detail + summary
        assert (
            summary
            == "Configuration check blocked: OpenAI is not ready: Missing API key."
        )
    else:
        assert rows["Key"] == "from env var OPENAI_API_KEY · present, not verified"
    assert "sk-env-value-canary" not in detail + summary


@pytest.mark.parametrize(
    ("provider", "app_config", "environ", "lead"),
    (
        # ADR-179: the key is set (env or saved) but the workspace URL is not.
        (
            "Databricks",
            {"api_settings": {}},
            {"DATABRICKS_TOKEN": "dapi-canary-value-0123456789"},
            ("Endpoint", "not set — enter the workspace URL in the Endpoint field"),
        ),
        (
            "Databricks",
            {
                "api_settings": {
                    "databricks": {"api_key": "dapi-canary-value-0123456789"}
                }
            },
            {},
            ("Endpoint", "not set — enter the workspace URL in the Endpoint field"),
        ),
        # _invalid_settings_readiness: a malformed provider table.
        (
            "QwenCloud",
            {"api_settings": {"qwencloud": "not-a-table"}},
            {"DASHSCOPE_API_KEY": "sk-canary-value-0123456789"},
            (
                "Config",
                (
                    "{name} is not ready: Invalid provider settings — fix this "
                    "provider's settings in Advanced Config"
                ),
            ),
        ),
    ),
)
def test_key_row_never_claims_missing_when_another_setting_blocks(
    provider, app_config, environ, lead
):
    """Round-1 I1 (AC#3): readiness drops the credential source whenever it
    blocks, so a non-key blocker must not read as a missing key."""
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness(provider, app_config, environ=environ)
    assert not readiness.ready
    detail, summary, _passed = screen._build_provider_readiness_findings(
        provider,
        "some-model",
        readiness,
        draft_endpoint="",
        dirty=set(),
    )

    rows = _assert_labelled_rows(detail)
    assert rows["Key"] == "not checked until the provider is ready"
    display = screen._provider_display_name(provider)
    assert _result_rows(detail)[1] == (lead[0], lead[1].format(name=display))
    if lead[0] != "Config":
        assert rows["Config"] == f"{display} is not ready"
    assert "api_settings" not in detail + summary
    assert "canary" not in detail + summary


def test_findings_never_print_a_custom_named_credential_query_param():
    """TASK-486 (absorbed by TASK-33002.2 AC#4): name-based redaction misses
    ``?mycred=``, so the Endpoint row shows the endpoint without its query."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama_cpp", app_config, environ={})

    for evidence in (
        None,
        ProviderProbeResult(endpoint="unreachable", model_ids=(), category="timeout"),
        ProviderProbeResult(endpoint="reachable", model_ids=("llama-3",)),
    ):
        detail, summary, _passed = screen._build_provider_readiness_findings(
            "llama_cpp",
            "llama-3",
            readiness,
            draft_endpoint="http://localhost:9099/v1?mycred=SEKRET",
            dirty={"endpoint"},
            evidence=evidence,
        )

        rows = _assert_labelled_rows(detail)
        assert rows["Endpoint"].startswith("http://localhost:9099/v1 (draft)")
        assert "SEKRET" not in detail and "mycred" not in detail
        assert "SEKRET" not in summary and "mycred" not in summary


def test_failed_probe_leads_with_the_failure_and_a_next_step():
    """TASK-33002.2 AC#2 + AC#6: an unreachable server leads with the failure
    and what to do, and the one-line toast says the same thing."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama_cpp", app_config, environ={})

    detail, summary, passed = screen._build_provider_readiness_findings(
        "llama_cpp",
        "llama-3",
        readiness,
        draft_endpoint="http://127.0.0.1:9099",
        dirty=set(),
        evidence=ProviderProbeResult(
            endpoint="unreachable", model_ids=(), category="connection_refused"
        ),
    )

    assert passed is True  # the configuration itself is complete
    _assert_labelled_rows(detail)
    label, text = _result_rows(detail)[1]  # [0] is Readiness (TASK-33005.3)
    assert label == "Endpoint"
    assert text == (
        "http://127.0.0.1:9099 · model listing failed (connection refused) "
        "— start the server or check the URL"
    )
    assert "configuration is complete" not in detail
    assert "\n" not in summary
    assert summary == (
        "Model listing failed (connection refused) — start the server or check "
        "the URL; generation not tested."
    )


def test_generation_row_and_in_flight_line_read_stored_generation_evidence():
    """TASK-33002 rider: the Generation row and the in-flight (checking) line
    report the stored generation fact, never a hard-coded "not tested"."""
    identity = _semantic_identity("http://127.0.0.1:9099", provider_key="llama_cpp")
    evidence = ProviderTestEvidence(
        # Qodo #2958 (rewritten on purpose): a paid test names its model.
        identity,
        "testing",
        (),
        generation="succeeded",
        generation_model="llama-3",
    )
    readiness = get_provider_readiness(
        "llama.cpp",
        {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}},
        environ={},
    )

    rows = dict(
        SettingsScreen._provider_test_rows(
            readiness,
            display_name="llama.cpp",
            model="llama-3",
            endpoint="http://127.0.0.1:9099",
            evidence=evidence,
            checking=True,
        )
    )

    assert rows["Endpoint"] == "http://127.0.0.1:9099 · checking the model listing"
    assert rows["Generation"] == "succeeded"
    assert "not tested" not in " ".join(rows.values())


def test_findings_never_print_endpoint_userinfo():
    """TASK-33002.2: the Endpoint row uses safe_endpoint_display, which never
    echoes user information (the endpoint contract rejects it anyway)."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp",
        "llama-3",
        readiness,
        draft_endpoint="http://user:hunter2@localhost:9099/v1",
        dirty={"endpoint"},
    )
    rows = _assert_labelled_rows(detail)
    assert "user" not in rows["Endpoint"] and "hunter2" not in rows["Endpoint"]
    assert "hunter2" not in detail and "hunter2" not in summary


def test_findings_no_draft_has_no_tags():
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("llama.cpp", app_config, environ={})
    detail, _summary, _passed = screen._build_provider_readiness_findings(
        "llama.cpp",
        "llama-3",
        readiness,
        draft_endpoint="http://localhost:8080",
        dirty=set(),
    )
    assert "(draft)" not in detail and "(unsaved)" not in detail
    assert "http://localhost:8080" in detail


def test_findings_avoid_ready_claim_when_blocked_on_missing_model():
    """TASK-366: a config-ready provider with no default model must not read
    'is ready' -- the blocking Model row leads (TASK-33002.2) and still
    explains the block."""
    app_config = {"api_settings": {"openai": {"api_key": "placeholder-not-a-real-key"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})
    assert readiness.ready is True  # config-level readiness is fine...

    detail, _summary, passed = screen._build_provider_readiness_findings(
        "OpenAI",
        "",
        readiness,
        draft_endpoint="",
        dirty=set(),
    )

    assert passed is False
    _assert_labelled_rows(detail)
    # TASK-33002.2 AC#2: the blocking fact leads.
    # [0] is the Readiness word (TASK-33005.3); the blocking fact leads the rest.
    assert _result_rows(detail)[:2] == [
        ("Readiness", "Not ready · no model"),
        ("Model", "not set — choose a default model"),
    ]
    assert "is ready" not in detail  # no contradictory ready claim


def test_findings_keep_configuration_only_verdict_when_passing():
    """A local pass describes configuration without claiming provider readiness."""
    app_config = {"api_settings": {"openai": {"api_key": "placeholder-not-a-real-key"}}}
    screen = _bare_settings_screen(app_config)
    readiness = get_provider_readiness("OpenAI", app_config, environ={})

    detail, summary, passed = screen._build_provider_readiness_findings(
        "OpenAI",
        "gpt-4o",
        readiness,
        draft_endpoint="",
        dirty=set(),
    )

    assert passed is True
    _assert_labelled_rows(detail)
    assert _result_rows(detail)[1] == (
        "Config",
        f"{screen._provider_display_name('OpenAI')} is configured",
    )
    assert "is ready" not in detail
    assert "status=ready" not in detail
    assert "configured" in summary.lower()
    assert "live generation has not been tested" in summary.lower()


def test_overview_shows_the_leading_test_row_and_the_endpoint_row():
    """TASK-33002.2: Settings Overview's one-line "Last connection test" row
    shows the result's leading row, never the whole multi-line table.
    Captures flag 10: when Config led, the row said only "Config: llama.cpp
    is configured" and never whether the endpoint was reached, so the
    Endpoint row follows the lead."""
    headline = SettingsScreen._provider_test_headline

    assert (
        headline(
            "Endpoint    http://127.0.0.1:9099 · model listing failed (timeout)\n"
            "Config      llama.cpp is configured\nGeneration  not tested"
        )
        == "Endpoint: http://127.0.0.1:9099 · model listing failed (timeout)"
    )
    assert headline(
        "Config      llama.cpp is configured\nKey         not required\n"
        "Endpoint    http://127.0.0.1:9198 · model listing reached\n"
        "Generation  not tested"
    ) == (
        "Config: llama.cpp is configured; "
        "Endpoint: http://127.0.0.1:9198 · model listing reached"
    )
    for sentinel in (
        SettingsScreen._PROVIDER_TEST_NOT_RUN_COPY,
        SettingsScreen._PROVIDER_TEST_STALE_COPY,
        "Configuration check cancelled; run again.",
    ):
        assert headline(sentinel) == sentinel


def test_mark_provider_test_result_stale_invalidates_prior_verdict():
    """TASK-366: editing a provider input must invalidate a prior Test result so
    a stale 'ready'/'blocked' verdict cannot linger while the form has changed.
    No-op when nothing has run or it is already stale."""
    screen = _bare_settings_screen({})
    screen._provider_test_result = (
        "Config      llama.cpp is configured\nModel       llama-3"
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
            identity,
            token,
        )

    assert store.evidence_for(identity) is None
    assert "checking" not in screen._provider_test_result.lower()
    assert "cancel" in screen._provider_test_result.lower()
    assert "secret-cancel-detail" not in screen._provider_test_result


@pytest.mark.asyncio
@pytest.mark.parametrize("draft_key", [None, "sk-draft-vllm-key"])
async def test_chat_settings_probe_worker_passes_explicit_chat_catalog_purpose(
    monkeypatch, draft_key
):
    """The probe carries the draft's key when it has one (TASK-33005.2 review
    I-1: a keyless probe of a keyed server read as "key rejected")."""
    screen = _bare_settings_screen({})
    screen._provider_current_draft_credential = lambda: draft_key
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
    )

    assert captured == {
        "base_url": "https://example.test/v1/chat/completions",
        "provider": "openai",
        "purpose": "chat_catalog",
        **({"api_key": draft_key} if draft_key else {}),
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
@private_profile_test
async def test_model_picker_prefix_search_finds_a_discovered_id(request):
    """TASK-369, rewritten on purpose for TASK-33007.3 AC#9: the ghost-text
    typeahead is gone; a prefix typed into the Default model picker lists the
    discovered gguf id as a visible row, and Enter chooses it."""
    from textual.widgets import OptionList

    gguf = "gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf"
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": ""}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}
    app.providers_models = {"llama_cpp": []}
    host = StyledSettingsDestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen._model_discovery_models = (
            SimpleNamespace(model_id=gguf),
            SimpleNamespace(model_id="mistral-7b-instruct.Q5_K_M.gguf"),
        )
        screen._refresh_model_picker_discovered()
        field = screen.query_one("#model-search-picker-input", Input)
        results = screen.query_one("#model-search-picker-results", OptionList)
        field.focus()
        await pilot.pause()
        await pilot.press(*"gemma")
        await pilot.pause()

        listed = [
            str(results.get_option_at_index(index).prompt)
            for index in range(results.option_count)
        ]
        assert listed == ["Served now", gguf]
        assert field.suggester is None
        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#settings-model-value", Input).value == gguf


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
@private_profile_test
async def test_test_provider_button_click_runs_the_check(request):
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
        assert result_text != "Configuration check has not run."
        rows = _assert_labelled_rows(result_text)
        assert rows["Endpoint"].endswith("· model listing reached")


@pytest.mark.asyncio
@private_profile_test
async def test_test_provider_button_runs_with_provider_input_focused(request):
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

        # TASK-33007.3, rewritten on purpose: the Model field users focus is
        # the Default model picker's (#settings-model-value is a hidden,
        # unfocusable adapter).
        model_input = screen.query_one("#model-search-picker-input", Input)
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

        _assert_labelled_rows(_provider_test_result_text(screen))


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

        # TASK-33007.3, rewritten on purpose: the Model field users focus is
        # the Default model picker's (#settings-model-value is a hidden,
        # unfocusable adapter).
        model_input = screen.query_one("#model-search-picker-input", Input)
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
@private_profile_test
async def test_model_edit_during_probe_rejects_late_old_model_result(
    request, monkeypatch
):
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
@private_profile_test
async def test_probe_worker_unexpected_exception_settles_bounded_failure(
    request, monkeypatch
):
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
@private_profile_test
async def test_test_provider_result_shows_draft_endpoint(request):
    """Wiring: a staged (unsaved) endpoint edit reaches the Test result.

    Exercises the widget-reading wrapper (``_provider_readiness_test_report``)
    that Task 2's unit tests (above) did not cover, by typing a draft endpoint
    into the real ``#settings-provider-endpoint-value`` input, firing its
    change handler (staging it dirty), then running the test via the button.

    TASK-33005.4 (AC#11, rewritten on purpose): a missing model no longer
    skips the listing, so the probe is stubbed and must receive the draft
    endpoint; the draft tag still threads through the Endpoint row.
    """
    probed = []

    async def probe(base_url, **kwargs):
        probed.append(base_url)
        return await _reachable_endpoint_probe(base_url)

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

        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            probe,
        ):
            await _click_scrolled_settings_button(
                screen, pilot, "#settings-test-provider"
            )
            await _wait_for_settings_text(screen, pilot, "model listing reached")

        detail = _provider_test_result_text(screen)
        rows = _assert_labelled_rows(detail)
        assert probed == ["http://localhost:9099"]
        assert (
            rows["Endpoint"] == "http://localhost:9099 (draft) · model listing reached"
        )
        assert _result_rows(detail)[1] == ("Model", "not set — choose a default model")


@pytest.mark.asyncio
@private_profile_test
async def test_custom_named_credential_query_param_never_reaches_rows_or_toast(
    request,
):
    """TASK-486 (absorbed by TASK-33002.2 AC#4), through the real screen: a
    typed endpoint carrying ``?mycred=SEKRET`` is probed as typed, but neither
    the result rows nor the toast ever print the credential."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")
    probed: list[str] = []

    async def refused_probe(base_url: str, **_kwargs: object):
        probed.append(base_url)
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            summary="unreachable: connection refused",
            category="connection_refused",
        )

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        toasts: list[str] = []
        host.notify = lambda message, **_kwargs: toasts.append(str(message))

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://localhost:9099/v1?mycred=SEKRET"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()

        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            refused_probe,
        ):
            screen.action_settings_test_category()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        result = _provider_test_result_text(screen)
        assert probed and "SEKRET" in probed[0]  # the probe got the typed URL
        assert "SEKRET" not in result and "mycred" not in result
        rows = _assert_labelled_rows(result)
        assert rows["Endpoint"].startswith("http://localhost:9099/v1 (draft)")
        assert toasts, "the Test produced no toast"
        assert all("SEKRET" not in toast and "mycred" not in toast for toast in toasts)
        assert toasts[-1].startswith("Model listing failed (connection refused)")


@pytest.mark.asyncio
@private_profile_test
async def test_identity_less_probe_finishing_after_an_edit_leaves_the_new_draft_stale(
    request,
):
    """Qodo #2878 finding 2: a keyless provider with a query-bearing endpoint
    forms no draft identity, so the evidence store never guards its probe. An
    edit made while it runs must still keep its outcome and toast off the new
    draft."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")
    release = asyncio.Event()

    async def held_probe(_base_url: str, **_kwargs: object):
        await release.wait()
        return await _reachable_endpoint_probe(_base_url)

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        toasts: list[str] = []
        host.notify = lambda message, **_kwargs: toasts.append(str(message))

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://localhost:9099/v1?tag=a"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()
        assert screen._provider_current_draft_identity() is None

        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            held_probe,
        ):
            screen.action_settings_test_category()
            await pilot.pause()
            assert "checking" in _provider_test_result_text(screen).lower()

            endpoint.value = "http://localhost:9100/v1"
            screen.handle_provider_endpoint_changed(
                Input.Changed(endpoint, endpoint.value)
            )
            await pilot.pause()
            toasts.clear()

            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert (
            _provider_test_result_text(screen)
            == SettingsScreen._PROVIDER_TEST_STALE_COPY
        )
        assert toasts == []


@pytest.mark.asyncio
@private_profile_test
async def test_revert_marks_the_discarded_drafts_test_rows_stale(request):
    """Captures flag 1 (settings-pm-test-after-revert): the Test rows kept
    describing a draft endpoint after Revert put the saved one back, and they
    survived a later Save."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://localhost:9"
        await pilot.pause()
        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            _reachable_endpoint_probe,
        ):
            await _click_scrolled_settings_button(
                screen, pilot, "#settings-test-provider"
            )
            await _wait_for_settings_text(screen, pilot, "model listing reached")
        assert "http://localhost:9 (draft)" in _provider_test_result_text(screen)

        screen.action_settings_revert_category()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        await pilot.pause()

        assert endpoint.value == "http://localhost:8080"
        assert (
            _provider_test_result_text(screen)
            == SettingsScreen._PROVIDER_TEST_STALE_COPY
        )


@pytest.mark.asyncio
@private_profile_test
async def test_wrapped_endpoint_row_stays_in_the_value_column_at_211x44(request):
    """Gap review Critical 1: the Test rows were one padded string, so a
    long Endpoint value wrapped back to column 0 under the labels
    (capture settings-pm-test-failed-211x44 put "URL" in the label column).
    Asserted on the painted frame: every continuation line is blank across
    the label column."""
    from textual.containers import VerticalScroll

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")
    long_endpoint = "http://127.0.0.1:9/" + "/".join(["segment"] * 20)

    async def refused_probe(_base_url: str, **_kwargs: object):
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            summary="unreachable: connection refused",
            category="connection_refused",
        )

    async with host.run_test(size=(211, 44)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = long_endpoint
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()
        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            refused_probe,
        ):
            screen.action_settings_test_category()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        result = screen.query_one("#settings-provider-test-result", Static)
        screen.query_one("#settings-detail-pane-body", VerticalScroll).scroll_to_widget(
            result, animate=False, immediate=True, top=True, force=True
        )
        await pilot.pause()
        region = result.region
        strips = host.screen._compositor.render_strips()
        painted = [
            strips[y].crop(region.x, region.right).text
            for y in range(region.y, region.bottom)
        ]
        labels = ("Readiness", "Endpoint", "Config", "Key", "Model", "Generation")
        label_cells = len("Generation  ")

        assert painted[0].startswith("Readiness"), painted  # TASK-33005.3
        assert painted[1].startswith("Endpoint"), painted
        continuations = [
            line for line in painted if line.strip() and not line.startswith(labels)
        ]
        assert continuations, f"the Endpoint value never wrapped: {painted}"
        assert all(
            len(line) - len(line.lstrip()) == label_cells for line in continuations
        ), "\n".join(painted)
        assert _result_rows(_provider_test_result_text(screen))[1][0] == "Endpoint"


async def _test_reachable_llama_cpp(
    screen, pilot, probe=_reachable_endpoint_probe
) -> str:
    with patch(
        "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
        probe,
    ):
        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _wait_for_settings_text(screen, pilot, "model listing reached")
    return _provider_test_result_text(screen)


async def _confirm_revert(screen, pilot) -> None:
    screen.action_settings_revert_category()
    await pilot.pause()
    await pilot.click("#confirm-button")
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_revert_of_a_temperature_only_draft_keeps_a_fresh_test_result(request):
    """Gap review Important 2: Revert marked the Test rows stale even when
    the discarded draft touched no tested field."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        tested = await _test_reachable_llama_cpp(screen, pilot)
        assert "http://localhost:8080 · model listing reached" in tested

        temperature = screen.query_one("#settings-model-profile-temperature", Input)
        temperature.value = "0.3"
        await pilot.pause()
        assert _provider_test_result_text(screen) == tested
        await _confirm_revert(screen, pilot)

        assert _provider_test_result_text(screen) == tested
        identity = screen._provider_current_draft_identity()
        evidence = screen._provider_evidence_store().evidence_for(identity)
        assert evidence is not None and evidence.endpoint == "reachable"


@pytest.mark.asyncio
@private_profile_test
async def test_probe_in_flight_at_revert_cannot_settle_onto_the_saved_identity(
    request,
):
    """Gap review Important 3: Revert left the discarded draft's probe token
    live, so its late result replaced the stale marker with rows crediting
    the saved endpoint with the draft's failure."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")
    release = asyncio.Event()

    async def held_refused_probe(_base_url: str, **_kwargs: object):
        await release.wait()
        return SettingsEndpointProbeOutcome(
            state="unreachable",
            summary="unreachable: connection refused",
            category="connection_refused",
        )

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://localhost:9"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))
        await pilot.pause()
        with patch(
            "tldw_chatbook.UI.Screens.settings_endpoint_probe.probe_settings_endpoint",
            held_refused_probe,
        ):
            screen.action_settings_test_category()
            await pilot.pause()
            assert "checking the model listing" in _provider_test_result_text(screen)

            await _confirm_revert(screen, pilot)
            assert endpoint.value == "http://localhost:8080"
            assert (
                _provider_test_result_text(screen)
                == SettingsScreen._PROVIDER_TEST_STALE_COPY
            )

            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

        assert (
            _provider_test_result_text(screen)
            == SettingsScreen._PROVIDER_TEST_STALE_COPY
        )


@pytest.mark.asyncio
@private_profile_test
async def test_test_result_grid_stays_selectable_and_copyable(request):
    """The label/value grid render made Widget.get_selection return None,
    so the Test result could no longer be selected or copied; the plain
    Static had returned its text (TASK-33002.2)."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        tested = await _test_reachable_llama_cpp(screen, pilot)
        result = screen.query_one("#settings-provider-test-result", Static)

        result.text_select_all()
        await pilot.pause()
        assert result.screen.get_selected_text() == tested

        result.screen.action_copy_text()
        assert pilot.app.clipboard == tested


def test_snapshot_restore_holds_no_private_evidence_copy():
    """TASK-33005.1 (AC#6, controller ruling 3): the late-handoff restore no
    longer copies the evidence store, so it can never roll back a result
    Chat settings or the Console settled in the shared owner meanwhile."""
    from dataclasses import fields

    from tldw_chatbook.UI.Screens.settings_screen import (
        _VllmDefaultPresentationSnapshot,
    )

    names = {item.name for item in fields(_VllmDefaultPresentationSnapshot)}
    assert "provider_test_evidence_store" not in names
    assert "provider_credential_revision" not in names
    assert "provider_draft_generation" in names


def test_settings_identity_stamps_the_key_digest_every_surface_computes():
    """TASK-33005.1 (AC#1, ruling 2): the credential revision is the digest
    of the key a send would use -- never Settings' own edit counter -- so the
    same saved key is the same connection in Settings, Chat settings and the
    Console, and a typed key is a different one."""
    from tldw_chatbook.Chat.provider_test_evidence import (
        connection_credential_revision,
    )

    app_config = {
        "api_settings": {
            "openai": {
                "api_url": "https://api.openai.com/v1",
                "api_key": "sk-saved-test-key",
                "credential_source": "stored",
            }
        }
    }
    screen = _bare_settings_screen(app_config)
    screen._settings_drafts = {}
    saved = connection_credential_revision(
        get_provider_readiness(
            "openai", app_config, background_credentials=True
        ).api_key
    )
    assert saved == connection_credential_revision("sk-saved-test-key")
    untouched = {"api_key": "", "credential_env_var": ""}
    assert (
        screen._provider_draft_credential_revision("openai", "stored", untouched)
        == saved
    )
    typed = {"api_key": "sk-typed-test-key", "credential_env_var": ""}
    assert screen._provider_draft_credential_revision(
        "openai", "draft", typed
    ) == connection_credential_revision("sk-typed-test-key")
    assert screen._provider_draft_credential_revision("openai", "none", typed) == 0


@pytest.mark.asyncio
@private_profile_test
async def test_returning_to_settings_shows_the_shared_test_result(request):
    """TASK-33005.1 (AC#6, AC#1): Settings is rebuilt on every visit, so its
    draft store starts empty; the result an earlier visit settled lives in
    the app's shared owner and shows again instead of "has not run"."""
    from tldw_chatbook.Chat.provider_test_evidence import (
        provider_connection_evidence,
    )

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://localhost:8080"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        first = _active_destination_screen(host)
        tested = await _test_reachable_llama_cpp(first, pilot)
        identity = first._provider_current_draft_identity()

        await host.pop_screen()
        await host.push_screen(SettingsScreen(app))
        await _open_settings_category(pilot, "#settings-category-providers-models")
        second = _active_destination_screen(host)

        assert second is not first
        assert _provider_test_result_text(second) == tested
        overview = second._settings_overview_presentation()
        rows = {
            row.key: row.value
            for row in (*overview.primary_rows, *overview.advanced_rows)
        }
        assert "model listing reached" in rows["last_connection_test"]
        shared = provider_connection_evidence(host).evidence_for(identity)
        assert shared is not None and shared.endpoint == "reachable"
        assert shared.observed_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llama_settings",
    [
        {"api_url": "http://localhost:8080"},
        # Review M-5: a stored key, so the credential revision is not 0.
        {"api_url": "http://localhost:8080", "api_key": "sk-llama-stored-test-key"},
        # Review finding 5: Chat settings reads the explicit source ("stored"),
        # Settings what resolves (nothing) -- still one keyless connection.
        {"api_url": "http://localhost:8080", "credential_source": "stored"},
    ],
    ids=["keyless", "stored-key", "stored-source-without-key"],
)
@private_profile_test
async def test_settings_test_result_reaches_chat_settings_for_the_same_connection(
    llama_settings, request
):
    """TASK-33005.1 (AC#1): a Settings 't' on the saved llama.cpp connection
    is what Chat settings shows for that connection -- the two surfaces keep
    their own draft stores but share settled evidence.

    TASK-33005.2 review: the Console keys it identically (finding 4), and the
    probe carries the saved key a send uses (I-1).

    TASK-33005.3 (AC#8): Settings' Readiness row, the Console (status row,
    rail and switcher all render ``readiness_words``) and Chat settings show
    the same word for the connection at the same moment."""
    from Tests.UI.test_console_session_settings import _readiness_text
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        ConsoleSettingsContextEstimate,
        build_console_settings_readiness,
        readiness_words,
    )
    from tldw_chatbook.Chat.provider_test_evidence import (
        provider_connection_evidence,
    )
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "llama-3"}
    app.app_config["api_settings"] = {"llama_cpp": dict(llama_settings)}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        sent = []

        async def probe(base_url, **kwargs):
            sent.append(kwargs.get("api_key"))
            return await _reachable_endpoint_probe(base_url)

        await _test_reachable_llama_cpp(screen, pilot, probe)
        tested = screen._provider_current_draft_identity()
        assert sent == [llama_settings.get("api_key")]
        console = build_console_settings_readiness(
            ConsoleSessionSettings(provider="llama_cpp", model="llama-3"),
            app_config=app.app_config,
            connection_evidence=provider_connection_evidence(host),
        )
        assert console.endpoint == "reachable"
        word = f"Ready · reachable {console.observed_at.astimezone():%H:%M}"
        assert readiness_words(console) == word
        assert _result_rows(_provider_test_result_text(screen))[0] == (
            "Readiness",
            word,
        )
        # Same endpoint and key; a revision-0 source is one keyless
        # connection whichever surface spelled it (Task 1 review F5).
        assert (
            console.connection.connection_identity,
            console.connection.credential_revision,
        ) == (tested.connection_identity, tested.credential_revision)

        await host.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(
                    provider="llama_cpp", model="llama-3", base_url=None
                ),
                app_config=app.app_config,
                providers_models={"llama_cpp": ["llama-3"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()
        modal = host.screen
        identity = modal._current_connection_probe_identity()
        evidence = modal._connection_evidence_store.evidence_for(identity)

        assert evidence is not None
        assert evidence.endpoint == "reachable"
        assert evidence.model_ids == ("llama-3",)
        assert "Endpoint · Reachable" in _readiness_text(modal)
        assert _readiness_text(modal).startswith(f"{word}\n")
        assert identity.credential_revision == tested.credential_revision
        assert (identity.credential_revision != 0) is ("api_key" in llama_settings)


@pytest.mark.asyncio
@private_profile_test
async def test_a_cloud_provider_is_one_connection_in_settings_and_chat_settings(
    request,
):
    """Review I-3: with no base URL, Chat settings keyed a cloud provider on a
    sentinel endpoint and Settings had no identity at all, so a cloud result
    could never cross surfaces. Both now key the endpoint a send uses."""
    from dataclasses import replace

    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        ConsoleSettingsContextEstimate,
    )
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    app.app_config["api_settings"] = {"openai": {"api_key": "sk-cloud-saved-test-key"}}
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        in_settings = _active_destination_screen(
            host
        )._provider_current_draft_identity()
        await host.push_screen(
            ConsoleSettingsModal(
                settings=ConsoleSessionSettings(provider="openai", model="gpt-4o"),
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4o"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            )
        )
        await pilot.pause()
        in_chat_settings = host.screen._current_connection_probe_identity()

    assert in_settings is not None and in_chat_settings is not None
    assert in_settings.connection_identity == canonical_connection_identity(
        "openai", "https://api.openai.com/v1"
    )
    assert replace(in_settings, draft_generation=0) == replace(
        in_chat_settings, draft_generation=0
    )
