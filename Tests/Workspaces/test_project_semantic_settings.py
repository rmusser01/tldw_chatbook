"""Independent project semantic consent and state."""

import sys
from hashlib import sha256

import pytest

from Tests.Workspaces.project_index_support import make_index_key, open_semantic_store
from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexPolicy,
    ProjectSemanticSettings,
)


def digest(value):
    return "sha256:v1:" + sha256(value.encode()).hexdigest()


def settings(destination="local-model-a"):
    return ProjectSemanticSettings(
        provider_id="huggingface",
        model_id="e5-small-v2",
        destination_fingerprint=digest(destination),
        config_fingerprint=digest("config-a"),
        approved_destination_fingerprint=digest(destination),
        provider_ref="provider:v1:huggingface",
        config_ref="model:v1:e5-small-v2",
    )


def test_semantic_consent_cas_preserves_keyword_scope_and_state(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.configure(
        key,
        ProjectIndexPolicy(("src",), (".py",)),
        enabled=True,
        expected_scope=store.load_status(key).scope,
    )
    keyword_before = store.load_status(key)
    first = store.configure_semantic(
        scope, settings(), enabled=True, expected_semantic_revision=0
    )
    assert first.enabled is True
    assert first.semantic_revision == 1
    assert first.settings == settings()
    assert store.load_status(key) == keyword_before
    with pytest.raises(ValueError, match="stale_semantic_revision"):
        store.configure_semantic(
            scope, settings("changed"), enabled=True, expected_semantic_revision=0
        )
    disabled = store.configure_semantic(
        scope, first.settings, enabled=False, expected_semantic_revision=1
    )
    assert disabled.semantic_revision == 2
    assert disabled.enabled is False
    assert store.load_status(key) == keyword_before
    store.close_native()
    reopened = open_semantic_store(tmp_path)
    assert reopened.load_semantic_state(scope) == disabled
    reopened.close_native()


def test_semantic_admission_refuses_observed_other_native_writer(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    other = store.db_path.parent.parent / ("b" * 64 + ".lifetime.lock")
    lease = BindingLifetime.acquire(other)
    try:
        with pytest.raises(ValueError, match="legacy_retirement_unproven"):
            store.configure_semantic(
                scope, settings(), enabled=True, expected_semantic_revision=0
            )
    finally:
        lease.close()
        store.close_native()
    reopened = open_semantic_store(tmp_path)
    try:
        assert reopened.load_semantic_state(scope).enabled is False
    finally:
        reopened.close_native()


def test_semantic_admission_requires_actual_current_native_lease(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    lease = BindingLifetime.acquire(store.db_path.parent.parent / key.lifetime_name)
    store._owned_lifetime_name = key.lifetime_name  # A name is not ownership.
    try:
        with pytest.raises(ValueError, match="legacy_retirement_unproven"):
            store.configure_semantic(
                scope, settings(), enabled=True, expected_semantic_revision=0
            )
        store.bind_native_lifetime(lease)
        result = store.configure_semantic(
            scope, settings(), enabled=True, expected_semantic_revision=0
        )
        assert result.enabled and result.semantic_revision == 1
    finally:
        lease.close()
        store.close_native()


def test_semantic_status_is_read_only_and_imports_no_embedding_backend(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    before = {
        name
        for name in sys.modules
        if name.startswith(
            ("sentence_transformers", "chromadb", "tldw_chatbook.Embeddings")
        )
    }
    state = store.load_semantic_state(scope)
    after = {
        name
        for name in sys.modules
        if name.startswith(
            ("sentence_transformers", "chromadb", "tldw_chatbook.Embeddings")
        )
    }
    assert state.semantic_revision == 0 and not state.enabled
    assert before == after
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM semantic_control").fetchone()[0] == 0
    store.close_native()


def test_full_revoke_disables_semantics_but_keyword_reconfigure_does_not(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    semantic = store.configure_semantic(
        scope, settings(), enabled=True, expected_semantic_revision=0
    )
    keyword = store.configure(
        key,
        ProjectIndexPolicy((), (".txt",)),
        enabled=True,
        expected_scope=scope,
    )
    assert (
        store.load_semantic_state(keyword).semantic_revision
        == semantic.semantic_revision
    )
    assert store.load_semantic_state(keyword).enabled
    revoked = store.revoke(key, clear=True, expected_scope=keyword)
    assert not store.load_semantic_state(revoked).enabled
    with pytest.raises(ValueError, match="stale_scope"):
        store.configure_semantic(
            scope,
            settings(),
            enabled=True,
            expected_semantic_revision=semantic.semantic_revision,
        )
    store.close_native()


def test_fresh_service_status_projects_off_without_creating_cache_or_gate(tmp_path):
    from contextlib import contextmanager

    from Tests.Workspaces.test_project_index_scanner import authority
    from tldw_chatbook.DB.Workspace_Project_Index_DB import (
        WorkspaceProjectIndexDB,
        index_store_path,
    )
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

    root = tmp_path / "project"
    root.mkdir()
    live = authority(root)
    key = live.key
    data = tmp_path / "cache"

    class Factory:
        def __call__(self, observed):
            path = index_store_path(data, observed)
            path.parent.mkdir(parents=True, exist_ok=True)
            return WorkspaceProjectIndexDB(path)

        def open_cleanup(self, observed):
            path = index_store_path(data, observed)
            if not path.exists():
                raise FileNotFoundError(path)
            return WorkspaceProjectIndexDB(path, cleanup_only=True)

    @contextmanager
    def fence(observed):
        yield live if observed == key else None

    service = ProjectIndexService(
        Factory(),
        ProjectScanner(),
        lambda observed: live if observed == key else None,
        fence,
    )
    status = service.status(key)
    assert status.state == "Off"
    assert status.scope.epoch == 0
    assert not (data / "project_index").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"provider_id": "api_key=secret"},
        {"model_id": "https://example.invalid/model"},
        {"provider_ref": "api_key=secret"},
        {"provider_ref": "provider:v1:other"},
        {"config_ref": "model:v1:api_key=secret"},
        {"config_ref": "model:v1:other"},
        {"destination_fingerprint": "token=secret"},
        {"config_fingerprint": "sha256:v1:" + "A" * 64},
        {"approved_destination_fingerprint": "https://example.invalid"},
    ],
)
def test_semantic_settings_refuse_credential_shapes_before_persistence(
    tmp_path, change
):
    from dataclasses import replace

    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    try:
        with pytest.raises(ValueError) as error:
            replace(settings(), **change)
        assert "secret" not in str(error.value)
        assert store.load_semantic_state(scope).semantic_revision == 0
        with store.connection() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM semantic_control").fetchone()[0] == 0
            )
    finally:
        store.close_native()


def test_semantic_settings_accept_versioned_refs_and_changed_approval(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    selected = settings()
    approved = ProjectSemanticSettings(
        selected.provider_id,
        selected.model_id,
        selected.destination_fingerprint,
        selected.config_fingerprint,
        digest("previous-destination"),
        selected.provider_ref,
        selected.config_ref,
    )
    try:
        state = store.configure_semantic(
            scope, approved, enabled=True, expected_semantic_revision=0
        )
        assert state.state == "Paused"
        assert state.reason == "destination_approval_required"
        assert state.settings == approved
    finally:
        store.close_native()
