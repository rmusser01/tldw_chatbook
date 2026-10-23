"""Installed embedding entrances refuse unissued project authority."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace

import pytest
from Tests.private_profile import private_profile_test

from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexAuthority,
    ProjectIndexKey,
    ProjectIndexPolicy,
    locator_fingerprint,
    root_identity_fingerprint,
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain


def test_invalid_project_carrier_refuses_before_source_or_model_effects(monkeypatch):
    effects = []
    monkeypatch.setattr(
        embeddings_wrapper,
        "source_paths",
        lambda service: effects.append("source") or (),
    )
    with pytest.raises(PermissionError):
        embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name="mock", device="cpu", execution_scope=object()
        )
    assert effects == []


def test_local_files_only_is_forwarded_to_actual_factory(monkeypatch):
    captured = []

    class Factory:
        def __init__(self, cfg, **kwargs):
            captured.append((cfg, kwargs))

        def close(self):
            pass

    monkeypatch.setattr(embeddings_wrapper, "_ensure_embeddings_imported", lambda: True)
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingFactory", Factory)
    from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingConfigSchema

    monkeypatch.setattr(
        embeddings_wrapper, "EmbeddingConfigSchema", EmbeddingConfigSchema
    )
    service = embeddings_wrapper.EmbeddingsServiceWrapper(
        model_name="openai/text-embedding-3-small",
        api_key="synthetic-key",
        device="cpu",
        local_files_only=True,
    )
    try:
        assert captured[0][1]["local_files_only"] is True
    finally:
        service.close()


def test_factory_local_only_refuses_missing_snapshot_before_loader(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Embeddings import Embeddings_Lib as lib

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    monkeypatch.setattr(lib, "_embeddings_rag_available", lambda: True)
    monkeypatch.setattr(lib, "_ensure_torch", lambda: None)
    monkeypatch.setattr(lib, "_ensure_transformers", lambda: None)
    monkeypatch.setattr(lib, "torch", object())
    monkeypatch.setattr(lib, "transformers", object())
    called = []
    monkeypatch.setattr(
        lib,
        "_HuggingFaceEmbedder",
        lambda cfg: called.append(cfg) or pytest.fail("loader reached"),
    )
    factory = lib.EmbeddingFactory(
        {
            "default_model_id": "default",
            "models": {
                "default": {
                    "provider": "huggingface",
                    "model_name_or_path": "missing/model",
                }
            },
        },
        local_files_only=True,
        allow_dynamic_hf=False,
    )
    try:
        with pytest.raises(
            FileNotFoundError, match="local_embedding_snapshot_required"
        ):
            factory.embed(["synthetic text"])
        assert called == []
    finally:
        factory.close()


def test_project_missing_local_package_is_setup_required(monkeypatch):
    from tldw_chatbook.Embeddings import Embeddings_Lib as lib
    from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg
    from tldw_chatbook.RAG_Search.projection_scope import ProjectProjectionRequired
    from tldw_chatbook.Workspaces.project_index.semantic import _local_snapshot

    def missing_package(_):
        raise ImportError("private-path-leak")

    monkeypatch.setattr(lib, "existing_local_hf_snapshot", missing_package)
    with pytest.raises(
        ProjectProjectionRequired, match="project_local_model_setup_required"
    ) as error:
        _local_snapshot(HFModelCfg(model_name_or_path="missing/model"))
    assert "private-path-leak" not in str(error.value)


@private_profile_test
async def test_issued_scope_uses_live_parent_and_rejects_copy_after_close(
    tmp_path, monkeypatch, request
):
    from copy import copy

    from tldw_chatbook.RAG_Search.projection_scope import ProjectionExecutionScope
    from tldw_chatbook.Workspaces.project_index.semantic import (
        _issue_project_projection_scope,
        _settings_for_project_model,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    root = tmp_path / "project"
    root.mkdir()
    chain = capture_directory_chain(root)
    factory = _AdmittedStoreFactory()
    key = ProjectIndexKey(
        factory.identity,
        "workspace",
        "binding",
        locator_fingerprint(root),
        root_identity_fingerprint(chain),
    )
    authority = ProjectIndexAuthority(key, root, chain, (), 1)

    @contextmanager
    def fence(selected):
        yield authority if selected == key else None

    service = ProjectIndexService(
        factory,
        ProjectScanner(),
        lambda selected: authority if selected == key else None,
        fence,
    )
    with service._store(key) as store:
        scope = store.configure(
            key,
            ProjectIndexPolicy((), (".txt",)),
            enabled=True,
            expected_scope=store.load_status(key).scope,
        )
        lease = service._lifetime(store, key)
        try:
            candidate = _settings_for_project_model("openai-3-small")
            settings = replace(
                candidate,
                approved_destination_fingerprint=candidate.destination_fingerprint,
            )
            state = store.configure_semantic(
                scope, settings, enabled=True, expected_semantic_revision=0
            )
        finally:
            lease.close()

    issued = _issue_project_projection_scope(
        service, scope, settings, state.semantic_revision
    )
    assert isinstance(issued, ProjectionExecutionScope)
    with pytest.raises(AttributeError):
        issued.model_name = "openai/forged"
    issued._config["api_key"] = "forged"
    assert issued.api_key == "synthetic-key"
    import numpy as np
    import threading

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingConfigSchema

    class Factory:
        def __init__(self, cfg, **kwargs):
            self.config = cfg

        def embed(self, texts, *, as_list=False):
            if texts == ["block"]:
                entered.set()
                assert release.wait(5)
                finished.set()
            result = np.ones((len(texts), 3), dtype=np.float32)
            return result.tolist() if as_list else result

        def close(self):
            pass

    monkeypatch.setattr(embeddings_wrapper, "_ensure_embeddings_imported", lambda: True)
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingFactory", Factory)
    monkeypatch.setattr(
        embeddings_wrapper, "EmbeddingConfigSchema", EmbeddingConfigSchema
    )
    monkeypatch.setattr(
        embeddings_wrapper,
        "source_paths",
        lambda service: pytest.fail("project scope reached Library source inference"),
    )
    wrapper = embeddings_wrapper.EmbeddingsServiceWrapper(
        model_name=issued.model_name,
        api_key=issued.api_key,
        base_url=issued.base_url,
        device=None,
        execution_scope=issued,
    )
    assert (await wrapper.create_embeddings_async(["synthetic text"])).shape == (1, 3)
    assert (await wrapper.create_embedding_async("synthetic text")).shape == (3,)
    with pytest.raises(PermissionError):
        await asyncio.to_thread(wrapper.create_embeddings, ["synthetic text"])
    caller = asyncio.current_task()

    async def cancel_during_native():
        await asyncio.to_thread(entered.wait)
        caller.cancel()
        caller.cancel()
        await asyncio.sleep(0.05)
        assert not finished.is_set()
        release.set()

    canceller = asyncio.create_task(cancel_during_native())
    with pytest.raises(asyncio.CancelledError):
        await wrapper.create_embeddings_async(["block"])
    await canceller
    assert finished.is_set()
    wrapper.close()
    with pytest.raises(PermissionError):
        embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name="openai/wrong-model",
            api_key=issued.api_key,
            base_url=issued.base_url,
            execution_scope=issued,
        )
    with pytest.raises(PermissionError):
        embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name=issued.model_name,
            api_key=issued.api_key,
            base_url=issued.base_url,
            device="cpu",
            execution_scope=copy(issued),
        )
    issued.close()
    with pytest.raises(PermissionError):
        embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name=issued.model_name,
            api_key=issued.api_key,
            base_url=issued.base_url,
            device="cpu",
            execution_scope=issued,
        )
    await service.aclose()


@private_profile_test
async def test_issued_scope_sync_entry_and_stale_revision_refuse(
    tmp_path, monkeypatch, request
):
    def run():
        from tldw_chatbook.Workspaces.project_index.semantic import (
            _issue_project_projection_scope,
            _settings_for_project_model,
        )
        from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
        from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
        from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
        root = tmp_path / "sync-project"
        root.mkdir()
        chain = capture_directory_chain(root)
        factory = _AdmittedStoreFactory()
        key = ProjectIndexKey(
            factory.identity,
            "workspace",
            "binding",
            locator_fingerprint(root),
            root_identity_fingerprint(chain),
        )
        authority = ProjectIndexAuthority(key, root, chain, (), 1)

        @contextmanager
        def fence(selected):
            yield authority if selected == key else None

        service = ProjectIndexService(
            factory,
            ProjectScanner(),
            lambda selected: authority if selected == key else None,
            fence,
        )
        with service._store(key) as store:
            scope = store.configure(
                key,
                ProjectIndexPolicy((), (".txt",)),
                enabled=True,
                expected_scope=store.load_status(key).scope,
            )
            lease = service._lifetime(store, key)
            try:
                candidate = _settings_for_project_model("openai-3-small")
                settings = replace(
                    candidate,
                    approved_destination_fingerprint=candidate.destination_fingerprint,
                )
                state = store.configure_semantic(
                    scope, settings, enabled=True, expected_semantic_revision=0
                )
            finally:
                lease.close()

        issued = _issue_project_projection_scope(
            service, scope, settings, state.semantic_revision
        )
        import numpy as np

        made = []

        class Factory:
            def __init__(self, cfg, **kwargs):
                made.append(1)

            def embed(self, texts, *, as_list=False):
                result = np.ones((len(texts), 3), dtype=np.float32)
                return result.tolist() if as_list else result

            def embed_one(self, text, *, as_list=False):
                result = np.ones((3,), dtype=np.float32)
                return result.tolist() if as_list else result

            def close(self):
                pass

        monkeypatch.setattr(
            embeddings_wrapper, "_ensure_embeddings_imported", lambda: True
        )
        monkeypatch.setattr(embeddings_wrapper, "EmbeddingFactory", Factory)
        from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingConfigSchema

        monkeypatch.setattr(
            embeddings_wrapper, "EmbeddingConfigSchema", EmbeddingConfigSchema
        )
        wrapper = embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name=issued.model_name,
            api_key=issued.api_key,
            base_url=issued.base_url,
            execution_scope=issued,
        )
        with pytest.raises(PermissionError):
            wrapper.rebind_execution_scope(issued)
        assert wrapper.create_embeddings(["synthetic text"]).shape == (1, 3)
        assert wrapper.create_embedding("synthetic text").shape == (3,)
        with service._store(key) as store:
            changed = store.configure_semantic(
                scope,
                settings,
                enabled=False,
                expected_semantic_revision=state.semantic_revision,
            )
        assert changed.semantic_revision != state.semantic_revision
        with pytest.raises(PermissionError):
            wrapper.create_embeddings(["synthetic text"])
        with service._store(key) as store:
            resumed = store.configure_semantic(
                scope,
                settings,
                enabled=True,
                expected_semantic_revision=changed.semantic_revision,
            )
        issued.close()
        assert not service._operations
        with pytest.raises(PermissionError):
            wrapper.create_embeddings(["synthetic text"])
        successor = _issue_project_projection_scope(
            service, scope, settings, resumed.semantic_revision
        )
        from copy import copy

        with pytest.raises(PermissionError):
            wrapper.rebind_execution_scope(copy(successor))
        wrapper.rebind_execution_scope(successor)
        assert wrapper.create_embeddings(["synthetic text"]).shape == (1, 3)
        successor.close()
        assert not service._operations
        later = _issue_project_projection_scope(
            service, scope, settings, resumed.semantic_revision
        )
        wrapper.rebind_execution_scope(later)
        assert wrapper.create_embeddings(["synthetic text"]).shape == (1, 3)
        assert len(made) == 1
        later.close()
        assert not service._operations
        monkeypatch.setenv("OPENAI_API_KEY", "rotated-key")
        changed_credential = _issue_project_projection_scope(
            service, scope, settings, resumed.semantic_revision
        )
        with pytest.raises(PermissionError):
            wrapper.rebind_execution_scope(changed_credential)
        changed_credential.close()
        wrapper.close()
        import asyncio

        asyncio.run(service.aclose())

    await asyncio.to_thread(run)


def test_selected_configured_catalog_entry_takes_precedence_and_rotates_credentials(
    monkeypatch,
):
    from tldw_chatbook import config
    from tldw_chatbook.Workspaces.project_index import semantic

    configured = {
        "provider": "openai",
        "model_name_or_path": "synthetic-model",
        "base_url": "http://127.0.0.1:17777/v1",
        "api_key": "config-key",
    }
    monkeypatch.setattr(
        config,
        "load_cli_config_and_ensure_existence",
        lambda: {"embedding_config": {"models": {"openai-3-small": configured}}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key-one")
    first = semantic._settings_for_project_model("openai-3-small")
    spec, destination = semantic._resolved_spec("openai-3-small")
    target = semantic._target(spec, destination)
    assert target.model_name == "openai/synthetic-model"
    assert target.api_key == "environment-key-one"
    assert target.base_url == "http://127.0.0.1:17777/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key-two")
    second = semantic._settings_for_project_model("openai-3-small")
    assert first == second
    assert "environment-key" not in repr(first)
    assert "config-key" not in repr(first)

    configured["base_url"] = "http://127.0.0.1:17778/v1"
    changed = semantic._settings_for_project_model("openai-3-small")
    assert changed.destination_fingerprint != first.destination_fingerprint

    configured["provider"] = "unsupported"
    with pytest.raises(PermissionError):
        semantic._settings_for_project_model("openai-3-small")


def test_library_wrapper_cannot_be_upgraded_by_rebind(monkeypatch):
    monkeypatch.setattr(embeddings_wrapper, "source_paths", lambda service: ())
    monkeypatch.setattr(
        embeddings_wrapper.EmbeddingsServiceWrapper,
        "_ensure_initialized",
        lambda self: None,
    )
    wrapper = embeddings_wrapper.EmbeddingsServiceWrapper(model_name="mock")
    with pytest.raises(PermissionError):
        wrapper.rebind_execution_scope(object())
