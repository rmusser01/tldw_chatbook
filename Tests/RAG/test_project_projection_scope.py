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


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
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


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
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
        from tldw_chatbook import config

        original_config = config.load_cli_config_and_ensure_existence
        monkeypatch.setattr(
            config,
            "load_cli_config_and_ensure_existence",
            lambda: {
                "embedding_config": {
                    "models": {
                        "openai-3-small": {
                            "provider": "openai",
                            "model_name_or_path": "synthetic-reroute",
                            "base_url": "http://127.0.0.1:17779/v1",
                            "api_key": "synthetic-key",
                        }
                    }
                }
            },
        )
        with pytest.raises(
            PermissionError, match="project_embedding_configuration_changed"
        ):
            wrapper.create_embeddings(["synthetic text"])
        monkeypatch.setattr(
            config, "load_cli_config_and_ensure_existence", original_config
        )
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
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
        closing = _issue_project_projection_scope(
            service, scope, settings, resumed.semantic_revision
        )
        wrapper.rebind_execution_scope(closing)
        wrapper.close()
        closing.close()
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


def test_selected_keyless_compatible_endpoint_and_default_key_requirement(monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Workspaces.project_index import semantic

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    selected = {
        "provider": "openai",
        "model_name_or_path": "synthetic-local",
        "base_url": "http://127.0.0.1:17777/v1",
        "api_key": None,
    }
    monkeypatch.setattr(
        config,
        "load_cli_config_and_ensure_existence",
        lambda: {"embedding_config": {"models": {"selected": selected}}},
    )
    settings = semantic._settings_for_project_model("selected")
    spec, destination = semantic._resolved_spec("selected")
    target = semantic._target(spec, destination)
    assert target.api_key is None
    assert target.base_url == selected["base_url"]
    assert settings.provider_ref == "provider:v1:openai"

    selected.pop("base_url")
    spec, destination = semantic._resolved_spec("selected")
    with pytest.raises(
        PermissionError, match="project_embedding_provider_setup_required"
    ):
        semantic._target(spec, destination)


def test_selected_custom_cache_root_binds_actual_snapshot_and_settings(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from tldw_chatbook import config
    from tldw_chatbook.Workspaces.project_index import semantic

    def complete_snapshot(cache, payload):
        snapshot = cache / "models--synthetic--model" / "snapshots" / "revision1"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(payload)
        return snapshot

    global_cache = tmp_path / "global-cache"
    selected_cache = tmp_path / "selected-cache"
    global_snapshot = complete_snapshot(global_cache, b"global")
    selected_snapshot = complete_snapshot(selected_cache, b"selected")
    monkeypatch.setenv("HF_HUB_CACHE", str(global_cache))
    selected = {
        "provider": "huggingface",
        "model_name_or_path": "synthetic/model",
        "revision": "revision1",
        "cache_dir": str(selected_cache),
    }
    monkeypatch.setattr(
        config,
        "load_cli_config_and_ensure_existence",
        lambda: {"embedding_config": {"models": {"selected": selected}}},
    )
    chosen = semantic._settings_for_project_model("selected")
    spec, destination = semantic._resolved_spec("selected")
    target = semantic._target(spec, destination)
    assert Path(target.model_name) == selected_snapshot
    assert target.config["model_name_or_path"] == str(selected_snapshot)

    selected["cache_dir"] = str(global_cache)
    other = semantic._settings_for_project_model("selected")
    assert chosen.config_fingerprint != other.config_fingerprint
    other_target = semantic._target(*semantic._resolved_spec("selected"))
    assert Path(other_target.model_name) == global_snapshot
    assert target.artifact_fingerprint != other_target.artifact_fingerprint


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
@private_profile_test
async def test_failed_project_factory_close_retains_real_parent_until_retry(
    tmp_path, monkeypatch, request
):
    def run():
        import gc
        import io
        import weakref
        import threading

        from loguru import logger
        from tldw_chatbook.Embeddings import Embeddings_Lib as lib
        from tldw_chatbook.RAG_Search.projection_scope import ProjectProjectionRequired
        from tldw_chatbook.Workspaces.project_index.models import (
            ProjectIndexSearchAuthority,
        )
        from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
        from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
        from tldw_chatbook.Workspaces.project_index.semantic import (
            _issue_project_projection_scope,
            _settings_for_project_model,
        )
        from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
        root = tmp_path / "close-project"
        root.mkdir()
        (root / "a.txt").write_text("synthetic keyword evidence", encoding="utf-8")
        chain = capture_directory_chain(root)
        store_factory = _AdmittedStoreFactory()
        key = ProjectIndexKey(
            store_factory.identity,
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
            store_factory,
            ProjectScanner(),
            lambda selected: authority if selected == key else None,
            fence,
        )
        initial = service.status(key)
        scope = service.enable(
            key,
            ProjectIndexPolicy(("",), (".txt",)),
            expected_scope=initial.scope,
        ).scope
        current = service.status(key)
        service.refresh(
            key,
            expected_scope=scope,
            expected_job_revision=current.job_revision,
        )
        for active_run in service._runs.values():
            if active_run.supervisor is not None:
                active_run.supervisor.join(30)
                assert not active_run.supervisor.is_alive()
        status = service.status(key)
        assert status.state == "Ready" and status.indexed == 1
        with service._store(key) as store:
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
        carrier = _issue_project_projection_scope(
            service, scope, settings, state.semantic_revision
        )
        monkeypatch.setattr(
            embeddings_wrapper, "_ensure_embeddings_imported", lambda: True
        )
        monkeypatch.setattr(
            embeddings_wrapper, "EmbeddingFactory", lib.EmbeddingFactory
        )
        monkeypatch.setattr(
            embeddings_wrapper, "EmbeddingConfigSchema", lib.EmbeddingConfigSchema
        )
        monkeypatch.setattr(lib, "_embeddings_rag_available", lambda: True)
        wrapper = embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name=carrier.model_name,
            api_key=carrier.api_key,
            base_url=carrier.base_url,
            execution_scope=carrier,
        )
        factory = wrapper.factory
        close_calls = []
        secret = "credential-and-private-path-sentinel"

        def model_close():
            close_calls.append(1)
            if len(close_calls) == 1:
                raise RuntimeError(secret)

        factory._cache["default"] = {
            "embed": lambda texts, as_list=False: [[1.0, 1.0, 1.0]],
            "close": model_close,
            "last": 0.0,
            "refs": 0,
        }
        capture = io.StringIO()
        sink = logger.add(capture)
        try:
            with pytest.raises(
                RuntimeError, match="project_embedding_close_failed"
            ) as failure:
                wrapper.close()
        finally:
            logger.remove(sink)
        assert failure.value.__cause__ is None
        assert secret not in capture.getvalue()
        assert wrapper.factory is factory
        assert factory._cache["default"]["close"] is model_close
        with pytest.raises(ProjectProjectionRequired):
            carrier.close()
        with pytest.raises(PermissionError):
            wrapper.create_embedding("synthetic text")
        with pytest.raises(
            ProjectProjectionRequired, match="project_embedding_close_pending"
        ):
            _issue_project_projection_scope(
                service, scope, settings, state.semantic_revision
            )
        with pytest.raises(
            ProjectProjectionRequired, match="project_embedding_close_pending"
        ):
            wrapper.rebind_execution_scope(carrier)
        keyword = ProjectIndexSearchAuthority(
            scope,
            authority.root,
            chain,
            (),
            1,
            lambda: True,
            None,
            lambda candidate: service._scanner.capture_file(
                authority, candidate.relative_path, stop=threading.Event()
            ),
        )
        found = service.search(keyword, "keyword")
        assert found.status == "ready"
        assert len(found.hits) == 1
        assert found.hits[0].excerpt == "synthetic keyword evidence"
        carrier_ref = weakref.ref(carrier)
        wrapper_ref = weakref.ref(wrapper)
        del carrier, wrapper
        gc.collect()
        assert carrier_ref() is not None
        assert wrapper_ref() is not None
        with pytest.raises(
            RuntimeError, match="project_embedding_close_pending"
        ) as pending:
            asyncio.run(asyncio.wait_for(service.aclose(), timeout=3))
        assert secret not in str(pending.value)
        wrapper_ref().close()
        assert close_calls == [1, 1]
        assert wrapper_ref().factory is None
        carrier_ref().close()
        assert not service._operations
        asyncio.run(service.aclose())

    await asyncio.to_thread(run)


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


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
@pytest.mark.parametrize("failed", [False, True])
@private_profile_test
async def test_peer_pending_close_does_not_poison_admitted_carrier_retirement(
    tmp_path, request, monkeypatch, failed
):
    def run():
        import threading
        from types import SimpleNamespace
        from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
        from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
        from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService
        from tldw_chatbook.RAG_Search.projection_scope import ProjectProjectionRequired
        from tldw_chatbook.Embeddings import Embeddings_Lib as lib
        from tldw_chatbook.Workspaces.project_index.semantic import (
            _issue_project_projection_scope,
            _settings_for_project_model,
        )

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
        rig = SimpleNamespace(key=key, service=service, close=service.aclose)
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
        scope = service.enable(
            key,
            ProjectIndexPolicy((), (".txt",)),
            expected_scope=service.status(key).scope,
        ).scope
        candidate = _settings_for_project_model("openai-3-small")
        selected = replace(
            candidate,
            approved_destination_fingerprint=candidate.destination_fingerprint,
        )
        with rig.service._store(rig.key) as store:
            state = store.configure_semantic(
                scope, selected, enabled=True, expected_semantic_revision=0
            )
        carrier = _issue_project_projection_scope(
            rig.service, scope, selected, state.semantic_revision
        )
        monkeypatch.setattr(
            embeddings_wrapper, "_ensure_embeddings_imported", lambda: True
        )
        monkeypatch.setattr(
            embeddings_wrapper, "EmbeddingFactory", lib.EmbeddingFactory
        )
        monkeypatch.setattr(
            embeddings_wrapper, "EmbeddingConfigSchema", lib.EmbeddingConfigSchema
        )
        monkeypatch.setattr(lib, "_embeddings_rag_available", lambda: True)
        wrapper = embeddings_wrapper.EmbeddingsServiceWrapper(
            model_name=carrier.model_name,
            api_key=carrier.api_key,
            base_url=carrier.base_url,
            execution_scope=carrier,
        )
        assert wrapper.factory.config.models["default"].allow_redirects is False
        peer_ready, peer_go, peer_done, retry = (threading.Event() for _ in range(4))
        errors, peer_owners = [], []

        def peer():
            admitted = _issue_project_projection_scope(
                rig.service, scope, selected, state.semantic_revision
            )
            peer_owners.append(admitted._parent.operation)
            peer_ready.set()
            assert peer_go.wait(45)
            try:
                admitted.close()
            except BaseException as error:
                errors.append(error)
                peer_done.set()
                assert retry.wait(45)
                admitted.close()
            else:
                peer_done.set()

        thread = threading.Thread(target=peer)
        thread.start()
        assert peer_ready.wait(30)
        attempts = []

        def close_model():
            attempts.append(threading.get_ident())
            if len(attempts) == 1:
                peer_go.set()
                assert peer_done.wait(45)
                # New model work remains denied during an ordinary slow close
                # as well as after an actual failure.
                with pytest.raises(
                    ProjectProjectionRequired, match="project_embedding_close_pending"
                ):
                    _issue_project_projection_scope(
                        rig.service, scope, selected, state.semantic_revision
                    )
                if failed:
                    raise RuntimeError("synthetic-close-failure")

        wrapper.factory._cache["default"] = {
            "embed": lambda *a, **k: [[1.0, 0.0]],
            "close": close_model,
            "last": 0.0,
            "refs": 0,
        }
        try:
            if failed:
                with pytest.raises(
                    RuntimeError, match="project_embedding_close_failed"
                ):
                    wrapper.close()
                assert rig.service.semantic_close_pending(rig.key)
                with pytest.raises(ProjectProjectionRequired):
                    carrier.close()
                wrapper.close()
            else:
                wrapper.close()
        finally:
            retry.set()
            thread.join(45)
            assert not thread.is_alive()
            if wrapper.factory is not None:
                wrapper.close()
            carrier.close()
            asyncio.run(rig.close())
        assert not errors, [str(error) for error in errors]
        assert peer_owners[0].done.is_set() and not peer_owners[0].retirement_unproven
        assert not rig.service._operations
        assert attempts == [threading.get_ident()] * (2 if failed else 1)

    await asyncio.to_thread(run)
