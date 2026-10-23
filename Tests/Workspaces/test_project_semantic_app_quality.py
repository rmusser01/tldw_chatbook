"""Offline quality and lifecycle qualification through the real app composition."""

import asyncio
from dataclasses import replace
import os
from types import SimpleNamespace
from unittest.mock import patch
from itertools import combinations


import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
async def test_real_app_model_free_four_page_stage_timing(
    tmp_path, request, monkeypatch
):
    """Measure actual registry/issuer/admission around four synthetic pages."""
    import json
    import time
    from collections import defaultdict
    from contextlib import contextmanager
    from hashlib import sha256

    import httpx
    import requests

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.runtime import (
        close_project_index_service,
        get_project_index_service,
        project_index_key,
    )
    from tldw_chatbook.Workspaces.project_index.semantic import (
        _settings_for_project_model,
    )

    load_config = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(load_config(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["task4-synthetic"] = {
            "provider": "openai",
            "model_name_or_path": "text-embedding-3-small",
            "base_url": "http://127.0.0.1:9/v1",
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)

    def blocked(*args, **kwargs):
        raise AssertionError("model_free_diagnostic_http_attempted")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)

    profile = tmp_path / "app-data"
    profile.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    text = "canine companion\n" * 128
    (project / "dog.txt").write_text(text, encoding="utf-8")
    app = _build_test_app(user_data_dir=profile)
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="timing-ws", name="Timing")
    binding = registry.add_folder_binding("timing-ws", project)
    service = get_project_index_service(app)
    key = project_index_key(app, "timing-ws", binding.binding_id, project)
    phases = defaultdict(list)
    batch_sizes = []
    closed = []
    issued = []

    class DeterministicWrapper:
        def __init__(self, **kwargs):
            issued.append(kwargs["execution_scope"])

        async def create_embeddings_async(self, texts):
            started = time.perf_counter()
            batch_sizes.append(len(texts))
            vectors = [[1.0, 0.0] for _ in texts]
            phases["wrapper"].append(time.perf_counter() - started)
            return vectors

        def close(self):
            closed.append("wrapper")

    original_source = WorkspaceProjectIndexDB.semantic_source_page
    original_append = WorkspaceProjectIndexDB.semantic_append
    original_retry = service._retry_read
    original_admitted = service._admitted
    original_registry = service.registry_connection_scope

    def source_page(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_source(self, *args, **kwargs)
        finally:
            phases["source_sql"].append(time.perf_counter() - started)

    def append_page(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_append(self, *args, **kwargs)
        finally:
            phases["append_sql"].append(time.perf_counter() - started)

    def retry_read(selected, operation):
        started = time.perf_counter()
        try:
            return original_retry(selected, operation)
        finally:
            phases[operation.__name__].append(time.perf_counter() - started)

    @contextmanager
    def admitted(*args, **kwargs):
        started = time.perf_counter()
        with original_admitted(*args, **kwargs) as live:
            phases["admitted_enter"].append(time.perf_counter() - started)
            yield live

    @contextmanager
    def registry_scope(*args, **kwargs):
        started = time.perf_counter()
        with original_registry(*args, **kwargs):
            phases["registry_enter"].append(time.perf_counter() - started)
            yield

    monkeypatch.setattr(WorkspaceProjectIndexDB, "semantic_source_page", source_page)
    monkeypatch.setattr(WorkspaceProjectIndexDB, "semantic_append", append_page)
    monkeypatch.setattr(service, "_retry_read", retry_read)
    monkeypatch.setattr(service, "_admitted", admitted)
    monkeypatch.setattr(service, "registry_connection_scope", registry_scope)
    monkeypatch.setattr(
        embeddings_wrapper, "EmbeddingsServiceWrapper", DeterministicWrapper
    )

    def run():
        off = service.status(key)
        enabled = service.enable(
            key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
        )
        lease = None
        try:
            with service._store(key) as store:
                lease = service._lifetime(store, key)
                with store.transaction() as conn:
                    conn.execute(
                        "INSERT INTO files VALUES(?,?,?,?,?,?)",
                        (
                            "dog.txt",
                            "identity-1",
                            123,
                            len(text),
                            sha256(text.encode()).hexdigest(),
                            text,
                        ),
                    )
                    for index in range(128):
                        chunk = conn.execute(
                            "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                            (
                                "dog.txt",
                                "canine companion",
                                index * 17,
                                index * 17 + 16,
                                index + 1,
                                index + 1,
                            ),
                        ).lastrowid
                        conn.execute(
                            "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                            (chunk, "dog.txt", "canine companion"),
                        )
                    conn.execute(
                        "UPDATE control SET state='Ready',manifest_revision=manifest_revision+1"
                    )
        finally:
            if lease is not None:
                lease.close()
        candidate = _settings_for_project_model("task4-synthetic")
        approved = replace(
            candidate,
            approved_destination_fingerprint=candidate.destination_fingerprint,
        )
        state = service.enable_semantic(
            key,
            approved,
            expected_epoch=enabled.scope.epoch,
            expected_semantic_revision=0,
        )
        started = time.perf_counter()
        job_id = service.refresh_semantic(
            key,
            expected_epoch=enabled.scope.epoch,
            expected_semantic_revision=state.semantic_revision,
        )
        job = service._semantic_runs[job_id]
        job.thread.join(120)
        elapsed = time.perf_counter() - started
        assert not job.thread.is_alive() and job.native_retired and job.error is None
        ready = service.semantic_state(key)
        assert ready.state == "Ready" and ready.ready_generation_id
        assert batch_sizes == [32] * 4
        assert len(phases["append_page"]) == 4
        assert len(issued) == 1 and closed == ["wrapper"]
        (tmp_path / "real-app-stage-timing.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": elapsed,
                    "source_chunks": 128,
                    "batch_sizes": batch_sizes,
                    "native_retired": job.native_retired,
                    "issued_scope": type(issued[0]).__name__,
                    "phases_seconds": dict(phases),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        await asyncio.to_thread(run)
    finally:
        await close_project_index_service(app)


@private_profile_test
async def test_all_saved_library_source_choices_survive_semantic_and_chat_changes(
    tmp_path, request
):
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Workspaces.project_index.runtime import (
        close_project_index_service,
        get_project_index_service,
        project_index_key,
    )
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.semantic import (
        _settings_for_project_model,
    )

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="library-independent", name="Independent")
    project = tmp_path / "project"
    project.mkdir()
    binding = registry.add_folder_binding("library-independent", project)
    screen = LibraryScreen(app)
    sources = ("media", "notes", "conversations", "prompts")

    def semantic_change():
        service = get_project_index_service(app)
        key = project_index_key(app, "library-independent", binding.binding_id, project)
        off = service.status(key)
        keyword = service.enable(
            key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
        )
        before = service.status(key)
        candidate = _settings_for_project_model("openai-3-small")
        paused = service.enable_semantic(
            key,
            candidate,
            expected_epoch=keyword.scope.epoch,
            expected_semantic_revision=0,
        )
        assert paused.state == "Paused" and service.status(key) == before
        return service, key, before

    service, key, keyword = await asyncio.to_thread(semantic_change)
    app.app_config.setdefault("chat_defaults", {})["provider"] = "openai"
    for count in range(5):
        for combo in combinations(sources, count):
            screen._rag_search_state.scope_deselected = set(combo)
            screen._rag_search_state.mode = "rag" if count % 2 else "search"
            before = screen.save_state()
            assert before["library_rag_scope_deselected"] == set(combo)
            assert service.status(key) == keyword
            assert screen.save_state() == before
    app.app_config["chat_defaults"]["provider"] = "anthropic"

    def disable():
        state = service.semantic_state(key)
        result = service.disable_semantic(
            key,
            expected_epoch=state.scope.epoch,
            expected_semantic_revision=state.semantic_revision,
        )
        assert result.state == "Off" and service.status(key) == keyword

    await asyncio.to_thread(disable)
    for count in range(5):
        for combo in combinations(sources, count):
            screen._rag_search_state.scope_deselected = set(combo)
            before = screen.save_state()
            assert before["library_rag_scope_deselected"] == set(combo)
            assert screen.save_state() == before
    await close_project_index_service(app)


@private_profile_test
async def test_cached_minilm_real_app_paraphrase_refresh_restart_and_revoke(
    tmp_path, request, monkeypatch
):
    from Tests.Agents.test_project_index_tool import invoke, provider
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook import config
    from tldw_chatbook.Workspaces.project_index.runtime import (
        close_project_index_service,
        get_project_index_service,
        project_index_key,
    )
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.semantic import (
        _settings_for_project_model,
    )

    model_id = os.environ.get("TLDW_TEST_OFFLINE_EMBEDDING_MODEL")
    if not model_id:
        pytest.skip("No selected offline embedding snapshot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    load_config = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(load_config(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["minilm-local"] = {
            "provider": "huggingface",
            "model_name_or_path": model_id,
            "device": "cpu",
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)
    network_attempts = []

    def blocked(*args, **kwargs):
        network_attempts.append(args)
        raise AssertionError("offline_embedding_http_attempted")

    import httpx
    import requests

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)

    user_data = tmp_path / "app-data"
    user_data.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "dog.txt").write_text(
        "A small puppy dozes beside the warm hearth.", encoding="utf-8"
    )
    (project / "finance.txt").write_text(
        "Quarterly revenue and gross margins improved.", encoding="utf-8"
    )
    query = "a sleeping canine near the fireplace"
    app = _build_test_app(user_data_dir=user_data)
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="quality-ws", name="Quality")
    binding = registry.add_folder_binding("quality-ws", project)

    def build_and_query(selected_app, *, fresh: bool):
        service = get_project_index_service(selected_app)
        key = project_index_key(selected_app, "quality-ws", binding.binding_id, project)
        if fresh:
            off = service.status(key)
            enabled = service.enable(
                key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
            )
            job_id = service.refresh(
                key,
                expected_scope=enabled.scope,
                expected_job_revision=enabled.job_revision,
            )
            job = service._runs[job_id]
            job.supervisor.join(60)
            assert not job.supervisor.is_alive()
            assert service.status(key).state == "Ready"
            candidate = _settings_for_project_model("minilm-local")
            approved = replace(
                candidate,
                approved_destination_fingerprint=candidate.destination_fingerprint,
            )
            semantic = service.enable_semantic(
                key,
                approved,
                expected_epoch=enabled.scope.epoch,
                expected_semantic_revision=0,
            )
            semantic_job = service.refresh_semantic(
                key,
                expected_epoch=enabled.scope.epoch,
                expected_semantic_revision=semantic.semantic_revision,
            )
            run = service._semantic_runs[semantic_job]
            run.thread.join(150)
            assert not run.thread.is_alive()
            assert run.error is None and run.native_retired
        state = service.semantic_state(key)
        assert state.state == "Ready" and state.ready_generation_id
        rig = SimpleNamespace(service=service, root=project, key=key)
        result = invoke(provider(rig), {"query": query})
        assert result.ok and "dog.txt" in result.content
        assert (
            result.content.find("dog.txt") < result.content.find("finance.txt")
            or "finance.txt" not in result.content
        )
        return service, key, state

    service, key, first = await asyncio.to_thread(build_and_query, app, fresh=True)
    assert not network_attempts

    def failed_refresh_keeps_previous_generation():
        from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import (
            EmbeddingsServiceWrapper,
        )

        async def failed_embedding(_wrapper, _texts):
            raise RuntimeError("offline_test_provider_failure")

        with patch.object(
            EmbeddingsServiceWrapper, "create_embeddings_async", failed_embedding
        ):
            semantic = service.semantic_state(key)
            job_id = service.refresh_semantic(
                key,
                expected_epoch=semantic.scope.epoch,
                expected_semantic_revision=semantic.semantic_revision,
            )
            run = service._semantic_runs[job_id]
            run.thread.join(150)
            assert not run.thread.is_alive() and run.native_retired
        failed = service.semantic_state(key)
        assert failed.state == "Ready"
        assert failed.ready_generation_id == first.ready_generation_id
        assert failed.reason.startswith("previous_index_refresh_failed")
        assert service.status(key).state == "Ready"

    await asyncio.to_thread(failed_refresh_keeps_previous_generation)
    await close_project_index_service(app)
    app.local_workspace_db.close()

    restarted = _build_test_app(user_data_dir=user_data)
    service, key, ready = await asyncio.to_thread(
        build_and_query, restarted, fresh=False
    )
    assert ready.ready_generation_id == first.ready_generation_id

    def refresh_and_revoke():
        state = service.semantic_state(key)
        job_id = service.refresh_semantic(
            key,
            expected_epoch=state.scope.epoch,
            expected_semantic_revision=state.semantic_revision,
        )
        run = service._semantic_runs[job_id]
        run.thread.join(150)
        assert not run.thread.is_alive() and run.native_retired and run.error is None
        refreshed = service.semantic_state(key)
        assert refreshed.state == "Ready" and refreshed.ready_generation_id
        keyword = service.status(key)
        disabled = service.disable_semantic(
            key,
            expected_epoch=keyword.scope.epoch,
            expected_semantic_revision=refreshed.semantic_revision,
        )
        assert disabled.state == "Off" and service.status(key).state == "Ready"
        result = invoke(
            provider(SimpleNamespace(service=service, root=project, key=key)),
            {"query": "puppy"},
        )
        assert result.ok and "dog.txt" in result.content
        cleared = service.clear(key, expected_scope=service.status(key).scope)
        assert cleared.state == "Off" and service.semantic_state(key).state == "Off"
        restarted.workspace_registry_service.remove_runtime_binding(binding.binding_id)
        return refreshed.ready_generation_id

    refreshed_id = await asyncio.to_thread(refresh_and_revoke)
    assert refreshed_id != first.ready_generation_id
    assert not network_attempts
    await close_project_index_service(restarted)
