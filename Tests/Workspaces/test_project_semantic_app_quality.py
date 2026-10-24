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

    def keyword_setup():
        service = get_project_index_service(app)
        key = project_index_key(app, "library-independent", binding.binding_id, project)
        off = service.status(key)
        service.enable(
            key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
        )
        return service, key, service.status(key)

    service, key, keyword = await asyncio.to_thread(keyword_setup)
    provider_config = app.app_config.setdefault("chat_defaults", {})
    candidate = _settings_for_project_model("openai-3-small")

    def transition(enable):
        state = service.semantic_state(key)
        if enable:
            result = service.enable_semantic(
                key,
                candidate,
                expected_epoch=state.scope.epoch,
                expected_semantic_revision=state.semantic_revision,
            )
            assert result.state == "Paused"
        else:
            result = service.disable_semantic(
                key,
                expected_epoch=state.scope.epoch,
                expected_semantic_revision=state.semantic_revision,
            )
            assert result.state == "Off"
        assert service.status(key) == keyword

    choices = (
        (count, combo) for count in range(5) for combo in combinations(sources, count)
    )
    for ordinal, (count, combo) in enumerate(choices):
        screen._rag_search_state.scope_deselected = set(combo)
        screen._rag_search_state.mode = "rag" if count % 2 else "search"
        before = screen.save_state()
        assert before["library_rag_scope_deselected"] == set(combo)
        await asyncio.to_thread(transition, ordinal % 2 == 0)
        assert screen.save_state() == before
        provider_config["provider"] = "openai" if count % 2 else "anthropic"
        assert screen.save_state() == before
        assert service.status(key) == keyword
    await close_project_index_service(app)


@private_profile_test
async def test_cached_minilm_local128_disable_retires_actual_held_producer(
    tmp_path, request, monkeypatch
):
    """Revoke while the real local producer has computed but not returned."""
    import json
    import sys
    import threading
    import time
    import traceback
    from pathlib import Path

    import httpx
    import psutil
    import requests

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook import config
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.runtime import (
        close_project_index_service,
        get_project_index_service,
        project_index_key,
    )
    from tldw_chatbook.Workspaces.project_index.semantic import (
        _settings_for_project_model,
    )

    model_id = os.environ.get("TLDW_TEST_OFFLINE_EMBEDDING_MODEL")
    if not model_id:
        pytest.skip("No selected offline embedding snapshot")
    from tldw_chatbook.Embeddings.Embeddings_Lib import _HuggingFaceEmbedder

    phase_path = tmp_path / "actual-local128-phases.jsonl"
    phase_lock = threading.Lock()

    def mark(phase, **values):
        row = {"phase": phase, "monotonic": time.monotonic(), **values}
        with phase_lock, phase_path.open("a", encoding="utf-8") as receipt:
            receipt.write(json.dumps(row, sort_keys=True) + "\n")

    def error_location(error):
        frames = traceback.extract_tb(error.__traceback__)
        last = frames[-1] if frames else None
        return {
            "error_class": type(error).__name__,
            "error_site": (
                f"{Path(last.filename).name}:{last.lineno}:{last.name}"
                if last is not None
                else None
            ),
        }

    sampled_rss = {"before_model": psutil.Process().memory_info().rss}
    mark("test_enter")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    load_config = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(load_config(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["minilm-local-128"] = {
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
        network_attempts.append(True)
        raise AssertionError("offline_embedding_http_attempted")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)

    original_init = _HuggingFaceEmbedder.__init__
    original_embed = _HuggingFaceEmbedder.embed
    encoded = threading.Event()
    release = threading.Event()
    call_sizes = []
    times = {}

    def observed_init(self, cfg):
        mark("hf_constructor_enter")
        try:
            original_init(self, cfg)
        except BaseException as error:
            mark("hf_constructor_error", **error_location(error))
            raise
        mark("hf_constructor_return", device=str(self._device), dtype=str(self._dtype))

    def held_real_embed(self, texts, *, as_list=False):
        mark("hf_embed_enter", text_count=len(texts))
        try:
            result = original_embed(self, texts, as_list=as_list)
        except BaseException as error:
            mark("hf_embed_error", **error_location(error))
            raise
        call_sizes.append(len(texts))
        times["actual_encode_completed"] = time.monotonic()
        mark("hf_encode_completed", text_count=len(texts), vector_count=len(result))
        encoded.set()
        if not release.wait(90):
            mark("hf_hold_expired")
            raise AssertionError("real_embed_hold_not_released")
        times["actual_embed_returned"] = time.monotonic()
        mark("hf_embed_return")
        return result

    monkeypatch.setattr(_HuggingFaceEmbedder, "__init__", observed_init)
    monkeypatch.setattr(_HuggingFaceEmbedder, "embed", held_real_embed)
    profile = tmp_path / "app-data"
    profile.mkdir()
    app = _build_test_app(user_data_dir=profile)
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="local-128", name="Local 128")
    project = tmp_path / "project"
    project.mkdir()
    (project / "large.txt").write_text(
        " ".join(f"distinctword{index:05d}" for index in range(40000)),
        encoding="utf-8",
    )
    binding = registry.add_folder_binding("local-128", project)
    service = get_project_index_service(app)
    key = project_index_key(app, "local-128", binding.binding_id, project)
    original_build = service._semantic_build
    original_worker = service._semantic_worker

    async def observed_build(selected_run, store):
        mark("semantic_build_enter")
        try:
            return await original_build(selected_run, store)
        except BaseException as error:
            mark("semantic_build_error", **error_location(error))
            raise
        finally:
            mark("semantic_build_exit")

    def observed_worker(selected_run, store):
        mark("semantic_worker_enter")
        try:
            return original_worker(selected_run, store)
        except BaseException as error:
            mark("semantic_worker_error", **error_location(error))
            raise
        finally:
            mark(
                "semantic_worker_exit",
                native_retired=selected_run.native_retired,
                worker_error=(
                    type(selected_run.error).__name__
                    if selected_run.error is not None
                    else None
                ),
            )

    monkeypatch.setattr(service, "_semantic_build", observed_build)
    monkeypatch.setattr(service, "_semantic_worker", observed_worker)
    run = None
    disable_task = None
    poll_task = None
    watchdog = None

    def disable_finished(task):
        if task.cancelled():
            mark("disable_task_done", outcome="cancelled")
        elif task.exception() is not None:
            mark(
                "disable_task_done",
                outcome="exception",
                **error_location(task.exception()),
            )
        else:
            mark("disable_task_done", outcome="returned", state=task.result().state)

    async def capture_waiters():
        await asyncio.sleep(10)
        frames = sys._current_frames()
        threads = []
        for owner in threading.enumerate():
            frame = frames.get(owner.ident)
            if frame is None:
                continue
            threads.append(
                {
                    "thread": owner.name,
                    "stack": [
                        f"{Path(item.filename).name}:{item.lineno}:{item.name}"
                        for item in traceback.extract_stack(frame)[-18:]
                    ],
                }
            )
        tasks = []
        for task in asyncio.all_tasks():
            tasks.append(
                {
                    "task": task.get_name(),
                    "done": task.done(),
                    "stack": [
                        f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}"
                        for frame in task.get_stack(limit=12)
                    ],
                }
            )
        mark(
            "disable_waiters_at_10s",
            disable_done=disable_task.done(),
            poll_done=poll_task.done(),
            threads=threads,
            tasks=tasks,
        )

    def start():
        mark("keyword_setup_enter")
        off = service.status(key)
        enabled = service.enable(
            key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=off.scope
        )
        keyword_job_id = service.refresh(
            key,
            expected_scope=enabled.scope,
            expected_job_revision=enabled.job_revision,
        )
        keyword_job = service._runs[keyword_job_id]
        keyword_job.supervisor.join(90)
        assert not keyword_job.supervisor.is_alive()
        assert service.status(key).state == "Ready"
        mark("keyword_ready")
        candidate = _settings_for_project_model("minilm-local-128")
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
        job_id = service.refresh_semantic(
            key,
            expected_epoch=enabled.scope.epoch,
            expected_semantic_revision=semantic.semantic_revision,
        )
        mark("semantic_job_started")
        return service._semantic_runs[job_id]

    try:
        mark("start_await_enter")
        run = await asyncio.to_thread(start)
        mark("start_await_return")
        mark("producer_wait_enter")
        assert await asyncio.to_thread(encoded.wait, 120), "real_producer_not_reached"
        mark("producer_wait_return")
        assert call_sizes == [128] and run.thread.is_alive()
        sampled_rss["held_after_actual_encode"] = psutil.Process().memory_info().rss
        mark("state_read_enter")
        state = await asyncio.to_thread(service.semantic_state, key)
        mark("state_read_return", state=state.state)
        disable_task = asyncio.create_task(
            asyncio.to_thread(
                service.disable_semantic,
                key,
                expected_epoch=state.scope.epoch,
                expected_semantic_revision=state.semantic_revision,
            )
        )
        disable_task.add_done_callback(disable_finished)
        mark("disable_submitted")
        poll_task = asyncio.create_task(asyncio.to_thread(service.semantic_state, key))
        watchdog = asyncio.create_task(capture_waiters())
        await watchdog
        current = poll_task.result() if poll_task.done() else None
        mark(
            "post_snapshot_state", state=current.state if current is not None else None
        )
        if current is None or current.state != "Off":
            release.set()
            mark("fail_fast_release")
            await asyncio.wait_for(asyncio.gather(disable_task, poll_task), timeout=120)
            run.thread.join(10)
            mark("fail_fast_drained", native_retired=run.native_retired)
            pytest.fail("durable_revocation_not_observed_while_actual_producer_held")
        assert current.state == "Off"
        times["durable_revocation"] = time.monotonic()
        mark("durable_revocation")
        assert not disable_task.done() and run.thread.is_alive()
        release.set()
        mark("producer_released")
        disabled = await asyncio.wait_for(disable_task, timeout=120)
        run.thread.join(10)
        times["physical_retirement"] = time.monotonic()
        mark("disable_returned", native_retired=run.native_retired)
        assert disabled.state == "Off"
        assert not run.thread.is_alive() and run.native_retired
        assert service.semantic_state(key).state == "Off"
        assert service.status(key).state == "Ready"
        assert not network_attempts
        sampled_rss["after_retirement"] = psutil.Process().memory_info().rss
        (tmp_path / "actual-local128-retirement.json").write_text(
            json.dumps(
                {
                    "hold_point": "after_actual_encode_before_vectors_return_to_service",
                    "provider_batch_sizes": call_sizes,
                    "sampled_absolute_rss_bytes": sampled_rss,
                    "actual_encode_completed": times["actual_encode_completed"],
                    "durable_revocation": times["durable_revocation"],
                    "actual_embed_returned": times["actual_embed_returned"],
                    "physical_retirement": times["physical_retirement"],
                    "native_retired": run.native_retired,
                    "semantic_state": service.semantic_state(key).state,
                    "keyword_state": service.status(key).state,
                    "network_attempts": len(network_attempts),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        mark("finally_enter")
        release.set()
        try:
            if watchdog is not None:
                await watchdog
            if poll_task is not None:
                await asyncio.wait_for(poll_task, timeout=120)
            if disable_task is not None:
                await asyncio.wait_for(disable_task, timeout=120)
        finally:
            mark("service_close_enter")
            await close_project_index_service(app)
            mark("service_close_return")


@private_profile_test
async def test_cached_minilm_real_app_paraphrase_refresh_restart_and_revoke(
    tmp_path, request, monkeypatch
):
    from Tests.Agents.test_project_index_tool import invoke, provider
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook import config
    from tldw_chatbook.Chat.rag_scope import (
        RagScope,
        ScopeItem,
        read_conversation_scope,
        write_conversation_scope,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
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
    chat_db_path = user_data / "quality-chats.sqlite"
    chat_db = CharactersRAGDB(chat_db_path, client_id="project-quality")
    app.chachanotes_db = chat_db
    conversation_id = chat_db.add_conversation({"title": "Project quality"})
    assert conversation_id is not None
    with chat_db.transaction(immediate=True) as cursor:
        cursor.execute(
            "INSERT INTO console_conversation_library_policy "
            "(conversation_id, auto_retrieve_on_send, assistant_library_access, "
            "policy_revision, updated_at) VALUES (?, 1, 1, 1, CURRENT_TIMESTAMP)",
            (conversation_id,),
        )
    conversation_scope = RagScope(
        (ScopeItem("note", "owned-note"), ScopeItem("media", "outside-media")),
        "2026-10-23T00:00:00Z",
    )
    workspace_scope = RagScope(
        (ScopeItem("note", "owned-note"), ScopeItem("media", "owned-media")),
        "2026-10-23T00:00:00Z",
    )
    write_conversation_scope(chat_db, conversation_id, conversation_scope)
    registry.set_workspace_scope("quality-ws", workspace_scope)

    def saved_library_state(selected_app, selected_db):
        row = (
            selected_db.get_connection()
            .execute(
                "SELECT auto_retrieve_on_send, assistant_library_access, "
                "policy_revision, updated_at FROM console_conversation_library_policy "
                "WHERE conversation_id = ?",
                (conversation_id,),
            )
            .fetchone()
        )
        return (
            tuple(row) if row is not None else None,
            read_conversation_scope(selected_db, conversation_id),
            selected_app.workspace_registry_service.get_workspace_scope("quality-ws"),
        )

    saved_before = saved_library_state(app, chat_db)
    assert saved_before[0][:3] == (1, 1, 1)
    assert saved_before[1] == conversation_scope
    assert saved_before[2] == workspace_scope

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
    assert saved_library_state(app, chat_db) == saved_before
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
    assert saved_library_state(app, chat_db) == saved_before
    await close_project_index_service(app)
    assert saved_library_state(app, chat_db) == saved_before
    chat_db.close()
    app.local_workspace_db.close()

    restarted = _build_test_app(user_data_dir=user_data)
    restarted_db = CharactersRAGDB(chat_db_path, client_id="project-quality")
    restarted.chachanotes_db = restarted_db
    assert saved_library_state(restarted, restarted_db) == saved_before
    service, key, ready = await asyncio.to_thread(
        build_and_query, restarted, fresh=False
    )
    assert ready.ready_generation_id == first.ready_generation_id
    assert saved_library_state(restarted, restarted_db) == saved_before

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
        assert saved_library_state(restarted, restarted_db) == saved_before
        keyword = service.status(key)
        disabled = service.disable_semantic(
            key,
            expected_epoch=keyword.scope.epoch,
            expected_semantic_revision=refreshed.semantic_revision,
        )
        assert disabled.state == "Off" and service.status(key).state == "Ready"
        assert saved_library_state(restarted, restarted_db) == saved_before
        result = invoke(
            provider(SimpleNamespace(service=service, root=project, key=key)),
            {"query": "puppy"},
        )
        assert result.ok and "dog.txt" in result.content
        cleared = service.clear(key, expected_scope=service.status(key).scope)
        assert cleared.state == "Off" and service.semantic_state(key).state == "Off"
        assert saved_library_state(restarted, restarted_db) == saved_before
        restarted.workspace_registry_service.remove_runtime_binding(binding.binding_id)
        assert saved_library_state(restarted, restarted_db) == saved_before
        return refreshed.ready_generation_id

    refreshed_id = await asyncio.to_thread(refresh_and_revoke)
    assert refreshed_id != first.ready_generation_id
    assert not network_attempts
    await close_project_index_service(restarted)
    assert saved_library_state(restarted, restarted_db) == saved_before
    restarted_db.close()
