"""Actual registry connection retirement before project model effects."""

from contextlib import contextmanager
import asyncio
import sqlite3

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
async def test_service_retires_actual_registry_handles_before_model_entry_and_close(
    tmp_path, request, monkeypatch
):
    from types import SimpleNamespace

    from Tests.Workspaces.test_project_index_service import Rig
    from Tests.Workspaces.test_project_semantic_generations import seed_keyword_ready
    from Tests.Workspaces.test_project_semantic_settings import digest, settings
    from tldw_chatbook.Backup_Recovery.participants import _core_cached_connection
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )

    rig = Rig(tmp_path)
    (rig.root / "dog.py").write_text("canine companion", encoding="utf-8")
    database = WorkspaceDB(tmp_path / "registry.db")
    database.close()  # The model worker has no borrowed caller handle.
    opened = []

    class Carrier:
        model_name = "deterministic"
        api_key = None
        base_url = None
        local_files_only = True
        artifact_fingerprint = digest("artifact-a")

        def close(self):
            pass

    def issue(*args):
        opened.append(database._held_connection())
        return Carrier()

    def closed(connection):
        assert _core_cached_connection(database, connection) is None
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(connection)

    class Wrapper:
        def __init__(self, **kwargs):
            closed(opened[0])

        async def create_embeddings_async(self, texts):
            opened.append(database._held_connection())
            return [[1.0, 0.0] for _ in texts]

        def close(self):
            closed(opened[1])

    rig.service._registry_connections = lambda: project_index_registry_connections(database)
    monkeypatch.setattr(semantic, "_issue_project_projection_scope", issue)
    monkeypatch.setattr(semantic, "_require_current_catalog", lambda *args: (None, None))
    monkeypatch.setattr(
        semantic, "_target",
        lambda *args: SimpleNamespace(artifact_fingerprint=digest("artifact-a")),
    )
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    try:
        status = seed_keyword_ready(rig)
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
        try:
            store.bind_native_lifetime(lease)
            state = store.configure_semantic(
                status.scope, settings(), enabled=True, expected_semantic_revision=0,
            )
        finally:
            store.close_native()
            lease.close()
        job = rig.service.refresh_semantic(
            rig.key, expected_epoch=status.scope.epoch,
            expected_semantic_revision=state.semantic_revision,
        )
        run = rig.service._semantic_runs[job]
        await asyncio.to_thread(run.thread.join, 20)
        assert not run.thread.is_alive() and run.native_retired
        assert len(opened) == 2
        assert rig.service.semantic_state(rig.key).ready_generation_id
    finally:
        await rig.close()
        database.close()


@private_profile_test
async def test_semantic_registry_callback_retires_each_owned_handle_and_preserves_borrowed(
    tmp_path, request, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.participants import _core_cached_connection
    from tldw_chatbook.DB import store_kernel
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.RAG_Search.projection_scope import ProjectProjectionRequired
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexRetirementUnproven
    from tldw_chatbook.Workspaces.project_index.runtime import (
        project_index_registry_connections,
    )

    database = WorkspaceDB(tmp_path / "registry.db")
    try:
        database.close()  # Constructor's thread-local handle is caller-owned.
        with project_index_registry_connections(database) as retire:
            first = database._held_connection()
            retire()
            assert _core_cached_connection(database, first) is None
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(first)
            second = database._held_connection()
            assert second is not first
            retire()
            assert _core_cached_connection(database, second) is None
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(second)

        with database.transaction() as borrowed:
            with project_index_registry_connections(database) as retire:
                with pytest.raises(ProjectProjectionRequired):
                    retire()
                assert database._held_connection() is borrowed
            assert borrowed.in_transaction
        database.close()

        original = store_kernel._core_closing

        @contextmanager
        def denied(repository, connection):
            with original(repository, connection) as allowed:
                yield False if repository is database else allowed

        with monkeypatch.context() as patch:
            patch.setattr(store_kernel, "_core_closing", denied)
            with pytest.raises(ProjectIndexRetirementUnproven):
                with project_index_registry_connections(database) as retire:
                    retained = database._held_connection()
                    retire()
            assert _core_cached_connection(database, retained) is retained
            assert sqlite3.Connection.in_transaction.__get__(retained) is False
    finally:
        database.close()
