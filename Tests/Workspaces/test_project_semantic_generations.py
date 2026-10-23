"""Bounded semantic generation behavior over real project SQLite."""

import math
from hashlib import sha256
import asyncio
import os
import subprocess
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Workspaces.project_index_support import (
    make_index_key,
    open_semantic_store,
)
from Tests.Workspaces.test_project_semantic_settings import digest, settings
from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime, LifetimeBusy
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
from Tests.private_profile import private_profile_test


def populated_store(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    lease = BindingLifetime.acquire(store.db_path.parent.parent / key.lifetime_name)
    store.bind_native_lifetime(lease)
    scope = store.configure(
        key,
        ProjectIndexPolicy((), (".py",)),
        enabled=True,
        expected_scope=store.load_status(key).scope,
    )
    text = "canine companion"
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO files VALUES(?,?,?,?,?,?)",
            ("src/dog.py", "identity-1", 123, len(text), sha256(text.encode()).hexdigest(), text),
        )
        chunk = conn.execute(
            "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
            ("src/dog.py", text, 0, len(text), 1, 1),
        ).lastrowid
        conn.execute("INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)", (chunk, "src/dog.py", text))
        conn.execute("UPDATE control SET state='Ready',manifest_revision=manifest_revision+1")
    semantic = store.configure_semantic(
        scope, settings(), enabled=True, expected_semantic_revision=0
    )
    return store, scope, semantic, lease


def seed_keyword_ready(rig):
    """Commit one real keyword source without an unrelated scanner startup."""
    rig.enable()
    file = rig.root / "dog.py"
    text = file.read_text(encoding="utf-8")
    store = rig.store(rig.key)
    lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
    try:
        store.bind_native_lifetime(lease)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO files VALUES(?,?,?,?,?,?)",
                ("dog.py", "identity-1", 123, len(text), sha256(text.encode()).hexdigest(), text),
            )
            chunk = conn.execute(
                "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                ("dog.py", text, 0, len(text), 1, 1),
            ).lastrowid
            conn.execute(
                "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                (chunk, "dog.py", text),
            )
            conn.execute("UPDATE control SET state='Ready',manifest_revision=manifest_revision+1")
        return store.load_status(rig.key)
    finally:
        store.close_native()
        lease.close()


def start_held_semantic(rig, monkeypatch, started, release, *, close=None):
    """Start a real service/SQLite/lease job with a held async embedding call."""
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic

    class Carrier:
        model_name = "deterministic"
        api_key = None
        base_url = None
        local_files_only = True
        artifact_fingerprint = digest("artifact-a")

        def close(self):
            pass

    class Wrapper:
        def __init__(self, **kwargs):
            pass

        async def create_embeddings_async(self, texts):
            started.set()
            assert await asyncio.to_thread(release.wait, 45)
            return [[1.0, 0.0] for _ in texts]

        def close(self):
            if close is not None:
                close()

    monkeypatch.setattr(semantic, "_issue_project_projection_scope", lambda *args: Carrier())
    monkeypatch.setattr(semantic, "_require_current_catalog", lambda *args: (None, None))
    monkeypatch.setattr(
        semantic, "_target",
        lambda *args: SimpleNamespace(artifact_fingerprint=digest("artifact-a")),
    )
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
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
    assert started.wait(45)
    return status, state, rig.service._semantic_runs[job]


@private_profile_test
async def test_clear_during_pending_semantic_close_revokes_without_waiting_for_retry(
    tmp_path, request, monkeypatch
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexRetirementUnproven, ProjectIndexSearchAuthority,
    )

    rig = Rig(tmp_path)
    (rig.root / "dog.py").write_text("canine companion", encoding="utf-8")
    started, release = threading.Event(), threading.Event()
    closes = []

    def close():
        closes.append(threading.get_ident())
        if len(closes) < 3:
            raise RuntimeError("injected_model_close_failure")

    status, _, run = start_held_semantic(
        rig, monkeypatch, started, release, close=close,
    )
    clear_task = None
    try:
        release.set()
        deadline = asyncio.get_running_loop().time() + 10
        while not rig.service.semantic_close_pending(rig.key):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert len(closes) == 2 and not run.native_retired
        clear_task = asyncio.create_task(
            asyncio.to_thread(rig.service.clear, rig.key, expected_scope=status.scope)
        )
        with pytest.raises(ProjectIndexRetirementUnproven, match="close_pending"):
            await asyncio.wait_for(asyncio.shield(clear_task), timeout=5)
        assert run.stop.is_set() and not run.native_retired
        assert rig.service.status(rig.key).scope.epoch > status.scope.epoch
        old_search = ProjectIndexSearchAuthority(
            status.scope, rig.live.root, rig.live.chain, (), 1,
            lambda: True, None, lambda candidate: None,
        )
        assert rig.service.search(old_search, "canine").status == "unavailable"
        with pytest.raises(ProjectIndexRetirementUnproven, match="close_pending"):
            await asyncio.wait_for(rig.service.aclose(), timeout=5)
        assert rig.service.retry_semantic_close(rig.key)
        await asyncio.to_thread(run.thread.join, 10)
        assert not run.thread.is_alive() and run.native_retired
        assert run.lease.closed and run.store is None
        assert len(closes) == 3 and len(set(closes)) == 1
    finally:
        release.set()
        # Test cleanup is bounded even when the expected immediate retry is
        # missing; it must not leave a non-daemon model owner in the child.
        for _ in range(3):
            if not run.thread.is_alive():
                break
            if rig.service.semantic_close_pending(rig.key):
                rig.service.retry_semantic_close(rig.key)
            await asyncio.to_thread(run.thread.join, 2)
        if clear_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(clear_task), timeout=10)
            except Exception:
                pass
        await rig.close()


@pytest.mark.parametrize("change", ["detach", "exclusion", "settings"])
@private_profile_test
async def test_held_semantic_refuses_publication_after_binding_or_settings_change(
    tmp_path, request, monkeypatch, change
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion
    from tldw_chatbook.Workspaces.project_index import semantic

    rig = Rig(tmp_path)
    (rig.root / "dog.py").write_text("canine companion", encoding="utf-8")
    started, release = threading.Event(), threading.Event()
    status, state, run = start_held_semantic(rig, monkeypatch, started, release)
    change_task = None
    try:
        if change == "detach":
            rig.live = None
            rig.service.binding_changed(rig.key.binding_id)
            assert run.stop.is_set()
        elif change == "exclusion":
            rig.live = replace(
                rig.live,
                exclusions=(SensitiveExclusion("exact", "dog.py"),),
                authority_revision=rig.live.authority_revision + 1,
            )
        else:
            changed = settings("changed-destination")
            monkeypatch.setattr(
                semantic, "_settings_for_project_model",
                lambda model_id: replace(
                    changed, approved_destination_fingerprint=None,
                ),
            )
            change_task = asyncio.create_task(asyncio.to_thread(
                rig.service.enable_semantic,
                rig.key, changed,
                expected_epoch=status.scope.epoch,
                expected_semantic_revision=state.semantic_revision,
            ))
            deadline = asyncio.get_running_loop().time() + 5
            while not run.stop.is_set():
                if change_task.done():
                    await change_task
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)
        assert not run.native_retired
        release.set()
        await asyncio.to_thread(run.thread.join, 10)
        if change_task is not None:
            await asyncio.wait_for(change_task, timeout=5)
        assert not run.thread.is_alive() and run.native_retired
        assert run.lease.closed and run.store is None
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
        try:
            store.bind_native_lifetime(lease)
            current = store.load_status(rig.key)
            assert store.load_semantic_state(current.scope).ready_generation_id is None
        finally:
            store.close_native()
            lease.close()
    finally:
        release.set()
        if change_task is not None:
            await asyncio.wait_for(asyncio.shield(change_task), timeout=10)
        await rig.close()


def test_generation_pages_checkpoint_and_ready_survives_failed_rebuild(tmp_path):
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector

    store, scope, state, lease = populated_store(tmp_path)
    try:
        manifest = store.load_status(scope.key).manifest_revision
        first = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-a"), claim_token="first"
        )
        rows = store.semantic_source_page(scope, first, "first", after=0)
        assert len(rows) == 1 and rows[0].text == "canine companion"
        assert store.semantic_append(scope, first, "first", rows, [encode_vector([1, 0])])
        assert store.semantic_publish(scope, first, "first")
        assert store.load_semantic_state(scope).ready_generation_id == first

        second = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-a"), claim_token="second"
        )
        assert store.semantic_fail(scope, second, "second", reason="provider_unavailable")
        failed = store.load_semantic_state(scope)
        assert failed.ready_generation_id == first
        assert failed.state == "Ready"
        assert failed.reason == "previous_index_refresh_failed_provider_unavailable"
        assert store.semantic_ready_generation(scope, failed, digest("artifact-a")) is not None
        from tldw_chatbook.Workspaces.project_index.semantic import rank_semantic
        assert rank_semantic(
            store, scope, failed, digest("artifact-a"), [1, 0], (), limit=10
        )[0].relative_path == "src/dog.py"
    finally:
        store.close_native()
        lease.close()


def test_failed_first_generation_has_no_ready_projection(tmp_path):
    store, scope, state, lease = populated_store(tmp_path)
    try:
        generation = store.semantic_claim(
            scope, state,
            manifest_revision=store.load_status(scope.key).manifest_revision,
            artifact_fingerprint=digest("artifact-a"), claim_token="first",
        )
        assert store.semantic_fail(scope, generation, "first", reason="provider_unavailable")
        failed = store.load_semantic_state(scope)
        assert failed.state == "Error"
        assert failed.ready_generation_id is None
        assert store.semantic_ready_generation(scope, failed, digest("artifact-a")) is None
    finally:
        store.close_native()
        lease.close()


def test_ready_zero_members_is_distinct_from_unavailable_semantics(tmp_path):
    from tldw_chatbook.Workspaces.project_index.semantic import rank_semantic

    store, scope, state, lease = populated_store(tmp_path)
    try:
        assert rank_semantic(
            store, scope, state, digest("artifact-a"), [1, 0], (), limit=10
        ) is None
        with store.transaction() as conn:
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM files")
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
        generation = store.semantic_claim(
            scope, state,
            manifest_revision=store.load_status(scope.key).manifest_revision,
            artifact_fingerprint=digest("artifact-a"), claim_token="empty",
        )
        assert store.semantic_source_page(scope, generation, "empty", after=0) == ()
        assert store.semantic_publish(scope, generation, "empty")
        ready = store.load_semantic_state(scope)
        assert ready.state == "Ready" and ready.ready_generation_id == generation
        assert rank_semantic(
            store, scope, ready, digest("artifact-a"), [1, 0], (), limit=10
        ) == ()
    finally:
        store.close_native()
        lease.close()


def test_identical_text_keeps_vectors_in_its_own_binding(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector

    first, scope, state, first_lease = populated_store(tmp_path)
    other_key = replace(scope.key, binding_id="binding-2")
    other_path = tmp_path / other_key.binding_namespace / other_key.namespace
    other_path.mkdir(parents=True)
    second = WorkspaceProjectIndexDB(other_path / "workspace_project_index.db")
    second_lease = BindingLifetime.acquire(other_path.parent / other_key.lifetime_name)
    try:
        second.bind_native_lifetime(second_lease)
        second_scope = second.configure(
            other_key, ProjectIndexPolicy((), (".py",)), enabled=True,
            expected_scope=second.load_status(other_key).scope,
        )
        text = "canine companion"
        with second.transaction() as conn:
            conn.execute(
                "INSERT INTO files VALUES(?,?,?,?,?,?)",
                ("src/dog.py", "identity-1", 123, len(text), sha256(text.encode()).hexdigest(), text),
            )
            chunk = conn.execute(
                "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                ("src/dog.py", text, 0, len(text), 1, 1),
            ).lastrowid
            conn.execute(
                "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                (chunk, "src/dog.py", text),
            )
            conn.execute("UPDATE control SET state='Ready',manifest_revision=manifest_revision+1")
        second_state = second.configure_semantic(
            second_scope, settings(), enabled=True, expected_semantic_revision=0,
        )
        for store, selected_scope, selected_state, token, vector in (
            (first, scope, state, "first", [1, 0]),
            (second, second_scope, second_state, "second", [0, 1]),
        ):
            generation = store.semantic_claim(
                selected_scope, selected_state,
                manifest_revision=store.load_status(selected_scope.key).manifest_revision,
                artifact_fingerprint=digest("artifact-a"), claim_token=token,
            )
            page = store.semantic_source_page(selected_scope, generation, token, after=0)
            assert len(page) == 1 and page[0].text == text
            assert store.semantic_append(
                selected_scope, generation, token, page, [encode_vector(vector)]
            )
            assert store.semantic_publish(selected_scope, generation, token)
        assert scope.key.binding_namespace != second_scope.key.binding_namespace
        first_member = first.semantic_member_page(
            scope, state, digest("artifact-a"), (), after=0
        )[0]
        second_member = second.semantic_member_page(
            second_scope, second_state, digest("artifact-a"), (), after=0
        )[0]
        assert first_member.relative_path == second_member.relative_path
        assert first_member.vector == encode_vector([1, 0])
        assert second_member.vector == encode_vector([0, 1])
    finally:
        first.close_native()
        second.close_native()
        first_lease.close()
        second_lease.close()


def test_failed_rebuild_with_changed_artifact_cannot_project_prior_ready(tmp_path):
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector

    store, scope, state, lease = populated_store(tmp_path)
    try:
        manifest = store.load_status(scope.key).manifest_revision
        first = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-a"), claim_token="first",
        )
        page = store.semantic_source_page(scope, first, "first", after=0)
        assert store.semantic_append(scope, first, "first", page, [encode_vector([1, 0])])
        assert store.semantic_publish(scope, first, "first")
        second = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-b"), claim_token="second",
        )
        assert store.semantic_fail(scope, second, "second", reason="provider_unavailable")
        failed = store.load_semantic_state(scope)
        assert failed.state == "Error"
        assert store.semantic_ready_generation(scope, failed, digest("artifact-b")) is None
    finally:
        store.close_native()
        lease.close()


def test_generation_rejects_source_manifest_change_before_checkpoint(tmp_path):
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector

    store, scope, state, lease = populated_store(tmp_path)
    try:
        manifest = store.load_status(scope.key).manifest_revision
        generation = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-a"), claim_token="claim"
        )
        rows = store.semantic_source_page(scope, generation, "claim", after=0)
        with store.transaction() as conn:
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
        assert not store.semantic_append(
            scope, generation, "claim", rows, [encode_vector([1, 0])]
        )
        assert not store.semantic_publish(scope, generation, "claim")
        assert store.load_semantic_state(scope).ready_generation_id is None
        assert store.semantic_fail(scope, generation, "claim", reason="source_changed")
        assert store.load_semantic_state(scope).state == "Error"
    finally:
        store.close_native()
        lease.close()


def test_stale_semantic_failure_cannot_settle_new_claim_or_reconfigured_control(tmp_path):
    store, scope, state, lease = populated_store(tmp_path)
    try:
        manifest = store.load_status(scope.key).manifest_revision
        first = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-a"), claim_token="first",
        )
        second = store.semantic_claim(
            scope, state, manifest_revision=manifest,
            artifact_fingerprint=digest("artifact-a"), claim_token="second",
        )
        assert not store.semantic_fail(scope, first, "first", reason="source_changed")
        assert store.load_semantic_state(scope).state == "Building"
        changed = store.configure_semantic(
            scope, settings(), enabled=True,
            expected_semantic_revision=state.semantic_revision,
        )
        assert changed.semantic_revision > state.semantic_revision
        assert not store.semantic_fail(scope, second, "second", reason="source_changed")
        assert store.load_semantic_state(scope).state == "Pending"
    finally:
        store.close_native()
        lease.close()


def test_semantic_checkpoint_refuses_binding_allocation_quota(tmp_path, monkeypatch):
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector
    from tldw_chatbook.Workspaces.project_index.service import binding_bytes
    import importlib

    store, scope, state, lease = populated_store(tmp_path)
    try:
        generation = store.semantic_claim(
            scope, state,
            manifest_revision=store.load_status(scope.key).manifest_revision,
            artifact_fingerprint=digest("artifact-a"), claim_token="quota",
        )
        rows = store.semantic_source_page(scope, generation, "quota", after=0)
        module = importlib.import_module("tldw_chatbook.DB.Workspace_Project_Index_DB")
        with monkeypatch.context() as patch:
            patch.setattr(
                module, "MAX_CACHE_BYTES", binding_bytes(store.db_path.parent.parent)
            )
            with pytest.raises(ValueError, match="index_cache_limit"):
                store.semantic_append(
                    scope, generation, "quota", rows,
                    [encode_vector([1.0] * 16384)],
                )
        assert store.load_semantic_state(scope).ready_generation_id is None
        assert store.semantic_fail(scope, generation, "quota", reason="quota_refused")
        assert store.load_semantic_state(scope).state == "Error"
    finally:
        store.close_native()
        lease.close()


@private_profile_test
async def test_service_builder_keeps_keyword_lookup_live_during_embedding_wait(
    tmp_path, request, monkeypatch
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ProjectIndexSearchAuthority,
    )

    started, release = threading.Event(), threading.Event()
    rig = Rig(tmp_path)
    file = rig.root / "dog.py"
    file.write_text("canine companion", encoding="utf-8")

    class Carrier:
        model_name = "deterministic"
        api_key = None
        base_url = None
        local_files_only = True
        artifact_fingerprint = digest("artifact-a")

        def close(self):
            pass

    class Wrapper:
        def __init__(self, **kwargs):
            pass

        async def create_embeddings_async(self, texts):
            started.set()
            assert await asyncio.to_thread(release.wait, 20)
            return [[1.0, 0.0] for _ in texts]

        def create_embedding(self, text):
            return [1.0, 0.0]

        def close(self):
            pass

    monkeypatch.setattr(semantic, "_issue_project_projection_scope", lambda *args: Carrier())
    monkeypatch.setattr(semantic, "_require_current_catalog", lambda *args: (None, None))
    monkeypatch.setattr(
        semantic, "_target",
        lambda *args: SimpleNamespace(artifact_fingerprint=digest("artifact-a")),
    )
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    try:
        rig.enable()
        rig.refresh()
        status = rig.settle()
        for run in rig.service._runs.values():
            await asyncio.to_thread(run.supervisor.join)
        from Tests.Workspaces.test_project_index_service import POLICY
        from tldw_chatbook.Tools.remote_root_types import LocalRoot
        from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
        from tldw_chatbook.Workspaces.project_index.models import (
            ProjectIndexAuthority, locator_fingerprint, root_identity_fingerprint,
        )
        from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
        from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

        other_root = tmp_path / "other-project"
        other_root.mkdir()
        other_chain = capture_directory_chain(other_root)
        other_namespace = replace(
            rig.key,
            locator_fingerprint=locator_fingerprint(other_root),
            root_identity_fingerprint=root_identity_fingerprint(other_chain),
        )
        assert other_namespace.namespace != rig.key.namespace
        assert other_namespace.binding_namespace == rig.key.binding_namespace
        other_live = ProjectIndexAuthority(
            other_namespace, LocalRoot(other_root), other_chain, (), 1
        )
        other_store = rig.store(other_namespace)
        other_lease = BindingLifetime.acquire(
            other_store.db_path.parent.parent / other_namespace.lifetime_name
        )
        try:
            other_store.bind_native_lifetime(other_lease)
            other_scope = other_store.configure(
                other_namespace, POLICY, enabled=True,
                expected_scope=other_store.load_status(other_namespace).scope,
            )
            other_status = other_store.load_status(other_namespace)
        finally:
            other_store.close_native()
            other_lease.close()
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
        try:
            store.bind_native_lifetime(lease)
            semantic_state = store.configure_semantic(
                status.scope, settings(), enabled=True, expected_semantic_revision=0
            )
        finally:
            store.close_native()
            lease.close()
        job_id = rig.service.refresh_semantic(
            rig.key, expected_epoch=status.scope.epoch,
            expected_semantic_revision=semantic_state.semantic_revision,
        )
        assert await asyncio.to_thread(started.wait, 10)
        def read(candidate):
            data = file.read_text(encoding="utf-8")
            return FileCapture(
                candidate.relative_path, candidate.file_identity, 123,
                len(data), sha256(data.encode()).hexdigest(), data,
            )
        authority = ProjectIndexSearchAuthority(
            status.scope, rig.live.root, rig.live.chain, (), 1,
            lambda: True, None, read,
        )
        result = await asyncio.to_thread(rig.service.search, authority, "canine")
        assert result.hits and result.hits[0].relative_path == "dog.py"
        with pytest.raises(Exception):
            rig.refresh()
        assert other_namespace.lifetime_name == rig.key.lifetime_name
        other_service = ProjectIndexService(
            rig.store, ProjectScanner(),
            lambda selected: other_live if selected == other_namespace else None,
            lambda selected: rig.fence(rig.key),
        )
        try:
            with pytest.raises(LifetimeBusy):
                other_service.refresh(
                    other_namespace, expected_scope=other_scope,
                    expected_job_revision=other_status.job_revision,
                )
        finally:
            await other_service.aclose()
        lifetime_path = store.db_path.parent.parent / other_namespace.lifetime_name
        child = subprocess.run(
            [
                sys.executable, "-c",
                "from pathlib import Path; from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime, LifetimeBusy; "
                "import sys; "
                "p=Path(sys.argv[1]); "
                "exec('try:\\n BindingLifetime.acquire(p)\\nexcept LifetimeBusy:\\n print(\"busy\")\\nelse:\\n raise SystemExit(3)')",
                str(lifetime_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        assert child.returncode == 0 and child.stdout.strip() == "busy"
        release.set()
        await asyncio.to_thread(rig.service._semantic_runs[job_id].thread.join)
        assert rig.service.semantic_state(rig.key).ready_generation_id
    finally:
        release.set()
        await rig.close()


@private_profile_test
async def test_unknown_registry_retirement_roots_model_and_native_owners(
    tmp_path, request, monkeypatch
):
    """An uncertain native close cannot unwind the actual model owner."""
    from contextlib import contextmanager
    import gc
    import sqlite3
    import weakref

    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexRetirementUnproven

    rig = Rig(tmp_path)
    (rig.root / "dog.py").write_text("canine companion", encoding="utf-8")
    observed = []
    native = []

    @contextmanager
    def registry():
        calls = 0

        def retire():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ProjectIndexRetirementUnproven("injected_native_close_uncertain")

        yield retire

    class Carrier:
        model_name = "deterministic"
        api_key = None
        base_url = None
        local_files_only = True
        artifact_fingerprint = digest("artifact-a")

        def close(self):
            raise AssertionError("carrier_close_with_uncertain_native_registry")

    class Wrapper:
        def __init__(self, **kwargs):
            observed.append((weakref.ref(self), weakref.ref(kwargs["execution_scope"])))

        async def create_embeddings_async(self, texts):
            native.append(sqlite3.connect(":memory:", check_same_thread=False))
            return [[1.0, 0.0] for _ in texts]

        def close(self):
            raise AssertionError("model_close_with_uncertain_native_registry")

    rig.service._registry_connections = registry
    monkeypatch.setattr(semantic, "_issue_project_projection_scope", lambda *args: Carrier())
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    try:
        status = seed_keyword_ready(rig)
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
        try:
            store.bind_native_lifetime(lease)
            state = store.configure_semantic(
                status.scope, settings(), enabled=True,
                expected_semantic_revision=0,
            )
        finally:
            store.close_native()
            lease.close()
        job = rig.service.refresh_semantic(
            rig.key, expected_epoch=status.scope.epoch,
            expected_semantic_revision=state.semantic_revision,
        )
        run = rig.service._semantic_runs[job]
        await asyncio.to_thread(run.thread.join, 10)
        assert not run.thread.is_alive() and not run.native_retired
        assert run.lease is not None and not run.lease.closed
        assert run.read_admission is not None
        assert run.store is not None
        gc.collect()
        assert observed and all(reference() is not None for reference in observed[0])
        assert native and sqlite3.Connection.in_transaction.__get__(native[0]) is False
        assert any(operation.retirement_unproven for operation in rig.service._operations)
        with pytest.raises(
            (ProjectIndexRetirementUnproven, RuntimeError),
            match="retirement_unproven",
        ):
            await asyncio.wait_for(rig.service.aclose(), timeout=5)
    finally:
        # This intentionally unprovable owner is quarantined until process exit.
        # A foreign-thread test cleanup would invalidate the ownership control.
        pass


@private_profile_test
async def test_retired_semantic_error_drops_traceback_but_keeps_live_join_record(
    tmp_path, request, monkeypatch
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.service import _SemanticRun

    rig = Rig(tmp_path)
    (rig.root / "dog.py").write_text("canine companion", encoding="utf-8")

    class Carrier:
        model_name = "deterministic"
        api_key = None
        base_url = None
        local_files_only = True
        artifact_fingerprint = digest("artifact-a")

        def close(self):
            pass

    class Wrapper:
        def __init__(self, **kwargs):
            pass

        async def create_embeddings_async(self, texts):
            raise ValueError("injected_provider_failure")

        def close(self):
            pass

    monkeypatch.setattr(semantic, "_issue_project_projection_scope", lambda *args: Carrier())
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    historic = []
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
        # The first old record still has a live owner join; the second is done.
        # Both lie before the bounded diagnostic tail when this job retires.
        for index in range(17):
            job = f"historic-{index}"
            old = _SemanticRun(job, status.scope, state, rig.live, threading.Event())
            old.native_retired = True
            old.thread = SimpleNamespace(is_alive=lambda i=index: i == 0)
            rig.service._semantic_runs[job] = old
            historic.append(job)
        job = rig.service.refresh_semantic(
            rig.key, expected_epoch=status.scope.epoch,
            expected_semantic_revision=state.semantic_revision,
        )
        run = rig.service._semantic_runs[job]
        await asyncio.to_thread(run.thread.join, 10)
        assert not run.thread.is_alive() and run.native_retired
        assert run.error is None and run.store is None
        assert rig.service.semantic_state(rig.key).state == "Error"
        assert historic[0] in rig.service._semantic_runs
        assert historic[1] not in rig.service._semantic_runs
    finally:
        for job in historic:
            rig.service._semantic_runs.pop(job, None)
        await rig.close()

@pytest.mark.parametrize("vector", [[], [math.nan], [math.inf], [1e50]])
def test_vector_codec_refuses_empty_nonfinite_and_float32_overflow(vector):
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector

    with pytest.raises(ValueError):
        encode_vector(vector)


def test_vector_codec_round_trips_finite_float32_and_refuses_corrupt_bytes():
    from tldw_chatbook.Workspaces.project_index.semantic_vectors import (
        decode_vector,
        encode_vector,
    )

    encoded = encode_vector([3.0, 4.0])
    assert decode_vector(encoded, 2) == (3.0, 4.0)
    with pytest.raises(ValueError):
        decode_vector(encoded[:-1], 2)
    with pytest.raises(ValueError):
        decode_vector(b"\x00\x00\xc0\x7f", 1)


@private_profile_test
async def test_actual_cached_local_model_builds_async_generation_offline(
    tmp_path, request, monkeypatch
):
    """Qualify the production async page path with an existing cached model."""
    from contextlib import contextmanager
    from dataclasses import replace

    from tldw_chatbook import config
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexAuthority, ProjectIndexKey, ProjectIndexPolicy,
        locator_fingerprint, root_identity_fingerprint,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.semantic import _settings_for_project_model
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

    model_id = os.environ.get("TLDW_TEST_OFFLINE_EMBEDDING_MODEL")
    if not model_id:
        pytest.skip("No pre-existing offline embedding snapshot selected")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    current = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(current(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["minilm-local"] = {
            "provider": "huggingface", "model_name_or_path": model_id,
            "device": "cpu",
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)
    attempted = []

    def blocked(self, *args, **kwargs):
        attempted.append(args)
        raise AssertionError("offline_embedding_http_attempted")

    import httpx
    import requests

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)

    def run():
        root = tmp_path / "project"
        root.mkdir()
        (root / "dog.txt").write_text("canine companion", encoding="utf-8")
        chain = capture_directory_chain(root)
        factory = _AdmittedStoreFactory()
        key = ProjectIndexKey(
            factory.identity, "workspace", "binding",
            locator_fingerprint(root), root_identity_fingerprint(chain),
        )
        authority = ProjectIndexAuthority(key, root, chain, (), 1)

        @contextmanager
        def fence(selected):
            yield authority if selected == key else None

        service = ProjectIndexService(
            factory, ProjectScanner(),
            lambda selected: authority if selected == key else None, fence,
        )
        try:
            status = service.status(key)
            scope = service.enable(
                key, ProjectIndexPolicy(("",), (".txt",)),
                expected_scope=status.scope,
            ).scope
            current_status = service.status(key)
            service.refresh(
                key, expected_scope=scope,
                expected_job_revision=current_status.job_revision,
            )
            for active in service._runs.values():
                active.supervisor.join(45)
                assert not active.supervisor.is_alive()
            assert service.status(key).state == "Ready"
            candidate = _settings_for_project_model("minilm-local")
            approved = replace(
                candidate,
                approved_destination_fingerprint=candidate.destination_fingerprint,
            )
            state = service.enable_semantic(
                key, approved, expected_epoch=scope.epoch,
                expected_semantic_revision=0,
            )
            job_id = service.refresh_semantic(
                key, expected_epoch=scope.epoch,
                expected_semantic_revision=state.semantic_revision,
            )
            job = service._semantic_runs[job_id]
            job.thread.join(150)
            assert not job.thread.is_alive()
            assert job.native_retired and job.error is None
            ready = service.semantic_state(key)
            assert ready.state == "Ready" and ready.ready_generation_id
            assert not attempted
        finally:
            asyncio.run(service.aclose())

    await asyncio.to_thread(run)


@pytest.mark.loopback_network
@private_profile_test
async def test_actual_http_provider_wait_allows_keyword_lookup(
    tmp_path, request, monkeypatch
):
    """Hold a local OpenAI-compatible response during a real async build."""
    from contextlib import contextmanager
    from dataclasses import replace
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json
    import time
    import requests
    from urllib.parse import urlsplit

    from tldw_chatbook import config
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexAuthority, ProjectIndexKey, ProjectIndexPolicy,
        ProjectIndexSearchAuthority, locator_fingerprint, root_identity_fingerprint,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.semantic import _settings_for_project_model
    from tldw_chatbook.Workspaces.project_index.service import (
        ProjectIndexRetirementUnproven, ProjectIndexService,
    )
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper

    entered, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/embeddings"
            entered.set()
            assert release.wait(60)
            body = json.dumps({"data": [{"embedding": [1.0, 0.0]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    current = config.load_cli_config_and_ensure_existence
    url = f"http://127.0.0.1:{server.server_port}/v1"

    def configured(*args, **kwargs):
        data = dict(current(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["held-openai"] = {
            "provider": "openai", "model_name_or_path": "held-vector",
            "base_url": url,
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)
    original = requests.sessions.Session.request

    def local_only(self, method, requested_url, **kwargs):
        assert urlsplit(requested_url).hostname == "127.0.0.1"
        return original(self, method, requested_url, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", local_only)
    observed_wrappers = []
    original_wrapper = embeddings_wrapper.EmbeddingsServiceWrapper

    class TrackingWrapper(original_wrapper):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            observed_wrappers.append(self)

    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", TrackingWrapper)

    def run():
        root = tmp_path / "project"
        root.mkdir()
        (root / "dog.txt").write_text("canine companion", encoding="utf-8")
        chain = capture_directory_chain(root)
        factory = _AdmittedStoreFactory()
        key = ProjectIndexKey(
            factory.identity, "workspace", "binding",
            locator_fingerprint(root), root_identity_fingerprint(chain),
        )
        authority = ProjectIndexAuthority(key, root, chain, (), 1)

        @contextmanager
        def fence(selected):
            yield authority if selected == key else None

        scanner = ProjectScanner()
        service = ProjectIndexService(
            factory, scanner,
            lambda selected: authority if selected == key else None, fence,
        )
        try:
            scope = service.enable(
                key, ProjectIndexPolicy(("",), (".txt",)),
                expected_scope=service.status(key).scope,
            ).scope
            status = service.status(key)
            service.refresh(key, expected_scope=scope, expected_job_revision=status.job_revision)
            for active in service._runs.values():
                active.supervisor.join(45)
                assert not active.supervisor.is_alive()
            assert service.status(key).state == "Ready"
            candidate = _settings_for_project_model("held-openai")
            approved = replace(candidate, approved_destination_fingerprint=candidate.destination_fingerprint)
            state = service.enable_semantic(
                key, approved, expected_epoch=scope.epoch,
                expected_semantic_revision=0,
            )
            job_id = service.refresh_semantic(
                key, expected_epoch=scope.epoch,
                expected_semantic_revision=state.semantic_revision,
            )
            assert entered.wait(45)
            assert len(observed_wrappers) == 1
            factory_ref = observed_wrappers[0].factory
            record = factory_ref._cache["default"]
            close_calls = []
            native_closed = []
            actual_native_close = record["close"]
            assert callable(actual_native_close)
            assert isinstance(actual_native_close.__self__, requests.Session)

            def fail_then_close():
                close_calls.append(threading.get_ident())
                if len(close_calls) < 3:
                    raise RuntimeError("native_close_injected")
                actual_native_close()
                native_closed.append(threading.get_ident())

            record["close"] = fail_then_close
            search = ProjectIndexSearchAuthority(
                scope, authority.root, chain, (), 1, lambda: True, None,
                lambda candidate: scanner.capture_file(
                    authority, candidate.relative_path, stop=threading.Event()
                ),
            )
            found = service.search(search, "canine")
            assert found.hits and found.hits[0].relative_path == "dog.txt"
            assert found.semantic_status == "unavailable"
            job = service._semantic_runs[job_id]
            cancelled_twice = threading.Event()
            observer_errors = []

            def observe_close():
                async def cancel_waiter_twice():
                    waiter = asyncio.create_task(service.aclose())
                    await asyncio.sleep(0.05)
                    waiter.cancel()
                    await asyncio.sleep(0.05)
                    waiter.cancel()
                    cancelled_twice.set()
                    try:
                        await waiter
                    except BaseException as error:
                        observer_errors.append(error)

                asyncio.run(cancel_waiter_twice())

            observer = threading.Thread(target=observe_close)
            observer.start()
            assert cancelled_twice.wait(5)
            assert observer.is_alive() and not job.native_retired
            release.set()
            deadline = time.monotonic() + 15
            while not service.semantic_close_pending(key):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert not job.native_retired
            observer.join(5)
            assert not observer.is_alive()
            assert len(observer_errors) == 1
            assert isinstance(observer_errors[0], ProjectIndexRetirementUnproven)
            with pytest.raises(ProjectIndexRetirementUnproven):
                asyncio.run(asyncio.wait_for(service.aclose(), timeout=5))
            assert len(close_calls) == 2
            assert service.retry_semantic_close(key)
            job.thread.join(30)
            assert not job.thread.is_alive() and job.native_retired
            assert job.lease.closed and job.store is None
            assert len(close_calls) == 3 and len(set(close_calls)) == 1
            assert native_closed == [close_calls[-1]]
            assert not factory_ref._cache
            assert observed_wrappers[0].factory is None
            assert observed_wrappers[0]._project_close_complete
            assert observed_wrappers[0]._projection_execution_scope._closed
            reopened = ProjectIndexService(
                factory, scanner,
                lambda selected: authority if selected == key else None, fence,
            )
            try:
                assert reopened.semantic_state(key).ready_generation_id is None
            finally:
                asyncio.run(reopened.aclose())
        finally:
            release.set()
            if "actual_native_close" in locals():
                # Restore the real close callback for bounded cleanup even if
                # the test assertion failed before its explicit retry.
                record["close"] = actual_native_close
            for active in service._semantic_runs.values():
                for _ in range(3):
                    if active.thread is None or not active.thread.is_alive():
                        break
                    if service.semantic_close_pending(key):
                        service.retry_semantic_close(key)
                    active.thread.join(2)
            if "observer" in locals():
                observer.join(5)
            asyncio.run(service.aclose())

    try:
        await asyncio.to_thread(run)
    finally:
        release.set()
        server.shutdown()
        server_thread.join(5)
        server.server_close()
