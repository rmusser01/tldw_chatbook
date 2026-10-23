"""Semantic ranking over immutable rows with current run eligibility."""

from hashlib import sha256
import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from Tests.Workspaces.test_project_semantic_generations import populated_store
from Tests.Workspaces.test_project_semantic_settings import digest, settings
from Tests.private_profile import private_profile_test
from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion
from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime
from tldw_chatbook.Workspaces.project_index.semantic_vectors import encode_vector
from tldw_chatbook.Workspaces.project_index.models import IndexCandidate


def test_rrf_deduplicates_metadata_before_live_validation():
    from tldw_chatbook.Workspaces.project_index.semantic import fuse_candidates

    def candidate(path, rank):
        return IndexCandidate(path, "identity", "hash", 1, 1, rank)

    lexical = (candidate("a.py", 0), candidate("b.py", 1))
    semantic = (candidate("b.py", 0), candidate("c.py", 1))
    fused = fuse_candidates(lexical, semantic)
    assert [item.relative_path for item in fused] == ["b.py", "a.py", "c.py"]
    assert len(fused) == 3


def test_rank_filters_forbidden_dominant_match_before_top_k(tmp_path):
    from tldw_chatbook.Workspaces.project_index.semantic import rank_semantic

    store, scope, state, lease = populated_store(tmp_path)
    try:
        text = "canine companion"
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO files VALUES(?,?,?,?,?,?)",
                ("private/secret.py", "identity-2", 123, len(text), sha256(text.encode()).hexdigest(), text),
            )
            chunk = conn.execute(
                "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                ("private/secret.py", text, 0, len(text), 1, 1),
            ).lastrowid
            conn.execute("INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)", (chunk, "private/secret.py", text))
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
        generation = store.semantic_claim(
            scope, state,
            manifest_revision=store.load_status(scope.key).manifest_revision,
            artifact_fingerprint=digest("artifact-a"), claim_token="rank",
        )
        rows = store.semantic_source_page(scope, generation, "rank", after=0)
        assert store.semantic_append(
            scope, generation, "rank", rows,
            [encode_vector([0.8, 0.2]), encode_vector([1, 0])],
        )
        assert store.semantic_publish(scope, generation, "rank")
        floor = (SensitiveExclusion("subtree", "private"),)
        ranked = rank_semantic(
            store, scope, state, digest("artifact-a"), [1, 0], floor, limit=1
        )
        assert ranked is not None
        assert [candidate.relative_path for candidate in ranked] == ["src/dog.py"]
        assert rank_semantic(
            store, scope, state, digest("artifact-b"), [1, 0], floor, limit=1
        ) is None
        with store.transaction() as conn:
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
        assert rank_semantic(
            store, scope, state, digest("artifact-a"), [1, 0], floor, limit=1
        )[0].relative_path == "src/dog.py"
        with store.transaction() as conn:
            conn.execute(
                "UPDATE semantic_members SET vector=? WHERE relative_path='src/dog.py'",
                (b"\xff" * 8,),
            )
        with pytest.raises(ValueError, match="invalid_semantic_vector"):
            rank_semantic(
                store, scope, state, digest("artifact-a"), [1, 0], floor, limit=1
            )
        with store.transaction() as conn:
            conn.execute(
                "UPDATE semantic_members SET vector=? WHERE relative_path='src/dog.py'",
                (b"\x00" * (4 * 16385),),
            )
        with pytest.raises(ValueError, match="invalid_semantic_vector"):
            rank_semantic(
                store, scope, state, digest("artifact-a"), [1, 0], floor, limit=1
            )
    finally:
        store.close_native()
        lease.close()


@private_profile_test
async def test_service_search_fuses_synonym_hit_then_live_validates(tmp_path, request, monkeypatch):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture, ProjectIndexSearchAuthority,
    )

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
        scope = rig.enable().scope
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
        try:
            store.bind_native_lifetime(lease)
            text = file.read_text(encoding="utf-8")
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
            status = store.load_status(rig.key)
            state = store.configure_semantic(
                scope, settings(), enabled=True, expected_semantic_revision=0
            )
            generation = store.semantic_claim(
                status.scope, state, manifest_revision=status.manifest_revision,
                artifact_fingerprint=digest("artifact-a"), claim_token="search",
            )
            rows = store.semantic_source_page(status.scope, generation, "search", after=0)
            assert store.semantic_append(
                status.scope, generation, "search", rows,
                [encode_vector([1.0, 0.0]) for _ in rows],
            )
            assert store.semantic_publish(status.scope, generation, "search")
        finally:
            store.close_native()
            lease.close()

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
        result = rig.service.search(authority, "hound")
        assert [hit.relative_path for hit in result.hits] == ["dog.py"]
    finally:
        await rig.close()


@private_profile_test
async def test_query_close_retry_keeps_keyword_available_and_exact_owner(tmp_path, request, monkeypatch):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture, ProjectIndexRetirementUnproven, ProjectIndexSearchAuthority,
    )

    rig = Rig(tmp_path)
    file = rig.root / "dog.py"
    file.write_text("canine companion", encoding="utf-8")
    closed = threading.Event()
    calls = []

    class Carrier:
        model_name = "deterministic"
        api_key = None
        base_url = None
        local_files_only = True
        artifact_fingerprint = digest("artifact-a")

        def close(self):
            closed.set()

    class Wrapper:
        def __init__(self, **kwargs):
            pass

        def create_embedding(self, text):
            return [1.0, 0.0]

        def close(self):
            calls.append(threading.get_ident())
            if len(calls) < 4:
                raise RuntimeError("injected_close_failure")

    monkeypatch.setattr(semantic, "_issue_project_projection_scope", lambda *args: Carrier())
    monkeypatch.setattr(semantic, "_require_current_catalog", lambda *args: (None, None))
    monkeypatch.setattr(semantic, "_target", lambda *args: SimpleNamespace(artifact_fingerprint=digest("artifact-a")))
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    try:
        rig.enable()
        rig.refresh()
        status = rig.settle()
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(store.db_path.parent.parent / rig.key.lifetime_name)
        try:
            store.bind_native_lifetime(lease)
            state = store.configure_semantic(status.scope, settings(), enabled=True, expected_semantic_revision=0)
            generation = store.semantic_claim(
                status.scope, state, manifest_revision=status.manifest_revision,
                artifact_fingerprint=digest("artifact-a"), claim_token="search",
            )
            rows = store.semantic_source_page(status.scope, generation, "search", after=0)
            assert store.semantic_append(status.scope, generation, "search", rows, [encode_vector([1, 0]) for _ in rows])
            assert store.semantic_publish(status.scope, generation, "search")
        finally:
            store.close_native()
            lease.close()

        def read(candidate):
            data = file.read_text(encoding="utf-8")
            return FileCapture(candidate.relative_path, candidate.file_identity, 123, len(data), sha256(data.encode()).hexdigest(), data)

        authority = ProjectIndexSearchAuthority(status.scope, rig.live.root, rig.live.chain, (), 1, lambda: True, None, read)
        pending_search = asyncio.create_task(asyncio.to_thread(rig.service.search, authority, "hound"))
        deadline = asyncio.get_running_loop().time() + 10
        while not rig.service.semantic_close_pending(rig.key):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert not closed.is_set()
        keyword = await asyncio.to_thread(rig.service.search, authority, "canine")
        assert keyword.hits and keyword.semantic_status == "unavailable"
        assert not rig.service.retry_semantic_close(replace(rig.key, binding_id="other"))
        assert rig.service.retry_semantic_close(rig.key)
        deadline = asyncio.get_running_loop().time() + 10
        while len(calls) < 3:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert rig.service.semantic_close_pending(rig.key)
        with pytest.raises(ProjectIndexRetirementUnproven):
            await asyncio.wait_for(rig.service.aclose(), timeout=5)
        assert rig.service.retry_semantic_close(rig.key)
        with pytest.raises(RuntimeError, match="project_index_closed"):
            await asyncio.wait_for(pending_search, timeout=5)
        assert closed.is_set() and len(set(calls)) == 1
        assert not rig.service.semantic_close_pending(rig.key)
        assert not rig.service.retry_semantic_close(rig.key)
        await rig.service.aclose()
    finally:
        if rig.service.semantic_close_pending(rig.key):
            rig.service.retry_semantic_close(rig.key)
