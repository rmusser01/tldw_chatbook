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


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
def test_rank_filters_forbidden_dominant_match_before_top_k(tmp_path):
    from tldw_chatbook.Workspaces.project_index.semantic import rank_semantic

    store, scope, state, lease = populated_store(tmp_path)
    try:
        text = "canine companion"
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO files VALUES(?,?,?,?,?,?)",
                (
                    "private/secret.py",
                    "identity-2",
                    123,
                    len(text),
                    sha256(text.encode()).hexdigest(),
                    text,
                ),
            )
            chunk = conn.execute(
                "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                ("private/secret.py", text, 0, len(text), 1, 1),
            ).lastrowid
            conn.execute(
                "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                (chunk, "private/secret.py", text),
            )
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
        generation = store.semantic_claim(
            scope,
            state,
            manifest_revision=store.load_status(scope.key).manifest_revision,
            artifact_fingerprint=digest("artifact-a"),
            claim_token="rank",
        )
        rows = store.semantic_source_page(scope, generation, "rank", after=0)
        assert store.semantic_append(
            scope,
            generation,
            "rank",
            rows,
            [encode_vector([0.8, 0.2]), encode_vector([1, 0])],
        )
        assert store.semantic_publish(scope, generation, "rank")
        floor = (SensitiveExclusion("subtree", "private"),)
        ranked = rank_semantic(
            store, scope, state, digest("artifact-a"), [1, 0], floor, limit=1
        )
        assert ranked is not None
        assert [candidate.relative_path for candidate in ranked] == ["src/dog.py"]
        assert (
            rank_semantic(
                store, scope, state, digest("artifact-b"), [1, 0], floor, limit=1
            )
            is None
        )
        with store.transaction() as conn:
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
        assert (
            rank_semantic(
                store, scope, state, digest("artifact-a"), [1, 0], floor, limit=1
            )[0].relative_path
            == "src/dog.py"
        )
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


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
@pytest.mark.parametrize("mutation", [None, "disable", "reconfigure"])
@private_profile_test
async def test_service_search_fuses_synonym_hit_then_live_validates(
    tmp_path, request, monkeypatch, mutation
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ProjectIndexSearchAuthority,
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

    monkeypatch.setattr(
        semantic, "_issue_project_projection_scope", lambda *args: Carrier()
    )
    monkeypatch.setattr(
        semantic, "_require_current_catalog", lambda *args: (None, None)
    )
    monkeypatch.setattr(
        semantic,
        "_target",
        lambda *args: SimpleNamespace(artifact_fingerprint=digest("artifact-a")),
    )
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    try:
        scope = rig.enable().scope
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(
            store.db_path.parent.parent / rig.key.lifetime_name
        )
        try:
            store.bind_native_lifetime(lease)
            text = file.read_text(encoding="utf-8")
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO files VALUES(?,?,?,?,?,?)",
                    (
                        "dog.py",
                        "identity-1",
                        123,
                        len(text),
                        sha256(text.encode()).hexdigest(),
                        text,
                    ),
                )
                chunk = conn.execute(
                    "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                    ("dog.py", text, 0, len(text), 1, 1),
                ).lastrowid
                conn.execute(
                    "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                    (chunk, "dog.py", text),
                )
                conn.execute(
                    "UPDATE control SET state='Ready',manifest_revision=manifest_revision+1"
                )
            status = store.load_status(rig.key)
            state = store.configure_semantic(
                scope, settings(), enabled=True, expected_semantic_revision=0
            )
            generation = store.semantic_claim(
                status.scope,
                state,
                manifest_revision=status.manifest_revision,
                artifact_fingerprint=digest("artifact-a"),
                claim_token="search",
            )
            rows = store.semantic_source_page(
                status.scope, generation, "search", after=0
            )
            assert store.semantic_append(
                status.scope,
                generation,
                "search",
                rows,
                [encode_vector([1.0, 0.0]) for _ in rows],
            )
            assert store.semantic_publish(status.scope, generation, "search")
        finally:
            store.close_native()
            lease.close()

        entered, release = threading.Event(), threading.Event()

        def read(candidate):
            if mutation is not None:
                entered.set()
                assert release.wait(45)
            data = file.read_text(encoding="utf-8")
            return FileCapture(
                candidate.relative_path,
                candidate.file_identity,
                123,
                len(data),
                sha256(data.encode()).hexdigest(),
                data,
            )

        authority = ProjectIndexSearchAuthority(
            status.scope,
            rig.live.root,
            rig.live.chain,
            (),
            1,
            lambda: True,
            None,
            read,
        )
        if mutation is None:
            result = rig.service.search(authority, "hound")
            assert [hit.relative_path for hit in result.hits] == ["dog.py"]
        else:
            task = asyncio.create_task(
                asyncio.to_thread(rig.service.search, authority, "hound")
            )
            try:
                assert await asyncio.to_thread(entered.wait, 30)
                before = rig.service.status(rig.key)
                if mutation == "disable":
                    await asyncio.to_thread(
                        rig.service.disable_semantic,
                        rig.key,
                        expected_epoch=scope.epoch,
                        expected_semantic_revision=state.semantic_revision,
                    )
                else:
                    with rig.service._store(rig.key) as changed:
                        changed.configure_semantic(
                            scope,
                            settings("new-destination"),
                            enabled=True,
                            expected_semantic_revision=state.semantic_revision,
                        )
                assert rig.service.status(rig.key) == before
            finally:
                release.set()
                result = await task
            assert result.status == "unavailable" and result.hits == ()
            assert result.semantic_status != "ready"
    finally:
        await rig.close()


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
@private_profile_test
async def test_query_close_retry_keeps_keyword_available_and_exact_owner(
    tmp_path, request, monkeypatch
):
    from Tests.Workspaces.test_project_index_service import Rig
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ProjectIndexRetirementUnproven,
        ProjectIndexSearchAuthority,
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

    monkeypatch.setattr(
        semantic, "_issue_project_projection_scope", lambda *args: Carrier()
    )
    monkeypatch.setattr(
        semantic, "_require_current_catalog", lambda *args: (None, None)
    )
    monkeypatch.setattr(
        semantic,
        "_target",
        lambda *args: SimpleNamespace(artifact_fingerprint=digest("artifact-a")),
    )
    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", Wrapper)
    try:
        rig.enable()
        rig.refresh()
        status = rig.settle()
        store = rig.store(rig.key)
        lease = BindingLifetime.acquire(
            store.db_path.parent.parent / rig.key.lifetime_name
        )
        try:
            store.bind_native_lifetime(lease)
            state = store.configure_semantic(
                status.scope, settings(), enabled=True, expected_semantic_revision=0
            )
            generation = store.semantic_claim(
                status.scope,
                state,
                manifest_revision=status.manifest_revision,
                artifact_fingerprint=digest("artifact-a"),
                claim_token="search",
            )
            rows = store.semantic_source_page(
                status.scope, generation, "search", after=0
            )
            assert store.semantic_append(
                status.scope,
                generation,
                "search",
                rows,
                [encode_vector([1, 0]) for _ in rows],
            )
            assert store.semantic_publish(status.scope, generation, "search")
        finally:
            store.close_native()
            lease.close()

        def read(candidate):
            data = file.read_text(encoding="utf-8")
            return FileCapture(
                candidate.relative_path,
                candidate.file_identity,
                123,
                len(data),
                sha256(data.encode()).hexdigest(),
                data,
            )

        authority = ProjectIndexSearchAuthority(
            status.scope, rig.live.root, rig.live.chain, (), 1, lambda: True, None, read
        )
        pending_search = asyncio.create_task(
            asyncio.to_thread(rig.service.search, authority, "hound")
        )
        deadline = asyncio.get_running_loop().time() + 10
        while not rig.service.semantic_close_pending(rig.key):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert not closed.is_set()
        keyword = await asyncio.to_thread(rig.service.search, authority, "canine")
        assert keyword.hits and keyword.semantic_status == "unavailable"
        assert not rig.service.retry_semantic_close(
            replace(rig.key, binding_id="other")
        )
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


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
@private_profile_test
async def test_incompatible_ready_artifact_skips_query_model(
    tmp_path, request, monkeypatch
):
    from contextlib import contextmanager

    from tldw_chatbook import config
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ProjectIndexAuthority,
        ProjectIndexKey,
        ProjectIndexPolicy,
        ProjectIndexSearchAuthority,
        locator_fingerprint,
        root_identity_fingerprint,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

    current_config = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(current_config(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["preflight-openai"] = {
            "provider": "openai",
            "model_name_or_path": "preflight-vector",
            "base_url": "http://127.0.0.1:1/v1",
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)
    root = tmp_path / "project"
    root.mkdir()
    file = root / "dog.txt"
    file.write_text("canine companion", encoding="utf-8")
    chain = capture_directory_chain(root)
    factory = _AdmittedStoreFactory()
    key = ProjectIndexKey(
        factory.identity,
        "workspace",
        "binding",
        locator_fingerprint(root),
        root_identity_fingerprint(chain),
    )
    live = ProjectIndexAuthority(key, root, chain, (), 1)

    @contextmanager
    def fence(selected):
        yield live if selected == key else None

    service = ProjectIndexService(
        factory,
        ProjectScanner(),
        lambda selected: live if selected == key else None,
        fence,
    )
    try:
        scope = service.enable(
            key,
            ProjectIndexPolicy(("",), (".txt",)),
            expected_scope=service.status(key).scope,
        ).scope
        with factory.retain(key) as pin, pin.bind():
            store = factory(key)
            lease = BindingLifetime.acquire(
                store.db_path.parent.parent / key.lifetime_name
            )
            try:
                store.bind_native_lifetime(lease)
                text = file.read_text(encoding="utf-8")
                content_hash = sha256(text.encode()).hexdigest()
                with store.transaction() as conn:
                    conn.execute(
                        "INSERT INTO files VALUES(?,?,?,?,?,?)",
                        ("dog.txt", "identity-1", 123, len(text), content_hash, text),
                    )
                    chunk = conn.execute(
                        "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                        ("dog.txt", text, 0, len(text), 1, 1),
                    ).lastrowid
                    conn.execute(
                        "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                        (chunk, "dog.txt", text),
                    )
                    conn.execute(
                        "UPDATE control SET state='Ready',manifest_revision=manifest_revision+1"
                    )
                status = store.load_status(key)
            finally:
                store.close_native()
                lease.close()

        candidate = semantic._settings_for_project_model("preflight-openai")
        approved = replace(
            candidate,
            approved_destination_fingerprint=candidate.destination_fingerprint,
        )
        state = service.enable_semantic(
            key,
            approved,
            expected_epoch=scope.epoch,
            expected_semantic_revision=0,
        )
        spec, destination = semantic._require_current_catalog(approved)
        assert semantic._target(spec, destination).artifact_fingerprint != digest(
            "artifact-a"
        )
        with factory.retain(key) as pin, pin.bind():
            store = factory(key)
            lease = BindingLifetime.acquire(
                store.db_path.parent.parent / key.lifetime_name
            )
            try:
                store.bind_native_lifetime(lease)
                generation = store.semantic_claim(
                    status.scope,
                    state,
                    manifest_revision=status.manifest_revision,
                    artifact_fingerprint=digest("artifact-a"),
                    claim_token="old-artifact",
                )
                rows = store.semantic_source_page(
                    status.scope, generation, "old-artifact", after=0
                )
                assert store.semantic_append(
                    status.scope,
                    generation,
                    "old-artifact",
                    rows,
                    [encode_vector([1.0, 0.0]) for _ in rows],
                )
                assert store.semantic_publish(status.scope, generation, "old-artifact")
            finally:
                store.close_native()
                lease.close()

        issued, constructed = [], []
        original_issue = semantic._issue_project_projection_scope

        def tracked_issue(*args):
            issued.append(True)
            return original_issue(*args)

        class NoQueryModel:
            def __init__(self, **kwargs):
                constructed.append(True)
                raise AssertionError("incompatible artifact constructed query model")

        monkeypatch.setattr(semantic, "_issue_project_projection_scope", tracked_issue)
        monkeypatch.setattr(
            embeddings_wrapper, "EmbeddingsServiceWrapper", NoQueryModel
        )

        def read(candidate):
            data = file.read_text(encoding="utf-8")
            return FileCapture(
                candidate.relative_path,
                candidate.file_identity,
                123,
                len(data),
                sha256(data.encode()).hexdigest(),
                data,
            )

        search = ProjectIndexSearchAuthority(
            status.scope,
            root,
            chain,
            (),
            1,
            lambda: True,
            None,
            read,
        )
        result = service.search(search, "canine")
        assert [hit.relative_path for hit in result.hits] == ["dog.txt"]
        assert result.semantic_status == "unavailable"
        assert issued == [True] and constructed == []
    finally:
        await service.aclose()


@pytest.mark.usefixtures("qualified_project_semantic_sqlite")
@pytest.mark.loopback_network
@private_profile_test
async def test_cancelled_query_tool_observers_keep_native_close_owner(
    tmp_path, request, monkeypatch
):
    """Cancelled tool waiters cannot retire a held provider on a foreign thread."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import requests
    from urllib.parse import urlsplit

    from Tests.Agents.test_project_index_tool import invoke, provider
    from tldw_chatbook import config
    from tldw_chatbook.RAG_Search.simplified import embeddings_wrapper
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.project_index import semantic
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexAuthority,
        ProjectIndexKey,
        ProjectIndexPolicy,
        locator_fingerprint,
        root_identity_fingerprint,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/embeddings"
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
    url = f"http://127.0.0.1:{server.server_port}/v1"
    current = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(current(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["query-openai"] = {
            "provider": "openai",
            "model_name_or_path": "query-vector",
            "base_url": url,
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)
    original_request = requests.sessions.Session.request

    def local_only(session, method, requested_url, **kwargs):
        assert urlsplit(requested_url).hostname == "127.0.0.1"
        return original_request(session, method, requested_url, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", local_only)
    close_calls, native_closed, observed = [], [], []
    original_wrapper = embeddings_wrapper.EmbeddingsServiceWrapper
    actual_native_close = None
    owner_thread = None

    class TrackingWrapper(original_wrapper):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            observed.append(self)

        def create_embedding(self, text):
            nonlocal actual_native_close, owner_thread
            vector = super().create_embedding(text)
            owner_thread = threading.get_ident()
            record = self.factory._cache["default"]
            actual_native_close = record["close"]
            assert callable(actual_native_close)
            assert isinstance(actual_native_close.__self__, requests.Session)

            def fail_then_close():
                close_calls.append(threading.get_ident())
                if len(close_calls) < 4:
                    raise RuntimeError("native_close_injected")
                actual_native_close()
                native_closed.append(threading.get_ident())

            record["close"] = fail_then_close
            return vector

    monkeypatch.setattr(embeddings_wrapper, "EmbeddingsServiceWrapper", TrackingWrapper)
    root = tmp_path / "project"
    root.mkdir()
    file = root / "dog.txt"
    file.write_text("canine companion", encoding="utf-8")
    chain = capture_directory_chain(root)
    factory = _AdmittedStoreFactory()
    key = ProjectIndexKey(
        factory.identity,
        "workspace",
        "binding",
        locator_fingerprint(root),
        root_identity_fingerprint(chain),
    )
    live = ProjectIndexAuthority(key, root, chain, (), 1)

    @contextmanager
    def fence(selected):
        yield live if selected == key else None

    scanner = ProjectScanner()
    capture = scanner.capture_file(live, "dog.txt", stop=threading.Event())
    assert capture is not None and scanner.retired
    service = ProjectIndexService(
        factory,
        scanner,
        lambda selected: live if selected == key else None,
        fence,
    )
    executor = ThreadPoolExecutor(max_workers=2)
    query_future = None
    try:
        scope = service.enable(
            key,
            ProjectIndexPolicy(("",), (".txt",)),
            expected_scope=service.status(key).scope,
        ).scope
        with factory.retain(key) as pin, pin.bind():
            store = factory(key)
            lease = BindingLifetime.acquire(
                store.db_path.parent.parent / key.lifetime_name
            )
            try:
                store.bind_native_lifetime(lease)
                text = file.read_text(encoding="utf-8")
                with store.transaction() as conn:
                    conn.execute(
                        "INSERT INTO files VALUES(?,?,?,?,?,?)",
                        (
                            capture.relative_path,
                            capture.identity,
                            capture.mtime_ns,
                            capture.size_bytes,
                            capture.content_sha256,
                            capture.text,
                        ),
                    )
                    chunk = conn.execute(
                        "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) "
                        "VALUES(?,?,?,?,?,?)",
                        ("dog.txt", text, 0, len(text), 1, 1),
                    ).lastrowid
                    conn.execute(
                        "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                        (chunk, "dog.txt", text),
                    )
                    conn.execute(
                        "UPDATE control SET state='Ready',manifest_revision=manifest_revision+1"
                    )
                status = store.load_status(key)
            finally:
                store.close_native()
                lease.close()
        candidate = semantic._settings_for_project_model("query-openai")
        approved = replace(
            candidate,
            approved_destination_fingerprint=candidate.destination_fingerprint,
        )
        state = service.enable_semantic(
            key,
            approved,
            expected_epoch=scope.epoch,
            expected_semantic_revision=0,
        )
        spec, destination = semantic._require_current_catalog(approved)
        artifact = semantic._target(spec, destination).artifact_fingerprint
        with factory.retain(key) as pin, pin.bind():
            store = factory(key)
            lease = BindingLifetime.acquire(
                store.db_path.parent.parent / key.lifetime_name
            )
            try:
                store.bind_native_lifetime(lease)
                generation = store.semantic_claim(
                    status.scope,
                    state,
                    manifest_revision=status.manifest_revision,
                    artifact_fingerprint=artifact,
                    claim_token="query-ready",
                )
                rows = store.semantic_source_page(
                    status.scope,
                    generation,
                    "query-ready",
                    after=0,
                )
                assert store.semantic_append(
                    status.scope,
                    generation,
                    "query-ready",
                    rows,
                    [encode_vector([1.0, 0.0]) for _ in rows],
                )
                assert store.semantic_publish(status.scope, generation, "query-ready")
            finally:
                store.close_native()
                lease.close()
        tool = provider(SimpleNamespace(service=service, key=key, root=root))
        query_future = executor.submit(
            invoke,
            tool,
            {"query": "canine"},
            "semantic-query",
        )
        deadline = asyncio.get_running_loop().time() + 30
        while not service.semantic_close_pending(key):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert query_future.running() and len(close_calls) == 2
        assert observed[0].factory is not None
        carrier = observed[0]._projection_execution_scope
        assert not carrier._closed and native_closed == []

        async def observe():
            return await asyncio.wrap_future(query_future)

        for _ in range(2):
            waiter = asyncio.create_task(observe())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert query_future.running() and service.semantic_close_pending(key)
        await asyncio.wait_for(asyncio.sleep(0), 1)
        keyword_future = executor.submit(
            invoke,
            tool,
            {"query": "canine"},
            "keyword-while-pending",
        )
        keyword = await asyncio.wait_for(asyncio.wrap_future(keyword_future), 30)
        assert keyword.ok, keyword.error
        keyword_body = json.loads(keyword.content)
        assert keyword_body["hits"][0]["relative_path"] == "dog.txt"
        assert keyword_body["semantic_status"] == "unavailable"
        assert service.retry_semantic_close(key)
        deadline = asyncio.get_running_loop().time() + 10
        while len(close_calls) < 3:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert service.semantic_close_pending(key) and not carrier._closed
        assert service.retry_semantic_close(key)
        result = await asyncio.wait_for(asyncio.wrap_future(query_future), 30)
        assert result.ok, result.error
        assert native_closed == [owner_thread]
        assert close_calls == [owner_thread] * 4
        assert observed[0].factory is None
        assert observed[0]._project_close_complete and carrier._closed
        assert not service.semantic_close_pending(key)
        assert not service._operations
    finally:
        if query_future is not None and not query_future.done():
            if actual_native_close is not None and observed[0].factory is not None:
                observed[0].factory._cache["default"]["close"] = actual_native_close
            for _ in range(2):
                if not service.semantic_close_pending(key):
                    break
                service.retry_semantic_close(key)
                await asyncio.sleep(0.05)
            try:
                await asyncio.wait_for(asyncio.wrap_future(query_future), 10)
            except Exception:
                pass
        executor.shutdown(wait=False)
        try:
            await asyncio.wait_for(service.aclose(), 10)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(5)
