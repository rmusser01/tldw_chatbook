"""Endpoint isolation and actual SQLite ownership for the embedding cache."""

import asyncio
import hashlib
import json as json_module
import sqlite3
import threading
import time

import pytest
import requests

from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import (
    EmbeddingsServiceWrapper,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def cache_db(tmp_path):
    db = RAGIndexingDB(tmp_path / "embeddings.db")
    try:
        yield db
    finally:
        db.close()


def test_same_openai_model_at_different_endpoints_does_not_reuse_vectors(
    cache_db, monkeypatch
):
    calls = []

    def respond(session, url, *, json, timeout):
        calls.append(url)
        response = requests.Response()
        response.status_code = 200
        vector = [1.0, 0.0] if "first.invalid" in url else [0.0, 1.0]
        response._content = json_module.dumps(
            {"data": [{"embedding": vector} for _ in json["input"]]}
        ).encode()
        return response

    # Keep the real OpenAI factory/configuration; replace only the external
    # HTTP boundary so no provider is contacted or charged.
    monkeypatch.setattr(requests.Session, "post", respond)
    first = EmbeddingsServiceWrapper(
        "openai/shared-alias",
        device="cpu",
        api_key="test-key",
        base_url="https://reader:secret@first.invalid/v1/",
        embedding_cache_db=cache_db,
    )
    second = EmbeddingsServiceWrapper(
        "openai/shared-alias",
        device="cpu",
        api_key="test-key",
        base_url="https://reader:secret@second.invalid/v1",
        embedding_cache_db=cache_db,
    )
    restarted = EmbeddingsServiceWrapper(
        "openai/shared-alias",
        device="cpu",
        api_key="test-key",
        base_url="https://reader:secret@second.invalid/v1/",
        embedding_cache_db=cache_db,
    )
    try:
        assert first.create_embeddings(["same query"]).tolist() == [[1.0, 0.0]]
        assert second.create_embeddings(["same query"]).tolist() == [[0.0, 1.0]]
        assert restarted.create_embeddings(["same query"]).tolist() == [[0.0, 1.0]]
        assert len(calls) == 2
        with cache_db.connection() as connection:
            namespaces = [
                row[0]
                for row in connection.execute("SELECT model_id FROM embedding_cache")
            ]
        assert len(namespaces) == 2
        assert all("secret" not in key and "invalid" not in key for key in namespaces)
    finally:
        for wrapper in (first, second, restarted):
            wrapper.close()


def test_mock_cache_keeps_existing_model_identity(cache_db):
    wrapper = EmbeddingsServiceWrapper(
        "mock", device="cpu", embedding_cache_db=cache_db
    )
    try:
        expected = wrapper.create_embeddings(["offline query"])
        digest = hashlib.sha256(b"offline query").hexdigest()
        actual = cache_db.get_cached_embeddings("mock", [digest])
        assert actual[digest] == pytest.approx(expected[0].tolist())
    finally:
        wrapper.close()


async def test_async_cache_waits_for_sqlite_write_before_propagating_repeated_cancellation(
    cache_db, monkeypatch
):
    from tldw_chatbook.RAG_Search.activation import execution

    started = threading.Event()
    release = threading.Event()
    opened = []
    original = cache_db.store_cached_embeddings

    def blocked_store(model_id, rows):
        with cache_db.connection() as connection:
            opened.append(connection)
            started.set()
            assert release.wait(2.0), "test did not release the finite SQLite write"
            original(model_id, rows)

    monkeypatch.setattr(cache_db, "store_cached_embeddings", blocked_store)
    wrapper = EmbeddingsServiceWrapper(
        "mock", device="cpu", embedding_cache_db=cache_db
    )

    async def embed_inside_existing_scope():
        # The outer activation guard intentionally reuses this accepted task.
        # The cache worker must therefore retain its own cancellation lifetime.
        with execution(wrapper):
            return await wrapper.create_embeddings_async(["cancelled write query"])

    pending = asyncio.create_task(embed_inside_existing_scope())
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        pending.cancel()
        await asyncio.sleep(0.02)
        assert not pending.done(), "cancellation escaped before SQLite work stopped"
        pending.cancel()
        await asyncio.sleep(0.02)
        assert not pending.done(), "repeated cancellation abandoned the SQLite worker"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        for connection in opened:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                sqlite3.Connection.in_transaction.__get__(connection)
        digest = hashlib.sha256(b"cancelled write query").hexdigest()
        assert digest in cache_db.get_cached_embeddings("mock", [digest])
    finally:
        release.set()
        if not pending.done():
            await pending
        wrapper.close()


async def test_async_cache_keeps_memory_database_on_its_owner_thread():
    db = RAGIndexingDB(":memory:")
    wrapper = EmbeddingsServiceWrapper("mock", device="cpu", embedding_cache_db=db)
    try:
        expected = await wrapper.create_embeddings_async(["memory owner query"])
        actual = await wrapper.create_embeddings_async(["memory owner query"])
        assert actual.tolist() == expected.tolist()
        digest = hashlib.sha256(b"memory owner query").hexdigest()
        assert digest in db.get_cached_embeddings("mock", [digest])
        assert wrapper._cache_hits == 1
    finally:
        wrapper.close()
        db.close()


async def test_async_cache_preserves_borrowed_worker_connection(cache_db):
    from tldw_chatbook.DB.base_db import operation_owned_connection

    def borrow_then_read():
        with cache_db.connection() as connection:
            with operation_owned_connection(cache_db):
                assert cache_db.get_cached_embeddings("mock", ["absent"]) == {}
            # A borrowed operation handle remains usable inside its owner's scope.
            assert (
                connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
                == 0
            )
        cache_db.close()

    await asyncio.to_thread(borrow_then_read)


async def test_invalid_cached_vector_is_rebuilt_without_failing_async_embedding(
    cache_db,
):
    digest = hashlib.sha256(b"damaged cache query").hexdigest()
    with cache_db.connection() as connection:
        connection.execute(
            "INSERT INTO embedding_cache VALUES (?, ?, ?, ?)",
            ("mock", digest, b"not-float32", "2026-10-01T12:00:00.000Z"),
        )
    wrapper = EmbeddingsServiceWrapper(
        "mock", device="cpu", embedding_cache_db=cache_db
    )
    try:
        embeddings = await wrapper.create_embeddings_async(["damaged cache query"])
        assert embeddings.shape == (1, 384)
        assert cache_db.get_cached_embeddings("mock", [digest])[
            digest
        ] == pytest.approx(embeddings[0].tolist())
    finally:
        wrapper.close()


def test_corrupt_vector_diagnostic_does_not_disclose_cache_identifiers(
    cache_db, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.DB import RAG_Indexing_DB as db_module

    model_path = str(tmp_path / "private-model-directory")
    content_hash = "private-cache-identifier"
    with cache_db.connection() as connection:
        connection.execute(
            "INSERT INTO embedding_cache VALUES (?, ?, ?, ?)",
            (model_path, content_hash, b"not-float32", "2026-10-01T12:00:00.000Z"),
        )
    diagnostics = []
    monkeypatch.setattr(
        db_module, "logger", SimpleNamespace(warning=diagnostics.append)
    )
    assert cache_db.get_cached_embeddings(model_path, [content_hash]) == {}
    assert diagnostics
    assert all(
        model_path not in message and content_hash not in message
        for message in diagnostics
    )


async def test_sqlite_cache_lock_does_not_stall_async_embedding_event_loop(cache_db):
    wrapper = EmbeddingsServiceWrapper(
        "mock", device="cpu", embedding_cache_db=cache_db
    )
    writer = sqlite3.connect(cache_db.db_path_str, check_same_thread=False)
    writer.execute("BEGIN IMMEDIATE")
    heartbeat = threading.Event()

    def release_writer():
        heartbeat.wait(1.0)
        writer.rollback()

    releaser = threading.Thread(target=release_writer)
    releaser.start()
    pending = asyncio.create_task(
        wrapper.create_embeddings_async(["uncached lock query"])
    )
    started = time.monotonic()
    try:
        await asyncio.sleep(0.05)
        delay = time.monotonic() - started
        heartbeat.set()
        embeddings = await pending
        assert embeddings.shape == (1, 384)
        assert delay < 0.3, f"SQLite stalled the event loop for {delay:.3f}s"
    finally:
        heartbeat.set()
        await pending
        releaser.join()
        writer.close()
        wrapper.close()


async def test_async_cache_retires_new_worker_sqlite_handles(cache_db, monkeypatch):
    opened = []
    original = cache_db._get_connection

    def observe_connection():
        connection = original()
        opened.append(connection)
        return connection

    monkeypatch.setattr(cache_db, "_get_connection", observe_connection)
    wrapper = EmbeddingsServiceWrapper(
        "mock", device="cpu", embedding_cache_db=cache_db
    )
    try:
        await wrapper.create_embeddings_async(["worker query"])
        assert opened, "SQLite cache operations should run on owned worker connections"
        for connection in opened:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                sqlite3.Connection.in_transaction.__get__(connection)
        digest = hashlib.sha256(b"worker query").hexdigest()
        assert digest in cache_db.get_cached_embeddings("mock", [digest])
    finally:
        wrapper.close()
