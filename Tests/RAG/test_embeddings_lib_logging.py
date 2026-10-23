"""Embeddings backend log lines must render their arguments.

TASK-32901 (tier-2 S10 P2): six ``logger.*`` calls in ``Embeddings_Lib``
used printf-style ``%s``/``%d`` placeholders with positional arguments.
loguru uses ``str.format`` and silently drops those arguments, so the two
``requests.RequestException`` handlers logged a literal ``%s`` line with no
endpoint and no cause -- a failing embedding backend was undiagnosable.
"""

from __future__ import annotations

import pytest

import tldw_chatbook.Embeddings.Embeddings_Lib as embeddings_lib


@pytest.fixture()
def captured_logs(monkeypatch):
    """Capture rendered loguru messages emitted during the test."""
    messages: list[str] = []
    sink_id = embeddings_lib.logger.add(
        lambda record: messages.append(record), level="DEBUG", format="{message}"
    )
    try:
        yield messages
    finally:
        embeddings_lib.logger.remove(sink_id)


def test_openai_embedder_failure_does_not_expose_endpoint_or_cause(
    monkeypatch, captured_logs
):
    """The exhausted-retry error classifies failure without revealing routing text."""
    monkeypatch.setattr(embeddings_lib, "_BACKOFF", (0,))

    class _FailingSession:
        headers: dict = {}

        def __init__(self) -> None:
            self.headers = {}

        def post(self, *args, **kwargs):
            raise embeddings_lib.requests.RequestException("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(embeddings_lib.requests, "Session", _FailingSession)

    cfg = embeddings_lib.OpenAICfg(
        provider="openai",
        model_name_or_path="text-embedding-3-small",
        dimension=1536,
        base_url="http://user:secret@127.0.0.1:1/v1?token=secret",
    )
    embed, close = embeddings_lib._openai_embedder(cfg)

    try:
        with pytest.raises(embeddings_lib.requests.RequestException) as failure:
            embed(["hello"])
    finally:
        close()

    rendered = "".join(captured_logs)
    assert "secret" not in rendered + str(failure.value)
    assert "127.0.0.1" not in rendered + str(failure.value)


def test_actual_openai_factory_without_key_refuses_before_provider_session(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(embeddings_lib, "_embeddings_rag_available", lambda: True)
    monkeypatch.setattr(
        embeddings_lib.requests,
        "Session",
        lambda: pytest.fail("provider session opened without a key"),
    )
    factory = embeddings_lib.EmbeddingFactory(
        {
            "default_model_id": "default",
            "models": {
                "default": {
                    "provider": "openai",
                    "model_name_or_path": "text-embedding-3-small",
                }
            },
        }
    )
    try:
        with pytest.raises(ValueError, match="embedding_provider_setup_required"):
            factory.embed(["synthetic text"])
    finally:
        factory.close()


def test_common_model_catalog_pins_a_revision_for_remote_code_models():
    """A catalog entry that executes Hub code must pin the commit it runs.

    TASK-32901 (tier-2 S10 P2): ``qwen3-embedding-4b`` shipped
    ``trust_remote_code=True`` with no ``revision``, while its sibling
    ``stella_en_1.5B_v5`` carries an explicit "Pinned for security" commit.
    Selecting the unpinned entry would execute whatever Python is at the HEAD
    of the Hub repository at download time.
    """
    unpinned = sorted(
        name
        for name, cfg in embeddings_lib.get_common_embedding_models().items()
        if getattr(cfg, "trust_remote_code", False)
        and not getattr(cfg, "revision", None)
    )
    assert unpinned == []


@pytest.mark.parametrize("outcomes", [("ok",), ("retry", "ok"), ("retry", "retry")])
def test_actual_factory_http_paths_hide_credentials(
    monkeypatch, captured_logs, outcomes
):
    monkeypatch.setattr(embeddings_lib, "_BACKOFF", (0, 0))
    monkeypatch.setattr(embeddings_lib, "_embeddings_rag_available", lambda: True)
    attempts = []
    sessions = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [1.0, 2.0]}]}

    class Session:
        def __init__(self):
            self.headers = {}
            self.close_count = 0
            sessions.append(self)

        def close(self):
            self.close_count += 1

        def post(self, url, *, json, timeout):
            attempts.append(url)
            outcome = outcomes[len(attempts) - 1]
            if outcome == "retry":
                raise embeddings_lib.requests.RequestException("secret-in-cause")
            return Response()

    monkeypatch.setattr(embeddings_lib.requests, "Session", Session)
    factory = embeddings_lib.EmbeddingFactory(
        {
            "default_model_id": "test",
            "models": {
                "test": {
                    "provider": "openai",
                    "model_name_or_path": "synthetic-model",
                    "api_key": "secret-header",
                    "base_url": "http://user:secret@127.0.0.1:1/v1?token=secret",
                }
            },
        },
        allow_dynamic_hf=False,
    )
    try:
        if outcomes[-1] == "retry":
            with pytest.raises(embeddings_lib.requests.RequestException) as failure:
                factory.embed(["synthetic text"])
            assert "secret" not in str(failure.value)
            assert "127.0.0.1" not in str(failure.value)
        else:
            assert factory.embed(["synthetic text"]).shape == (1, 2)
        assert len(attempts) == len(outcomes)
        rendered = "".join(captured_logs)
        assert "secret" not in rendered
        assert "127.0.0.1" not in rendered
    finally:
        factory.close()
    assert len(sessions) == 1 and sessions[0].close_count == 1


def test_default_factory_retains_openai_session_after_close_failure(monkeypatch):
    monkeypatch.setattr(embeddings_lib, "_embeddings_rag_available", lambda: True)
    sessions = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [1.0, 2.0]}]}

    class Session:
        def __init__(self):
            self.headers = {}
            self.close_calls = 0
            sessions.append(self)

        def post(self, *args, **kwargs):
            return Response()

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected_close_failure")

    monkeypatch.setattr(embeddings_lib.requests, "Session", Session)
    factory = embeddings_lib.EmbeddingFactory(
        {
            "default_model_id": "test",
            "models": {
                "test": {
                    "provider": "openai",
                    "model_name_or_path": "synthetic-model",
                    "base_url": "http://127.0.0.1:1/v1",
                }
            },
        },
        allow_dynamic_hf=False,
    )
    assert factory.embed(["synthetic text"]).shape == (1, 2)
    with pytest.raises(RuntimeError, match="injected_close_failure"):
        factory.close()
    assert len(sessions) == 1 and sessions[0].close_calls == 1
    assert "test" in factory._cache
    factory.close()
    assert sessions[0].close_calls == 2
    assert not factory._cache
