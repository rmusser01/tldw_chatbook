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


def test_openai_embedder_failure_log_names_endpoint_and_cause(
    monkeypatch, captured_logs
):
    """The exhausted-retry error names the endpoint and the underlying cause."""
    monkeypatch.setattr(embeddings_lib, "_BACKOFF", (0,))

    class _FailingSession:
        headers: dict = {}

        def __init__(self) -> None:
            self.headers = {}

        def post(self, *args, **kwargs):
            raise embeddings_lib.requests.RequestException("connection refused")

    monkeypatch.setattr(embeddings_lib.requests, "Session", _FailingSession)

    cfg = embeddings_lib.OpenAICfg(
        provider="openai",
        model_name_or_path="text-embedding-3-small",
        dimension=1536,
        base_url="http://127.0.0.1:1/v1",
    )
    embed = embeddings_lib._openai_embedder(cfg)

    with pytest.raises(embeddings_lib.requests.RequestException):
        embed(["hello"])

    rendered = "".join(captured_logs)
    assert "%s" not in rendered and "%d" not in rendered
    assert "http://127.0.0.1:1/v1/embeddings" in rendered
    assert "connection refused" in rendered


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
