"""B26: per-search pipeline overrides must not mutate the global table.

``BUILTIN_PIPELINES`` entries were shallow-copied before the reranker-model
override ran ``step.setdefault("config", {})["model"] = ...`` against nested
dicts -- every subsequent search inherited the override (global state
corruption). The hybrid builder already deep-copies; the plain and semantic
builders must too. Also: the OpenRouter streaming summarize logged one INFO
line per received chunk ("OpenRouter Stream: Content received"); the line is
removed, so zero per-chunk log records remain.
"""

from __future__ import annotations

import copy
import json

import pytest

import tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events as chat_rag_events
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    BUILTIN_PIPELINES,
)
import tldw_chatbook.LLM_Calls.Summarization_General_Lib as general_summarization

_WATCHED_PIPELINES = ("plain", "semantic", "hybrid")


@pytest.fixture(autouse=True)
def _restore_global_pipelines():
    """Restore the module-global table after each test.

    The bug under test mutates global state; without a restore the first
    polluted test would fail every later test in the session.
    """
    snapshot = copy.deepcopy(BUILTIN_PIPELINES)
    yield
    BUILTIN_PIPELINES.clear()
    BUILTIN_PIPELINES.update(snapshot)


def _snapshot() -> dict:
    return copy.deepcopy(
        {name: BUILTIN_PIPELINES[name] for name in _WATCHED_PIPELINES}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "builder",
    (
        chat_rag_events.perform_plain_rag_search,
        chat_rag_events.perform_full_rag_pipeline,
        chat_rag_events.perform_hybrid_rag_search,
    ),
)
async def test_reranker_override_does_not_mutate_global(
    builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _snapshot()
    captured: dict = {}

    async def fake_execute_pipeline(config, *args, **kwargs):
        captured["config"] = config
        return [], "context"

    monkeypatch.setattr(chat_rag_events, "execute_pipeline", fake_execute_pipeline)

    await builder(
        None,
        "test query",
        {"media": False, "notes": False},
        enable_rerank=True,
        reranker_model="nondefault-model",
    )

    after = _snapshot()
    assert before == after, (
        "module-global BUILTIN_PIPELINES was mutated by a per-search override"
    )
    # and the override landed on the per-call copy only
    rerank_steps = [
        step
        for step in captured["config"]["steps"]
        if step.get("function") == "rerank_results"
    ]
    assert rerank_steps, "expected a rerank step on the per-call config"
    assert all(
        step["config"]["model"] == "nondefault-model" for step in rerank_steps
    )
    global_overrides = [
        (name, step)
        for name in _WATCHED_PIPELINES
        for step in BUILTIN_PIPELINES[name]["steps"]
        if isinstance(step, dict)
        and step.get("function") == "rerank_results"
        and step.get("config", {}).get("model") == "nondefault-model"
    ]
    assert not global_overrides, global_overrides


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "builder",
    (
        chat_rag_events.perform_plain_rag_search,
        chat_rag_events.perform_full_rag_pipeline,
        chat_rag_events.perform_hybrid_rag_search,
    ),
)
async def test_rerank_removal_does_not_mutate_global(
    builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _snapshot()

    async def fake_execute_pipeline(config, *args, **kwargs):
        return [], "context"

    monkeypatch.setattr(chat_rag_events, "execute_pipeline", fake_execute_pipeline)

    await builder(
        None,
        "test query",
        {"media": False, "notes": False},
        enable_rerank=False,
    )

    after = _snapshot()
    assert before == after
    # the global table still carries its own rerank step untouched
    assert any(
        step.get("function") == "rerank_results"
        for step in BUILTIN_PIPELINES["plain"]["steps"]
    )


def _install_stream_fakes(monkeypatch: pytest.MonkeyPatch, chunk_count: int):
    # owned_json_post yields one record per SSE line with the "data: " prefix
    # already stripped; .data is the JSON payload text (or "[DONE]").
    payloads = [
        json.dumps({"choices": [{"delta": {"content": f"chunk-{i}"}}]})
        for i in range(chunk_count)
    ]
    payloads.append("[DONE]")
    closed = {"records": 0}

    class _Record:
        def __init__(self, data) -> None:
            self.data = data

    class _Records:
        def __iter__(self):
            return iter([_Record(payload) for payload in payloads])

        def close(self):
            closed["records"] += 1

    monkeypatch.setattr(
        general_summarization, "owned_json_post", lambda **kwargs: _Records()
    )
    monkeypatch.setattr(
        general_summarization,
        "HostedHTTPTransportConfig",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        general_summarization, "load_and_log_configs", lambda: None
    )

    def fake_get_cli_setting(section, key, default=None, **kwargs):
        if key == "api_retries":
            return 0
        if key == "api_retry_delay":
            return 0
        return default

    monkeypatch.setattr(general_summarization, "get_cli_setting", fake_get_cli_setting)
    return closed


def test_openrouter_stream_emits_no_per_chunk_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_count = 5
    _install_stream_fakes(monkeypatch, chunk_count)
    info_calls: list[str] = []
    real_info = general_summarization.logging.info

    def spy_info(msg, *args, **kwargs):
        info_calls.append(str(msg))
        real_info(msg, *args, **kwargs)

    monkeypatch.setattr(general_summarization.logging, "info", spy_info)

    result = general_summarization.summarize_with_openrouter(
        "fixed-openrouter-key",
        "fixed input",
        "fixed prompt",
        streaming=True,
    )

    assert result == "chunk-0chunk-1chunk-2chunk-3chunk-4"
    per_chunk_logs = sum(
        1 for msg in info_calls if "OpenRouter Stream: Content received" in msg
    )
    assert per_chunk_logs == 0, (
        f"expected 0 per-chunk INFO logs, got {per_chunk_logs} "
        f"(one per received chunk before the fix)"
    )
