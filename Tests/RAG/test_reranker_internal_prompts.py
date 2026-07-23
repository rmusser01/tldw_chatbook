# Tests/RAG/test_reranker_internal_prompts.py
"""Registry overrides must reach the reranker's LLM boundary; caller-supplied
config must still win. Fake lives ONLY at _call_llm.

The `scratch_config` fixture comes from Tests/RAG/conftest.py, which
re-exports the canonical definition in Tests/Internal_Prompts/conftest.py
(plain import, not `pytest_plugins` — see the comment in Tests/RAG/conftest.py
and Tests/Web_Scraping/conftest.py for why)."""

import pytest

from tldw_chatbook.RAG_Search.reranker import PointwiseReranker, RerankingConfig
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult


def _result(i):
    return SearchResult(
        id=str(i), document=f"doc {i}", metadata={"doc_title": f"t{i}"}, score=0.5
    )


@pytest.mark.asyncio
async def test_override_reaches_llm_boundary(scratch_config, monkeypatch):
    scratch_config(
        "[internal_prompts.rag_reranker]\n"
        'pointwise_template = "RATE {query} | {title} | {content} | end{reasoning}"\n'
        'pointwise_system = "CUSTOM SYSTEM"\n'
    )
    reranker = PointwiseReranker(RerankingConfig())
    assert reranker.config.system_prompt == "CUSTOM SYSTEM"

    captured = []

    async def fake_call_llm(prompt, system_prompt=None):
        captured.append(prompt)
        return '{"score": 0.9}'

    monkeypatch.setattr(reranker, "_call_llm", fake_call_llm)
    await reranker.rerank("my query", [_result(0), _result(1)])

    assert captured, "reranker never called the LLM"
    assert captured[0].startswith("RATE my query | t0 | doc 0 | end")


@pytest.mark.asyncio
async def test_caller_supplied_config_beats_registry(scratch_config, monkeypatch):
    scratch_config(
        '[internal_prompts.rag_reranker]\npointwise_system = "REGISTRY SYSTEM"\n'
    )
    config = RerankingConfig(system_prompt="CALLER SYSTEM")
    reranker = PointwiseReranker(config)
    assert reranker.config.system_prompt == "CALLER SYSTEM"


def test_default_when_no_config(scratch_config):
    reranker = PointwiseReranker(RerankingConfig())
    assert reranker.config.system_prompt.startswith(
        "You are a search result relevance evaluator."
    )
    assert 'Return JSON: {"score": 0.0-1.0{reasoning}}' in (
        reranker.config.scoring_prompt_template
    )
