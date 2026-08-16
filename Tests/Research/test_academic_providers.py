"""Academic search providers (task-16326).

httpx-based arXiv + Semantic Scholar runners with configurable timeouts,
retry-with-backoff, config-driven API keys, constants-driven endpoints, DOI
normalization, and the DOI-level dedup merge that feeds papers into the same
evidence pool as web results. HTTP is faked at the client seam; parsing,
retry, and merge logic run real.
"""

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

from tldw_chatbook.Research_Interop.academic_providers import (
    ARXIV_API_ENDPOINT,
    SEMANTIC_SCHOLAR_API_ENDPOINT,
    AcademicProviderError,
    merge_evidence_pools,
    papers_to_evidence,
    resolve_semantic_scholar_api_key,
    search_arxiv,
    search_papers,
    search_semantic_scholar,
)

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>Persistent Agents  Checkpointing</title>
    <published>2024-01-02T00:00:00Z</published>
    <summary>We show   checkpoints survive.</summary>
    <author><name>A. Author</name></author>
    <link rel="related" title="pdf" type="application/pdf" href="http://arxiv.org/pdf/2401.00001v2"/>
  </entry>
</feed>
"""


def _client_returning(responses):
    """Fake httpx client whose .request pops canned responses; records calls."""
    client = MagicMock()
    client.request.calls = []
    queue = list(responses)

    def request(method, url, **kwargs):
        client.request.calls.append({"method": method, "url": url, **kwargs})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client.request.side_effect = request
    return client


def _response(status_code=200, text="", headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.content = text.encode("utf-8")
    response.headers = headers or {}
    return response


def test_arxiv_uses_constants_endpoint_with_timeout_and_parses_atom(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=_ATOM)])

    out = search_arxiv(query="agents", client=client, timeout=12.5)

    call = client.request.calls[0]
    assert call["url"] == ARXIV_API_ENDPOINT
    assert call["timeout"] == 12.5
    item = out["items"][0]
    assert item["title"] == "Persistent Agents Checkpointing"
    assert item["doi"] == "10.48550/arxiv.2401.00001"
    assert item["pdf_url"] == "http://arxiv.org/pdf/2401.00001v2"
    assert item["source"] == "arxiv"
    assert out["total_results"] == 1


def test_arxiv_retries_on_5xx_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        sleeps.append,
    )
    client = _client_returning([_response(status_code=503), _response(text=_ATOM)])

    out = search_arxiv(query="agents", client=client)

    assert len(client.request.calls) == 2
    assert out["items"][0]["doi"] == "10.48550/arxiv.2401.00001"
    assert sleeps  # backoff actually ran between attempts


def test_arxiv_raises_structured_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(status_code=500)] * 3)

    with pytest.raises(AcademicProviderError):
        search_arxiv(query="agents", client=client, max_retries=2)

    assert len(client.request.calls) == 3


def test_semantic_scholar_sends_api_key_header_when_provided(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text='{"total": 0, "data": []}')])

    search_semantic_scholar(query="agents", api_key="s2-key", client=client)

    call = client.request.calls[0]
    assert call["url"] == SEMANTIC_SCHOLAR_API_ENDPOINT
    assert call["headers"] == {"x-api-key": "s2-key"}


def test_semantic_scholar_no_key_means_no_header(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text='{"total": 0, "data": []}')])

    search_semantic_scholar(query="agents", client=client)

    assert client.request.calls[0]["headers"] == {}


def test_semantic_scholar_retries_on_429(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    payload = (
        '{"total": 1, "data": [{"paperId": "p1", "title": "T", "abstract": "A",'
        ' "externalIds": {"DOI": "10.1000/x"}}]}'
    )
    client = _client_returning([_response(status_code=429), _response(text=payload)])

    out = search_semantic_scholar(query="agents", client=client)

    assert len(client.request.calls) == 2
    assert out["items"][0]["doi"] == "10.1000/x"
    assert out["items"][0]["source"] == "semantic_scholar"


def test_semantic_scholar_transport_error_retries(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning(
        [httpx.ConnectError("boom"), _response(text='{"total": 0, "data": []}')]
    )

    out = search_semantic_scholar(query="agents", client=client)

    assert out["total_results"] == 0


def test_api_key_resolution_env_wins(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-key")
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.get_cli_setting",
        lambda section, key, default=None: "toml-key",
    )
    assert resolve_semantic_scholar_api_key() == "env-key"


def test_api_key_resolution_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.get_cli_setting",
        lambda section, key, default=None: "toml-key",
    )
    assert resolve_semantic_scholar_api_key() == "toml-key"


def test_papers_to_evidence_maps_to_search_result_shape():
    papers = [
        {
            "title": "T",
            "abstract": "A",
            "doi": "10.1/x",
            "url": "https://doi.org/10.1/x",
            "pdf_url": None,
            "authors": "A. Author",
            "published_date": "2024-01-02",
            "source": "arxiv",
        }
    ]

    evidence = papers_to_evidence(papers)

    assert evidence[0]["title"] == "T"
    assert evidence[0]["url"] == "https://doi.org/10.1/x"
    assert evidence[0]["content"] == "A"
    assert evidence[0]["metadata"]["doi"] == "10.1/x"
    assert evidence[0]["metadata"]["source"] == "academic"


def test_merge_dedups_papers_by_doi_and_keeps_web_results():
    web = [{"title": "Web", "url": "https://web.example/", "content": "w"}]
    papers = [
        {"title": "P1", "abstract": "a", "doi": "10.1/x", "url": "https://doi.org/10.1/x",
         "source": "arxiv"},
        {"title": "P1 duplicate", "abstract": "a", "doi": "10.1/x",
         "url": "https://other.example/10.1/x", "source": "semantic_scholar"},
        {"title": "No DOI", "abstract": "b", "doi": None, "url": "https://nodoi.example/",
         "source": "arxiv"},
    ]

    merged = merge_evidence_pools(web, papers)

    titles = [item["title"] for item in merged]
    assert titles == ["Web", "P1", "No DOI"]  # DOI dup dropped, no-DOI kept
    assert merged[1]["metadata"]["doi"] == "10.1/x"


# --- dual-provider paper search (task-16328) ------------------------------------

_S2_PAYLOAD = (
    '{"total": 1, "data": [{"paperId": "p1", "title": "S2 Paper", "abstract": "s2 abs",'
    ' "externalIds": {"DOI": "10.48550/arxiv.2401.00001"}}]}'
)


def test_search_papers_queries_both_providers_and_dedups_by_doi(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    calls = []

    def fake_arxiv(**kwargs):
        calls.append("arxiv")
        return search_arxiv(query="agents", client=_client_returning([_response(text=_ATOM)]))

    def fake_s2(**kwargs):
        calls.append("s2")
        return search_semantic_scholar(
            query="agents", client=_client_returning([_response(text=_S2_PAYLOAD)])
        )

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.search_arxiv", fake_arxiv
    )
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.search_semantic_scholar", fake_s2
    )

    papers = asyncio.run(search_papers("agents"))

    assert calls == ["arxiv", "s2"]
    # The S2 paper shares the arXiv DOI -> deduped to one paper.
    assert len(papers) == 1
    assert papers[0]["doi"] == "10.48550/arxiv.2401.00001"
    assert papers[0]["source"] == "arxiv"  # first provider wins the DOI


def test_search_papers_degrades_when_one_provider_fails(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )

    def boom_arxiv(**kwargs):
        raise AcademicProviderError("arxiv down")

    def ok_s2(**kwargs):
        return search_semantic_scholar(
            query="agents", client=_client_returning([_response(text=_S2_PAYLOAD)])
        )

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.search_arxiv", boom_arxiv
    )
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.search_semantic_scholar", ok_s2
    )

    papers = asyncio.run(search_papers("agents"))

    assert len(papers) == 1
    assert papers[0]["title"] == "S2 Paper"


def test_search_papers_raises_only_when_all_providers_fail(monkeypatch):
    def boom(**kwargs):
        raise AcademicProviderError("down")

    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.search_arxiv", boom
    )
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers.search_semantic_scholar", boom
    )

    with pytest.raises(AcademicProviderError):
        asyncio.run(search_papers("agents"))


def test_paper_query_normalizes_questions_to_topics():
    from tldw_chatbook.Research_Interop.academic_providers import _paper_query

    assert _paper_query("What is retrieval augmented generation?") == "retrieval augmented generation"
    assert _paper_query("What are the main differences between HTTP/2 and HTTP/3?") == "differences between HTTP/2 and HTTP/3"
    assert _paper_query("How does SQLite FTS5 ranking work?") == "SQLite FTS5 ranking work"
    assert _paper_query("already a topic query") == "already a topic query"


def test_arxiv_phrases_multi_word_queries(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=_ATOM)])

    search_arxiv(query="retrieval augmented generation", client=client)

    sent = client.request.calls[0]["params"]
    assert sent["search_query"] == 'all:"retrieval augmented generation"'


def test_arxiv_single_word_queries_stay_unquoted(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=_ATOM)])

    search_arxiv(query="transformers", client=client)

    assert client.request.calls[0]["params"]["search_query"] == "all:transformers"


def test_search_papers_runs_providers_concurrently(monkeypatch):
    import threading
    from tldw_chatbook.Research_Interop import academic_providers as ap

    arxiv_started = threading.Event()
    s2_release = threading.Event()

    def blocking_arxiv(**kwargs):
        arxiv_started.set()
        assert s2_release.wait(timeout=5), "serial execution would deadlock here"
        return {"items": []}

    def releasing_s2(**kwargs):
        s2_release.set()  # proves S2 ran WHILE arxiv was in flight
        return {"items": []}

    monkeypatch.setattr(ap, "search_arxiv", blocking_arxiv)
    monkeypatch.setattr(ap, "search_semantic_scholar", releasing_s2)

    papers = ap.search_papers_sync_for_test() if hasattr(ap, "search_papers_sync_for_test") else None
    import asyncio
    papers = asyncio.run(ap.search_papers("query"))
    assert papers == []
