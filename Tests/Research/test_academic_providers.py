"""Academic search providers (task-16326).

httpx-based arXiv + Semantic Scholar runners with configurable timeouts,
retry-with-backoff, config-driven API keys, constants-driven endpoints, DOI
normalization, and the DOI-level dedup merge that feeds papers into the same
evidence pool as web results. HTTP is faked at the client seam; parsing,
retry, and merge logic run real.
"""

import asyncio
import json
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
    search_biorxiv,
    search_crossref,
    search_figshare,
    search_openalex,
    search_osf,
    search_zenodo,
    search_pubmed,
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


# --- BioRxiv / MedRxiv / PubMed providers (task-16790) -----------------------------

_BIORXIV_PAGE = {
    "messages": [{"count": 2, "total": 2}],
    "collection": [
        {"doi": "10.1101/2026.01.01.000001", "title": "Protein folding advances",
         "authors": "A. Author; B. Author", "category": "bioinformatics",
         "date": "2026-01-02", "abstract": "We study folding dynamics.",
         "server": "biorxiv", "version": 1},
        {"doi": "10.1101/2026.01.01.000002", "title": "Unrelated climate paper",
         "authors": "C. Author", "date": "2026-01-03", "abstract": "Ice cores.",
         "server": "biorxiv", "version": 1},
    ],
}


def test_search_biorxiv_filters_and_normalizes(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_BIORXIV_PAGE))])

    out = search_biorxiv(query="protein folding", client=client)

    call = client.request.calls[0]
    assert "api.biorxiv.org/details/biorxiv/" in call["url"]
    item = out["items"][0]
    assert item["title"] == "Protein folding advances"
    assert item["doi"] == "10.1101/2026.01.01.000001"
    assert item["source"] == "biorxiv"
    assert item["url"] == "https://www.biorxiv.org/content/10.1101/2026.01.01.000001v1"
    assert item["pdf_url"].endswith(".full.pdf")
    # Client-side query filter drops the unrelated paper.
    assert len(out["items"]) == 1


def test_search_biorxiv_medrxiv_switch(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_BIORXIV_PAGE))])

    search_biorxiv(query="folding", server="medrxiv", client=client)

    assert "/details/medrxiv/" in client.request.calls[0]["url"]


_ESEARCH = {"esearchresult": {"idlist": ["39123456", "39123457"], "count": "2"}}
_ESUMMARY = {
    "result": {
        "uids": ["39123456"],
        "39123456": {
            "title": "CRISPR base editing efficiency",
            "fulljournalname": "Nature Biotech",
            "pubdate": "2026 Jan",
            "authors": [{"name": "D. Researcher"}],
            "articleids": [
                {"idtype": "doi", "value": "10.1038/nbt.2026.01"},
                {"idtype": "pmc", "value": "PMC9999888"},
            ],
        },
    }
}


def test_search_pubmed_two_step_esearch_then_esummary(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning(
        [_response(text=json.dumps(_ESEARCH)), _response(text=json.dumps(_ESUMMARY))]
    )

    out = search_pubmed(query="CRISPR base editing", client=client)

    first, second = client.request.calls
    assert "esearch.fcgi" in first["url"]
    assert "CRISPR" in first["params"]["term"]
    assert "esummary.fcgi" in second["url"]
    assert second["params"]["id"] == "39123456,39123457"
    item = out["items"][0]
    assert item["doi"] == "10.1038/nbt.2026.01"
    assert item["url"] == "https://pubmed.ncbi.nlm.nih.gov/39123456/"
    assert item["pdf_url"] == "https://pmc.ncbi.nlm.nih.gov/9999888/pdf"
    assert item["source"] == "pubmed"
    assert item["authors"] == "D. Researcher"


def test_search_pubmed_empty_idlist(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning(
        [_response(text=json.dumps({"esearchresult": {"idlist": [], "count": "0"}}))]
    )

    out = search_pubmed(query="nothing", client=client)

    assert out["items"] == [] and out["total_results"] == 0


def test_search_papers_provider_set_filtering(monkeypatch):
    from tldw_chatbook.Research_Interop import academic_providers as ap

    calls = []

    def fake_arxiv(**kw):
        calls.append("arxiv")
        return {"items": [{"title": "A", "abstract": "a", "doi": "10.1/a",
                           "url": "https://doi.org/10.1/a", "source": "arxiv"}]}

    def fake_pubmed(**kw):
        calls.append("pubmed")
        return {"items": [{"title": "P", "abstract": "p", "doi": "10.2/p",
                           "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
                           "source": "pubmed"}]}

    def unused_s2(**kw):
        calls.append("semantic_scholar")
        return {"items": []}

    monkeypatch.setattr(ap, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(ap, "search_semantic_scholar", unused_s2)
    monkeypatch.setattr(ap, "search_pubmed", fake_pubmed)

    papers = asyncio.run(ap.search_papers("topic", providers=["arxiv", "pubmed"]))

    assert sorted(calls) == ["arxiv", "pubmed"]
    assert {p["source"] for p in papers} == {"arxiv", "pubmed"}


def test_default_academic_providers_from_config(monkeypatch):
    from tldw_chatbook.Research_Interop import academic_providers as ap

    monkeypatch.setattr(
        ap, "get_cli_setting",
        lambda section, key, default=None: "arxiv, pubmed, biorxiv",
    )
    assert ap._default_academic_providers() == ["arxiv", "pubmed", "biorxiv"]

    monkeypatch.setattr(
        ap, "get_cli_setting", lambda section, key, default=None: default
    )
    assert ap._default_academic_providers() == ["arxiv", "semantic_scholar"]


def test_search_papers_dedupes_and_rejects_unknown_defaults(monkeypatch):
    from tldw_chatbook.Research_Interop import academic_providers as ap

    calls = []

    def fake_arxiv(**kw):
        calls.append("arxiv")
        return {"items": []}

    monkeypatch.setattr(ap, "search_arxiv", fake_arxiv)
    # Duplicates collapse; unknown tokens now RAISE (a config typo must not
    # silently narrow the provider set).
    monkeypatch.setattr(
        ap, "_default_academic_providers",
        lambda: ["arxiv", "arxiv"],
    )
    papers = asyncio.run(ap.search_papers("q"))
    assert calls == ["arxiv"]  # deduped: one call despite the duplicate
    assert papers == []

    monkeypatch.setattr(
        ap, "_default_academic_providers",
        lambda: ["arxiv", "not_a_provider"],
    )
    with pytest.raises(ValueError, match="unknown research source or category"):
        asyncio.run(ap.search_papers("q"))


# --- catalog-lane providers: OpenAlex/Crossref/Zenodo/Figshare/OSF (task-16792) ----

_OPENALEX_PAGE = {
    "results": [
        {
            "id": "W1", "doi": "https://doi.org/10.1234/oa.1",
            "title": "Graph of scholarship",
            "abstract_inverted_index": {
                "Scholarship": [0], "is": [1], "vast": [2],
            },
            "authorships": [{"author": {"display_name": "E. Scholar"}}],
            "publication_year": 2026,
            "primary_location": {
                "landing_page_url": "https://doi.org/10.1234/oa.1",
                "pdf_url": "https://example.org/oa1.pdf",
            },
        }
    ],
    "meta": {"count": 1},
}


def test_search_openalex_reconstructs_abstract_from_inverted_index(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_OPENALEX_PAGE))])

    out = search_openalex(query="scholarship", client=client)

    assert "api.openalex.org/works" in client.request.calls[0]["url"]
    item = out["items"][0]
    assert item["abstract"] == "Scholarship is vast"
    assert item["doi"] == "10.1234/oa.1"  # https://doi.org/ prefix stripped
    assert item["url"] == "https://doi.org/10.1234/oa.1"
    assert item["pdf_url"] == "https://example.org/oa1.pdf"
    assert item["authors"] == "E. Scholar"
    assert item["source"] == "openalex"


_CROSSREF_PAGE = {
    "message": {
        "total-results": 1,
        "items": [
            {
                "DOI": "10.5555/cr.1",
                "title": ["Registry metadata study"],
                "author": [{"given": "F.", "family": "Registrar"}],
                "issued": {"date-parts": [[2025]]},
                "abstract": "<jats:p>DOI metadata at scale.</jats:p>",
            }
        ],
    }
}


def test_search_crossref_strips_jats_from_abstract(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_CROSSREF_PAGE))])

    out = search_crossref(query="registry", client=client)

    assert "api.crossref.org/works" in client.request.calls[0]["url"]
    item = out["items"][0]
    assert item["abstract"] == "DOI metadata at scale."
    assert item["authors"] == "F. Registrar"
    assert item["source"] == "crossref"


_ZENODO_PAGE = {
    "hits": {
        "total": 1,
        "hits": [
            {
                "id": 998877,
                "doi": "10.5281/zenodo.998877",
                "metadata": {
                    "title": "Dataset of things",
                    "description": "<p>A big dataset.</p>",
                    "creators": [{"name": "G. Curator"}],
                    "publication_date": "2026-02-01",
                },
            }
        ],
    }
}


def test_search_zenodo_normalizes_repository_records(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_ZENODO_PAGE))])

    out = search_zenodo(query="dataset", client=client)

    assert "zenodo.org/api/records" in client.request.calls[0]["url"]
    item = out["items"][0]
    assert item["title"] == "Dataset of things"
    assert item["abstract"] == "A big dataset."  # HTML stripped
    assert item["url"] == "https://zenodo.org/records/998877"
    assert item["source"] == "zenodo"


_FIGSHARE_BODY = [
    {
        "id": 776655,
        "title": "Figures for the paper",
        "description": "<p>Twelve figures.</p>",
        "doi": "10.6084/m9.figshare.776655",
        "url_publication": "https://figshare.com/articles/figure/776655",
        "authors": [{"full_name": "H. Artist"}],
    }
]


def test_search_figshare_posts_search_body(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_FIGSHARE_BODY))])

    out = search_figshare(query="figures", client=client)

    call = client.request.calls[0]
    assert call["method"] == "POST"
    assert "api.figshare.com/v2/articles/search" in call["url"]
    item = out["items"][0]
    assert item["abstract"] == "Twelve figures."
    assert item["url"] == "https://figshare.com/articles/figure/776655"
    assert item["source"] == "figshare"


_OSF_PAGE = {
    "data": [
        {
            "id": "abc12",
            "attributes": {
                "title": "Registered analysis plan",
                "description": "A preprint with a registration.",
                "date_created": "2026-03-03T00:00:00.000000Z",
            },
            "links": {"html": "https://osf.io/preprints/abc12"},
        }
    ]
}


def test_search_osf_normalizes_preprint_records(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Research_Interop.academic_providers._sleep_backoff",
        lambda attempt: None,
    )
    client = _client_returning([_response(text=json.dumps(_OSF_PAGE))])

    out = search_osf(query="registration", client=client)

    assert "api.osf.io/v2/preprints" in client.request.calls[0]["url"]
    item = out["items"][0]
    assert item["title"] == "Registered analysis plan"
    assert item["url"] == "https://osf.io/preprints/abc12"
    assert item["published_date"] == "2026-03-03"
    assert item["source"] == "osf"


def test_search_papers_accepts_categories(monkeypatch):
    from tldw_chatbook.Research_Interop import academic_providers as ap

    calls = []

    def fake_pubmed(**kw):
        calls.append("pubmed")
        return {"items": []}

    def unused(**kw):
        calls.append("other")
        return {"items": []}

    monkeypatch.setattr(ap, "search_biorxiv", unused)  # serves biorxiv + medrxiv lanes
    for name in ("arxiv", "semantic_scholar",
                 "openalex", "crossref", "zenodo", "figshare", "osf"):
        monkeypatch.setattr(ap, f"search_{name}", unused)
    monkeypatch.setattr(ap, "search_pubmed", fake_pubmed)

    papers = asyncio.run(ap.search_papers("topic", providers=["biomedical"]))

    assert calls == ["pubmed"]
    assert papers == []
