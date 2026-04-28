from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    PaperSearchDetailRequest,
    PaperSearchIngestRequest,
    PaperSearchOperationResponse,
    PaperSearchRequest,
    WebSearchRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_research_search_client_routes_websearch_and_paper_search(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "web_search_results_dict": {"results": [{"title": "A", "url": "https://example.com"}]},
                "sub_query_dict": {"queries": ["mcp governance"]},
            },
            {
                "query_echo": {"query": "agents", "author": None, "year": None},
                "items": [
                    {
                        "id": "2401.00001",
                        "title": "Agent Governance",
                        "authors": "A. Researcher",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "query_echo": {"query": "agents"},
                "items": [
                    {
                        "paperId": "abc",
                        "title": "Agent Governance",
                        "authors": [{"name": "A. Researcher"}],
                    }
                ],
                "total_results": 1,
                "offset": 0,
                "limit": 10,
                "next_offset": None,
                "page": 1,
                "total_pages": 1,
            },
            {
                "query_echo": {"q": "genomics", "server": "biorxiv"},
                "items": [
                    {
                        "doi": "10.1101/2026.01.01.000001",
                        "title": "Preprint Governance",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "doi": "10.1101/2026.01.01.000001",
                "title": "Preprint Governance",
            },
            {
                "query_echo": {"q": "governance"},
                "items": [
                    {
                        "pmid": "12345678",
                        "title": "Clinical Governance",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "pmid": "12345678",
                "title": "Clinical Governance",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    web = await client.research_websearch(WebSearchRequest(query="mcp governance", engine="searxng"))
    arxiv = await client.search_arxiv_papers(query="agents", page=1, results_per_page=10)
    semantic = await client.search_semantic_scholar_papers(query="agents", page=1, results_per_page=10)
    biorxiv = await client.search_biorxiv_papers(q="genomics", server="biorxiv", page=1, results_per_page=10)
    biorxiv_detail = await client.get_biorxiv_paper_by_doi(
        doi="10.1101/2026.01.01.000001",
        server="biorxiv",
    )
    pubmed = await client.search_pubmed_papers(q="governance", free_full_text=True, page=1, results_per_page=10)
    pubmed_detail = await client.get_pubmed_paper_by_id(pmid="12345678")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/research/websearch")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "query": "mcp governance",
        "engine": "searx",
        "result_count": 10,
        "content_country": "US",
        "search_lang": "en",
        "output_lang": "en",
        "searx_json_mode": False,
        "include_archived": False,
        "subquery_generation": False,
        "user_review": False,
        "aggregate": False,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/paper-search/arxiv")
    assert mocked.await_args_list[1].kwargs["params"] == {"query": "agents", "page": 1, "results_per_page": 10}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/paper-search/semantic-scholar")
    assert mocked.await_args_list[2].kwargs["params"] == {"query": "agents", "page": 1, "results_per_page": 10}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/paper-search/biorxiv")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "q": "genomics",
        "server": "biorxiv",
        "page": 1,
        "results_per_page": 10,
    }
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/paper-search/biorxiv/by-doi")
    assert mocked.await_args_list[4].kwargs["params"] == {
        "doi": "10.1101/2026.01.01.000001",
        "server": "biorxiv",
    }
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/paper-search/pubmed")
    assert mocked.await_args_list[5].kwargs["params"] == {
        "q": "governance",
        "free_full_text": True,
        "page": 1,
        "results_per_page": 10,
    }
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/paper-search/pubmed/by-id")
    assert mocked.await_args_list[6].kwargs["params"] == {"pmid": "12345678"}

    assert web.web_search_results_dict["results"][0]["title"] == "A"
    assert arxiv.items[0].id == "2401.00001"
    assert semantic.items[0].paperId == "abc"
    assert biorxiv.items[0].doi == "10.1101/2026.01.01.000001"
    assert biorxiv_detail.title == "Preprint Governance"
    assert pubmed.items[0].pmid == "12345678"
    assert pubmed_detail.title == "Clinical Governance"


def test_websearch_request_normalizes_aliases_and_rejects_unknown_engines():
    request = WebSearchRequest(query="test", engine="searxng")
    assert request.engine == "searx"

    with pytest.raises(ValueError):
        WebSearchRequest(query="test", engine="unknown")


def test_paper_search_allowlists_cover_server_provider_contract():
    server_get_endpoints = {
        "acm",
        "acm/by-doi",
        "arxiv",
        "arxiv/by-id",
        "biorxiv",
        "biorxiv-pubs",
        "biorxiv-pubs/by-doi",
        "biorxiv/by-doi",
        "biorxiv/funder",
        "biorxiv/pub",
        "biorxiv/publisher",
        "biorxiv/raw/details",
        "biorxiv/raw/funder",
        "biorxiv/raw/pub",
        "biorxiv/raw/pubs",
        "biorxiv/raw/reports/summary",
        "biorxiv/raw/reports/usage",
        "biorxiv/reports/summary",
        "biorxiv/reports/usage",
        "chemrxiv/categories",
        "chemrxiv/items",
        "chemrxiv/items/by-doi",
        "chemrxiv/items/by-id",
        "chemrxiv/licenses",
        "chemrxiv/oai",
        "chemrxiv/version",
        "earthrxiv",
        "earthrxiv/by-doi",
        "earthrxiv/by-id",
        "figshare",
        "figshare/by-doi",
        "figshare/by-id",
        "figshare/oai",
        "hal",
        "hal/by-id",
        "hal/raw",
        "iacr/conf",
        "iacr/conf/raw",
        "ieee",
        "ieee/by-doi",
        "ieee/by-id",
        "medrxiv",
        "medrxiv/by-doi",
        "medrxiv/raw/details",
        "medrxiv/raw/pub",
        "medrxiv/raw/pubs",
        "osf",
        "osf/by-doi",
        "osf/by-id",
        "osf/raw",
        "osf/raw/by-id",
        "pmc-oa/fetch-pdf",
        "pmc-oa/identify",
        "pmc-oa/query",
        "pmc-oai/get-record",
        "pmc-oai/identify",
        "pmc-oai/list-identifiers",
        "pmc-oai/list-records",
        "pmc-oai/list-sets",
        "pubmed",
        "pubmed/by-id",
        "repec/by-handle",
        "repec/citations",
        "scopus",
        "scopus/by-doi",
        "semantic-scholar",
        "semantic-scholar/by-id",
        "springer",
        "springer/by-doi",
        "vixra/by-id",
        "vixra/search",
        "wiley",
        "wiley/by-doi",
        "zenodo",
        "zenodo/by-doi",
        "zenodo/by-id",
        "zenodo/oai",
    }
    server_post_endpoints = {
        "arxiv/ingest",
        "earthrxiv/ingest",
        "figshare/ingest",
        "figshare/ingest-by-doi",
        "hal/ingest",
        "ingest/batch",
        "ingest/by-doi",
        "osf/ingest",
        "pmc-oa/ingest-pdf",
        "pubmed/ingest",
        "semantic-scholar/ingest",
        "vixra/ingest",
        "zenodo/ingest",
    }

    for endpoint in server_get_endpoints:
        assert PaperSearchRequest(endpoint=endpoint).endpoint == endpoint
        assert PaperSearchDetailRequest(endpoint=endpoint).endpoint == endpoint
    for endpoint in server_post_endpoints:
        assert PaperSearchIngestRequest(endpoint=endpoint).endpoint == endpoint


@pytest.mark.asyncio
async def test_research_search_client_routes_generic_paper_search_gateway(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"items": [{"id": "cat-1", "name": "Biology"}], "total_results": 1},
            {"id": "record-1", "title": "OAI Record"},
            {"media_id": 12, "status": "queued"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.paper_search(PaperSearchRequest(endpoint="chemrxiv/categories", params={"q": "bio"}))
    detail = await client.paper_search_detail(
        PaperSearchDetailRequest(endpoint="pmc-oai/get-record", params={"identifier": "oai:pmc:1"})
    )
    ingested = await client.paper_search_ingest(
        PaperSearchIngestRequest(endpoint="pmc-oa/ingest-pdf", payload={"pmcid": "PMC1"})
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/paper-search/chemrxiv/categories")
    assert mocked.await_args_list[0].kwargs["params"] == {"q": "bio"}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/paper-search/pmc-oai/get-record")
    assert mocked.await_args_list[1].kwargs["params"] == {"identifier": "oai:pmc:1"}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/paper-search/pmc-oa/ingest-pdf")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"pmcid": "PMC1"}

    assert listed.items[0]["name"] == "Biology"
    assert detail.title == "OAI Record"
    assert isinstance(ingested, PaperSearchOperationResponse)
    assert ingested.data["status"] == "queued"


@pytest.mark.asyncio
async def test_research_search_client_routes_paper_detail_and_medrxiv_aliases(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": "2401.00001",
                "title": "Agent Governance",
                "authors": "A. Researcher",
            },
            {
                "paperId": "abc",
                "title": "Agent Governance",
                "authors": [{"name": "A. Researcher"}],
            },
            {
                "query_echo": {"q": "genomics", "server": "medrxiv"},
                "items": [
                    {
                        "doi": "10.1101/2026.01.01.000001",
                        "title": "Clinical Preprint Governance",
                    }
                ],
                "total_results": 1,
                "page": 1,
                "results_per_page": 10,
                "total_pages": 1,
            },
            {
                "doi": "10.1101/2026.01.01.000001",
                "title": "Clinical Preprint Governance",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    arxiv_detail = await client.get_arxiv_paper_by_id(id="2401.00001")
    semantic_detail = await client.get_semantic_scholar_paper_by_id(paper_id="abc")
    medrxiv = await client.search_medrxiv_papers(q="genomics", page=1, results_per_page=10)
    medrxiv_detail = await client.get_medrxiv_paper_by_doi(doi="10.1101/2026.01.01.000001")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/paper-search/arxiv/by-id")
    assert mocked.await_args_list[0].kwargs["params"] == {"id": "2401.00001"}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/paper-search/semantic-scholar/by-id")
    assert mocked.await_args_list[1].kwargs["params"] == {"paper_id": "abc"}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/paper-search/medrxiv")
    assert mocked.await_args_list[2].kwargs["params"] == {
        "q": "genomics",
        "page": 1,
        "results_per_page": 10,
    }
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/paper-search/medrxiv/by-doi")
    assert mocked.await_args_list[3].kwargs["params"] == {"doi": "10.1101/2026.01.01.000001"}

    assert arxiv_detail.id == "2401.00001"
    assert semantic_detail.paperId == "abc"
    assert medrxiv.items[0].doi == "10.1101/2026.01.01.000001"
    assert medrxiv_detail.title == "Clinical Preprint Governance"
