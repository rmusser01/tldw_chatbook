"""Research source catalog with categories (task-16792; port of the
server's Research/discovery catalog)."""

import pytest

from tldw_chatbook.Research_Interop.research_source_catalog import (
    CATEGORIES,
    SOURCES_BY_CATEGORY,
    catalog_entries,
    expand_source_selection,
)


def test_catalog_covers_all_ten_sources_with_server_fields():
    entries = {e.source_id: e for e in catalog_entries()}

    assert set(entries) == {
        "openalex", "semantic_scholar", "crossref",
        "arxiv", "biorxiv", "medrxiv",
        "pubmed", "zenodo", "figshare", "osf",
    }
    assert entries["openalex"].category == "open_research_graph"
    assert entries["arxiv"].category == "preprints"
    assert entries["pubmed"].category == "biomedical"
    assert entries["zenodo"].category == "repositories"
    # Server-parity metadata on a representative entry.
    pubmed = entries["pubmed"]
    assert pubmed.display_name == "PubMed"
    assert "abstracts" in pubmed.content_types
    assert pubmed.access_level == "open_metadata"
    assert pubmed.trust_notes
    assert pubmed.priority > 0


def test_categories_match_server_groupings():
    assert SOURCES_BY_CATEGORY["open_research_graph"] == (
        "openalex", "semantic_scholar", "crossref"
    )
    assert SOURCES_BY_CATEGORY["preprints"] == ("arxiv", "biorxiv", "medrxiv")
    assert SOURCES_BY_CATEGORY["biomedical"] == ("pubmed",)
    assert SOURCES_BY_CATEGORY["repositories"] == ("zenodo", "figshare", "osf")
    assert set(CATEGORIES) == {
        "open_research_graph", "preprints", "biomedical", "repositories",
    }


def test_expand_mixes_ids_and_categories_with_dedupe():
    assert expand_source_selection(["biomedical", "arxiv"]) == [
        "pubmed", "arxiv",
    ]
    assert expand_source_selection(["preprints", "arxiv"]) == [
        "arxiv", "biorxiv", "medrxiv",
    ]
    assert expand_source_selection([]) == []


def test_expand_rejects_unknown_tokens():
    with pytest.raises(ValueError, match="unknown research source or category"):
        expand_source_selection(["biomedical", "not_a_thing"])


def test_paper_provider_constants_in_sync_with_catalog():
    # Qodo (PR 1724): the service-level provider listings must not drift
    # from the catalog the lane actually runs.
    from tldw_chatbook.Research_Interop.local_research_search_service import (
        LOCAL_SUPPORTED_PAPER_PROVIDERS,
    )

    catalog_ids = {e.source_id for e in catalog_entries()}
    assert set(LOCAL_SUPPORTED_PAPER_PROVIDERS) <= catalog_ids
    # Every catalog source is runnable by the lane (full parity, not a subset).
    assert set(LOCAL_SUPPORTED_PAPER_PROVIDERS) == catalog_ids


# --- source-kind classification (task-17066) ---------------------------------------

def test_source_kind_for_provider_maps_categories():
    from tldw_chatbook.Research_Interop.research_source_catalog import (
        source_kind_for_provider,
    )

    assert source_kind_for_provider("zenodo") == "repository"
    assert source_kind_for_provider("figshare") == "repository"
    assert source_kind_for_provider("osf") == "repository"
    assert source_kind_for_provider("openalex") == "metadata"
    assert source_kind_for_provider("crossref") == "metadata"
    # Papers/preprints stay strict; graph members (which include
    # semantic_scholar/openalex by catalog category) take the metadata note.
    assert source_kind_for_provider("arxiv") == "paper"
    assert source_kind_for_provider("pubmed") == "paper"
    assert source_kind_for_provider("biorxiv") == "paper"
    assert source_kind_for_provider("semantic_scholar") == "metadata"


def test_source_kind_unknown_and_missing_default_to_paper():
    from tldw_chatbook.Research_Interop.research_source_catalog import (
        source_kind_for_provider,
    )

    assert source_kind_for_provider("not_a_provider") == "paper"
    assert source_kind_for_provider("") == "paper"
    assert source_kind_for_provider(None) == "paper"


def test_source_kind_classifier_is_cached():
    from tldw_chatbook.Research_Interop import research_source_catalog as rsc

    first = rsc._entries_by_id()
    second = rsc._entries_by_id()
    assert first is second  # memoized, not rebuilt per call
