"""Research source catalog with categories (task-16792).

Port of tldw_server dev's ``Research/discovery/catalog.py``: the categorized
source list behind discovery-lane selection. The chatbook's academic lane
consumes it through ``expand_source_selection`` — callers may name individual
sources ("pubmed") or whole categories ("biomedical", "repositories") — and
the window/baseline/Console surfaces accept the same tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "CATEGORIES",
    "SOURCES_BY_CATEGORY",
    "ResearchSourceCatalogEntry",
    "catalog_entries",
    "expand_source_selection",
]


@dataclass(frozen=True)
class ResearchSourceCatalogEntry:
    """One selectable research source (server catalog-entry parity).

    Attributes:
        source_id: Stable selection token (also the provider lane name).
        display_name: Human-readable name.
        category: Coarse grouping used for category-based selection.
        subcategory: Optional finer grouping.
        content_types: What the source returns (papers, datasets, ...).
        access_level: Server-parity access descriptor
            (``open_metadata`` / ``open_full_text`` / ``open_repository``).
        priority: Lower runs earlier in fan-outs (server ordering parity).
        provider_adapter: The executing provider lane (== source_id here).
        trust_notes: One-line provenance/trust description.
    """

    source_id: str
    display_name: str
    category: str
    content_types: Tuple[str, ...]
    access_level: str
    priority: int
    trust_notes: str
    subcategory: str | None = None
    provider_adapter: str | None = None


def _entry(
    *,
    source_id: str,
    display_name: str,
    category: str,
    content_types: Tuple[str, ...],
    access_level: str,
    priority: int,
    trust_notes: str,
    subcategory: str | None = None,
) -> ResearchSourceCatalogEntry:
    return ResearchSourceCatalogEntry(
        source_id=source_id,
        display_name=display_name,
        category=category,
        content_types=content_types,
        access_level=access_level,
        priority=priority,
        trust_notes=trust_notes,
        subcategory=subcategory,
        provider_adapter=source_id,
    )


def catalog_entries() -> Tuple[ResearchSourceCatalogEntry, ...]:
    """The default catalog (server ``_default_entries`` parity, plus the
    BioRxiv/MedRxiv preprint pair the server splits by server param).

    Returns:
        All catalog entries ordered by priority.
    """
    return (
        _entry(
            source_id="openalex",
            display_name="OpenAlex",
            category="open_research_graph",
            content_types=("works", "authors", "institutions", "venues"),
            access_level="open_metadata",
            priority=10,
            trust_notes="Open scholarly graph with broad metadata coverage.",
        ),
        _entry(
            source_id="semantic_scholar",
            display_name="Semantic Scholar",
            category="open_research_graph",
            content_types=("papers", "citations", "recommendations"),
            access_level="open_metadata",
            priority=20,
            trust_notes="Open metadata and citation graph with API rate limits.",
        ),
        _entry(
            source_id="crossref",
            display_name="Crossref",
            category="open_research_graph",
            content_types=("works", "publishers", "funders"),
            access_level="open_metadata",
            priority=30,
            trust_notes="Publisher DOI metadata registry.",
        ),
        _entry(
            source_id="arxiv",
            display_name="arXiv",
            category="preprints",
            subcategory="general_preprints",
            content_types=("preprints", "papers"),
            access_level="open_full_text",
            priority=40,
            trust_notes="Open preprint repository with direct full-text access.",
        ),
        _entry(
            source_id="biorxiv",
            display_name="BioRxiv",
            category="preprints",
            subcategory="biomedical_preprints",
            content_types=("preprints", "papers"),
            access_level="open_full_text",
            priority=45,
            trust_notes="Biology preprint server; shared API with MedRxiv.",
        ),
        _entry(
            source_id="medrxiv",
            display_name="MedRxiv",
            category="preprints",
            subcategory="biomedical_preprints",
            content_types=("preprints", "papers"),
            access_level="open_full_text",
            priority=46,
            trust_notes="Health-sciences preprint server; shared API with BioRxiv.",
        ),
        _entry(
            source_id="pubmed",
            display_name="PubMed",
            category="biomedical",
            content_types=("papers", "abstracts", "biomedical_metadata"),
            access_level="open_metadata",
            priority=50,
            trust_notes="Biomedical literature metadata from NCBI.",
        ),
        _entry(
            source_id="zenodo",
            display_name="Zenodo",
            category="repositories",
            content_types=("datasets", "software", "papers"),
            access_level="open_repository",
            priority=60,
            trust_notes="Open research repository operated by CERN.",
        ),
        _entry(
            source_id="figshare",
            display_name="Figshare",
            category="repositories",
            content_types=("datasets", "figures", "papers"),
            access_level="open_repository",
            priority=70,
            trust_notes="Research repository for datasets, figures, and papers.",
        ),
        _entry(
            source_id="osf",
            display_name="OSF",
            category="repositories",
            content_types=("projects", "registrations", "preprints"),
            access_level="open_repository",
            priority=80,
            trust_notes="Open Science Framework project and registration metadata.",
        ),
    )


def sources_by_category() -> dict[str, Tuple[str, ...]]:
    """Category -> member source ids, in priority order.

    Returns:
        The grouping used for category-based selection.
    """
    grouped: dict[str, list[str]] = {}
    for entry in catalog_entries():
        grouped.setdefault(entry.category, []).append(entry.source_id)
    return {category: tuple(ids) for category, ids in grouped.items()}


SOURCES_BY_CATEGORY = sources_by_category()
CATEGORIES = tuple(SOURCES_BY_CATEGORY)


def expand_source_selection(tokens: list[str]) -> list[str]:
    """Expand a mix of source ids and category names to provider ids
    (deduped, first-seen order — category order is catalog priority).

    Args:
        tokens: Source ids ("pubmed") and/or category names ("biomedical").

    Returns:
        The expanded, deduplicated provider-id list.

    Raises:
        ValueError: Naming any token that is neither a known source id nor
            a known category.
    """
    ids_by_category = sources_by_category()
    known_sources = {e.source_id for e in catalog_entries()}
    unknown = [
        token for token in tokens
        if token not in known_sources and token not in ids_by_category
    ]
    if unknown:
        raise ValueError(
            f"unknown research source or category: {', '.join(unknown)}"
        )
    expanded: list[str] = []
    for token in tokens:
        candidates = (
            ids_by_category[token] if token in ids_by_category else (token,)
        )
        for source_id in candidates:
            if source_id not in expanded:
                expanded.append(source_id)
    return expanded
