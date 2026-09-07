---
id: TASK-16792
title: Port the server's research source catalog with categories
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 14:31'
updated_date: '2026-08-16 14:35'
labels:
  - research
  - web-tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_server's Research discovery module carries a categorized source catalog - open_research_graph (OpenAlex, Semantic Scholar, Crossref), preprints (arXiv, BioRxiv, MedRxiv), biomedical (PubMed), repositories (Zenodo, Figshare, OSF) - with category-based selection, per-source metadata, and a selection cap. The chatbook's academic lane has five flat providers and no catalog, no categories, and is missing OpenAlex, Crossref, Zenodo, Figshare, and OSF entirely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source catalog module mirrors the server's entry fields (source_id, display_name, category, subcategory, content_types, access_level, priority, trust notes) for all ten sources,Category-based selection expands to member providers with dedupe and an unknown-token error path,OpenAlex, Crossref, Zenodo, Figshare, and OSF providers are implemented over the shared httpx retry ladder, keyless, normalizing to the DOI-dedup paper shape (OpenAlex abstracts reconstructed from the inverted index),search_papers accepts source ids AND category names in its providers selection,The window providers input and the baseline script accept categories too,Tests cover the catalog, category expansion, each new provider with mocked HTTP, and category-driven search_papers
<!-- AC:END -->
