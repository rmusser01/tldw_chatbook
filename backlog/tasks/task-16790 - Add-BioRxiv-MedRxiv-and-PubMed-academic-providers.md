---
id: TASK-16790
title: 'Add BioRxiv, MedRxiv, and PubMed academic providers'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 13:21'
updated_date: '2026-08-16 13:24'
labels:
  - research
  - web-tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The academic lane covers arXiv and Semantic Scholar only; tldw_server's pipeline also searches BioRxiv/MedRxiv (shared details API) and PubMed (ESearch+ESummary), so biomedical questions have no local coverage. Port both providers and make the provider set configurable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] search_biorxiv queries the details API with a default 30-day window, client-side query filtering, and MedRxiv via the server parameter,search_pubmed does the ESearch then ESummary two-step and normalizes items (DOI from articleids, PMC PDF links when present),Both providers use the shared httpx retry ladder and constants-driven endpoints and normalize to the established paper shape for DOI dedup,search_papers fans out across a configurable provider set (config [SearchSettings] research_academic_providers, default arxiv+semantic_scholar) with the providers parameter filtering lanes and per-provider degradation preserved,Tests with mocked HTTP cover both providers, filtering, the medrxiv switch, and provider-set configuration
<!-- AC:END -->

## Implementation Notes

- `search_biorxiv` (port of the server's Third_Party/BioRxiv.py, simplified to the lane's needs): details API over a default 30-day window (server's default), client-side case-insensitive query filtering over title/abstract, MedRxiv via the `server` parameter (shared API, matching the server's MedRxiv aliasing), canonical content/PDF URLs from DOI+version. `search_pubmed` (port of Third_Party/PubMed.py): ESearch→ESummary two-step with relevance sort, DOI/PMCID extraction from articleids, PMC PDF links when present. Both ride the shared `_request_with_retries` ladder (timeouts, capped backoff, 429/5xx retries, 4xx fail-fast), constants-driven endpoints, injectable clients, and normalize to the established paper shape for DOI dedup.
- `search_papers` became a registry fan-out: lanes for arxiv/semantic_scholar/biorxiv/medrxiv/pubmed, all selected providers concurrent via gather, per-provider degradation preserved, unknown provider names warn and drop. `providers=` filters the set per call; the default set comes from `_default_academic_providers()` (`[SearchSettings] research_academic_providers`, comma list, default arxiv+semantic_scholar).
- The existing concurrency/degradation tests still pass unchanged; 6 new provider tests (mocked httpx) cover both providers, medrxiv switch, provider-set filtering, and config defaults.
