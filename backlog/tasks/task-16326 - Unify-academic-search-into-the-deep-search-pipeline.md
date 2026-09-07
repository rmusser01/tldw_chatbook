---
id: TASK-16326
title: Unify academic search into the deep-search pipeline
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:16'
updated_date: '2026-08-15 16:00'
labels:
  - research
  - web-tools
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
LocalResearchSearchService implements arXiv and Semantic Scholar inline with urllib, hardcoded 30s timeouts, no API-key support, no retry or backoff, and no connection to the deep-search evidence pool. tldw_server treats academic as just another lane feeding one evidence graph with DOI dedup. Migrate the runners to httpx with retries and let academic results feed the same evidence pool as web results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 arXiv and Semantic Scholar runners use httpx with configurable timeouts and retry with backoff
- [x] #2 API keys are read from config when available for providers that support them
- [x] #3 Academic results feed the same evidence pool as web results with DOI-level dedup
- [x] #4 Provider endpoints and engine lists are config or constants driven rather than inline literals
- [x] #5 Tests with mocked HTTP cover retry and dedup behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the current inline urllib arXiv and Semantic Scholar runners and their tests
2. TDD a shared Research_Interop/academic_providers.py: httpx clients with configurable timeouts, retry with exponential backoff and jitter, config-driven API keys (Semantic Scholar), constants-driven endpoints, normalized paper records carrying DOI for dedup
3. TDD a DOI-level dedup merge that combines web results and academic papers into one evidence pool
4. Wire the engine collecting phase to optionally include academic lanes (paper results join merged_results with DOI dedup against web URLs)
5. Tests with mocked HTTP for retry and dedup plus lint plus task close
ADR required: no - provider adapters behind the existing engine seams
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- New `Research_Interop/academic_providers.py`: `search_arxiv` and `search_semantic_scholar` over httpx with configurable timeouts (default 30s), retry with exponential backoff + jitter (capped at 8s; retries on transport errors and 429/5xx; 4xx fails fast via `AcademicProviderError`), constants-driven endpoints (`ARXIV_API_ENDPOINT`, `SEMANTIC_SCHOLAR_API_ENDPOINT`), and DOI normalization (arXiv DOIs derived `10.48550/arxiv.<id>` from the entry id; S2 DOIs from `externalIds.DOI`). Response shapes match the legacy runners exactly (items additionally carry `doi`/`source`/`url`) so `LocalResearchSearchService`'s default runners now just delegate — its injectable-runner seams and existing tests are untouched.
- API keys: `resolve_semantic_scholar_api_key()` = `SEMANTIC_SCHOLAR_API_KEY` env var, then `[API] semantic_scholar_api_key` via `get_cli_setting` (house env → config.toml precedence); the key rides as an `x-api-key` header when present. arXiv needs no key.
- Evidence-pool unification: `papers_to_evidence` maps papers to the search-result shape (`{title, url, content, metadata.source=academic, doi, ...}`) and `merge_evidence_pools` combines web results + papers with DOI-level dedup (cross-provider duplicates collapse; no-DOI papers are kept). The engine gained an optional `paper_search_fn` seam: when set, each round's queries also fetch papers which join `merged_results` with DOI dedup across providers AND rounds; a lane error is a warning (the web lane already collected), never a run failure.
- Verified TDD: 11 provider tests (mocked httpx client at the request seam: timeout plumbing, 5xx/429/transport retry with recorded backoff, retry exhaustion, api-key header presence/absence, key resolution precedence, evidence mapping, DOI dedup) + 1 engine lane test, all written first and watched failing; full `Tests/Research/` = 91 passed; ruff clean.
- Known follow-up: the window does not yet expose an academic toggle — the lane activates wherever an engine is constructed with `paper_search_fn` (e.g. `LocalResearchSearchService` runners are the natural default wiring in a future UI task).
<!-- SECTION:NOTES:END -->
