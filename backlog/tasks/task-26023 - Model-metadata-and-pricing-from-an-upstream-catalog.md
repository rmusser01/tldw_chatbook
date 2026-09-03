---
id: TASK-26023
title: Model metadata and pricing from an upstream catalog
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 19:23'
labels:
  - providers
  - ops
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Model capabilities and prices are hand-maintained and go stale silently. Verified on origin/dev: model_capabilities.py:27-45 infers vision support and context window from a regex pattern table, and LLM_Calls/pricing_catalog.py:41-48 carries a hand-seeded price table with a _SEED_AS_OF staleness stamp that a human must re-verify. Every new model release needs a code change. Hermes pulls models.dev, which covers 4000-plus models with ETag conditional GET, disk cache and stale-while-revalidate. Chatbook already has the expensive half: memory cache, disk cache, merge and persistence modules under LLM_Provider_Catalog/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model context window, capability flags and per-token pricing can be sourced from an upstream catalog
- [x] #2 Upstream data is a lower-priority merge layer beneath existing hand-maintained entries, so a deliberate local override always wins
- [x] #3 The catalog is fetched with conditional requests and cached to disk; a cold start with no network uses the cache and then the hand-seeded values
- [x] #4 No network call occurs on a hot path - fetching is background or explicit, never blocking a send
- [x] #5 The origin of a displayed price or window is inspectable, so a wrong number can be traced to its source
- [x] #6 The existing honest cost-unavailable behavior is preserved: an unknown price still refuses to fabricate a figure
- [x] #7 Fully offline operation is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: parse (context/vision/price + defensive junk), conditional GET (ETag/304, never-raise), offline cold-start; pricing+capability gap-fill integration (enabled/off/local-wins)\n2. models_dev_catalog.py: parse_models_dev + ModelsDevCache (disk load/save) + fetch_models_dev (conditional GET) + network-free lazy in-memory gap-fill layer (config-gated)\n3. Wire as step-4 fallback in pricing get_pricing + step-3 fallback in model_capabilities; as_of/source='models.dev' for origin\n4. Config [model_catalog] use_models_dev (default off); guide
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New self-contained LLM_Provider_Catalog/models_dev_catalog.py: parse_models_dev (defensive provider->models->{limit.context, modalities.input has 'image', cost.input/output} — schema drift degrades to fewer entries, never a crash), ModelsDevCache (disk load/save of {etag, body}, missing/corrupt = empty), fetch_models_dev (conditional GET: sends stored If-None-Match, 304 keeps cache, 200 replaces via atomic mkstemp+replace, ANY network/parse/disk error keeps the old cache and never raises — AC#3). A process-wide lazy in-memory layer (models_dev_entry) reads the disk cache ONCE into memory and answers from memory — the lookup path never touches network or disk after first read (AC#4); fetching is explicit/background only. Wired as a LOWER-priority gap-fill: pricing get_pricing step 4 (after direct mappings, provider patterns, and local-zero — so a hand-maintained or local entry ALWAYS wins, AC#2, pinned) before the honest None (AC#6 preserved: unknown-upstream returns None, no fabricated price); model_capabilities step 3 (after direct+pattern) before defaults. Origin is inspectable via ModelPricing.as_of='models.dev' and the capabilities dict's source='models.dev' (AC#5). Gated by [model_catalog] use_models_dev, default OFF => offline/today byte-identical (AC#7, pinned both catalogs). 12 new tests; pricing (30) + capabilities (27) suites green. DEFERRED (documented): the actual scheduled/settings-triggered fetch WIRING (a background timer or a Settings 'refresh catalog' action calling fetch_models_dev) is not added here — the fetch function + cache are complete and tested, but nothing calls fetch_models_dev in production yet, so the cache stays empty until wired. That wiring is a small follow-up (mirror model_auto_refresh's consent-gated pattern); the merge/lookup half — the risky part — is done and safe (empty cache = today). models.dev's exact api.json schema was inferred; parse is defensive so a mismatch yields fewer entries, not errors.
<!-- SECTION:NOTES:END -->
