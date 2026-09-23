---
id: TASK-32925
title: Large model catalogs (OpenRouter) list fully, stay searchable and survive restarts
status: Done
created_date: 2026-09-23 17:54
assignee:
- '@claude'
labels:
- model-catalog
- openrouter
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need OpenRouter's full model list pulled and searchable. Measured live on 2026-09-23: 456 models in 748,851 bytes. Discovery failed closed past 512 models or 1 MiB, and when it did it listed nothing, so ordinary catalog growth would soon have emptied OpenRouter's list for every user. The disk cache also refused any snapshot over 100 models, so OpenRouter's list was never saved and re-downloaded every launch. The Settings endpoint probe read through the same 1 MiB bound.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A catalog of 3x OpenRouter's live size (1,368 records, over 1 MiB) discovers every model
- [x] #2 A full discovered list is saved to disk and reloads intact after a restart
- [x] #3 One full-size list does not evict other providers' cached lists
- [x] #4 Every stage after discovery (disk, memory, probe read) is pinned by a test to hold what discovery accepts, with headroom over the live catalog
- [x] #5 Over-bound catalogs still fail closed (no partial list is cached)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Discovery count limit 512 -> 4096 (~8x OpenRouter)
2. Shared model-response byte limit 1 MiB -> 8 MiB (4096 live-sized records fit)
3. Disk cache per-entry 100 -> 4096; in-memory total 4096 -> 8192
4. Scale test with an OpenRouter-shaped fixture; bounds-relationship pins
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Raised four limits and pinned how they relate. `DISCOVERED_MODEL_MAX_COUNT` 512 -> 4096. `MODEL_PROBE_RESPONSE_MAX_BYTES` 1 MiB -> 8 MiB. This one constant also backs discovery, the Settings endpoint probe and local probes; raising it beat threading a size parameter through five callers. `MODEL_CATALOG_DISK_MAX_MODELS_PER_ENTRY` 100 -> 4096. `ModelDiscoveryCache` total 4096 -> 8192. Over-bound catalogs still fail closed; the existing reject tests use the constants and still pass.

`test_openrouter_scale_catalog.py` uses records with the live API's field set (~1.7 KB each). It covers full discovery, a disk round-trip across restart, and no cross-provider eviction. It also pins every downstream bound to at least the discovery bound, since the 100-model disk limit showed a list can be lost AFTER discovery succeeds. Worst-case memory per read goes from 1 MiB to 8 MiB.

Unchanged by design: the search picker filters the whole list and shows the top 20 matches; config write-through still skips oversized first fetches (`merge_cap`).

Deliberate test pins updated to the new values: `_EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES` (local discovery + Settings probe tests), the llama.cpp raw-body limit test (renamed `..._at_eight_mib_...`), and `_EXPECTED_MAX_MODELS_PER_ENTRY`. Two disk-cache fixtures built 120-char IDs as a 3-digit index + 116 chars; past index 999 they became 121 chars, so they now use a 4-digit index + 115. The 2 MiB whole-file disk budget is unchanged: a worst-case entry (4096 IDs x 120 emoji) is about 1.98 MB, and real ASCII IDs of about 40 chars make a full list about 170 KB. Compared against clean origin/dev, the same 36 touching test files give identical failure sets (the local ADR-126 RecoveryRequired gate plus a dev-red `recursive_json` test).

Qodo review fixes, confirmed by tests that failed first:
- Anthropic pagination was capped at 10 pages of 100. At the old 512 limit the count check failed closed first; at 4096, a 1,001-4,096 model catalog silently kept only 1,000. The page cap is now derived from the model limit (41 pages).
- A last permitted page that still reports `has_more` now fails closed. Before, short pages that never finished returned partial "success".
- All pages now share one `MODEL_DISCOVERY_RESPONSE_MAX_BYTES` budget through a new optional `max_bytes` on `read_bounded_model_response`. The per-page bound had let ten pages hold about 28 MB.
- Added a Google-style docstring to `ModelDiscoveryCache.__init__`.

Files: `Chat/local_server_discovery.py`, `LLM_Provider_Catalog/{openai_compatible_model_discovery,model_discovery_disk_cache,model_discovery_cache}.py`, `Tests/LLM_Provider_Catalog/test_openrouter_scale_catalog.py`, `Tests/UI/test_settings_endpoint_probe.py` (deliberate pin updated).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
