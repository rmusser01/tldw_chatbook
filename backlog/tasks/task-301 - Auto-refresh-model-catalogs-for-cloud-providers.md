---
id: TASK-301
title: Auto-refresh model catalogs for cloud providers
status: Done
assignee: []
created_date: '2026-07-18 05:43'
updated_date: '2026-07-19 16:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh model lists for OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, and Mistral on app startup via the LLM_Provider_Catalog discovery pipeline. Disk-backed TTL cache feeds selectors (capped merge + search picker); per-provider opt-in write-through appends new models to [providers]. Amends ADR-002; see ADR-020 and Docs/superpowers/specs/2026-07-17-model-catalog-auto-refresh-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup auto-refresh fetches stale providers in a background worker without blocking app start
- [x] #2 Fetched small catalogs (<=50) merge into chat selector and survive restarts
- [x] #3 Large catalogs stay saved-list-only in dropdown and are searchable via picker
- [x] #4 Write-through is append-only with baseline guard for oversized first fetch
- [x] #5 Global toggle off / opt-out / no key / fresh cache each skip fetching
- [x] #6 Failures degrade to cached models with at most one notification
- [x] #7 Anthropic uses x-api-key + pagination; Z.AI ships config defaults
- [x] #8 ADR-020 linked; tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md
ADR required: yes
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md (amends ADR-002)
Reason: amends ADR-002 explicit-persistence-only stance for opt-in write-through
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-020 hybrid: startup background discovery → disk TTL cache → capped merge into selectors, with per-provider opt-in write-through to `[providers]`. All failures degrade to cached/saved models; startup is never blocked.

Per task (plan Docs/superpowers/plans/2026-07-17-model-catalog-auto-refresh.md):
1. Discovery client: per-provider auth profiles (Anthropic x-api-key + cursor pagination, others Bearer), endpoint path policy, keyless OpenRouter.
2. Default discovery endpoints for all six providers; Z.AI + `[model_catalog]` config defaults.
3. `ModelCatalogDiskStore`: atomic disk TTL cache (`model_catalog_cache.json` in user data dir, IDs + timestamps only), hardened against temp-file races and future stamps.
4. `[model_catalog]` settings parsing (enable/opt-out/TTL/merge cap 50/write-through flags).
5. Refresh report types + startup refresher with TTL/opt-out/no-key skips, diff write-through, and oversized-first-fetch baseline guard.
6. App wiring: cache load before selectors build, startup worker, at-most-one failure notification, hardened refresher loop.
7. Capped merge (50) via `resolve_provider_model_options`; merged options in chat screen; selection preservation.
8. Type-to-filter `ModelSearchPicker` with transient option insertion; relocated into the live Alt+M `ConsoleModelPopover` after discovering the classic sidebar is dormant.
9. Settings UI toggles for auto-refresh and per-provider write-through.

Key decisions/trade-offs: hybrid cache+write-through per ADR-020 (cache-first default keeps ADR-002's explicit-persistence stance unless the user opts in); append-only write-through with baseline guard so a huge first fetch never floods config.toml; OpenRouter refreshes keyless (public catalog); picker placed on the live Console popover surface rather than the dormant classic sidebar.

Files: new — `LLM_Provider_Catalog/model_discovery_disk_cache.py`, `model_catalog_settings.py`, `model_auto_refresh.py`, `local_llm_provider_catalog_service.py`, `Widgets/model_search_picker.py`; modified — `openai_compatible_model_discovery.py` (auth/pagination), `app.py` (startup wiring), `config.py` (defaults), `UI/Screens/chat_screen.py`, `UI/Screens/provider_model_resolution.py`, `UI/Screens/settings_screen.py`, `Widgets/Console/console_model_popover.py`, `Widgets/settings_sidebar.py`. Tests: 6 new/extended test modules under Tests/LLM_Provider_Catalog/, Tests/UI/, Tests/Widgets/, Tests/ (~2900 LOC total branch delta incl. ADR-020, spec, plan).

AC evidence: #1 Tests/LLM_Provider_Catalog/test_app_model_catalog_wiring.py; #2 Tests/UI/test_provider_model_resolution.py + test_model_discovery_disk_cache.py; #3 cap boundary tests + test_popover_model_search_inserts_transient_option; #4 test_model_auto_refresh.py baseline/second-fetch tests; #5 skip-condition tests in test_model_auto_refresh.py; #6 notification tests (at most one); #7 test_openai_compatible_model_discovery.py + test_config_model_catalog_defaults.py; #8 ADR-020 linked above; targeted suite 187 passed/1 skipped.

Smoke check (2026-07-19): OpenRouter live fetch succeeded keyless — 338 models (sample: thinkingmachines/inkling, openrouter/auto-beta, moonshotai/kimi-k3). The five keyed providers (OpenAI, Anthropic, MistralAI, Moonshot, ZAI) were verified at spec time (2026-07-17: all routes return 401 pre-auth, i.e. endpoints exist; Anthropic pagination contract verified against anthropic-sdk-python source) but cannot be live-smoked without keys — configured app keys will exercise them on next launch.
<!-- SECTION:NOTES:END -->
