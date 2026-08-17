---
id: TASK-16198
title: 'Fix dev-red knowledge_entry test: real network egress in teardown'
status: Done
assignee: ['@claude']
created_date: '2026-08-14 03:05'
labels:
  - tests
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The knowledge_entry suite fails on pristine dev with a network-egress teardown error naming real remote IPs (`104.18.3.115:443`, `104.18.2.115:443`) — reproduced byte-for-byte on clean base `c3ed2854a` during TASK-15471. This is the egress-guard (TASK-15211 programme: tests must not reach the network; the guard now blocks by default) catching a genuine leak: something in the knowledge_entry path opens a real connection during teardown. Find the egress source, stub or gate it, and make the suite green under the guard. Absent from known-red batch task-15766. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The knowledge_entry suite passes on a pristine checkout with the egress guard active
- [x] #2 The egress source is named in the notes (what connects, from where, why in teardown)
- [x] #3 No weakening of the egress guard itself
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce at HEAD: run `Tests/UI/test_product_maturity_phase3_knowledge_entry.py` under the venv python with PYTHONPATH pinned to this worktree; capture verbatim output to a file.
2. Trace the egress source: identify what opens a real connection during teardown (atexit hook, `__del__` closer, session-close flush, library telemetry). Name module → call chain → destination.
3. Fix at the source layer (stub/gate/env-var the connecting component in the suite's fixtures), narrowest scope; do NOT touch the egress guard.
4. Prove: (a) knowledge_entry suite green under the active guard; (b) guard integrity — a deliberately seeded connection in a scratch test still trips the guard.
5. ruff check + format on touched files; commit; task notes + Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The egress source (module → call chain → destination).** The destination is
**openrouter.ai:443** — `dig openrouter.ai A` returns exactly `104.18.3.115`
and `104.18.2.115`, the two recorded IPs (httpx attempts each resolved A
record in turn; DNS via getaddrinfo is C-level and outside the guard, which
is why numeric IPs appear in the record). The call chain is the ADR-020
startup catalog refresh: `TldwCli.on_mount` (app.py:9962) →
`_schedule_startup_model_catalog_refresh` → worker
`_refresh_model_catalogs` (app.py:10016) →
`LocalLLMProviderCatalogService.refresh_stale_configured_providers`
(LLM_Provider_Catalog/local_llm_provider_catalog_service.py:610) →
`discover_models` → `discover_openai_compatible_models` → real httpx `GET
https://openrouter.ai/api/v1/models`. **OpenRouter is the one provider
fetched WITHOUT credentials** — the loop's skip reads `if api_key is None
and provider_key != "openrouter"` ("OpenRouter's catalog is public") — so a
sandbox with zero API keys still egresses to exactly this host, matching the
incident IPs.

**Why "in teardown".** The service wraps each provider fetch in a broad
per-provider `try/except` (the block becomes a "failed/request_failed"
outcome), so the guard's `BlockedNetworkAccess` (an OSError) is swallowed
mid-test; the autouse `_no_network_io` fixture (Tests/conftest.py:510) then
fails the test at TEARDOWN on the non-empty blocked-attempt record — the
record-then-fail design that makes the guard unswallowable. The teardown
timing is attribution, not the connect's timing.

**Why intermittent / why it needed a full-app boot.** The refresh is gated on
`[model_catalog] refresh_consent_recorded` (default false) AND
`auto_refresh_enabled` (default true) AND per-provider disk-cache staleness
(24h TTL). On a healthy per-test sandbox consent is always false, so the
worker branch never runs (headless boots skip the consent modal instead) —
the red required consented settings reaching the test process (shared
`TLDW_TEST_CONFIG_ROOT` between concurrent sessions / config-cache
pollution), plus the refresh worker racing test teardown. Only
`test_study_screen_consumes_pending_initial_section` boots the real
`TldwCli`, which is why the file's other four tests never fired it.

**Fix layer.** Autouse `_disable_model_catalog_refresh` fixture in
`Tests/UI/conftest.py` patching `TldwCli._refresh_model_catalogs` to an async
no-op — the identical pattern the repo already uses for this exact seam in
`Tests/ProductionApp/conftest.py` (autouse) and
`Tests/RuntimePolicy/test_runtime_policy_full_app.py:93` (per-instance);
Tests/UI was the one full-app-boot surface without it. This stubs the
product's phone-home at its single seam regardless of settings content, so
no consent/cache/timing combination can re-open it. The egress guard itself
(`Tests/network_guard.py`, `_no_network_io`) is untouched.

**Regression test.** New
`Tests/UI/test_app_startup_model_catalog_offline.py` drives the worst case
on purpose: consented settings pinned via `app_module.load_settings`, a real
`TldwCli` boot, the startup schedule and its `model-catalog-refresh` worker
awaited to completion, then asserts zero recorded attempts. Born-red proof:
with the new conftest fixture temporarily disabled, this test reproduces the
incident byte-for-byte — `test attempted network egress (blocked):
socket.connect -> 104.18.2.115:443, socket.connect -> 104.18.3.115:443` from
the guard's teardown assertion — green again with the fixture restored.

**Guard integrity (AC #3).** A scratch test (run, then deleted) seeded
`socket.create_connection(("104.18.3.115", 443))`: the connect was denied
in-body (`network access blocked in tests` OSError) AND the teardown record
still failed the test (`socket.create_connection -> 104.18.3.115:443`) —
both guard layers intact, nothing loosened.

**Tests.** knowledge_entry suite 5/5 + new test 1/1 green under the guard;
collateral: phase1_first_run + settings model-catalog toggles/save +
consent modal + provider_model_resolution + full Tests/LLM_Provider_Catalog
= 406 passed (the class-attribute patch does not disturb the stub-host
schedule tests or the direct service tests, which live outside Tests/UI);
TTS_Events + phase3 product-maturity + study_dashboard batch = 99 passed.
ruff check + format clean on both touched files.

**Modified/added files.** `Tests/UI/conftest.py` (autouse fixture),
`Tests/UI/test_app_startup_model_catalog_offline.py` (new regression test).
<!-- SECTION:NOTES:END -->
