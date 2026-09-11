---
id: TASK-32308
title: Console custom endpoint registry (named endpoints from templates)
status: Done
assignee: []
created_date: '2026-09-11 03:43'
updated_date: '2026-09-11 22:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the named custom-endpoint registry per Docs/superpowers/specs/2026-09-10-console-custom-endpoint-registry-design.md: config-owned [custom_endpoints.<slug>] entries mapped onto existing execution families, creation-from-template flow in Conversation Settings, F9 Settings management, optional one-way convert for the custom/custom_2 slots. Integrates with the TASK-30012 connection-first modal recomposition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Registry entries persist in [custom_endpoints.<slug>] and survive restart,custom-ep:<slug> ids resolve through the family execution path with pinned endpoint,Endpoint created from any provider template inside the modal without config editing,Selecting a registry entry never hits the unsaved-endpoint block,F9 Settings rename/edit/delete with reference guard + detach,custom and custom_2 keep working; optional convert action,Unit + Pilot tests per spec testing section,ADR authored and linked before implementation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR: backlog/decisions/146-console-custom-endpoint-registry.md

Executed as the 8-task SDD plan `Docs/superpowers/plans/2026-09-10-task-32308-console-custom-endpoint-registry.md` (tasks 1-7 plus plan-amendment task 5.5) on branch `feat/custom-endpoint-registry`; per-task reports and review diffs in `.superpowers/sdd/2026-09-10-task-32308-console-custom-endpoint-registry/`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach** (per ADR-146): a config-owned registry of named endpoints mapped onto existing execution families. New pure module `Chat/custom_endpoint_registry.py` owns `[custom_endpoints.<slug>]` load/validate/mutate — `custom-ep:<slug>` provider ids, family mapping onto `llama_cpp` / `ollama` / `custom` execution keys, slug derivation with uniquifying suffix, `api_key` kept out of reprs/displays. Creation happens where the need arises: `Widgets/Console/console_endpoint_template_modal.py` (the Console settings modal's "New endpoint…" button) templates off any provider, an existing entry (duplicate), or a blank OpenAI-compatible start; Create persists atomically, switches the modal's provider, and probes models. Entries surface as first-class provider options (`build_console_provider_options`), flow through readiness/provider-settings seams (`Chat/console_session_settings.py`, `Chat/console_provider_support.py`), persist through Save/Save-as-default (`Widgets/Console/console_settings_modal.py`), and route through the gateway with family execution + pinned endpoint + credential resolution (`Chat/console_provider_gateway.py`). Full management lives in F9 ▸ Providers & Models ▸ Custom endpoints (`UI/Screens/settings_provider_view_model.py` seams + `UI/Screens/settings_screen.py` panel): rename (display name only), edit URL/env-var/models, delete guarded by a conversation-reference check with detach-to-conversation-only recovery, and a one-way convert action for the two built-in `custom`/`custom_2` slots. Convert carries the slot's URL, models, and `api_key_env` reference but never its stored `api_key` (documented in the User Guide).

**Plan amendment (Task 5.5):** review of Task 5 found that declared entry credentials (`api_key_env`/`api_key`) never reached the gateway's `resolution.api_key`, so resolved-key entries would have sent unauthenticated; the plan was amended (commit `2857f83efb`) and Task 5.5 (`ded6202d1e`) wired entry credentials through gateway resolution before Task 6 built on it.

**Dual-shape config contract:** `load_settings()` never projects unknown top-level tables, so the registry reader accepts both config shapes — a top-level `custom_endpoints` table (raw CLI config) wins, with the projection nested under `COMPREHENSIVE_CONFIG_RAW` (the normalized app shape) as fallback, mirroring the gateway's `_caching_config_value` both-shapes precedent. Without it, no runtime seam receiving the app-config shape could see entries. Documented in the spec's Data model section.

**Files (high level):** new `tldw_chatbook/Chat/custom_endpoint_registry.py` and `tldw_chatbook/Widgets/Console/console_endpoint_template_modal.py`; seams in `console_session_settings.py`, `console_provider_support.py`, `console_provider_gateway.py`, `console_settings_modal.py`, `settings_provider_view_model.py`, `settings_screen.py`; tests in `Tests/Chat/test_custom_endpoint_registry.py`, `Tests/Chat/test_console_session_settings.py`, `Tests/Chat/test_console_provider_gateway.py`, `Tests/Widgets/test_console_endpoint_template_modal.py`, `Tests/UI/test_settings_custom_endpoints.py`, `Tests/UI/test_console_session_settings.py`; docs in `Docs/User_Guide/settings.md` (Custom endpoints) and `Docs/User_Guide/console.md` (Conversation Settings paragraph).

**Verification:** per-task RED/GREEN with targeted regressions, all review-clean (reports in `.superpowers/sdd/2026-09-10-task-32308-console-custom-endpoint-registry/task-{1..7,5.5}-report.md`). Final sweep in the snapshot env — `pytest Tests/Chat/test_custom_endpoint_registry.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_provider_gateway.py Tests/Widgets/test_console_endpoint_template_modal.py Tests/UI/test_settings_custom_endpoints.py Tests/UI/test_console_session_settings.py -q` — **2 failed, 580 passed**: both failures are the documented pre-existing `Tests/UI/test_console_session_settings.py` pair (inspector staged-context ordering, unmount-timeout repair); the compaction test that failed during Task 7 (`test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`, `Tests/Chat/test_console_session_settings.py:2117`) passed in this run.

**ADR:** [backlog/decisions/146-console-custom-endpoint-registry.md](../decisions/146-console-custom-endpoint-registry.md) — [spec](../../Docs/superpowers/specs/2026-09-10-console-custom-endpoint-registry-design.md).
<!-- SECTION:NOTES:END -->
