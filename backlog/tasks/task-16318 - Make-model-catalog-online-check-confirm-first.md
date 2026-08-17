---
id: TASK-16318
title: Make model catalog online check confirm-first
status: Done
assignee: []
created_date: '2026-08-15 04:19'
updated_date: '2026-08-15 04:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Privacy: startup auto-refresh contacted 7 cloud providers without asking. Gate it behind a one-time consent dialog persisted to [model_catalog] refresh_consent_recorded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Startup refresh requires recorded consent
- [x] No network calls before consent
- [x] Decline persists auto_refresh_enabled=false
- [x] Tests cover gating and modal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add refresh_consent_recorded setting
2. Consent modal
3. App wiring
4. Tests + ADR-020 amendment
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: new persisted `[model_catalog] refresh_consent_recorded` gate (default `false`, privacy-safe: only explicit boolean `true` counts). Startup scheduler shows `ModelCatalogConsentModal` (UI/Screens/model_catalog_consent.py) instead of refreshing; Allow persists consent + runs the refresh worker, Deny persists `auto_refresh_enabled = false` so the question is never re-asked.
- Defense in depth: `TldwCli._refresh_model_catalogs` itself re-checks consent, so no code path refreshes unconsented. Settings-screen save no-op guard compares all fields except consent (only the dialog records consent) so a consented config doesn't trigger per-keystroke config rewrites.
- Default config.toml template ships `refresh_consent_recorded = false`; existing users (key absent) are prompted exactly once.
- Modified files: tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py, tldw_chatbook/app.py, tldw_chatbook/UI/Screens/model_catalog_consent.py (new), tldw_chatbook/UI/Screens/settings_screen.py, tldw_chatbook/config.py; tests in Tests/LLM_Provider_Catalog/, Tests/UI/, Tests/test_config_model_catalog_defaults.py.
- ADR: amended backlog/decisions/020-automatic-model-catalog-refresh.md (confirm-first amendment).
- Verification: full affected suites pass (LLM_Provider_Catalog + consent modal + first-run + config defaults + settings catalog suites, 390 passed); one pre-existing flaky voice-step failure in the live-contract file was verified on the unmodified baseline; ruff clean.
<!-- SECTION:NOTES:END -->

