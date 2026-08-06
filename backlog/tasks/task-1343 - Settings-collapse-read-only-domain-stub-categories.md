---
id: TASK-1343
title: Settings collapse read-only domain stub categories
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 05:21'
labels:
  - settings
  - ux
  - ia
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
8 of 19 sidebar categories (Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP Defaults, ACP Defaults) are read-only ownership notes ('X owns this / Follow-up: add defaults later'). Half the sidebar is locked doors; every persona clicks each once before learning to skip it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 8 stub categories are replaced by a single read-only Domain ownership map (selector or grouped disclosure),Sidebar drops to ~11 categories all of which are functional,Editable Library & RAG stays reachable in its current group
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: pure settings information-architecture change (sidebar grouping + detail-pane rendering); no storage/schema/migration, sync policy, data ownership, provider boundary, or service contract changes. The underlying domain contract data is kept verbatim.

1. settings_config_models.py: add SettingsCategoryId.DOMAIN_OWNERSHIP ("domain-ownership"); KEEP the 8 stub enum members (contract data keys + legacy deep-link ids).
2. settings_screen.py: add READ_ONLY_DOMAIN_CATEGORY_IDS + legacy-id redirect map; replace the 8 stub summaries with one 'Domain Ownership' read-only summary (12 sidebar categories: 19 - 8 + 1); Domain Defaults group becomes (LIBRARY_RAG, DOMAIN_OWNERSHIP) so editable Library & RAG stays in its group.
3. New _render_domain_ownership_detail(): grouped disclosure rendering all 8 stub contracts (owner destination, sources, boundary rows, follow-up) in one read-only pane; wire into _render_detail_pane dispatch.
4. Add DOMAIN_OWNERSHIP entries: ownership record, state banner, guided-action message, inspector guidance. Footer hint frozensets untouched (new category is read-only, advertises nothing).
5. Deep links: canonicalize legacy stub category values -> domain-ownership in apply_navigation_context / restore_state / _select_category; all 8 contracts visible in the combined pane, so no extra selection state is needed.
6. Tests: sweep + footer-hints derive category ids from the sidebar summaries (count 12); configuration_hub records test becomes superset check (legacy stub records stay keyed by stub ids); grouped test asserts stubs left the sidebar; new tests for combined pane rendering and deep-link mapping.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Collapsed the 8 read-only domain stub sidebar categories (Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP Defaults, ACP Defaults) into one read-only 'Domain Ownership' category rendered as a flat grouped list with per-domain headings (chosen over an in-pane selector: no extra selection state, all 8 contracts visible at once, so legacy deep links need no domain sub-target).

Approach:
- settings_config_models.py: added SettingsCategoryId.DOMAIN_OWNERSHIP; the 8 stub enum members are KEPT as contract-data keys and legacy deep-link ids.
- settings_screen.py: sidebar summaries 19 -> 12 (Domain Defaults group is now Library & RAG + Domain Ownership; Library & RAG stays editable in place); SETTINGS_DOMAIN_CATEGORY_CONTRACTS data untouched; new _render_domain_ownership_detail() renders every stub contract (owner destination, sources, boundary rows, follow-up); DOMAIN_OWNERSHIP got its own ownership record, state banner, guided-action message, and inspector guidance; READ_ONLY_DOMAIN_CATEGORY_IDS + SETTINGS_LEGACY_DOMAIN_CATEGORY_REDIRECTS canonicalize retired stub ids to domain-ownership in apply_navigation_context, restore_state, and _select_category (deep links keep working). New category is in neither footer-hint capability frozenset, so it advertises no dead keys (ADR-031). The stub summary description lists all 8 domain names so category search still surfaces them.

Tests: sweep + footer-hints now derive category ids from the sidebar summaries (12); narrow-layout wrap probe repointed from mcp-defaults to domain-ownership; configuration_hub ownership-records test is now a superset check (stub-keyed records remain), grouped test asserts stubs left the sidebar; 4 new tests cover sidebar composition, legacy-id redirect (deep link + persisted state), combined-pane rendering of all 8 contracts, and stub button removal. Results: configuration_hub 246, sweep/footer/save-commit 18, narrow-layout 9, remaining settings files 53 -- all green, no failures. Ruff: 2 pre-existing F401/F811 on save_setting_to_cli_config imports (not from this change).

Files: tldw_chatbook/UI/Screens/settings_config_models.py, tldw_chatbook/UI/Screens/settings_screen.py, Tests/UI/test_settings_category_sweep.py, Tests/UI/test_settings_configuration_hub.py, Tests/UI/test_settings_footer_hints.py, Tests/UI/test_settings_narrow_layout.py.

ADR: not required (documented in plan) - IA-only change; no storage/schema/boundary/service-contract changes; contract data preserved verbatim.
<!-- SECTION:NOTES:END -->
