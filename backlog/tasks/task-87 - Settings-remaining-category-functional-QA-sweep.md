---
id: TASK-87
title: Settings remaining category functional QA sweep
status: Done
assignee: []
created_date: 2026-06-09 01:03
updated_date: 2026-06-09 19:35
labels:
- settings
- ux
- qa
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit and verify the Settings screen categories not covered by recent focused Settings slices through actual rendered app use. The goal is to identify remaining broken controls, placeholder states, and category ownership gaps before claiming Settings is a real configuration hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual CDP/Textual-web walkthrough covers the remaining Settings categories that still rely on read-only contracts or incomplete mutation paths: Overview, Storage, Privacy & Security, Diagnostics, Advanced Config, Server/Sync/Workspace/Handoff, and Domain Defaults for Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP, and ACP.
- [x] #2 Every category records whether it is guided-editable, read-only status/recovery, owned by another destination, or an explicit WIP placeholder.
- [x] #3 Confirmed usability blockers receive a failing regression before fixes, or become child Backlog tasks with evidence when they are outside this sweep.
- [x] #4 Dropdown/input focus, save/revert/test feedback, and keyboard operation are verified for each editable category.
- [x] #5 Actual screenshots and QA notes are recorded, and user approval is captured before a functional Settings PR is created.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 ADR check is documented before implementation starts; create or link an ADR only if the sweep changes storage, runtime, provider, sync, or ownership boundaries.
- [x] #2 Focused automated regressions cover every in-scope code change made from confirmed Settings blockers.
- [x] #3 Actual CDP/Textual-web screenshots and QA notes are attached for each approved Settings fix or category sweep result.
- [x] #4 Verification commands and residual risks are recorded in Implementation Notes before the task can move to Done.
<!-- DOD:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: This closeout records actual-use QA evidence and the approved Storage action reachability correction. It does not introduce new storage, runtime, provider, sync, or ownership boundaries beyond existing ADRs.

1. Review current `origin/dev`, the existing post-merge actual-use Settings sweep, and merged follow-up regressions.
2. Preserve the approved Storage action reachability screenshot as tracked QA evidence.
3. Update Settings QA notes with the remaining-category state map, verification evidence, and residual risks.
4. Mark `TASK-87` criteria complete only after focused Settings verification and diff hygiene pass.

## Implementation Notes

Closed out the remaining Settings category functional QA sweep. The QA notes now map each Settings category to its verified state: guided-editable, read-only status/recovery, destination-owned read-only, or guarded expert path. The approved Storage screenshot from the PR #495 correction is tracked at `Docs/superpowers/qa/product-maturity/screen-qa/settings/remaining-category-functional-qa-2026-06-09/settings-storage-check-visible-before-paths.png`, showing `Check Storage` visible before the long database path editor.

Focused verification from this closeout was `python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_library_rag_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_privacy_security.py Tests/UI/test_console_session_settings.py Tests/UI/test_product_maturity_phase1_navigation_smoke.py::test_top_level_navigation_activates_visible_tab_border_from_cached_console_screen --tb=short`, which reported `298 passed, 1 warning`. `git diff --check` was clean. Residual risks are intentionally scoped: domain defaults remain read-only until owning destinations define writable contracts, and runtime operations for MCP, ACP, Schedules, Workflows, Skills, Personas, and sync/handoff stay outside Settings.
