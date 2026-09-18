---
id: TASK-32739
title: Preserve model discovery selection and result ownership in Settings
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 17:35'
updated_date: '2026-09-17 18:01'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review of Settings model discovery so checked choices, save feedback and asynchronous results remain attached to the provider the user is working with.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard Discover, selection, Save selected and Clear preserve exact model IDs and the existing active-model contract across dark/light wide/compact layouts.
- [x] #2 Checked discovery rows survive same-screen pane rebuilds; failed operations retain truthful recovery feedback and can be retried.
- [x] #3 Delayed discovery and save completions cannot replace another provider or changed endpoint form with stale results, receipts or active-model edits.
- [x] #4 Targeted production-CSS regressions, private native loopback/config journeys, documentation, static checks and review evidence qualify the bounded changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing backlog/decisions/002-openai-compatible-model-discovery.md and backlog/decisions/020-automatic-model-catalog-refresh.md; ADR-031/150/161 govern keys and component patterns. Reason: preserve existing Settings-owned manual discovery, exact provider-list persistence and staged active-model behavior without new ownership or service boundaries.
1. Trace discovery/list selection, Save selected and failure/retry paths; reproduce provider/endpoint races and pane-rebuild selection loss with production-CSS tests before production edits.
2. Repair confirmed defects inside the existing Settings owner and handlers, retaining service persistence semantics and selected-model identity. Review the narrow diff and run relevant targeted tests.
3. Verify dark/light 170x48 and 80x24 keyboard journeys in a private native app using real loopback discovery and isolated config persistence; inspect one capture batch, verify shutdown/default isolation, update guide/audit/task and commit. No full suite or merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved checked discovered-model IDs across pane rebuilds and unchanged connection-field mount echoes. Added monotonic revision guards to discovery/save/clear so delayed results and errors cannot overwrite an edited provider form; successful old-provider saves still update that exact runtime catalog. Clear failures retain rows and report retry, while successful clearing also removes typeahead.

Validation: 176 distinct targeted cases pass (24 new production-CSS cases; 32 discovery/typeahead total, 79 provider regressions, 65 service/governance). Two F8 assertions reproduced against HEAD method bodies and were corrected to existing F3 copy. Four private native dark/light 170x48 and 80x24 journeys pass with real loopback 503 recovery, config persistence and cache clearing; eight captures inspected, eleven SQLite databases healthy, zero messages/conversations, unchanged default-file hashes, app/server shutdown verified. Static checks add no lint debt; independent review found no actionable issues. No full suite or merge.

ADR required: no; existing backlog/decisions/002-openai-compatible-model-discovery.md and 020-automatic-model-catalog-refresh.md preserve the service/ownership contract, with ADR-031/150/161 for UI conventions. Changed settings_screen.py, the new model-discovery journeys, two legacy test expectations, Settings guide, workflow audit and live-verification lesson. QA and explicit fixture limitations: Docs/superpowers/qa/2026-09-17-settings-model-discovery/README.md. Saved workstream to draft PR #2704 against dev before continuing; integration conflicts remain. Closeout ID scan: 258 refs/27 worktrees, no collision.
<!-- SECTION:NOTES:END -->
