---
id: TASK-36944
title: 'Work stream: dependency direction between Chat, Agents, LLM_Calls and the
  UI'
status: Done
created_date: 2026-10-26 01:07
updated_date: '2026-11-20 14:55'
dependencies:
- TASK-36942.3
labels:
- architecture-review-2026-10-25
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Agents and Chat, and Chat and LLM_Calls, import each other because shared provider types live in Chat, and importing anything in Chat runs its package init, which loads the DB. Separately, most domain-to-presentation imports pull pure code that happens to live in Textual widget or screen files, and one Chat module identity-checks UI-private callbacks.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F7, F12; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every subtask of this stream is Done
- [x] #2 The layer test's allowlist shrinks by every edge these subtasks remove
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-20 on dev `d85e90bc9e`. AC #1: all ten subtasks (TASK-36944.1 to .10) are Done. AC #2: `test_layer_direction.py::test_allowlist_has_no_stale_rows` fails on any allowlist row that no longer matches an upward import, so each edge a subtask removed had to leave `layer_direction_allowlist.tsv` in the same change. The allowlist went from 191 rows when it was added (`a812f78e28`) to 158. That is net across the programme: other streams also removed rows, and three rows were added (`96671c682c`, `dec7997795`). The guard files `test_layer_direction.py`, `test_module_size_ratchet.py`, `test_app_string_lookup_census.py`, `test_agent_runtime_import_footprint.py` and `test_screen_size_ratchet.py` passed serially on dev `d85e90bc9e` through the scratch wrapper: 314 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
