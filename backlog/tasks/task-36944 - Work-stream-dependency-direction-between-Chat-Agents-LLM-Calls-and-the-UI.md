---
id: TASK-36944
title: 'Work stream: dependency direction between Chat, Agents, LLM_Calls and the
  UI'
status: To Do
created_date: 2026-10-26 01:07
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
- [ ] #1 Every subtask of this stream is Done
- [ ] #2 The layer test's allowlist shrinks by every edge these subtasks remove
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
