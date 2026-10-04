---
id: TASK-34361
title: Expose learned response rules in composer and scoped settings
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-04 13:31'
labels: []
dependencies:
  - TASK-34360
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 8. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Composer commands dispatch natively and refusals preserve the draft.
- [x] #2 Chat, Workspace and global settings reach one scoped manager with explicit promotion and tested revisions.
- [x] #3 Stop remains reachable during checking and token-backed controls render at 80x24 and 120x35.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: native commands and scoped local rule manager. Task 8 RED-GREEN: real composer command dispatch and draft preservation, shared Chat/Workspace/profile manager, public inactive Test then exact CAS Save, binding changes and promotion preview, Stop/status/focus, token-backed painted terminal layouts and targeted governance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented native /omfg and /rules commands, shared Chat/Workspace/current-profile manager, exact tested CAS Save, private example inspection, promotion preview, exclusion and Stop/status/focus. Thin UI adapters retain app-owned runtime and token/component patterns; generated CSS rebuilt through its owner. Public targeted run: 55 passes plus the corrected active-run/Workspace witnesses (2 passes); real HTTP literal-send and saved-chat journeys pass. Static checks pass. Required full owner/governance and composer attempts retain independently verified baseline exclusions, documented in Docs/superpowers/reviews/2026-10-03-console-response-rules.md. ADR-219 governs storage, scope and admission; independent whole-branch review belongs to qualification task 34362.
<!-- SECTION:NOTES:END -->
