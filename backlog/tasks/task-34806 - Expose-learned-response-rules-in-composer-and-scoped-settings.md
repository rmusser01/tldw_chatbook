---
id: TASK-34806
title: Expose learned response rules in composer and scoped settings
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-12 03:43'
labels: []
dependencies:
  - TASK-34805
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 8. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Composer commands dispatch natively and refusals preserve the draft.
- [ ] #2 Chat, Workspace and global settings reach one scoped manager with explicit promotion and tested revisions.
- [ ] #3 Stop remains reachable during checking and token-backed controls render at 80x24 and 120x35.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: native commands and scoped local rule manager. Task 8 RED-GREEN: real composer command dispatch and draft preservation, shared Chat/Workspace/profile manager, public inactive Test then exact CAS Save, binding changes and promotion preview, Stop/status/focus, token-backed painted terminal layouts and targeted governance.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented native /omfg and /rules commands, shared Chat/Workspace/current-profile manager, exact tested CAS Save, private example inspection, promotion preview, exclusion and Stop/status/focus. Thin UI adapters retain app-owned runtime and token/component patterns; generated CSS rebuilt through its owner. Public targeted run: 55 passes plus the corrected active-run/Workspace witnesses (2 passes); real HTTP literal-send and saved-chat journeys pass. Static checks pass. Required full owner/governance and composer attempts retain independently verified baseline exclusions, documented in Docs/superpowers/reviews/2026-10-03-console-response-rules.md. ADR-228 governs storage, scope and admission; independent whole-branch review belongs to qualification task 34807.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Source TASK-34361 at codex/wave2-source-pr-3010 (73823fde3a4e) was created on 2026-10-04. The older current-dev task keeps 34361; this preserved younger response-rules task is TASK-34806. Original implementation history remains above; references and dependency identities follow this remap.
