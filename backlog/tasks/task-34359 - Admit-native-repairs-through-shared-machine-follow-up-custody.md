---
id: TASK-34359
title: Admit native repairs through shared machine follow-up custody
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-12 03:43'
labels: []
dependencies:
  - TASK-34358
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 6. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native and genuine hook feedback produce one atomic follow-up and shared receipt.
- [ ] #2 Stale, forged, duplicate or historical assessments cannot authorize repairs.
- [ ] #3 Two native and three shared continuation limits, foreground priority and permission gates hold.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: native/hook shared machine admission and atomic receipt custody. Follow Task 6 RED-GREEN: current assessment lookup, real Stop composition, retained counters/deduplication, same-cursor receipt and fingerprint, rollback/foreground/veto/recovery tests; retain old hook protocol and permissions.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Native references resolve current runtime-owned assessments and pinned rules, compose only owned Stop results, preserve hook receipts and share one queue custody. Source versions and strict host counters join the acceptance fingerprint; shared receipt writes on the message/checkpoint cursor. Duplicate callbacks, vetoes, required barriers, foreground input and limits refuse repair. ADR-228. RED-GREEN includes hybrid chain identity and missing genuine hook provenance. 25 focused cases; 38 existing queue cases; 5 supported hook/agent budget controls; 54 passing DB/UI owner cases (9 exact baseline DB failures excluded). External Stop tests also fail on unchanged source in this Windows environment; no assertion/guard weakened. black/ruff/mypy pass.
<!-- SECTION:NOTES:END -->
