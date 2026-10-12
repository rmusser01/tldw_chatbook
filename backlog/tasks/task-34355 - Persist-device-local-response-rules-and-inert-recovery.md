---
id: TASK-34355
title: Persist device-local response rules and inert recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:53'
updated_date: '2026-10-12 03:44'
labels: []
dependencies:
  - TASK-34354
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/228-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 2. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saved Chat rules reopen locally and temporary Chat rules adopt atomically on save.
- [ ] #2 Scoped drafts remain inspectable after invalid or failed learning and binding edits use compare-and-swap.
- [ ] #3 Imported rules are inactive without changing preserved local owners or rollback snapshots.
- [ ] #4 Current installed profiles and every supported recovery lineage upgrade to the actual next response-rule schema with exact catalog parity, fail-closed voice sidecar residue and inactive imports.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: direct implementation of accepted ADR-228. Follow plan Task 2: failing SQLite storage/adoption/restore tests, migrate v75 to v76, scoped CAS store and deletion/adoption hooks, imported-only inert recovery, exact schema qualification and targeted regression.

Current-dev PR3010 integration: follow Docs/superpowers/plans/2026-10-11-pr3010-current-dev-integration.md on the actual post-prior-merge base. ADR required: yes. ADR path: backlog/decisions/228-console-learned-response-rules.md. Reason: preserve the approved local storage/recovery, helper custody and shared native/hook acceptance; source ADR219 remapped after the older claimant. Reproduce reviewed failures before fixes; qualify actual next schema, six voice residues, exact private/index/table catalogs, preparation-through-transport custody, lazy first-use, current Goal/Personal Context owners and separately labelled real-provider/native UI evidence. Historical source qualification above is retained; reopened acceptance awaits current integrated verification and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented private SQLite v76 revisions, scoped CAS bindings, drafts, evidence and inert imported restoration under ADR-228. Temporary Chat save adopts rules on the transcript cursor and clears memory only after commit; close removes temporary state. Deletion removes source evidence while independently promoted definitions survive. Exact schema qualification preserves 533 prior objects and adds 11. New storage and restore cases and native save regressions pass; Black, Ruff and mypy pass. Two existing recovery cases fail on the unchanged baseline in this environment (Windows symlink privilege and selected-prompts subprocess unavailable); recorded in the plan ledger and excluded from the passing completion gate.
<!-- SECTION:NOTES:END -->
