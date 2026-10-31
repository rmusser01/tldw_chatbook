---
id: TASK-35583
title: Rebase archive recovery PR3162 onto current dev and qualify integration
status: Done
created_date: 2026-10-18 20:02
assignee:
- '@codex'
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3162
- TASK-33164
- TASK-33020
documentation:
- backlog/decisions/166-agent-assisted-archive-recovery.md
- backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
updated_date: 2026-10-18 20:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the approved archive-recovery feature with current dev so PR3162 applies without merge conflicts and its affected persistence, authority, capture and UI behavior remains verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR3162 is rebased onto the fetched dev tip without dropping current dev behavior or the approved archive search/confirmation/restore/Open boundaries; migrations use unoccupied current versions.
- [x] #2 Affected storage, runtime, capture and mounted UI targeted tests pass on the rebased source, with failures attributed and scoped static/generated-artifact checks recorded.
- [x] #3 The rebased branch is pushed using an exact force-with-lease, PR3162 reports no merge conflicts, and its validation and integration notes reflect the actual new head.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; current dev lifecycle/ownership ADRs remain binding.
Reason: compatibility integration of the approved feature with existing current-dev boundaries; renumber only PR-added migrations, preserve existing authority and ownership policies.
1. Rebase fa86ea508a onto fetched dev9af0c59263dce00fa44be4a74d5d7e78712346bf; retain original ea5c335462 branch and exact remote lease. Inventory33conflicts and current interfaces before manual resolution.
2. Resolve storage, runtime and UI in disjoint file scopes under one root Git-index owner. Preserve current dev behavior, native callback/DB handle custody and source/permission gates. Root resolves controller/trace and documentation. Do not resurrect removed CSS or replace large files wholesale.
3. Port archive chronology after current schema versions and update exact recovery declarations/tests together. Preserve native/fenced recovery identity, exact current-user capture ownership, immutable old traces and one-shot target/scope confirmation.
4. Before tests, parse all changed production Python, check duplicate merged definitions, clear all conflict markers and rebuild CSS from sources. No production imports before scratch/test isolation.
5. Run targeted storage/search/reference/coordinator/tool-policy/capture and mounted UI checks using the shared Python3.12, zero network and explicit fresh basetemp. Investigate each failure at its source; add only focused integration regressions. Keep the full suite and additional model requests outside scope.
6. Independently review the integration boundaries, verify scoped static/generated checks, update architecture/docs/task evidence and finish rebase with explicit staging. Push only the PR branch using an exact force-with-lease to fa86ea508a, update PR notes, verify dev ancestry and no merge conflicts; leave original branch/worktree preserved.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Resolved 33 conflict paths against fetched dev 9af0c59263dce00fa44be4a74d5d7e78712346bf while preserving current received-input, native ownership, Canvas/response-rules, extracted review and CSS structures. Chronology now appends chat 82→83 and workspace 8→9 with exact recovery catalogs/authorizers. Repaired proof transfer, cancellation teardown and opposing thinking-owner alias normalization; aligned old test fixtures with current profile/gateway/close-completion APIs without weakening guards.

Targeted evidence: runtime 184 distinct passes; root original 308-case selection now has 303 distinct passing cases and five unchanged trace-baseline failures, plus 19 current review/kill-switch passes; UI 160 functional passes/one existing xfail, CSS 28 passes/three unchanged baseline failures. Storage main 254 passes/six corrected import failures, corrected query 7 passes, recovery 28 passes, focused 2 passes, final pins 21 passes/two unchanged Evals guard failures. Counts overlap across runs; do not sum them.

Independent source review approved after exact alias RED/GREEN (one failed/seven passed → nine passed), with no remaining actionable integration findings. Scoped syntax/duplicate/critical Ruff, no-new-diagnostic comparisons, source whitespace and 65-module CSS build/sync qualify current source. Ten unchanged baseline failures and cumulative descriptor warnings 295/213/255 remain explicit; dedicated retirement/canceled-disposal checks pass. No full sweep, new live/model call, installed build or long resource repeat. Existing ADR166/097 and current lifecycle ADRs apply; no new ADR. Updated architecture, user guides, lessons, design/plan/task records; historical QA remains immutable.

Durable validation: Docs/superpowers/qa/console/2026-10-18-archive-recovery-pr3162-rebase.md and its integration-review/validation artifacts. Rebase completed and exact-lease push published source commit 5660e9cb51c168575ca629d16a16986ac94a4481. GitHub verified current dev 9af0c59263dce00fa44be4a74d5d7e78712346bf, exact new head and MERGEABLE/no-conflict status; PR validation notes updated, draft retained, auto-merge off. All 72 tested source blobs match the committed tree; original verified branch remains preserved. Documentation-only closeout records these final results.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
