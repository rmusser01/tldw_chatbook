---
id: TASK-32704
title: Verify Library conversation export journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 22:11'
updated_date: '2026-09-16 22:30'
labels:
  - ui
  - conversations
  - export
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qualify the shared ZIP export flow from Conversations so scope, recovery controls and the written bundle agree, while source conversations remain intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Whole-source and selected-row exports contain the advertised conversations and exact fixture messages, with no unrelated note content.
- [x] #2 Destination cancellation, suffix normalization and explicit overwrite remain usable at compact and wide sizes in both themes, with truthful completion receipts.
- [x] #3 Returning from Export preserves the originating Conversations query; source records and current Console context remain unchanged.
- [x] #4 Targeted tests, bounded native captures, artifact inspection and lifecycle evidence are recorded with clear limits.
- [x] #5 Stale export validation expectations follow the documented inline refusal contract, and row-selection coverage exercises mounted controls rather than the incomplete fake.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/147-conversation-archive-and-exact-resume.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: verify existing whole-source/explicit-selection export and local archive contracts; no schema, ownership or interaction redesign.

1. Trace Conversations entry, explicit selection, shared destination form and writer/receipt boundaries; run the directly relevant automated checks. Reconcile demonstrated stale test expectations with TASK-32251 inline destination refusal and mounted multi-select behavior, preserving or strengthening coverage.
2. Seed a private native profile with three conversations (one archived), exact message bodies and an unrelated note. From a filtered Conversations view, verify the advertised whole-source scope and selected-row scope through real export workers and FileSave controls.
3. Exercise destination cancellation, .zip normalization, overwrite and Escape return at 80x24 and 170x48 in both themes. Inspect each resulting ZIP manifest and message payload; compare source identity/version/content and Console context.
4. Inspect one batched native capture set, verify normal process exit and private persistence, and request independent evidence review. File demonstrated product defects before expanding repair scope.
5. Record limits, update the review ledger, and close the task only after its criteria pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified Conversations Export through real native FileSave and LocalChatbookService at 170x48 and 80x24 in textual-dark/light. Whole-source Export advertises and writes both active conversations despite the Alpha text filter; Export selected writes only Alpha. Picker cancellation, suffix normalization, explicit replacement and Escape return retain the intended scope/query. Eight ZIP snapshots pass independent manifest length/identity and ordered message ID/body/role/length checks; four final destinations match their replacement hashes.

No production source or CSS changed. Repaired three stale test failures in two files: replaced the incomplete fake row test with stronger assertions in the existing mounted selection journey, and aligned invalid destination tests with TASK-32251's inline refusal while preserving unrelated form fields. Baseline: 120 passed/3 failed. Final: 122 passed in 40.39s, exit 0. Native/checker/destination tests pass Ruff; multi-select retains its one inherited lint finding with no new diagnostics; edited-range formatting, evidence hashes/XML/JSON/links and diff whitespace pass. No full suite was run.

Native run-002 returns normally with exit 0 and PID 71650 independently absent before owned terminal cleanup. Ten private DB integrity checks pass; all three conversation identities/versions, six message bodies/owners and the unrelated note remain intact. Three default-profile configuration/state hashes are unchanged; logs contain no ERROR/CRITICAL lines, traceback headers or faulthandler output. Twelve captures were rendered and inspected once; independent review has no remaining findings.

Evidence: Docs/superpowers/qa/2026-09-16-conversation-export/README.md with guarded native runner, post-exit artifact checker, receipts and scope limits. The Library workflow ledger now points to exact Resume next. Programmatic focus plus Enter does not qualify full Tab traversal; native Cancel covers the picker, not cancellation during an active write. Empty Console draft preservation is bounded; attachment/graph/provider fidelity is outside this fixture.

Plan refinement: AC#5 records demonstrated stale-test repairs before editing them. An early runner assertion compared a multiline marked-up row label with compositor paint; it exited normally and was replaced by a fresh-profile run checking the identifying title plus focus. The post-exit checker strengthened native set/dict comparisons after review. A transient read-only SQLite opening failure is disclosed, with successful final mode=ro checks using the project interpreter; no unproven cause is claimed.

ADR required: no. Existing ADR-011, ADR-147 and ADR-150 govern this verification and test-only correction. No new general lesson or architecture decision was needed.
<!-- SECTION:NOTES:END -->
