# Console Current and Next-Send Spend — Completed Implementation Record

The approved [design](../specs/2026-09-04-console-current-and-next-send-spend-design.md) shipped in [PR #2397](https://github.com/rmusser01/tldw_chatbook/pull/2397), merged into `dev` as `c418d4516c` on 2026-09-05 UTC.

This record supersedes the unmerged local implementation checklist. [TASK-31591](../../../backlog/tasks/task-31591%20-%20Close-Console-context-spend-workstream-and-repair-context-control-tests.md) owns documentation reconciliation and the six pre-existing context-control test repairs. The original local TASK-31382 collided with dev's unrelated ask_user-attribution task; that existing task is unchanged. The closeout itself was renumbered from TASK-31568 after the older Library Reader task landed during CI; its task record preserves that provenance.

## Completed scope

- [x] Preserve inherited compaction settings and persist deliberate Off → Automatic reselection.
- [x] Show context fullness, settled Current spend, and estimated additional On next send input charge with exact full/compact labels.
- [x] Separate draft/staged inputs from settled spend, preserving independent price and context fallbacks.
- [x] Handle dispatch/preparation/recovery ownership, including persisted versus transient IDs and provider-reported usage replacing local user estimates.
- [x] Filter historical media through canonical provider admission, capabilities, and image budgeting without serializing bytes.
- [x] Coalesce idle draft refreshes, cancel during active sends/teardown, and update labels on resize.
- [x] Address all six Qodo findings and the introduced startup import regression.
- [x] Verify required CI and merge the implementation into `dev`.

## Final implementation decisions

Pure formatting/history logic lives in `UI/Console_Modules/console_spend_projection.py` and the existing cost/context builders. Controller wiring owns the refresh timer. ChatScreen supplies captured live inputs and uses metadata-only provider rows for media admission.

The initial plan's raw historical-attachment scan was corrected during review: excluded media must not suppress available text estimates. Capturing send configuration for an empty chat was also removed after the UI-ready census exposed eager RAG imports. Neither correction changed the approved additional-input-charge semantics.

ADR required: no.
ADR paths: [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md), [ADR-095](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md).
Reason: bug fixes and display/test refinements inside accepted ownership and policy boundaries.

## Verification at PR #2397 merge

- 204 focused cost, status-chip, wiring, projection, rail, and mounted-screen tests passed.
- After the startup fix, 10 UI-ready census/media regressions passed at 972/972 modules with the existing limit unchanged.
- Scoped Ruff lint/format, compilation, CSS reproduction, and diff checks passed. Reviewed diagnostic changes were recorded in the generated inventory.
- Qodo reported zero bugs and zero rule violations on final head `24bffc3f40`; all review threads were resolved.
- GitHub fast-lane, derived-artifact, CSS, platform-import, and performance checks passed before merge.
- The additional context-control suite had 25 passes and six failures identical to clean dev. Those stale fixtures/assertions are the follow-up work in TASK-31591, not a feature regression.
- The full repository suite was not run; repository policy requires explicit opt-in.

## Cleanup scope

After the closeout PR merges, remove only this workstream's temporary worktrees and topic/backup branches. Preserve a recoverable copy of the abandoned conflicted port and any superseded local task/design records before removing them. The shared main checkout and all unrelated worktrees, edits, branches, and task records remain outside cleanup scope.
