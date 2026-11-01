# PR3298 trace-capture review repairs

> Execute inline with a final independent `superpowers:requesting-code-review` review.

**Goal:** Preserve captured sends after Edit & resend or Compact now when an unrelated completed-turn witness has the same descriptor count.

**Architecture:** Keep completed-turn and rewritten-tail proofs separate. Select a proof using its actual applicability, then retain its full validation and final transaction recheck. Preserve current native continuation custody and completed replay.

**Spec:** TASK-33621.7, TASK-33621.29, and the rewritten-tail amendment of `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`.

ADR required: no new ADR.
ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`.
Reason: Correct proof selection within the existing bounded contract; no storage, authority or runtime boundary changes.

## Constraints

- Non-draft PRs only, oldest-first merges. PR3267/3297 excluded. No release or main promotion.
- Hosted CI and size limits waived; exact measured pins, targeted local evidence and independent review remain required.
- Use this isolated worktree. Preserve all unrelated current-dev changes and concurrent remote work.
- No whole-suite runs, real providers, user profile/keyring access or guard/deadline weakening. Use native private test profiles; freeze sources during runs.
- Keep the existing 256 stored-sequence bound, immutable historical heads, whole-tail messages-only shape, branch/memory anchor, source/final-value binding, typed witness exclusivity, and dispatch recheck.
- Existing wider-history, plain compaction, effective-memory, continuation-edit and broader end-to-end follow-ups remain explicit. Correct the incoming duplicate TASK-37600 identity without changing the existing dev SSE task.

## Task 1: Repair proof selection and qualify the integrated branch

Files: `console_trace_service.py`, `console_trace_final_values.py` only as needed for integration; existing rewritten-tail planner/history tests; affected size metadata, existing tasks and QA notes.

- [x] Rebase onto merged PR3294 (`2561efd6587a263de6eb564b4254a09883f7f9b3`). Preserve both ADR amendments, native custody/ordinals and completed replay. Initial integrated head `0d646b9daad48dc1fe70b1747dea77b801bbb081`; 2,850 non-overlap dev files preserved exactly.
- [x] Add a real-ledger planner regression with equal descriptor count and unrelated source/response identities; assert a rewritten witness, no completed witness, and exact incoming descriptors.
- [x] Add real controller/store/gateway cases: dictionary-transformed last user, Edit & resend, then captured send on agent/plain paths. Assert exact trace/provider equality and one reservation using existing helpers.
- [x] Add a transformed-last-turn Compact now case on the agent path; prove the retained transformed source and successful memory update before the captured send.
- [x] Run these cases before changing selection. Investigate fixture failures separately; retain RED receipts.
- [x] Apply the smallest descriptor/range applicability correction. Do not catch arbitrary proof errors or relax completed-turn ownership checks.
- [x] Run affected rewritten-tail/history/planner and final-value files, the two existing dictionary/source-pin controls, the safe completed-history/native-rounds control, stale/rollback custody negatives, and project-turn controls if selection changes them. Check the specific workflow target pin.
- [x] Verify changed diagnostic projections and exact size pins. Resolve duplicate incoming task ID through existing backlog tooling and update its references.
- [x] Record source hashes and precise results/limitations, update task status only when its criteria are met, commit, and obtain independent exact-head review before publication.

Review focus: equal-count unrelated witnesses; matching descriptors with a wider compaction range; current-turn dictionary transformation; mixed completed/native witness types; final-binding refusal after durable state changes. Existing negative tests qualify the latter boundaries rather than introducing a new proof mechanism.

Pre-flight: one repair task; no cross-task interface dependencies. PR3295 will merge first and requires a final refresh/rebase before publication.

RED qualification: private receipt `../r-98c08663` froze all selected trace sources. The equal-count planner failed with `completed_tool_turn_unavailable` as expected. The three send cases stopped at their transformation assertion: the inherited dictionary fixture accepted two arguments, but current dev supplies a third frozen-context argument and catches the resulting TypeError. Correct that fixture signature and rerun those three cases before claiming their production failure. The two existing Ruff F811 diagnostics in that fixture file are unchanged.

Corrected RED: `../r-558a3b87` ran the three send cases with unchanged source manifests: both edit paths failed at `completed_tool_turn_revision`, and compaction failed at `completed_tool_turn_range` (three failures, zero errors). The fix now uses exact descriptor matching and the shared permitted-start predicate before selecting a completed proof. Its full durable validator remains mandatory and unchanged in meaning. The independently proven rewritten-tail fallback remains bounded by the existing gates.

Task identity: the incoming trace follow-up is now TASK-38298; the existing dev TASK-37600 SSE fix remains untouched. The read-only 300-snapshot filename audit found maximum 38268, and the local filename/frontmatter helper confirms 38298 was free. The trace ADR, guide and two owning tasks now reference the corrected ID.

Prepared integration: all 99 selected cases have passing results across frozen receipts, including native completed replay and stale/rollback negatives after correcting private-profile/signature/Windows socketpair fixture issues. Detailed failures and qualifications: `Docs/superpowers/qa/2026-10-31-pr3298-backlog-integration.md`. Production source review and fixture review report no blockers; final rebase onto PR3295 merge and exact-head approval remain.

Final source review cleared integrated head `1c4f01847871f61baceb095ba3e35ca2659330d3` on `3926ab8e672cef0b7ab8eba20b7e5c8800f4b929`. Native completed replay and critical workflow-pin checks passed2/2 with frozen sources (`../r-6ebc2c79`). Final completion metadata is reviewed before publication; hosted CI and release remain outside this task.
