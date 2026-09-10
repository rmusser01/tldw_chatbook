# Console archive recovery verification — 2026-09-10

PR integration base: `origin/dev` at `068535986ed005e85045cf6e08397c372377cb28`.
Branch: `codex/console-archive-recovery`.
ADR: [147](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md).

The implementation was ported from the original checkout using feature-only deltas. It preserves current dev's asynchronous Console hydration, canonical Library Reader, switcher modes, workspace persona annotations and tree-wide close-loss accounting. The archive migration is 70→71. No unrelated changes from the original checkout are included.

## Passing focused checks

- Archive persistence/service/action tests: 21 passed. Real SQLite covers archive/restore/Undo, restart, counts, paging, body search, retained identity/history, concurrent writes and cancellation-safe reservations.
- Conversation service and character seek compatibility: 100 passed, including archive scope combined with workspace-union and seek filters.
- Console archive publication/action checks: 15 passed; refresh ownership, intervening archive receipts, in-flight changes and navigation away during hydration are covered. Existing cache-neighbor checks also pass.
- Library state/canvas regressions: 83 passed. Five new recovery checks, nine canonical Reader regressions and nine source/link compatibility checks pass; mounted tests cover saved-body search, complete-transcript Find, archive/Undo, separate source reuse and original Resume at 100×30 and 160×44.
- Four Console workflow cases pass: compact/wide archive → Undo → exact resume → full search; and actual Restore & resume confirmation → simulated next send for both a global conversation and an archived workspace with a reused name. The simulated response and prior transcript persist under the original conversation ID, and another draft stays intact. Fixtures use the real Console persistence path, including current dev's durable Library policy metadata.
- Twenty existing switcher trust/reuse/mode regressions pass. Four additional activation-boundary and compact-viewport checks pass.
- Selected workspace registry, Console lifecycle, close and Settings tests pass. Settings tests wait for the exact rebuilt controls instead of fixed delays.

## Verification limits

These are targeted runs, not a full-suite result. Next-send checks use a deterministic provider gateway; they do not establish live provider behavior. Compact/wide SVG captures were inspected for review/action placement. Minimal Library harness captures deliberately lack unrelated source services and are not evidence of source-service availability.

A broader historical-migration run was stopped after encountering failures outside the archive tests, and its interruption also caused pytest teardown errors. That run is not counted as passing evidence. Migration failure triage is recorded below. All new archive migration checks pass against isolated databases; no real user database was migrated.

### Historical-migration triage

The interrupted run showed 44 failing cases. Three version/column expectation tests were corrected to distinguish their historical migration from the current schema, and all three pass in isolation.

Forty cases point to unchanged dev fixtures/behavior: 28 v41 partial objects lack `_db_diagnostic_ref` (the migration implementation is unchanged and both default-shape cases reproduce on dev); 11 thinking/quick-note/retention cases reproduce using unchanged dev DB code (existing semantic-authorization triggers or modern methods called against historical schemas); one bare-open content-hash case also reproduces on dev. These are outside the archive change.

One bare-open SIGKILL test did not reach its v48 child marker within 60 seconds. Its child-process baseline was not established, so it remains unverified rather than being described as a pre-existing failure. The broader interrupted run is not a clean test-suite result.

### Static and UI checks

New Python files pass Ruff and formatting. Changed existing files add no Ruff findings over the dev baseline; fatal-error checks and `git diff --check` pass. Consolidated CSS was regenerated successfully. A final cross-module review was performed before publication.

Final review correction: archive refusal now checks all session branches, including hidden unsaved/pending messages. The new regression failed before the fix; all 13 action tests pass afterward. The focused re-review checks this correction.


## PR #2576 Qodo review corrections

The first Qodo review identified thirteen actionable findings. The corrections cover:

- Portable screenshot paths; guarded Reader child lookups during recomposition; independent conversation/workspace lifecycle assertions.
- Capturing unsent Console text for its owning session before archive checks; invalidating saved-row caches after workspace lifecycle changes; clearing only the restored workspace's Undo receipt.
- Recoverable initial workspace-read failures during Resume, retaining original-ID retry behavior.
- One executable migration artifact; local-only archive transitions with optimistic Undo versions; bounded exact-title pagination and explicit service scope-forwarding assertions.
- Shared strict archive-scope validation and public UI handler docstrings.

ADR147 remains the governing decision. Review verification uses targeted tests; the historical-migration limits above remain applicable.

Review verification: the combined DB/service/scope/Library run passed 117 cases; the shared action/workspace regressions passed 25. The 76-case mounted UI run passed 74 and exposed two test assumptions: exact row equality omitted the intentional workspace label, and the rename test observed the screen before its action control mounted. After correcting those expectations, all six focused rerun cases passed, including both full restore/resume/send workflows. Changed files introduce no Ruff diagnostics over the reviewed head; new files pass Ruff, and diff checks pass.

The retention review also exercised existing privacy and maintenance coverage. Two raw hard-delete fixture cases stop at existing message semantic-authorization guards before reaching retention; both failures reproduced using the unchanged pre-review DB implementation. They are excluded from the focused retention acceptance run, not counted as passing. Archive-cycle, stale-Undo, late-obsolete-payload, soft-delete, restore and maintenance checks exercise the revised retention boundary.

Final archive/retention acceptance run: 41 passed, 2 baseline-reproduced hard-delete cases deselected.


### Latest dev integration

Rebased onto dev at `98704acc283b2e09229c7b9b475153c372a581db`, retaining the shared Library pager and explanatory keyboard behavior from PR2569. Source reuse keeps workspace membership checks and its own disabled marker; Resume uses the loaded original identity and fresh storage reads. A stale list no longer silently vetoes an otherwise eligible Resume.

The rebased core/architecture/pager run passed 86 cases. The mounted UI run passed 79 of 80; its remaining failure was the archive test observing a pushed confirmation screen before its button mounted. The test now waits for that action control. All 17 final handoff tests pass, including button/key dispatch with stale browse state and current-storage deleted-record refusal.

Final compact/wide recovery rerun: 3 passed after waiting for actual actionable controls across both modal mounting and canvas recomposition. Conversation characterization: 3 passed. No unresolved failures remain in the targeted acceptance checks.


### Second Qodo review and required CI correction

The next review added fourteen findings. Corrections cover asynchronous workspace reads/writes and retained Undo receipts, failure/retry feedback, concurrent-restore conflict reporting, shared workspace-name validation, archived-name import conflicts, complete public contracts and typed recovery records, contextual recovery diagnostics, confirmation preflight and explicit partial-recovery reporting, and current-screen checks during existing-session Resume.

A durable send-state read now replaces a stale cached archive flag, guarded against a local archive transaction completing during the read. The synchronous admission gate no longer treats a presentation cache as authoritative; archived conversations are still refused by the awaited check before provider work. Mounted restored-conversation workflows verify the original history can continue despite an obsolete cached archived flag.

The required derived-artifact job failed because the new archive index lacked a census entry. Actual production Library count/page queries use `idx_conversations_archive` with no `sqlite_stat1`, verified for Active and Archived on a 128-conversation corpus. The index is now plan-pinned in `scripts/index_plan_pin_census.tsv`.

Diagnostic inventory review used the checker’s statement comparison against its committed inventory baseline: six new structured archive exception calls contain validated conversation/workspace identities only; the workspace changes are constant-text archive-refresh and switcher-failure diagnostics. No content, credentials, filesystem paths or new sink destinations were added. Regenerated the inventory only after that review.

Targeted integration evidence so far: 108 registry/name/import/boundary/Library/index checks passed; 25 Console workflow and shared archive-action checks passed. The latter includes the corrected synchronous/durable admission contract. The earlier historical-migration and raw-delete fixture limitations remain unchanged.

Final second-review verification: 108 core/backend/Library/boundary/query-plan tests and 25 Console/send-action tests passed. The expanded workspace/Settings run passed 46 cases, including both started-write cancellation cases. Its sole failure, `test_compact_overview_keeps_a_painted_recovery_action`, also fails when run with the prior HEAD Settings implementation: it inspects the Theme button inside the hidden compact inspector (zero region). No archive change touches that layout, and experimental test-only waits were reverted. This pre-existing overview assertion is not claimed as passing. Diagnostic inventory regeneration was followed by a successful check (596 owners, 7665 TASK-494 calls); changed-file Ruff diagnostic delta is empty and `git diff --check` passes. Latest fetched dev remains `98704acc283b2e09229c7b9b475153c372a581db`.

A subsequent remote check found dev at `41d14d1f741c91a1a0e035c1dce05d63299e2bbf` despite GitHub still reporting the earlier PR base OID. Rebased all four commits onto that fetched ref; range-diff confirms the only conflict resolution was the additive diagnostic census total (7666). Post-rebase Library/Console recovery, handoff, and pager regression run passed; CSS reproduction, index-plan census, and diagnostic inventory checks also passed.

### Third Qodo review

Addressed six findings on `1a346f83`: Trash/deletion-inclusive searches include deleted chats independently of their archive flag; failed new Library changes clear prior Undo targets while failed Undo preserves its retry; existing-tab Resume rereads saved conversation/workspace state and routes archived targets through restore confirmation; even initially active requests revalidate before staging. Restore as serializes the Unicode case-folded duplicate-name check under an immediate write transaction. Reader composition and in-place updates label workspace recovery consistently, and the archive navigation API documents both parameters.

Evidence: six initial database/receipt/Unicode regression cases failed before fixes; two new UI/recovery regressions also failed before fixes. Final core run passed 138 cases, boundary run passed 20 (including the additional initially-active race), and mounted/handoff verification passed 21 distinct cases (19 initially passing plus the corrected call-count cases in a 4-pass rerun). Production next-send flow passed in both compact and wide layouts. The first combined run was interrupted by host disk exhaustion and is not counted as evidence; only this task's completed temporary test directory was removed before clean bounded reruns. Changed-code lint and whitespace checks pass; diagnostic inventory passes without regeneration. Existing documented baseline limits remain. ADR147 clarifies Trash scope rather than introducing a new lifecycle.

### Latest-dev Backlog collision

Backlog Guard found two TASK-32273 records after the reasoning-history rebase. Preserved the landed reasoning task and renumbered this PR's unmerged archive lifecycle record to TASK-32300, carrying provenance and updating its plan/ADR references. The other archive records remain TASK-32274–32276. This is task metadata only; no runtime behavior changed.

### Fourth Qodo review

Six further findings are addressed: resume intent IDs use the shared Pydantic boundary without changing their exact spelling policy; in-memory workspace enrichment stays on the SQLite owning thread; the persisted-session close test confirms the action and proves the original SQLite conversation/message remain available. Both warm and cold resume paths reread conversation/workspace lifecycle state. Recovery callbacks carry a revision guard and request token through confirmation and storage completion so an older restore cannot replace or navigate over a newer Resume. Console archive-receipt Undo delegates to the same async, error-handled restore path with an expected-record check.

Evidence: three focused regressions failed before fixes and then passed. The final focused suite passed 99 tests; both confirmation-time and write-completion supersession cases pass (2-case run, including one added case). Confirmed close with real SQLite passed, and 14 mounted Console archive/workspace lifecycle tests passed. Read-error receipt recovery is covered through the actual receipt callback; memory-backed enrichment is tested using WorkspaceDB(":memory:"). Scoped lint/format/whitespace checks pass; persistent diagnostic inventory verifies without regeneration. Existing ADR147 remains the governing design.

Fourth-review integration rebased onto dev `0292293d25373669ac53a00b3d5a321f59253aaa`; all code applied unchanged and both appended testing lessons were preserved. 25 post-rebase Library recovery/import tests passed; Backlog ID and CSS reproduction checks pass. Corrected the remaining archive testing-lesson reference to TASK-32300.

### Fifth Qodo review

Cancellation during Console send admission returns only that attempt's unaccepted keyboard stash to its owning session, retaining later edits and other tabs. Console conversation archive and Undo retain completion tasks so a committed write still invalidates caches and offers recovery; navigation suppresses stale modal publication and reports the completed archive with the Archived chats recovery route.

The import conflict test now checks that the renamed conversation was actually created. Its prior broad success assertion hid an unexpected-keyword failure; the strengthened assertion failed before correcting the mock contract. Settings recovery tests wait for durable state and mounted controls, and off-loop storage tests use blocked storage/event-loop signals rather than a scheduling-speed threshold. Console screenshots use per-test temporary directories. Archive-state batching has one named bound, public archive controls document their arguments, and ADR147 distinguishes archive guards from explicitly confirmed session close.

The claimed failure in `test_existing_resume_releases_claim_after_navigation` does not reproduce: releasing a current claim clears its in-flight marker and requeues the same revision. The unchanged test passes; changing it to expect a missing retry would violate the intended recovery contract.

Verification: the four new parent cancellation regressions failed before fixes; 49 send-snapshot/cancellation/recovery boundary tests passed afterward, followed by four passing final cancellation cases after persisting restored visible drafts. The consolidated import, workspace, Settings, switcher and test-reliability run passed 52 cases. These targeted runs do not supersede the baseline limits documented above.

Library verification passed 134 reader/recovery/handoff/controller cases and 20
archive-review cases. Additional same-ID/new-generation ownership coverage failed
before its guard, then all 33 affected reader/recovery cases passed. Lifecycle
completion updates the retained metadata, versions and page rows without
invalidating a newer reader request. Find waits for the target row before
reporting a match and reports a message number rather than a misleading
normalized-character offset. Mounted source-action verification asserts the
actual payload and distinguishes it from Resume.

Integration review added a closed-owner send-cancellation regression: removing
the session during preflight raised KeyError instead of propagating cancellation.
The handler now re-resolves the live owner before any restoration; all six final
cancellation tests pass, including closed-background and closed-still-visible
cases. The original accepted/consumed stash remains untouched.

Final workspace verification passed 47 cases (24 new cancellation, storage,
receipt and request-ownership regressions; 18 existing review cases; five mounted
lifecycle cases). Strong application-owned references retain started completion
through cancellation. Owned conversation reservations last until workspace
archive settles, and generic storage failures have recovery feedback. Console
receipts remain accessible after a late dialog dismissal or the next workspace
switcher visit. The late existing-session Resume branch carries its predicate
through token preparation, with positive and superseded controls.

Two additional pre-existing resume fixture failures were checked using the exact
HEAD opener/resume methods in the same harness: `_NoMountScreen` lacks
`_build_console_provider_selection` before the feature's ownership check runs.
Both failures reproduce unchanged; they are not included among passing cases.
The baseline substitution script and log are retained as
`/private/tmp/archive-v5-workspace-resume-head-baseline.{py,log}`.

Final scoped Ruff comparison adds no diagnostics. CSS reproduction, the 283-index
census, Backlog ID uniqueness, whitespace, and the diagnostic inventory pass.
The inventory now verifies 596 owners and 7672 TASK-494 calls; newly added logs
contain fixed operation descriptions without user content interpolation.

Final combined Console cancellation, workspace completion and mounted original-ID
archive/Undo/resume/send run: 34 passed.

Fifth-review rebase onto dev `0741f53ea9c28bf709b950eeeacfab2ee3476f8a`
preserved both independently appended testing lessons. Post-rebase incoming
Library notes onboarding plus archive reader/recovery tests passed 19 cases;
all 3671 Backlog records remain unique and Windows-compatible.
