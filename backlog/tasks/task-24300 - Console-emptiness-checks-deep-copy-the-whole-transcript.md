---
id: TASK-24300
title: >-
  Console emptiness checks deep-copy the whole transcript, making typing O(N) in
  messages
status: In Progress
assignee: []
created_date: '2026-08-28 23:30'
updated_date: '2026-09-28 03:28'
labels:
  - performance
  - console
  - chat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleChatStore.messages_for_session` materialises every stream buffer and returns a fresh
snapshot of every message in the session. Four call sites use it purely as a predicate --
"does this session have any messages?" -- and one of them sits on the composer keystroke path,
where it runs 3.27 times per printable key.

Measured on dev `3a3383123e` (40 keys, app-side cProfile attribution restricted to `tldw_chatbook`
frames): the draft-edit handler costs 6.51 ms/key on an empty conversation and 39.70 ms/key at
400 messages, of which `messages_for_session` is 0.005 ms and 34.32 ms respectively. The cost is
pure O(N) in transcript length and is paid on every keystroke.

Precondition, verified rather than assumed: the guard above the hot call short-circuits when
`session.has_user_work` is true, and appending messages does NOT set that flag -- only renaming a
session, replacing its settings, or persisting a non-empty draft do. A session restored from
screen state comes back with the flag false. So this fires for resumed conversations and for
sessions still on untouched defaults, not for every user on every keystroke.

The two `reversed(messages_for_session(...))` scans are the same defect in a second shape: they
snapshot N messages in order to walk backwards and usually stop at the first match.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A session-emptiness question is answerable without allocating a snapshot of the transcript
- [x] #2 The four predicate call sites no longer materialise message snapshots
- [x] #3 Typing in a 400-message conversation costs no more per keystroke than typing in an empty one, measured by app-side attribution and pinned by call count rather than wall clock
- [x] #4 The reverse scans stop at the first match instead of snapshotting the whole transcript
- [x] #5 A guard fails if a predicate-shaped use of the snapshot API returns to the keystroke path
- [x] #6 Mounted typing with a settled 400-message transcript performs constant transcript/context/cost projection work per key, with an unchanged-tick census that fails on any full-history walk
- [x] #7 Context and spend displays remain exact after append, edit, branch switch, streaming, and usage attached after a terminal answer
- [x] #8 Context and spend projections refresh when provider continuation publication changes assistant history eligibility or final content
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add O(1) `message_count` / `has_messages` and a lazy `iter_messages_newest_first` to `ConsoleChatStore`.
2. Convert the four predicate sites and the two reverse scans.
3. Guard by CALL COUNT (wall clock is unusable here), and mutation-test the guard.
4. Extend the mounted census to count all transcript snapshots and projection traversal, not only `messages_for_session` calls; establish the failing baseline.
5. Add a store-owned projection revision that advances on transcript/payload, streaming, and late usage mutations. Keep screen-owned settled context and cost aggregates keyed to that revision and session/settings identity; count draft text as a separate incremental contribution.
6. Test invalidation and display parity for late terminal usage, edits, and branch changes. Run focused mounted/projection/store tests and lint only.
7. Rebase integration: preserve dev's one-second active-stream context estimate bound with a constant-time key checked before transcript materialization; retain immediate draft/payload/settings/run-owner invalidation and test the combined cache behavior.
8. Independent review fixes: prove warm history invalidates on ordinary and dispatch provider-continuation publication, then advance the display revision at those publication points. Preserve lazy tokenizer import and character-estimate fallback when an installed tokenizer cannot import. Run failing regressions before source fixes and scoped verification afterward; retain the recorded mounted census because unchanged typing behavior is unaffected.

ADR required: yes
ADR path: backlog/decisions/190-console-incremental-display-projections.md
Reason: the store-to-screen invalidation contract and projection ownership are cross-module interfaces; ADR-088 governs history selection and remains authoritative.
Reservation: ADR-188 was collision-free before implementation; a later live PR (#2862) claimed it. ADR-190 was checked against origin/dev 7cda012822 and all live PR file lists on 2026-09-27 and reserved for this task before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added three accessors to `ConsoleChatStore` beside `messages_for_session`:
`message_count` and `has_messages` (O(1), reading the same active-path view the
projection walks, so the two can never disagree about emptiness) and
`iter_messages_newest_first` (materialises and snapshots one message at a time,
so a caller that breaks on the first match pays for one).

Converted six call sites: four emptiness predicates
(`Console_Modules/session.py` x2, `Console_Modules/message.py` x2) and two
newest-first scans (`chat_screen.py`, `Console_Modules/prompt_queue.py`).

**Measured, deterministic (call counts, not wall clock).** Per printable
keystroke at 400 messages: `messages_for_session` 3.27 calls/key -> 0, and
1,310 message snapshots/key -> 0. Interleaved wall-clock A/B against a
pristine merge-base worktree, 3 rounds: draft-edit handler 16.61 / 11.74 /
10.97 ms per key -> 0.727 / 0.734 / 0.706. The fixed arm's variance collapsed
because the term that scaled with the transcript is gone.

**The precondition was verified, not assumed.** The guard above the hot site
short-circuits on `session.has_user_work`, and appending messages does NOT set
that flag -- only renaming a session, replacing its settings, or persisting a
non-empty draft do. Typing does not set it either (the draft lives on the
composer widget; `store.session_draft` stayed empty through a 10-key burst).
So this fired for resumed conversations and sessions on untouched defaults.

**Trade-off.** `messages_for_session` is unchanged -- 49 genuine full-list
consumers still need the snapshot. The fix is additive.

Files: `Chat/console_chat_store.py`, `UI/Console_Modules/session.py`,
`UI/Console_Modules/message.py`, `UI/Console_Modules/prompt_queue.py`,
`UI/Screens/chat_screen.py`, `Tests/Chat/test_console_chat_store_message_counts.py` (new),
`Tests/Performance/test_console_keystroke_work_census.py` (new).

Review continuation (2026-09-27): added [ADR-190](../decisions/190-console-incremental-display-projections.md)'s store-owned display revision and screen-owned settled history/context/spend projections. Unchanged draft edits count only the live draft. Payload, active-path, streaming, and terminal usage mutations invalidate the detached projections; late usage does not need a payload revision. Pricing catalog replacement and settings changes also invalidate settled spend.

Focused evidence: the mounted empty-versus-400-message census passed with exact census equality (397.77 s), zero transcript snapshot/history/cost/context row traversals in both arms, and at most one draft row in any context estimate. The one-app 400-message guard also passed (272.51 s). Store/context/history mutation and token-parity cases and all six cost-cache regressions passed in scoped runs, including a terminal usage update that reprices Current to $0.60 without changing the payload revision. Focused Ruff/format checks and edited source ranges passed; no full test sweep was run.

The census records maximum estimator input rows because Textual coalesces a variable number of one-row draft repaints; summing those calls gave 19 versus 17 with zero history work. The exact comparison and zero-history assertions remain. Windows widget admission receives a 120 s watchdog before the burst after the default 30 s wait timed out; this changes no work assertion. Task status remains In Progress pending independent review, push, and protected CI.
Rebased onto dev `7cda012822a9ea26be84b446ec68d83e48b299c5`: the two original performance commits replayed unchanged. Resolved the testing-lessons append conflict by preserving both entries, and integrated the active-stream context TTL with the incremental projections. The one-second key is checked before snapshots; payload/settings/run-owner changes and equal-length draft edits still invalidate immediately. The newer tool lifecycle, deferred stream persistence, context-window probe, and other stall fixes remain in the base. The added cache-order regression plus focused store/config/tokenizer tests and three upstream deferred-persistence cases passed together (36 tests, 6.61 s). All six mounted cost-cache tests passed (330.77 s), including late terminal usage and warm projected cost. The first post-rebase two-app census passed (373.71 s); direct cost aggregation and warm-projection estimator row counters were then added to cover aggregate-only walks over cached rows, and both exact census dictionaries are retained as child JUnit properties.

The bundled-tokenizer child now removes inherited `TIKTOKEN_CACHE_DIR` and `DATA_GYM_CACHE_DIR` overrides. A combined run exposed that earlier parity cases had selected the bundled cache in their parent; the child inherited that override and correctly bypassed import-hook arming, making its bundled-hook assertion order-dependent. Removing the overrides inside that specific child restored the combined 36-test run. Production override behavior is preserved.

The final strengthened post-rebase two-app gate passed (384.60 s; 382.96 s call). Each arm typed 24 keys; the loaded arm was verified to contain 400 messages before measurement. Both child JUnit properties recorded this exact dictionary:

```json
{
  "context_estimate_max_rows": 1,
  "context_rows": 0,
  "cost_projection_estimate_rows": 0,
  "cost_rows": 0,
  "cost_snapshot_rows": 0,
  "messages_for_session": 0,
  "settings_readiness_builds": 0,
  "snapshot_rows": 0,
  "snapshots": 0,
  "spend_history_rows": 0,
  "template_default_builds": 0
}
```

Post-rebase static evidence: modified test files passed Ruff, eight full files plus the added gateway-method range passed formatting, source undefined-name checks and the edited context-method range passed, and `git diff --check` passed. The cost-chip helper file retains three pre-existing assertion formatting differences outside the edits. No full test sweep was run. ADR-190 was rechecked against the actual decisions trees on all 24 live PR heads and live dev `3af9f9121d27d3dd637ebdd345afae7b9d67ef11`: no collision. This branch remains based on the requested `7cda012822` for independent review and has not been pushed.

Independent review corrections (2026-09-27): a materialized assistant prefix followed by an ordinary ToolBatchReady event kept the display revision at 8, leaving cached request history eligible while the fresh lifecycle projection excluded the active continuation. FinalContinuation also changed eligibility/content without invalidating the cache. Both ordinary publication and committed dispatch handoff now advance the display revision when live fields are published, including before a separate durability barrier can report failure. The screen cache and one-second streaming TTL are unchanged.

Four regressions failed before the source fixes: warm ordinary ToolBatchReady and FinalContinuation (8 > 8), committed dispatch handoff (2 > 2), and installed tokenizer metadata with a deferred ImportError. All four passed after the fixes (7.47 s). The tokenizer import is inside the existing encoding error boundary, preserving character estimation when an optional native dependency cannot load.

Fresh focused verification: 46 tests passed in 21.65 s across the store count, default-settings memo, context parity, display history, config-path memo, lazy tokenizer, selected deferred persistence, continuation ownership/durability and dispatch failure/settlement cases. Full checks passed for the two edited regression files; source undefined-name checks, edited source/handoff format ranges and git diff --check passed. Reports: D:/Codex-UAT/pr2196-review-fixes/{red,green,focused}.xml. No full suite or long mounted cohort was repeated. The recorded mounted exact-census/cost results remain evidence on 6c98a43a8a329ce0b14bfee7ecc3fd8a7abb3f10, distinct from these fresh review-fix runs.

ADR required: no new ADR. Existing ADR-190 governs continuation display invalidation; the tokenizer change restores the existing optional-import fallback without a new runtime boundary. Added the materialized-baseline lesson in backlog/docs/lessons-testing-evidence.md. Task remains In Progress pending final integration review, push and protected CI.

Final local rebase (2026-09-27): inspected live dev 97b16d4fb684f2c1fc5b92d8906e12a1cdba2e4e before replay. Its bounded delta adds external MCP character reads and extracts application destination/handoff helpers; the only shared source path is config.py's unrelated expose_character_tools flag, away from the path-resolution memo. ADR-190 remains unallocated in dev. All five existing PR commits replayed cleanly, and range-diff reported identical patches; the review fix became 7b041d35164f398a126369c1e07e2b0118aa72f6.

Fresh post-rebase evidence: the same 46 focused cases passed in 21.73 s (D:/Codex-UAT/pr2196-review-fixes/focused-rebased.xml). Source undefined-name checks, full lint/format checks for the two regression files, edited continuation/tokenizer/handoff format ranges and base-to-head diff --check passed. These are fresh results on the latest dev integration; the long mounted O(1)/cost evidence remains explicitly tied to the earlier 6c98a43a8a head and was not repeated. Only this evidence note follows the tested source commit. The checkout is kept local for final integration review; no push or merge was performed, and protected CI for this new head is pending.
<!-- SECTION:NOTES:END -->
