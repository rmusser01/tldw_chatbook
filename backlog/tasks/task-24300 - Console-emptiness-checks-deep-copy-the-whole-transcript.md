---
id: TASK-24300
title: >-
  Console emptiness checks deep-copy the whole transcript, making typing O(N) in messages
status: In Progress
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - console
  - chat
priority: high
dependencies: []
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add O(1) `message_count` / `has_messages` and a lazy `iter_messages_newest_first` to `ConsoleChatStore`.
2. Convert the four predicate sites and the two reverse scans.
3. Guard by CALL COUNT (wall clock is unusable here), and mutation-test the guard.
4. Extend the mounted census to count all transcript snapshots and projection traversal, not only `messages_for_session` calls; establish the failing baseline.
5. Add a store-owned projection revision that advances on transcript/payload, streaming, and late usage mutations. Keep screen-owned settled context and cost aggregates keyed to that revision and session/settings identity; count draft text as a separate incremental contribution.
6. Test invalidation and display parity for late terminal usage, edits, and branch changes. Run focused mounted/projection/store tests and lint only.
7. Rebase integration: preserve TASK-33081's one-second active-stream context estimate bound with a constant-time key checked before transcript materialization; retain immediate draft/payload/settings/run-owner invalidation and test the combined cache behavior.

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

The census records maximum estimator input rows because Textual coalesces a variable number of one-row draft repaints; summing those calls gave 19 versus 17 with zero history work. The exact comparison and zero-history assertions remain. Windows widget admission receives a 120 s watchdog before the burst after the default 30 s wait timed out; this changes no work assertion. Task status remains In Progress pending coordinated commit, rebase, push, and protected CI.
<!-- SECTION:NOTES:END -->
