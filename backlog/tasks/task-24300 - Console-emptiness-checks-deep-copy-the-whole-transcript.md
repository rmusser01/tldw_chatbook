---
id: TASK-24300
title: >-
  Console emptiness checks deep-copy the whole transcript, making typing O(N) in messages
status: Done
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add O(1) `message_count` / `has_messages` and a lazy `iter_messages_newest_first` to `ConsoleChatStore`.
2. Convert the four predicate sites and the two reverse scans.
3. Guard by CALL COUNT (wall clock is unusable here), and mutation-test the guard.
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
<!-- SECTION:NOTES:END -->
