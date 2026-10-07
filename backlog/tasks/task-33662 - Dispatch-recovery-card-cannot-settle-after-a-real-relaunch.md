---
id: TASK-33662
title: Dispatch-recovery card cannot settle after a real relaunch
status: Done
assignee:
  - '@claude'
created_date: '2026-10-01 21:10'
updated_date: '2026-10-03 01:40'
labels:
  - console
  - recovery
  - bug
dependencies:
  - TASK-33661
references:
  - tldw_chatbook/Chat/console_conversation_hydration.py
  - tldw_chatbook/Chat/console_chat_controller.py
  - Tests/Chat/test_console_dispatch_recovery.py
  - qa/task-33661-resend/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: found while building TASK-33661 (Resend) on 2026-10-01; this bug predates that change.

If the app quits while a reply is still in flight, the next launch shows a dispatch-recovery card ("Retry response" / "Retry anyway" / "Discard"). After a REAL relaunch, both Retry and Discard refuse with "That response recovery action is unavailable." That leaves the chat stuck: sends stay refused with "Finish or discard the pending response…", which is exactly the broken-chat case the owner wants to be able to resume.

Cause, as traced by the implementer:
- The recovery state keeps the persisted assistant message id.
- Hydration (console_messages_from_conversation_tree → _ingest_full_tree) gives every restored node a fresh native id.
- claim_dispatch_recovery_action → _message_or_raise then raises KeyError.

Reproduced in a controller test and live; see the e2/e3 captures in qa/task-33661-resend/.

Tests/Chat/test_console_dispatch_recovery.py restores nodes whose native id equals the persisted id, so the suite never sees the bug.

Until this is fixed, the Resend path "Discard, then Resend" works only within a single app session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a real relaunch, Retry response and Discard on a dispatch-recovery card both settle the recovery. Retry streams into the pending reply; Discard settles it as discarded and keeps the user message.
- [x] #2 After Discard following a relaunch, the user message offers Resend (TASK-33661), and Resend re-runs the turn in place.
- [x] #3 A regression test drives the production hydration path, where restored nodes get fresh native ids. It fails on the current code and passes after the fix.
- [x] #4 Every existing dispatch-recovery test still passes, and any test that relied on native id == persisted id is rewritten on purpose and named in the notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: regression tests that relaunch through the production hydration (fresh native ids) after a real mid-reply crash, then Retry / Discard / Discard→Resend.
2. Root-cause fix at the one point where restored nodes get native ids (ConsoleChatStore._ingest_full_tree): the recovery owners keep their persisted id as native id, the invariant a live session already holds.
3. Prove AC#1/#2/#4 with the plain pytest command against the merge base; rewrite the resend test that relied on native id == persisted id.
4. Live tmux check at 211x44 on a scratch profile with a local stub; captures in qa/task-33662-recovery-relaunch/.
5. Ratchets, census, preflight, User Guide, task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
After a real relaunch, a dispatch-recovery card's Retry and Discard now settle. The fix is in `ConsoleChatStore`: a restored recovery owner keeps its persisted id as its native id, and the claim looks only in its own session's tree.

**Root cause (confirmed).** The recovery and its checkpoint name the owner rows by persisted id. A live turn's native ids equal those ids, because `_hydrate_durable_turn_owner_messages` uses the commit's ids. The production hydration (`console_messages_from_conversation_tree` → `_ingest_full_tree`) gives every restored node a fresh native id, so `claim_dispatch_recovery_action` → `_message_or_raise` raised KeyError and returned None. The miss was not only in the claim. About 15 other sites treat the recovery's ids as native ids:
- the store: release, prepare, settle, the generation-token bind, the terminal settle, the continuation handoff, `_normalize_restored_provider_continuation`;
- the controller: `_provider_messages_for_session(before_message_id=...)`, `store.get_message(checkpoint.user_message_id)`, `begin_generation_attempt`, `_stream_assistant_response(assistant_message_id=...)`.

**What changed** (`tldw_chatbook/Chat/console_chat_store.py`, net 0 lines):
- `_ingest_full_tree`, the one place every restore gives nodes their native ids: when the session holds a recovery, its assistant owner and its checkpoint's user owner keep their persisted id as their native id. The exception is an id another session already holds.
- `claim_dispatch_recovery_action` looks the owner up in its own session's tree (`_nodes_by_session[session_id]`), not through the store-wide index.
- The line budget was paid by deleting `_persist_message_projection`, which has had no caller since 771f2dc347. That removed one diagnostic call, so `Docs/security/production-diagnostic-inventory.json` was regenerated: 94 → 93 calls. `--statements` showed that the only removed statement was that method's `logger.warning`.

**Rulings**
- **Keep the persisted id as the native id for the owners, at `_ingest_full_tree`.** The alternatives were rejected:
  - Resolving persisted → native at `_message_or_raise` fixes only lookups. The equality comparisons (`recovery.assistant_message_id == message.id`) would stay broken.
  - Recording the native id in the recovery would split it from its checkpoint. The checkpoint's ids go to the repository CAS and settlement, and every later `_hydrate_dispatch_recovery` re-reads them from SQLite.

  There is precedent: `_hydrate_provider_continuations_from_persistence` already sets `node.id = persisted_id` for restored thinking rows. Cost: in one store, only one session can hold the owner ids. A second open of the same conversation (a fresh-session open, `reuse_existing=False`) gets fresh ids. Its card refuses with "That response recovery action is unavailable.", which is the old behaviour. The first session settles it.
- **Session-scoped claim.** Keeping the persisted ids made them resolvable store-wide. Without this guard, that second open's Discard was accepted, and it rewrote the FIRST session's node (RED: `test_a_second_open_of_the_same_chat_never_settles_the_first_ones_reply`). The guard is in the shared claim, so it covers both Retry and Discard.
- **Continuation owners are included.** The rule keys on the recovery's own ids, not its kind. A restored legacy provider-continuation owner is therefore found and normalized, as the hand-built suite pins. Before, through the production hydration, it stayed unnormalized with its actions disabled (RED: `test_relaunched_legacy_continuation_owner_is_normalized_before_actions`, which reads `('accepted', 7)` at base).
- **The test relaunch uses `hydrate_console_session`.** Retry re-checks the frozen Library authority, which needs the durable policy that function hydrates, and the model the chat was saved with. The ACCEPTED-shape fixture freezes `direct_library_tools=True`, which is what this controller's own send freezes.

**Tests**
- New: `Tests/Chat/test_console_dispatch_recovery_relaunch.py` (module-level `bootstrap_profile`). The crash shapes are real: a controller sends a turn whose provider hangs, and the SQLite file is copied at that moment. A fresh store and controller then relaunch from the copy through `hydrate_console_session`. Each test first asserts that the healthy turn's nodes got fresh native ids.
  - Retry anyway (DISPATCH_STARTED) streams into the same pending reply. The provider sees the full history.
  - Discard keeps the user message and settles the reply as discarded, with no provider call.
  - Retry response (ACCEPTED).
  - Discard, then Resend, re-runs the turn in place, with and without a second relaunch. One live child row in the database.
  - The legacy continuation owner.
  - The second-open guard.
- **RED** at merge base `ef8fd5d38a`: all 7 new tests fail.
  - Six fail with `AssertionError: That response recovery action is unavailable.` At base, the second-open test fails because even the first session cannot discard.
  - The continuation test fails with `('accepted', 7) == ('continuation_active', 8)`.
  - The second-open test was also RED on the first half of the fix alone: the second session's Discard was accepted. That was before the claim guard.
- **Rewritten pinned test:** `Tests/Chat/test_console_turn_resend.py::test_resend_after_a_discarded_dispatch_recovery[False|True]`. It restored through `_restored_store` with the comment "a hydrated relaunch cannot claim the card". It now restores through the production hydration (`_restore`). At base it fails with the same "unavailable" copy.
- **Not rewritten:** the shared `_restored_store` helper (38 call sites in 11 files).
  - 30 call sites restore a conversation written by `_insert` (the `user-1` / `assistant-1` owner pair). For those rows the fix makes production assign exactly the ids that the helper hand-builds.
  - The other 8 restore other shapes: `test_console_dispatch_cursor_recovery.py`, fix rounds 1 and 4, and `test_console_trace_first_send_atomicity.py`.
  - None of these tests fails differently from base (see Evidence). The new relaunch file covers the production path instead.

**Evidence (2026-10-02, base `ef8fd5d38a`, plain `pytest <files>` with `PYTHONPATH=<tree>`).** Every base-side number below comes from a fresh `git archive` of the merge base in a task-named scratch directory. A first base tree at the generic `scratchpad/base` was overwritten and then deleted by another agent mid-run. Every base result taken from it after 15:16 was discarded and re-run.
- New file: 7 passed. With the rewritten Resend test: 9 passed at head, 9 failed at base.
- AC#4, the dispatch-recovery family plus the Resend tests (15 `test_console_dispatch_*` files + `test_console_turn_resend.py`):
  - Under the plain command, many fail at both base and head with "Hooks unavailable; review or disable hooks before sending." That is ADR-126 profile-selection admission inside the per-test sandbox.
  - With a scratch `-p` plugin that adds `bootstrap_profile` (comparison only): head 275 passed / 1 failed; base 268 passed / 1 failed.
  - The same test fails at both, with the same message: `test_console_dispatch_recovery_fix_round2.py::test_explicit_retry_resumes_every_unfinished_postcommit_effect_before_provider[checkpoint_transition]` ("Accepted turn is retained for recovery.").
  - The 7 extra head passes are the new tests.
- Covering suite: every file under `Tests/` that names `restore_persisted_session`, `hydrate_console_session`, `dispatch_recovery`, `_restored_store` or `console_messages_from_conversation_tree`. That is 91 files, run as 3 chunks per side under the plain command:
  - chunk 1: head 457 failed / 1655 passed; base 457 failed / 1648 passed.
  - chunk 2: 177 failed / 730 passed on both sides.
  - chunk 3: head 764 failed / 415 passed / 20 errors; base the same.
  - Totals: head 1398 failed / 2800 passed / 20 errors; base 1398 failed / 2793 passed / 20 errors. The 7 extra passes are the new tests.
  - The failing sets are identical, and so is every `FAILED`/`ERROR` summary line, message included. The failures are environmental and the same on both sides: ADR-126 admission (for example the "Hooks unavailable" copy above), and 300 s timeouts while a backup storage-admission thread waits.
- `Tests/Chat/test_console_local_citation_boundary.py` hangs 300 s per test at both base and head, alone too. It was run apart with `--timeout=20`: 87 failed / 8 passed on both sides, with identical summary lines.
- `Tests/UI/test_console_turn_resend_ui.py` + `Tests/Chat/test_console_message_actions.py`: 10 failed / 131 passed on both sides, same set.
- Size ratchets (`-p no:xdist`): the same 11 failures at base and head, with identical messages. `console_chat_store.py` 22550 → 22550 lines (21180 → 21180 non-blank). The controller and `console_transcript.py` are untouched.
- `_ui_ready` census with `PYTHONPATH=<worktree>`: 1033/1033, 4 passed. Base reads the same, with the same "+5/-2" snapshot-drift warning. No module was added.
- `PYTHONPATH=$PWD ./scripts/preflight.sh`: rc 0, after the inventory regeneration.

**Live check** (`qa/task-33662-recovery-relaunch/`): 211x44, scratch profile (HOME, XDG and `TLDW_CONFIG_PATH`), local llama.cpp-shaped stub. The app was killed mid-reply and relaunched each time:
- the card appeared;
- **Retry anyway** streamed into the pending reply;
- **Discard** settled it and kept the message;
- the message offered **Resend**, and `r` re-ran it in place, with one live reply in the database.

The `f` captures repeat these shapes on the final build. The real config hash and the `~/.local/share/tldw_cli` listing were unchanged.

**Found, not fixed (pre-existing at base):** quitting with Ctrl+Q → **Quit** while a reply hangs settles that reply as `stopped`, but the process never exits after unmount. The merge-base build reproduced this.

**Files:** `tldw_chatbook/Chat/console_chat_store.py`, `Docs/security/production-diagnostic-inventory.json`, `Docs/User_Guide/console/chat-basics.md` (Discard → Resend after reopening restored; **Retry response** described), `Tests/Chat/test_console_dispatch_recovery_relaunch.py` (new), `Tests/Chat/test_console_turn_resend.py`, `qa/task-33662-recovery-relaunch/`, `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`.
<!-- SECTION:NOTES:END -->
