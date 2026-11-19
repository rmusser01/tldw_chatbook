# PR3309 recovery integration implementation plan

> For agentic workers: use superpowers:executing-plans for inline execution of this existing review repair, followed by the user-requested independent requesting-code-review gate.

**Goal:** A legacy/imported chat whose missing Library policy is repaired at commit must retain ordinary response recovery after restart.
**Architecture:** Preserve the existing atomic commit and original dispatch checkpoint. Extend only the established first-save authority comparison to recognize the unchanged safe missing-row repair; keep all other frozen authority comparisons.
**Tech Stack:** Python 3.12, SQLite, pytest, Textual.
**Spec:** TASK-33620.3 AC4 and new restart/rollback criteria; ADR-079 Console Library authority; initial review at `C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3309-initial-review.md` (sibling worktree evidence).

ADR required: no. Existing `backlog/decisions/079-console-library-conversation-authority.md` and `033-application-session-state-ownership.md` govern this correction. No new storage or permission boundary.

## Global constraints

- No draft PRs, PR3267 or PR3297, release or main promotion.
- Rebase first, focused local tests, independent review, then merge under the user's hosted-CI and size waivers.
- Never widen Library access: only source missing/revision None with Never/Blocked may normalize to durable revision1; every other frozen field must match.
- Preserve native physical retirement, reservation ownership, accepted-row identity and transaction rollback.
- Use private test profiles, freeze source during test runs, and retain raw receipts. No full sweep.

## Review focus

Restart after accepted versus dispatch_started, real hydration's new native IDs,
changed durable policy revision, changed destination/Library authority, and late
transaction failure are exercised below. Existing background-start and custody
exit follow-ups remain explicit and are reviewed as separate pre-existing paths.

### Task 1: Repair response recovery after a safe missing-row commit

**Files:** `tldw_chatbook/Chat/console_chat_controller.py`,
`Tests/Chat/test_console_dispatch_recovery_relaunch.py`,
`Tests/Chat/test_console_dispatch_recovery_fix_round1.py`,
`Tests/Chat/test_console_durable_turn_acceptance.py`.
**Interfaces:** Existing static `_dispatch_authority_matches(current, frozen) -> bool`; no new public API.

- [x] Add a parameterized real SQLite/relaunch regression for accepted and dispatch_started checkpoints after a missing-policy commit. Assert repair row(0,0,1), frozen source missing, real hydration, one retry provider call, and identical durable user/assistant IDs.
- [x] Run that focused test against the rebased unfixed candidate. Expected: recovery refuses with changed Library authority, not setup/fixture errors.
- [x] Extend the matcher to admit only missing/None Never/Blocked -> durable/1. Preserve existing new_session normalization and full equality of all other fields.
- [x] Add negative controls for later revisions, unsafe/changed policy, unavailable/temporary source and destination changes; add a real late-commit failure rollback control using the existing missing-row fixture. Expected: refusals remain, no repaired row/partial turn survives rollback.
- [x] Run focused restart, first-save and durable-acceptance modules. Expected: all targeted cases pass.
- [x] Run affected native-commit and mounted recovery tests plus census/diagnostic contracts. Review failures against exact base before any unrelated changes.
- [x] Correct the stale duplicated census comment and preserve all1258base entries plus5incoming entries. Record QA and finish existing task viaCLI.
Final gate: obtain independent review of the saved integrated head, then publish with exact lease and merge after fresh non-draft/head/base/thread checks. The external exact-head report and merge handoff record this post-commit gate.

Pre-flight: no new shared interfaces. User's standing authorization already selects inline implementation plus independent review; no repeated approval gate.
