---
id: TASK-22690
title: Closing a chat mid-turn raises Durable continuation owner changed
status: To Do
assignee: []
created_date: '2026-08-26 23:51'
labels:
  - console
  - durable-turns
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A fourth raise site of the class TASK-22587 fixed. Closing a Console session while a durable turn's postcommit sequence is running can reach the tail of `resume_durable_postcommit` with its continuation already gone, and the owner check turns that into `RuntimeError('Durable continuation owner changed.')` (console_chat_controller.py:6276).

The check is right in intent -- a continuation whose OWNER changed must not be settled by the wrong turn. What it does not distinguish is 'the owner is legitimately GONE because the user closed the chat', which is the same conflation TASK-22587 removed for the fingerprint guard: an ordinary close and a genuine mismatch shared one raise.

Found while converting `test_console_local_citation_boundary` to durable sessions (TASK-22301). TASK-22587 predicted these two tests would reach a durable-path failure once the sessions stopped being ephemeral; it fixed the fingerprint raise, and this is the next one the same scenario hits:
  - test_citation_repair_session_close_privacy_sentinels
  - test_citation_repair_close_during_collection_never_resurrects_session_or_message

Both are xfail(strict=False) naming this task until it is fixed.

The retirement tombstone TASK-22587 introduced already makes the two cases decidable: a preparation with a matching tombstone was retired by a close, so an absent continuation there is expected rather than anomalous.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Closing a chat during an in-flight durable postcommit does not raise Durable continuation owner changed
- [ ] #2 A continuation whose owner genuinely CHANGED still raises (negative control, mutation-proven)
- [ ] #3 The two xfail markers in test_console_local_citation_boundary are removed and the tests pass
<!-- AC:END -->

## Numbering provenance

`backlog task create` assigned task-22618 from a stale local view; a sweep of
all remotes and worktrees showed the real maximum was 22660 AND that 22618 was
already taken. Renumbered to 22690 (max+30) under the leapfrog rule before the
file was ever committed.
