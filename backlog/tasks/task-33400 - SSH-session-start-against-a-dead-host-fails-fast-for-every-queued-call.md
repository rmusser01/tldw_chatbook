---
id: TASK-33400
title: SSH session start against a dead host fails fast for every queued call
status: To Do
assignee: []
created_date: '2026-09-28 20:30'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a host dies or blackholes, concurrent tool calls queued behind one failing session start each start their own session in turn, so the last caller waits roughly k times the connect or handshake timeout, while one-shot calls fail together. The handshake also ignores the call's time budget (fixed 30 s). Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Callers queued behind a session start that fails transport-class all get that failure without starting another session
- [ ] #2 A call arriving after such a failure still tries a fresh session (no sticky failure)
- [ ] #3 A session handshake never outlives the call's budget plus grace, and a stalled handshake is classified exactly like the one-shot path
- [ ] #4 Any session start failure leaves no ssh process running and no open pipe on the laptop
<!-- AC:END -->
