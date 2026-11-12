---
id: TASK-32368
title: Local-tool Deny refusal names its authority instead of the bare blocked word
status: Done
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - console
  - approvals
dependencies: []
priority: medium
---

## Description

`LOCAL_DENY_REFUSAL` in the local tool provider cannot say who refused, so `console_agent_bridge._refusal_statuses()` maps it to the bare `blocked` status and the transcript shows `· blocked` for a Deny the user pressed on a workspace tool, while the same Deny on an MCP tool shows "denied by you". Lane C's task-32280 made the Audit row honest (`denied`); the transcript word is still generic. Lane D's docs now document the bare `· blocked` as a fourth vocabulary word; this rider removes the need for it on the user-deny path.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] A user Deny on a local workspace tool renders the same "denied by you" transcript status as an MCP deny
- [ ] `LOCAL_DENY_REFUSAL` (or its successor) carries the authority so `classify_activity_status` can name it
- [ ] Docs (`agent-runs-and-tools.md` refusal vocabulary) updated to drop the local-Deny case from the bare `· blocked` bullet

## Closeout audit 2026-10-11

Board audit (branch `chore/board-prune-prune-d`, dev at a3c47c6c72): this task's work is
landed on `origin/dev` but the status field was never flipped. Verdict: **Done**.

Closed by the rider's parent fix: "fix(console): give a local user Deny its own refusal
string (Qodo #7 / task-32280)" whose body ends "Closes rider task-32368". Tree check:
`LOCAL_USER_DENY_REFUSAL` in `Chat/console_agent_bridge.py` mapping to "denied" "matching the MCP
pair" (AC#1), the same split applied to `VirtualCliProvider` and `gateway_runtime._LOCAL_FAILURES`
(AC#2), and `Docs/User_Guide/console/agent-runs-and-tools.md` updated - "denied by you" now covers
local workspace tools and `blocked (Off)` is its own bullet (AC#3).

AC checkboxes are left as found; this section is the audit's evidence record.
