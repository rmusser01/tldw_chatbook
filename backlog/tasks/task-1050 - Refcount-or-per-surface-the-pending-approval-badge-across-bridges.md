---
id: TASK-1050
title: Refcount or per-surface the pending-approval badge across bridges
status: To Do
assignee: []
created_date: '2026-07-27 14:30'
labels:
  - console
  - approvals
  - concurrency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The per-session pending-approval badge (`_pending_approvals`, toggled via `set_run_pending_approval`) is a single boolean shared across every approval surface a session can have outstanding at once -- MCP tool approvals, skill-install confirms, and skill-script confirms are three independent bridges that all write it. Each bridge's teardown (`request_mcp_approvals`'s `finally` around `console_chat_controller.py` ~:2100-2110, `request_skill_install_confirm`'s `finally` ~:2493-2517, `request_skill_script_confirm`'s `finally` ~:2687-2712) sets the badge to `False` for its own session_id independently of whether a sibling round from a DIFFERENT bridge is still outstanding for that same session. Whichever round finishes first clears the badge even though another one is still pending, so the fleet UI can show "nothing needs attention" for a session that in fact still has a live confirm waiting.

Compounding this, the skill-install and skill-script bridges' `finally` blocks already compute a `still_armed_same_session` guard (checking their OWN round map for a sibling round belonging to the same session) before deciding whether to clear the MOUNTED card -- but the session-keyed parked-payload pop (`_parked_skill_install_payloads.pop(session_id, None)` / `_parked_skill_script_payloads.pop(session_id, None)`) and the `set_run_pending_approval(session_id, False)` call right above it run UNCONDITIONALLY, ahead of that guard, with no equivalent check. The mounted-card clear is guarded; the badge clear and payload pop are not, even though they exist in the same teardown block for the same reason.

This is production-unreachable today: per-session tool dispatch is sequential, so a single session can only ever have one approval round of any kind outstanding at a time. It becomes a live correctness bug the moment same-session concurrent rounds are ever allowed to exist (e.g. an agent that can request an MCP approval and a skill-install confirm in the same turn, or two backgrounded runs sharing a session).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The pending-approval badge for a session stays set while ANY surface (MCP approval, skill-install confirm, skill-script confirm) has an outstanding round for that session, and only clears once none remain.
- [ ] #2 The session-keyed parked-payload maps are only cleared for a bridge's own round, never in a way that discards a sibling surface's still-armed payload for the same session.
- [ ] #3 A regression test exercises two concurrent same-session rounds from different bridges (or, if dispatch is truly sequential today, a direct unit test of the teardown logic) and proves the badge/payload survive the first round's resolution.
<!-- AC:END -->
