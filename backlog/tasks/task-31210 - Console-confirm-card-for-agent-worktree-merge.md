---
id: TASK-31210
title: Console confirm card for agent-worktree merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 11:45'
updated_date: '2026-09-12 20:13'
labels:
  - agents
  - console
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-28238 phase 2 shipped `merge_agent_worktree`/`discard_agent_worktree` with full controller/bridge plumbing (`set_pending_worktree_merge`, `request_worktree_merge_confirm`), but no Console widget ever assigns the controller hook, so in production the tools fail closed ("no approval surface") and — per the disclosure gate — are not offered to the model at all. Wire the actual card so worktree merge-back becomes usable.

Scope notes from the phase-2 review record:
- The rendering hook precedent is `set_pending_skill_script`: wired via `CONSOLE_VIEW_HOOK_SLOTS` in `UI/…/console_runtime.py` and the per-view hook dict in `chat_screen.py`. `set_pending_worktree_merge` needs the equivalent slot + a card that renders the diffstat payload the controller already provides.
- Preview/live parity: the two preview call sites of `build_console_first_request_plan` (`build_project_instruction_preview_request`, `build_personal_context_preview_snapshot`) currently omit `worktree_merge_enabled`; once a real surface exists they must thread the same flag or preview token accounting diverges from live.
- One deferred test gap worth closing here: no single test threads the real bound `controller.request_worktree_merge_confirm` through a real `AgentService.run_turn` (each hop is pinned separately today).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a fleet-active Console session, an agent's `merge_agent_worktree` call raises a visible confirm card showing the diffstat preview, and Allow/Deny drive the real merge/refusal.
- [ ] #2 `merge_agent_worktree`/`discard_agent_worktree` are disclosed to the model exactly when the card surface is wired, and preview/live plan builders agree on disclosure.
- [ ] #3 Card parks/remounts across session switch like the skill-script confirm card.
- [ ] #4 An end-to-end test threads the real controller callable through a real `run_turn`.
- [ ] #5 Worktree creation and confirmed mutations use the exact current writable named binding and contained root-pinned operations; stale, read-only, ambiguous, scratch or unsupported authority refuses before mutation.
- [ ] #6 Merge/discard eligibility requires durable positive physical-owner drain proof as well as terminal run ownership; a cancelled-but-live child remains protected.
- [ ] #7 Confirmed discard removes the agent changes and exact branch while retaining a detached baseline checkout, and the card and result explicitly disclose retained cleanup; no automatic or forced pathname root deletion occurs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/155-agent-worktree-recovery.md. Reason: closed mutating Git authority, positive physical drain and exact visible confirmation. Execute Docs/superpowers/plans/2026-09-12-agent-worktree-recovery.md Tasks 1-3 in order under its spec: contained authority/operations, durable ownership and drain fences, then actual card/disclosure integration. No visible mutation capability is enabled before prerequisite review. Real temporary Git, SQLite and mounted tests qualify POSIX behavior; unsupported platforms refuse honestly.

Execution qualification remains open: actual Git2.39.5 temp-repository tests show linked administrative paths can reopen a replaced parent despite both root pins. Only the test-isolation prerequisite has been implemented and independently reviewed (32557b80f2; four affected tests passed). Do not enable the new mutation capability or mark these tasks Done until the execution boundary is redesigned and qualified. See backlog/docs/agent-orchestration-followups-2026-09-12.md for current status.
<!-- SECTION:PLAN:END -->
