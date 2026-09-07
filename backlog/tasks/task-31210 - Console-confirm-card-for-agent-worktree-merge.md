---
id: TASK-31210
title: Console confirm card for agent-worktree merge
status: To Do
assignee: []
created_date: '2026-09-03 11:45'
labels:
  - agents
  - console
  - ui
dependencies: []
priority: high
---

## Description

TASK-28238 phase 2 shipped `merge_agent_worktree`/`discard_agent_worktree` with full controller/bridge plumbing (`set_pending_worktree_merge`, `request_worktree_merge_confirm`), but no Console widget ever assigns the controller hook, so in production the tools fail closed ("no approval surface") and — per the disclosure gate — are not offered to the model at all. Wire the actual card so worktree merge-back becomes usable.

Scope notes from the phase-2 review record:
- The rendering hook precedent is `set_pending_skill_script`: wired via `CONSOLE_VIEW_HOOK_SLOTS` in `UI/…/console_runtime.py` and the per-view hook dict in `chat_screen.py`. `set_pending_worktree_merge` needs the equivalent slot + a card that renders the diffstat payload the controller already provides.
- Preview/live parity: the two preview call sites of `build_console_first_request_plan` (`build_project_instruction_preview_request`, `build_personal_context_preview_snapshot`) currently omit `worktree_merge_enabled`; once a real surface exists they must thread the same flag or preview token accounting diverges from live.
- One deferred test gap worth closing here: no single test threads the real bound `controller.request_worktree_merge_confirm` through a real `AgentService.run_turn` (each hop is pinned separately today).

## Acceptance Criteria

- [ ] With a fleet-active Console session, an agent's `merge_agent_worktree` call raises a visible confirm card showing the diffstat preview, and Allow/Deny drive the real merge/refusal.
- [ ] `merge_agent_worktree`/`discard_agent_worktree` are disclosed to the model exactly when the card surface is wired, and preview/live plan builders agree on disclosure.
- [ ] Card parks/remounts across session switch like the skill-script confirm card.
- [ ] An end-to-end test threads the real controller callable through a real `run_turn`.
