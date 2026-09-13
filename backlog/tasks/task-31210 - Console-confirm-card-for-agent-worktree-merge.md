---
id: TASK-31210
title: Console confirm card for agent-worktree merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 11:45'
updated_date: '2026-09-13 01:29'
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
- [ ] #5 Worktree creation and confirmed mutations use the exact selected writable named binding with fresh binding and root identity checks; missing selection and observed stale/read-only/ambiguous/scratch authority refuse. Ordinary Git does not claim atomic protection against concurrent external metadata replacement. Existing child filesystem containment remains intact.
- [ ] #6 Merge/discard eligibility requires durable positive physical-owner drain proof as well as terminal run ownership; a cancelled-but-live child remains protected.
- [ ] #7 Confirmed discard removes the agent changes and exact branch while retaining a detached baseline checkout, and the card and result explicitly disclose retained cleanup; no automatic or forced pathname root deletion occurs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amendment. ADR path: backlog/decisions/155-agent-worktree-recovery.md. Reason: restore ordinary local Git with exact selected writable authority and explicit limits, then finish confirmation and durable ownership/recovery. Current spec: Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md. First slice: Docs/superpowers/plans/2026-09-12-agent-worktree-creation-restoration.md. Keep automatic deletion disabled. Application checks do not promise atomic protection against an external process replacing Git metadata during a command. The former qualified-backend blocker is superseded by the user-approved restoration scope. Tasks remain In Progress until their actual acceptance criteria and targeted verification are complete. Recovery migration remains 19 to 20 after caps (ADR-158).
<!-- SECTION:PLAN:END -->
