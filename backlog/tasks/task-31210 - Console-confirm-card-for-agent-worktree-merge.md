---
id: TASK-31210
title: Console confirm card for agent-worktree merge
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 11:45'
updated_date: '2026-09-13 04:30'
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
- [x] #1 With a fleet-active Console session, an agent's `merge_agent_worktree` call raises a visible confirm card showing the diffstat preview, and Allow/Deny drive the real merge/refusal.
- [x] #2 `merge_agent_worktree`/`discard_agent_worktree` are disclosed to the model exactly when the card surface is wired, and preview/live plan builders agree on disclosure.
- [x] #3 Card parks/remounts across session switch like the skill-script confirm card.
- [x] #4 An end-to-end test threads the real controller callable through a real `run_turn`.
- [x] #5 Worktree creation and confirmed mutations use the exact selected writable named binding with fresh binding and root identity checks; missing selection and observed stale/read-only/ambiguous/scratch authority refuse. Ordinary Git does not claim atomic protection against concurrent external metadata replacement. Existing child filesystem containment remains intact.
- [x] #6 Merge/discard eligibility requires durable positive physical-owner drain proof as well as terminal run ownership; a cancelled-but-live child remains protected.
- [x] #7 Confirmed discard removes the agent changes and exact branch while retaining a detached baseline checkout, and the card and result explicitly disclose retained cleanup; no automatic or forced pathname root deletion occurs.
- [x] #8 Both Console preview paths disclose queued agent progress tools exactly as the live first request does.
- [x] #9 Both Console previews forward the available virtual CLI and raw shell provider schemas consistently with the live first request, without creating runtime resources.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, existing amended decision.
ADR path: backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md.
Reason: ordinary local Git with exact selected writable authority, durable original-base/ownership, positive physical completion, and user-confirmed recovery.
Spec: Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md.
1. Creation restoration (Docs/superpowers/plans/2026-09-12-agent-worktree-creation-restoration.md): implemented and independently reviewed at a11db916ff/d63d4f2a19.
2. Durable storage and completion callbacks (Docs/superpowers/plans/2026-09-12-agent-worktree-records-and-drain.md): implemented and reviewed at4756f88793/94f7110db1; schema20 retains caps.
3. Connect actual child ownership (Docs/superpowers/plans/2026-09-12-agent-worktree-ownership-integration.md), then confirmed operations (Docs/superpowers/plans/2026-09-12-agent-worktree-confirmed-operations.md).
4. Wire visible confirmation and retained manual recovery (Docs/superpowers/plans/2026-09-12-agent-worktree-console-recovery.md), verify real Git/SQLite and mounted wide/narrow UI, independently review and close only when all AC are met.
Automatic deletion remains disabled. Command-boundary checks do not promise atomic protection against concurrent external Git metadata replacement. The former qualified-backend blocker is superseded by the user-approved scope. Current review/evidence/rulings: Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the real retained Console worktree confirmation card, exact immutable Allow/Deny request IDs, independent park/remount behavior, and actual surface-driven tool disclosure. Both previews now match the live first-request schemas for tested worktree/progress/virtual CLI/raw-shell availability without consuming messages or creating runtime resources. The actual picker reaches the card; accepted session close fences queued admission while refused/provisional closes preserve recovery.

Ordinary selected-authority Git creation, schema 20 original-base/ownership records, positive physical completion and confirmed apply/merge/discard are integrated. Final review corrections also retire partial Git-reader startup and use an atomic pre-entry gate for manual executor rejection/queueing, keeping admitted workers owned through physical completion. Discard explicitly retains a detached baseline checkout; automatic deletion remains disabled.

Independent task reviews, the broad restoration review and its scoped final correction review are complete through e135a085f2. Final correction evidence:11 behavioral RED failures;88 affected/capacity neighbors and2 actual reopened card/Git flows pass. Prior reviewed Console selections:25 mounted UI,23 preview/raw-shell and60 lifetime tests; counts overlap and are not summed. Native wide/narrow evidence was inspected in two root rounds and independent task review.

Verification is targeted, not a full-suite or live-provider claim. The inherited Requests warning remains; UI-ready passes at 973/973 with zero headroom. Existing ChatScreen size/no-growth guards (including the disclosed 21-line/3-method addition) and eight stale historical diagnostic-label expectations remain failures. The current diagnostic inventory/sink guard passes, and changed-file static comparisons add no diagnostic identities with edited-range formatting passing.

ADR check: existing ADR-155 governs recovery and ADR-158 migration order; existing ADR-136 progress and ADR-150 design language remain unchanged. The full review, source/test file inventory, earlier chronology qualifications and all rulings are in Docs/superpowers/reviews/2026-09-12-agent-worktree-restoration.md. User guide and testing lessons are updated. Work remains on the local codex/agent-orchestration-remaining branch.
<!-- SECTION:NOTES:END -->
