---
id: TASK-16482
title: Enforce checkpointed autonomy for local research runs
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 03:19'
updated_date: '2026-08-16 03:28'
labels:
  - research
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-068 deferred local checkpoint enforcement: runs record autonomy_mode (default checkpointed) but the engine always executes autonomously, and local checkpoint approval in the window uses a placeholder id against a scope path that only supports the server. Port the server's review-loop semantics: the engine pauses at phase boundaries to create reviewable checkpoints, approval (with validated patches) advances the run, and sources approval can loop back to collecting for recollection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The service stores checkpoints (type, status, proposed payload, user patch) with create, list, latest-pending, and patch-and-approve operations recording run events and versioning rows,The engine honors checkpointed autonomy: it creates a plan review checkpoint before collecting and a sources review checkpoint before synthesizing, entering a non-terminal awaiting state and exiting cleanly (no report yet, partial artifacts preserved),Approving a checkpoint validates patches per type (unknown keys rejected; sources patches must reference the proposed inventory) and resumes the run past that boundary on re-execution without recreating an already-approved checkpoint,Sources approval with recollect enabled loops the run back to collecting; autonomous runs (autonomy_mode autonomous) never pause,The window's Approve Checkpoint action works for local runs against the latest pending checkpoint and restarts the engine,Tests cover the service operations and validation, the engine pause/resume matrix, and the window approval path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD LocalResearchService checkpoints: research_checkpoints table plus create, list, latest-pending, and patch-and-approve with per-type patch validation, run events, and optimistic versioning
2. TDD engine enforcement: checkpointed runs pause at plan-review (before collecting) and sources-review (before synthesizing) boundaries in a non-terminal awaiting state with partial artifacts; approved checkpoints let re-execution pass without recreating; sources recollect loops back to collecting; autonomous runs never pause
3. Wire the scope service local checkpoint path and the window approve flow (latest pending for local, engine restart after approval); ADR-068 gains a dated addendum noting the deferred item is now implemented
4. Tests plus lint plus task close
ADR required: no - implements ADR-068's explicitly deferred checkpoint clause; a dated addendum records the activation
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Service**: `research_checkpoints` table (soft-delete + versioning, house pattern) with `create_checkpoint`, `list_checkpoints`, `latest_pending_checkpoint`, `approved_checkpoint(run_id, type)` (the engine's pass-through signal), and `patch_and_approve_checkpoint` with per-type validation — plan patches allow `limits`; sources patches allow `pinned_source_ids`/`dropped_source_ids`/`recollect` with inventory membership and disjointness checks; non-pending approvals and wrong-run checkpoints raise. Approvals record `checkpoint_approved` events. The scope service's checkpoint routing needed no change — it already dispatched via `getattr` on the backend, so the local method lights it up.
- **Engine**: checkpointed runs (the schema default) pause at a `plan_review` boundary before any search spend and a `sources_review` boundary before synthesis — `_await_review` creates the pending checkpoint, parks the run non-terminally (`control_state=awaiting_<type>`), and exits via the `_RunAwaitingReview` path (partial artifacts preserved; report never produced early). Re-execution passes an APPROVED boundary without recreating it (existence-based, not patch-truthiness — approving with no patch passes). An approved plan `limits` patch supersedes the run's originals for ledger enforcement; an approved sources patch drops the named sources; `recollect.enabled` re-collects and presents a fresh sources review (server parity). Autonomous runs never pause; all pre-existing engine tests were migrated to `autonomy_mode="autonomous"` since the default is checkpointed. Outline review stays unimplemented: the local engine has no separate outline phase.
- **Window**: local approval resolves the latest PENDING checkpoint (the placeholder-id era ends), reports clearly when none is pending, and restarts the engine on approval so the run advances. The old "local checkpoints unavailable" pin updated to the new contract and renamed.
- **ADR-068** gained a dated addendum recording the activation, semantics, and the outline-review non-mapping.
- Verified TDD: 4 service tests + 5 engine tests (pause, plan→sources advance, dropped-filter completion, recollect loop, empty-patch pass) + 1 window test written first and watched failing; suites 174 passed; ruff clean.
<!-- SECTION:NOTES:END -->
