# Agent orchestration burn-down implementation plan

**Goal:** Close the remaining routing/progress defects, reconcile completed tickets, and deliver the requested optional orchestration capabilities under explicit reviewed contracts.

**Base:** dev 64579cce2c8dc64053fb50c00eb4f59b56716b01.
**Authority:** User requested all listed items on 2026-09-29. Existing task acceptance criteria govern each repair. Existing ADR-134/135/136/146/147/155 govern budgets, delivery, communication, routing and recovery.

## Global constraints

- Use the isolated codex/agent-orchestration-burndown worktree; preserve other checkouts and work.
- Preserve endpoint identity independently from execution spellings; explicit built-in child routing must still select the built-in destination.
- Preserve configured caps, cancellation, approval and physical-owner boundaries. No automatic replay of uncertain effects.
- Diagnostics and ordinary metadata contain no communication bodies or private URLs/paths.
- Follow ADR-150/161 UI tokens and existing rendering patterns; targeted verification only.
- Reproduce defects before minimal fixes; review changes independently before closing tasks.

## Repair sequence

- [x] TASK-32929: thread raw selection identity into primary run dispatch and keep explicit child target resolution independent. ADR required: no; direct repair under ADR-146/147, with clarification of the existing identity contract.
- [x] TASK-33001.9: reuse GENERATION_FIELD_REQUEST_KEYS and verify top_p through real provider projection. ADR required: no; implement the existing shared generation-field contract.
- [x] TASK-32639 / TASK-32517: keep count/navigation synchronization active when the button is absent; mounted regressions for removal/remount. ADR required: no; lifecycle bug fix.
- [x] TASK-32499: reconcile admission-refusal counting; surface routing level; move parameter parsing to existing Chat/sampling_params.py. ADR required: no; preserve ADR-147 behavior and consolidate a shared helper.
- [x] TASK-32497: project saved resolved provider/model for live/historical rows with honest legacy fallback and rendered regression. ADR required: no; direct ADR-147 follow-up.
- [x] TASK-22061 / TASK-32477 / TASK-32520: compare existing implementation/evidence to current criteria, verify affected behavior, repair remaining gaps and reconcile task status.

## Requested extensions

- [x] TASK-32508: specify and implement explicit pre-tool provider fallback chains. ADR required: yes; amend ADR-147 or create a focused canonical successor before implementation; keep budget/continuation/snapshot semantics explicit.
- [x] Direct sibling addressing, durable progress inboxes and progress-triggered wakes: inspect existing communication and delivery owners; create atomic Backlog tasks and ADR/spec before implementation. Reuse existing queues, SQLite ownership and wake budgets. Restart persistence preserves messages, never live execution authority.

## Completion

- [x] Targeted combined verification, static/derived checks and independent code review.
- [x] Update the workstream ledger with exact dispositions, acceptance evidence and remaining limits.

## Extension decisions and execution order

ADR required: yes
ADR paths: backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md; backlog/decisions/200-preset-pre-tool-fallback-targets.md
Reason: explicit new communication authority, persistent state and fallback policy.

1. TASK-32508: frozen preset fallback targets, schema migration and existing-runtime integration.
2. TASK-33430: scoped peers using existing steering and private metadata.
3. TASK-33431: durable inbox after fallback schema settles.
4. TASK-33432: progress wake source after durable enqueue is authoritative.

All four are in the authorized burn-down; run targeted checks and independent review for each.

## September 29 review checkpoint

All 13 tasks are Done with checked acceptance criteria and implementation notes. Durable progress passes 247 affected checks, 82 final native/modal/hydration checks and 74 final report/tool/queue checks; these overlapping selections are not summed. Independent final review approves all four worker cache corrections with 13 checks and actual participant drain. Routing/UI and schema/ledger selections pass 116 and 137 checks. Changed-code static verification passes; existing size-ratchet and source debt remain disclosed. Final latest-dev integration and draft PR publication follow this local completion.
