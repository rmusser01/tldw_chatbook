# Agent orchestration burn-down implementation plan

**Goal:** Close the remaining routing/progress defects, reconcile completed tickets, and deliver the requested optional orchestration capabilities under explicit reviewed contracts.

**Implementation base:** dev 64579cce2c8dc64053fb50c00eb4f59b56716b01.
**Integration base:** latest fetched dev 31d4f9b76492120706ba8e3ad7d355f1aa4e0273.
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

All 13 tasks are Done with checked acceptance criteria and implementation notes. Durable progress passes 247 affected checks, 82 final native/modal/hydration checks and 74 final report/tool/queue checks; these overlapping selections are not summed. Independent final review approves all four worker cache corrections with 13 checks and actual participant drain. Routing/UI and schema/ledger selections pass 116 and 137 checks. Changed-code static verification passes; existing size-ratchet and source debt remain disclosed. The clean latest-dev rebase passes 217 targeted provider/search/routing/sampling/startup checks in 87.14s; changed-code checks remain clean. Delivery is through a draft PR against dev. No merge is claimed.

## October 1 CI and integration checkpoint

Rebased PR #2918 onto dev `31d4f9b76492120706ba8e3ad7d355f1aa4e0273`. Reopened TASK-33430/33431/33432 for their existing acceptance criteria and CI qualification. ADR required: no; direct repairs and verification under ADR-173/199.

- [x] Preserve exact runtime-tool inventory, including both scoped peer tools.
- [x] Use the existing canonical UTC millisecond timestamp writer for durable progress.
- [x] Pin real populated FIFO and progress-claim cleanup query plans, register both indexes and allowlist the new Chat table.
- [x] Audit changed production diagnostics before refreshing their inventory.
- [x] Repair the upstream hook-refusal early return through the existing exact wake authorization and preacceptance refund path; qualify real plain/agent retry and nonreplay.
- [x] Independently review overlapping runtime/routing/recovery boundaries and run final targeted checks.

Affected messaging/schema checks pass 145 cases; fallback/sampling/saved-close/hydration/mounted progress/token/guide checks pass 147 cases. The exact CI correction selection passes three cases. Hook repair passes six authority/real-gateway cases and three existing ledger guards; independent review passes both real provider paths and three ledger guards. The upstream MCP click correction passes all four focused cases. These selections overlap and are not summed. Final static qualification covers 75 changed Python files and ten new files with zero added-line/new-file findings. Delivery remains PR #2918 against dev; no merge is claimed.

## October 1 Qodo and boot CSS follow-up

- [x] Preserve the existing boot CSS ratchet by consolidating the two preset editor selectors into one scoped class and qualifying actual painted fields at standard/narrow widths.
- [x] Complete public peer documentation and provider-error annotations.
- [x] Add strict Pydantic peer argument validation through the existing text validator; preserve refusal codes, quotas, privacy and capability custody.
- [x] Replace global-close mutable membership traversal with existing immutable published membership after the global authority fence; qualify controlled removal and actual mounted disposal.
- [x] Independently re-review all four Qodo findings and close the reopened tasks after fresh targeted/static/diagnostic checks.

ADR required: no; direct qualification and mechanical repairs under ADR-097/150/173/199/200. All 13 tasks are Done. Fresh evidence:17 CSS/authoring/token/bundle cases, 88 peer/inventory/fallback cases, 2 startup guards, 44 queue cases and 4 mounted lifecycle cases. Independent final review approves 22 focused cases; selections overlap and are not summed. Final static scan covers 76 changed Python files and ten new files with zero added-line/new-file findings; diagnostics match the audited inventory. Publish the follow-up to ready PR #2918 and require fresh checks on that head. No merge is claimed.
