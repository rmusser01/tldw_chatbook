# PR3311 integration evidence

Base: `dd7c4cf43d46b011fd8b698039a6e1d3767fd20b`. Incoming head: `34533e3c2023d1934e3a17ee255269dea3f279ec`. Hosted CI and size limits are waived by the user. Draft PRs and PR3267/3297 are excluded; no release work is authorized in this session.

## Repairs and preserved contracts

- Retained current acknowledgement-first input custody, pending received-row handling, typed attachment blockers, attach reconciliation and effective memory presentation. The rebase preserved both PR3310's refusal-height tests and this PR's refusal repaint test.
- Stop commands consume only their original captured draft after a successful write. Actual later typing, identical retyping and navigation to another chat retain newer input. Failed or refused writes preserve input.
- An app admits one physical stop mutation at a time across command and chip routes. Competing mutations receive a visible retry message. Repeated caller cancellation waits for physical retirement and outcome publication before releasing admission; it does not announce success or clear the draft. Executor rejection, including enqueue-then-raise, cannot write the sentinel.

ADR required: no new ADR. Existing [ADR094](../../../backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md) and [ADR230](../../../backlog/decisions/230-console-send-preparation-and-io-ownership.md) govern view lifetime, captured input and physical ownership. Diagnostic inventory governance is [ADR029](../../../backlog/decisions/029-local-private-data-boundary.md). Plan: [PR3311 integration](../plans/2026-11-01-pr3311-backlog-integration.md).

## Focused verification

Receipts are retained under the session's isolated `pr3311-resync` worktree parent. Every completed receipt below records unchanged source hashes during execution. No full sweep was run.

| Receipt | Result |
| --- | --- |
| `r-90661408` | Seven selected failures before repair: six causal draft/concurrent-write/cancellation failures and one chat-switch harness failure. |
| `r-7dbca796` | Corrected chat-switch control failed on unrepaired source rebased onto the exact current base: the other chat's draft was erased. |
| `r-96f78825` | Ownership, cached view and captured-command suites: 39 passed. |
| `r-155d34a8` | Stop, scheduler heartbeat and presentation suites: 90 passed, eight failed, zero errors/skips. |
| `pr3310-resync/r-0d222f3a` | All eight failed IDs/messages reproduce on the exact base tree: six old hand-built scheduler fixture failures and two onboarding copy assertions. |
| `r-0117c732` | Original emergency-stop UI, acknowledgement and two executor rejection controls: 26 passed, two failed, zero errors/skips. |
| `r-f5310084` | Persisted-stop UI passed after waiting for the deferred empty-row repaint. Post-Clear send still timed out before provider entry. |
| `r-a3455be0` | Post-Clear send completed with exactly one provider call and the expected reply. The bounded completion wait recorded 8.844 seconds. |

The persisted-stop test previously treated a painted header as proof that the separately scheduled empty-row reconciliation had finished. It now waits for the actual body and retains all original content assertions. The post-Clear test now uses the existing wall-clock polling helper with a 60-second completion allowance. This is functional completion evidence, not a latency qualification. Its original short-wait failure and the earlier durable-commit RuntimeError remain recorded; a later successful run does not establish the RuntimeError's cause or classify it as a proven baseline defect.

An earlier whole-app ownership harness run (`r-83a06192`) was stopped after teardown errors and is incomplete evidence. The final ownership tests use the existing real ChatScreen ConsoleHarness with file-backed persistence and the production sentinel writer. The repeated-cancellation control uses two unmounted screen stand-ins sharing an app; it does not certify native screen destruction or application quit. Chip competition calls the real Clear handler directly; the original UI suite separately exercises the clickable chip.

The diagnostic inventory verifies unchanged: 716 owners, 1063 TASK-492 calls, 56 TASK-31551 calls, 7790 TASK-494 calls and 16 sink files. Scoped Ruff comparison reports zero introduced diagnostics across 19 changed Python files; the repaired module and tests also pass scoped Ruff directly. `git diff --check` passes. Existing size pins remain under the explicit size waiver.

## Remaining limits

Existing TASK-33620.1.1-.7 remain open: queued/Goal/Buddy/mobile entry points, the disabled Send-button command route while held, the hidden narrow-width reason, heartbeat lag, typed-reason presentation, scheduler test fixture debt and timing-sensitive module census. The parent's unrestricted visible-reason criterion remains unchecked and the task stays In Progress because the 80-column limitation is not repaired here. Header/chip/tooltip indications remain available. No all-entrypoint stop, cross-process mutation serialization, native terminal performance or new live-provider claim is made.

Independent exact-head review is retained as `pr3311-integration-review.md` in the worktree parent before publication.
