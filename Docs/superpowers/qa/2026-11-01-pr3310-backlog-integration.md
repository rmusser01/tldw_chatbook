# PR3310 integration evidence

Base: `b1e19a1dba935a599186d8a252cb01a5ecf54abf`. Incoming head: `98538d60674ac970778bc1b74c670b956c1a1fd8`. Rebased series: `3d39549ecea6c792c4a62d2a4c3e5b2a16da173c`, followed by the review repairs recorded below. Hosted CI and size limits are waived by the user; no release is part of this work.

## Preserved behavior and repairs

- Retained current acknowledgement-first receipt, captured-input custody, typed attachment blockers, guard warnings and Terminal focus behavior. The census retains the entire 1,263-node base prefix and adds the five incoming selectors once (floor 1,268).
- Independent review reproduced a refused press claiming the composer's height hold. Only a press that obtains its own acknowledgement token now marks its captured draft; Queue full, chainless run hold and another turn's Preparing state cannot mark newer unsent input.
- Height retention recognizes the two existing Preparing presentations, including received-intent preparation. Their displayed strings are unchanged and share named constants. Other refusals re-wrap normally, and edits retain their existing revision/generation protection.
- Existing stock sends now also take acknowledgement custody. Both layout oracles require echo and reply placeholder to first appear in the same reflow.

ADR required: no new ADR. Existing [ADR094](../../../backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md) and [ADR230](../../../backlog/decisions/230-console-send-preparation-and-io-ownership.md) govern the preserved lifetime and exact-input boundaries. Plan: [PR3310 integration](../plans/2026-11-01-pr3310-backlog-integration.md).

## Current focused evidence

Receipts are retained under the session's isolated `pr3310-resync` worktree parent; exact-base comparisons are under `pr3309-resync`. Every receipt reports unchanged source hashes during execution. No full sweep was run.

| Receipt | Result |
| --- | --- |
| `r-aba2932f` / `r-8d647c70` | Six acknowledgement/refusal ownership controls red, then all six green. |
| `r-31b33c9f` | Mounted Queue full/run-hold cases red; idle control green. |
| `r-a9f16f5e` | Received-Preparing positive control red (two rows instead of one); acknowledged control green. |
| `r-f657524e` | Final acknowledgement suite plus both Preparing and all refusal geometry controls: 21 passed. |
| `r-23f47606` | Revised changed-file UI run plus navigation control: 22 passed, two failed, no errors/skips. Acknowledged mounted layout and unknown-command resize passed. Navigation failed at the same custody-retirement assertion as exact dev. Stock layout timed out before provider entry. |
| `r-cd4d46b6` | Census checker, replay and count contracts: 112 passed. |
| `r-fd78f570` / base `r-442376db` | Initial focused run: 25 passed, 13 failed, one teardown error. All 11 existing failed nodes reproduce on exact dev; two additional failures were the new layout cases timing out. The teardown error did not reproduce on base or in the later candidate navigation run and is not classified as proven baseline. |
| Base `r-7e7695e6` | Identical transplanted stock test also times out before provider entry on exact dev. No production base source was modified. |
| `r-4532c89e` | Separate stock-layout diagnostic passed every original geometry/reflow assertion on final production source. It permits up to 120 seconds for provider readiness; the committed measured-send test retains its 15-second deadline. This qualifies layout only, not the normal deadline or a latency improvement. |

The normal test's first two warm-up sends now allow 60 seconds to prepare an overflowing chat; its measured third send remains bounded by 15 seconds. The longer diagnostic is retained outside the repository as `pr3310-stock-qualification-probe.py`, with its exact receipt, and is not shipped as a replacement test. Its postcommit timing labels cover complete nested effects, including provider work, and are not additive latency measurements.

Scoped Ruff comparison reports no introduced diagnostics across eight changed Python files. The two changed UI test files are formatted. Existing formatting failures and inherited diagnostics remain unchanged; `git diff --check` passes. Current composer size pin is 7,195 under the user's size waiver.

## Limits retained

TASK-33620.15.36 tracks the mounted echo/placeholder timing oracle; deterministic projection controls also pass. TASK-33620.15.37 tracks a quiet stand-in release awaiting another invalidation or forced sync; no universal three-second release bound is claimed. Historical task notes describe older source bases and external captures. This integration adds no new native terminal, real-provider performance, or app-wide latency claim. The normal stock test's deadline remains a documented failure on this host.

Independent review: `pr3310-integration-review.md` in the worktree parent; final exact-head verdict is retained there before publication.
