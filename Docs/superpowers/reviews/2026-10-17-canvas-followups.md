# Canvas follow-up closeout — 2026-10-17

Base: `272cad956855feda4d78efb9db9fce7e909d53db` (`dev`).
Branch: `codex/canvas-followups-20261017`.
This records local implementation qualification, not a new hosted-CI or merge claim.

## Completed maintenance

- TASK-32897.1: independently pin the trusted Canvas shell's JS, CSS and HTML
  alongside the existing V2 worker/renderer. Tampering each shell file failed
  the new regression before the fix; deliberate pin updates restore acceptance.
- TASK-32380 and overlapping TASK-32377: check in all eight original authenticated
  Mermaid/Unicode inputs outside application package data. Default verification
  rebuilds offline, without skipping or fetching. Explicit refresh downloads only
  to a new directory and preserves existing destinations. Original manifests,
  packaged assets and immutable profile bytes are unchanged.
- TASK-34968: qualify the merged Canvas/Personal Context budget and ownership
  behavior on current source. Fix a demonstrated native connection leak in the
  shared first-request agent-roster read using `operation_owned_connection`.
  Preserve borrowed handles and active transactions; use the actual production
  maintenance worker wrapper in the live-dispatch fixture. Gate the new native
  regression in the existing required PR contract and its two inventories.

The source-input decision is [ADR-238](../../../backlog/decisions/238-offline-authenticated-canvas-build-inputs.md).
The bounded roster correction reuses ADR-121/202/228/126; it changes no budget,
permission, storage schema or preview-publication policy. The main checkout's
unrelated work was left untouched.

## Verification and review

Native CPython 3.12.11; application/profile-core imports resolve to this worktree.
Each pytest run used a unique temporary directory and disabled the parent cache.

| Selection | Result |
| --- | --- |
| Asset checker, derived-workflow contracts, Personal Context gate membership, queue-pressure contract | 137 passed, exit 0 |
| Complete Canvas profile-preview module | 27 passed, zero errors/skips, exit 0 |
| Required membership and Inspector selection/currentness controls | 20 passed, exit 0 |
| Additional production project-preview/no-execution shared-caller controls | 2 passed, exit 0 |
| Root final confirmation of new owner/borrower cases plus the preceding controls | 24 passed, exit 0 |
| `scripts/check_canvas_mermaid_assets.py` | Six reproduced outputs, four derived outputs and five independent asset pins verified |
| `scripts/preflight.sh` | All derived-artifact guards passed, exit 0 |

The actual offline-rebuild test rejects any network opener. New fetch-path and
shell-tamper controls also have pre-fix RED receipts. The native roster pair was
RED/GREEN: a newly acquired owner remained usable before the fix; its borrowed
active-transaction control passed both before and after. An external passive
observer retained original SQLite objects without closing them: the final 27
isolated children recorded 608 native connection observations, all closed on
their creating threads, zero survivors or wrong imports. These are observations,
not a unique application-owner count.

Changed checker/test/CI Python files pass scoped Ruff checks. Changed checker,
asset-test, gate-test and preview-test formatting passes. The queue-pressure
file's pre-existing formatting failure and the bridge's exact 28 pre-existing
Ruff findings reproduce on base; neither is presented as a clean whole-file
static result. Whitespace checks pass for owned source changes. The vendored
Unicode emoji data and UAX HTML contain six original trailing-whitespace lines;
authenticated third-party input bytes are preserved, not reformatted.

Independent build/offline and context reviews found no remaining findings.
Historical [PR 2862](https://github.com/rmusser01/tldw_chatbook/pull/2862)
merged on 2026-10-12 with a successful required Derived Artifacts aggregate;
that resolves the old task's publication gate, not this branch's hosted status.

The root's final 137-case rerun and 24-case confirmation also passed after the
task closeout edits. Backlog ID and readable-section checks pass for all 5,496
task records. Staged input blobs match the exact working-file bytes.

Raw logs, JUnit, RED controls and native observation receipts are under
`/private/tmp/chatbook-canvas-followups.FrzDFL`. The detailed local context report
is `.superpowers/canvas-followups-2026-10-17/context-closeout.md`.
No full suite, cross-platform run, live provider call or fresh hosted check was run.
Unrelated Personal Context/Notes qualification debt remains outside this closeout.

## Roadmap and open decision (pre-approval snapshot)

Created six independent To Do tasks: TASK-35370 React, TASK-35371 D3,
TASK-35372 multi-file V3/VFS, TASK-35373 network, TASK-35374 filesystem,
TASK-35375 partitioned cookies/persistent state. Future capability work requires
its own bounded design/ADR, explicit authority and adversarial qualification.
The current zero-egress runtime remains unchanged.

The following paragraph records the first local closeout, before the user’s later
2026-10-17 approval. Current status: TASK-31003 is Done as a contract-design task,
ADR-239 is Accepted, and the approved design’s delivery map tracks the separate
client/server implementation tasks.

TASK-31003 remains In Progress with all original criteria open. The
[sync contract proposal](../specs/2026-10-17-canvas-server-sync-contract-design.md)
and [Proposed ADR-239](../../../backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md)
are design evidence only. Approval must cover server-readable source, retained
concurrent revisions with storage/archive compatibility changes, and explicit
delete/restore semantics. Companion servers have no Canvas adapters yet.
Contract approval must be separated from coordinated client/server implementation;
no synchronization or new runtime capability has been enabled.
