# Textual publication and Notes export follow-ups

This implements TASK-37266, TASK-37267 and TASK-37268 from the
[source-pinned audit](2026-10-28-textual-book-after-pr3283.md). The
[implementation plan](../plans/2026-10-28-textual-37266-37268.md) preserves the
existing Evals, Research Runs and Library owners. The requested delivery is a
reviewed open PR against `dev`; hosted checks and final branch review are still
pending at this local integration checkpoint.

## Behavior and local evidence

| Task | Result | Native evidence |
| --- | --- | --- |
| TASK-37266 | Character Save refreshes metadata while retaining later raw input, editing context and newer selection. An editor-local monotonic revision also handles equal-valued Saves and returned drafts. | Six intended assertion REDs for the review repair; final 31 parent passes, including all 26 new actual child passes. Earlier intended defect controls and existing editor/selection controls are retained separately. |
| TASK-37267 | Research timer publication applies only to its captured source and selection revision. Source changes immediately render empty detail; accepted progress retains current payloads. | Three intended assertion RED children; seven final parent/actual child passes; 13 legacy behavior controls pass. Static controls remain 46 passes/8 failures, qualified below. |
| TASK-37268 | Native Notes export workers capture content, target and operation, offload only destination writing, and publish completion/error/collision results on the App loop. | One intended same-thread/blocked-dispatch assertion RED; ten final parent/actual child passes, including ordinary/held new/Replace, collision, error/retry and retired/changed-view outcomes. Old compatibility controls: 18 passes/one pre-export failure; exact retry repeats that failure. |

Tests use the installed Python 3.12.10 and Textual 8.2.8, native-private external
profile/TEMP roots and the unchanged 180-second limit. They release holds and
drain accepted work inside the App lifetime. Parent wrappers and actual children
are counted separately. The Notes source is a seeded service stand-in, not a
real Notes SQLite claim. Research payload controls arrange data from real scope
reads; they do not claim the separate artifact action works.

The portable evidence folder contains full hashed XML/JSON/source snapshots,
commands, invalid attempts and reviews. Raw `.log` files remain private; XML
includes its original captured output. Each archive's member hashes were
independently verified after writing.

| Task | Receipt map and raw archive | Independent review |
| --- | --- | --- |
| TASK-37266 | [Map](2026-10-28-textual-followups-implementation/task1-local-evidence.json), [archive](2026-10-28-textual-followups-implementation/task1-local-receipts.tar.gz), [report](2026-10-28-textual-followups-implementation/task1-implementation-report.md) | [Initial](2026-10-28-textual-followups-implementation/task1-review.md), [approved repair](2026-10-28-textual-followups-implementation/task1-fix1-review.md) |
| TASK-37267 | [Map](2026-10-28-textual-followups-implementation/task2-local-evidence.json), [archive](2026-10-28-textual-followups-implementation/task2-local-receipts.tar.gz), [report](2026-10-28-textual-followups-implementation/task2-implementation-report.md) | [Approved](2026-10-28-textual-followups-implementation/task2-review.md) |
| TASK-37268 | [Map](2026-10-28-textual-followups-implementation/task3-local-evidence.json), [archive](2026-10-28-textual-followups-implementation/task3-local-receipts.tar.gz), [report](2026-10-28-textual-followups-implementation/task3-implementation-report.md) | [Approved](2026-10-28-textual-followups-implementation/task3-review.md) |

The implementation reports describe historical source checkpoints; their final
source pins and approved repair sections supersede earlier attempts. Original
absolute private paths in receipts map to the archived member names in each
public map. Rebased Git identities may differ without changing executed blobs.
Two description parameter IDs subsequently replace a space with a hyphen so the
existing census can read them. Parameter values and test bodies are unchanged;
the [final collection map](2026-10-28-textual-followups-implementation/parameter-id-collection.json)
records all 44 actual current IDs and the aliases to historical receipts.

## Complete census and capacity

[ADR-103](../../../backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md)
extends its existing replay/reorder/add-capacity policy. All 1,140 inherited
selectors remain exactly once, plus 43 concrete new behavior cases and the
unchanged canonical-class guard. Actual collection found all 44 new selections;
collection is not a passing test run. The floor grows to 1,184 and split-file
contracts require all cases in the three new files.

The [capacity replay](2026-10-28-textual-followups-implementation/capacity-replay.json)
retains complete costs, exact ordering, source hashes, input hashes and every
candidate from 28 through 36. The corrected inherited workload is 17,138.65
seconds: prior worst costs plus only the last qualified run's 28 UI job logs.
Eight unrelated job logs are excluded; an earlier provisional model counted
shared selectors twice and was discarded before any CI mutation.

New parent-case measurements add 2,004.23 seconds, including native subprocess
and setup/call/teardown costs. The total is 19,142.88 seconds. Slot-constrained
deterministic longest-processing-time placement selects 32 shards, forecasting
`599.34 * 1.15 + 310 = 999.24` seconds against the existing 1,020-second target.
Even ideal balance at 31 gives 1,020.14 seconds. These are estimates: local
Windows timings are not hosted capacity proof, and session/collection overhead
is outside case times and covered by the existing reserve.

The UI ceiling remains 17 simultaneous jobs, with 38 eligible application jobs
and a peak of 23 application runners (24 with Perf Guard). Workflow structure
is conserved except for matrix membership; minimal dependencies, every selected
test, serial shards, 20-minute job cap, 180-second test timeout and required
aggregation remain unchanged. The official replay's algorithm is unchanged.

## Qualifications and follow-up

- Task-scoped reviews approve all three changes; final whole-branch review and
  actual final-head CI remain root-owned closure gates.
- The old `test_note_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes`
  fails locally during unchanged Notes source loading before export. It stays
  locally unqualified and enrolled in full. Its exact hosted result must pass.
- The canonical-class guard fails locally because Windows path separators differ
  from its slash-based inventory. Exact baseline inputs are conserved; this
  unchanged guard is newly enrolled for actual Linux qualification.
- Research's timer/worker static comparison retains the same 120 semantic timer
  sites/75 roots and existing findings. Windows path handling is not the only
  cause: inherited ProjectIndexBindingCard inventory gaps also remain. No green
  static-suite or whole-repository clearance is claimed.
- Scoped lint/format passes. The large pre-existing Library shell test retains
  its exact three E721 findings and formatting debt; only its changed function
  is format-qualified. Pytest-asyncio configuration, Pydantic, audioop, optional
  backend and inherited splash warnings remain disclosed, not suppressed.
  The CI workflow/queue contract files retain their exact inherited formatting
  regions; the remaining five checked CI scripts/tests are format-clean. All
  seven CI files and the renamed-ID regression pass scoped lint.
- The exporter retains all four sanitized diagnostic calls; its Notes warning
  moves from `_write` to `_fail`. Root must regenerate the latest-dev inventory,
  prove every other owner/sink conserved, and capture a successful real check.
- [TASK-37330](<../../../backlog/tasks/task-37330 - Restore-Research-Runs-artifact-loading-through-the-real-scope-contract.md>)
  separately tracks the pre-existing positional/keyword Research artifact API
  mismatch. The [ownership proof](2026-10-28-textual-followups-implementation/artifact-followup-ownership.json)
  checked 412 refs, 366 trees and 55 worktrees before creation, and links existing
  TASK-16483/21127/19300/37267 and ADR-033/068/078. This PR does not fix that action.
