# Textual publication and Notes export follow-ups

This implements TASK-37266, TASK-37267 and TASK-37268 from the
[source-pinned audit](2026-10-28-textual-book-after-pr3283.md). The
[implementation plan](../plans/2026-10-28-textual-37266-37268.md) preserves the
existing Evals, Research Runs and Library owners. The requested delivery is a
reviewed open PR against `dev`. The implementation source checkpoint passed
hosted CI; a later metadata-only head exposed two existing Library control
preparation races, documented below. The final reviewed head is checked before
delivery; its run links and result are recorded in the PR description without
creating another evidence-only commit and CI cycle.

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

- [PR #3290](https://github.com/rmusser01/tldw_chatbook/pull/3290) remains open
  against `dev`. Source head `7c2b8f5aab066035ed9d06f9719a93d184f8e5c3` passed
  all **40 Derived Artifacts jobs** and **Perf Guard**:
  [application/required run](https://github.com/rmusser01/tldw_chatbook/actions/runs/40810284866),
  [performance run](https://github.com/rmusser01/tldw_chatbook/actions/runs/40810284730).
  All 32 UI shards passed; their longest actual execution was **679 seconds**
  under the unchanged 1,200-second cap. All 44 added selections have actual
  execution rows in successful jobs, including the canonical-class guard on
  shard22 and the exact existing Notes Escape/Replace case on shard8. There were
  zero unresolved GitHub review threads. [Hashed hosted qualification](2026-10-28-textual-followups-implementation/hosted-source-head.json)
  records jobs, case ownership, timings and retained private-log hashes. This
  establishes hosted parent outcomes; it does not claim independently extracted
  hosted child XML. Local actual children are archived above.
- The [whole-branch Superpowers review](2026-10-28-textual-followups-implementation/final-review.md)
  approved implementation with **0 Critical/Important/new Minor** findings.
  The two inherited Minor categories remain disclosed below. The
  [decision ledger](2026-10-28-textual-followups-implementation/decision-ledger.md)
  records every scope ruling and the final review's declined-to-judge items.
- The [integration map](2026-10-28-textual-followups-implementation/integration-local-evidence.json)
  and [raw archive](2026-10-28-textual-followups-implementation/integration-local-receipts.tar.gz)
  retain 262 distinct contract cases with passing latest observations across
  successive targeted runs, plus nine backlog/diagnostic-engine/Evals-size
  passes. This is not one all-green batch: the first run exposed the two
  whitespace IDs and old checkpoint expectation; the next fixed those but two
  partition reads overlapped the root's rebase. Their frozen exact retry passed.
  Original failed results and the source transitions remain in the evidence.
- Rebase onto `3efe8476cebf70eceafa605d1abed76a4aad4e0e` changed none of the
  owned committed product, test, workflow or evidence blobs. Only the appended
  testing-lessons conflict needed resolution; both branches' entries survive.
- Task-scoped and whole-branch reviews approve all three changes. Task closure
  followed the successful source-head hosted qualification: all three are Done
  via CLI, with all13 ACs checked and implementation notes/final summaries present.
  [Closure and source conservation](2026-10-28-textual-followups-implementation/backlog-closure.json)
  records the unchanged product/test/workflow/security inputs. Final delivery
  also verifies the final reviewed head and records its result in the PR description.
- The old `test_note_export_onto_an_existing_file_asks_escape_keeps_it_replace_writes`
  fails locally during unchanged Notes source loading before export. It stays
  locally unqualified and enrolled in full. Its exact hosted result passed on
  the qualified source head; the local failures are not relabeled passes.
- The canonical-class guard fails locally because Windows path separators differ
  from its slash-based inventory. Exact baseline inputs are conserved; this
  unchanged guard is newly enrolled and passed actual Linux qualification.
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
  moves from `_write` to `_fail`. The latest-dev inventory was regenerated and
  every other owner, privacy rule/candidate and sink was conserved. The
  [actual command receipt](2026-10-28-textual-followups-implementation/diagnostic-inventory.json)
  records drift/statements/write/postcheck exit codes `1/0/0/0`; the final check
  passes. The earlier empty stdout-only file is not used as passing evidence.
- [TASK-37330](<../../../backlog/tasks/task-37330 - Restore-Research-Runs-artifact-loading-through-the-real-scope-contract.md>)
  separately tracks the pre-existing positional/keyword Research artifact API
  mismatch. The [ownership proof](2026-10-28-textual-followups-implementation/artifact-followup-ownership.json)
  checked 412 refs, 366 trees and 55 worktrees before creation, and links existing
  TASK-16483/21127/19300/37267 and ADR-033/068/078. This PR does not fix that action.

## Final-head CI feedback and bounded control repair

Metadata-only head `3e414ff824f88bf8e13a5aef1003be3525481a87` retained the
previously qualified source inputs, but its [Derived run](https://github.com/rmusser01/tldw_chatbook/actions/runs/40815454377)
finished with 37 successful jobs and three failures: UI12, UI19 and their
dependent required aggregate. [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40815454364)
passed. Two unrelated one-off workflows were intentionally skipped; no review
thread was unresolved. The [historical failure receipt](2026-10-28-textual-followups-implementation/ci-repairs/failed-metadata-head.json)
keeps the actual assertions, job URLs and raw log hashes. The earlier source-head
pass and this metadata-head failure are distinct results.

UI12's Reader identity test prepared a filter with separate key bursts. Native
paced keys independently reproduced query `t` instead of `Product`: each
debounced browse publication rebuilt the focused Input and selected its prefix.
The Reader test now prepares its filter with native App Paste, preserving every
original Reader/write/toast assertion. A private wrong-target negative still
fails the original write-ID assertion. Ordinary typing remains a product defect,
tracked separately in [TASK-37360](<../../../backlog/tasks/task-37360 - Preserve-Media-filter-typing-and-editing-context-across-browse-publication.md>).
Its [ownership search](2026-10-28-textual-followups-implementation/ci-repairs/media-typing-ownership.json)
checked 417 refs, 368 trees, 8,493 bodies and 55 worktrees before creation and
links the older browse/editing tasks and ADRs. Paste is not a typing repair.

UI19's full note-link journey failed its Navigation reopen after its original
picker and autosave assertions passed. Native delivery tracing showed a row
pressed while a sibling folder result was still pending; recomposition retired
its parent before the queued Pressed message could bubble. Local link-test
preparation now waits for the acknowledged Notes visit, complete expected
branches and actual current tree DOM publication, then presses one reacquired
current row. The initial title setup explicitly clicks/focuses its real Input
before its unchanged typing assertions. A selected Preview keyboard consumer
also exposed Enter dispatch during native Work/Items recomposition. Only local
Preview preparation now awaits current focused/painted DOM publication before
returning; its original keyboard/link/picker oracles remain unchanged. Broader
production gesture delivery remains the existing TASK-34000.85 obligation.

The [Media investigation](2026-10-28-textual-followups-implementation/ci-repairs/media-filter-investigation.md),
[Notes investigation](2026-10-28-textual-followups-implementation/ci-repairs/note-open-investigation.md)
and [native investigation archive/map](2026-10-28-textual-followups-implementation/ci-repairs/native-investigations.json)
retain the matching mechanisms and distinguish invalid attempts. The
[combined repair report](2026-10-28-textual-followups-implementation/ci-repairs/ci-fix-wave-report.md)
and [hashed raw repair map/archive](2026-10-28-textual-followups-implementation/ci-repairs/ci-fix-wave.json)
preserve all run outcomes, commands and executed sources. Six distinct targeted
journeys have qualified latest passing outcomes across bounded runs, not one
all-green batch. Held-folder and held-Preview controls fail with old preparation
and pass with the new preparation; holds drain inside App without a watchdog.
The optional old-helper Preview failure at a later picker assertion is not
treated as equivalent to the earlier Enter failure. The amended full keyboard
journey and Preview sibling each pass with their original assertions. Scoped
Ruff passes; exact inherited formatting regions and unaffected AST owners/oracle
tails are conserved. An invalid outside-UI collection's raw config capture stays
private and is excluded from the portable archive.

The [scoped Superpowers re-review](2026-10-28-textual-followups-implementation/ci-repairs/ci-fix-wave-review.md)
checks this three-test-file repair after the approved whole-branch review. No
product, fixture/service/getter, workflow, census, timeout or privacy guard changes
are included in the repair. These native controls qualify preparation and its
outcome assertions; final delivery still requires actual green CI and resolved
feedback on the final pushed head, recorded in the PR description.
