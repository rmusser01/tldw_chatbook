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

## Latest-dev integration

The repaired head `f73ab0247974114c13f53d6569484b9a81c76a07` subsequently passed
all 40 Derived Artifacts jobs and Perf Guard. All 32 UI shards passed, maximum
697 seconds; all 44 added selections, exact old Notes Escape/Replace and the
canonical-class guard have successful execution rows. There were no unresolved
threads. Its [qualification receipt](2026-10-28-textual-followups-implementation/dev-sync/qualified-predecessor.json)
is preserved as a historical result. Dev advanced to
`38c9d2edcf10bc69d0d7597514fa319db7e4ccd2` during qualification and produced
an integration conflict, so delivery rebases and checks the integrated head.

The two manual conflicts were the census and its floor. The integrated census
retains all **1,145 latest-dev selectors plus 44 PR additions = 1,189**, each
once. The previous 1,184-entry replay block/order remains intact; the five new
dev selectors append outside that block, preserving all existing shard slots.
The floor is 1,189. Dev's split-file contracts and three new required Console
targets remain. Matrix/concurrency, fail-closed required aggregation and all
timeouts are unchanged. The [measured append replay](2026-10-28-textual-followups-implementation/dev-sync/capacity.json)
uses complete setup/call/teardown rows from successful merged dev PR jobs for
the five additions: 13.37, 7.52, 18.04, 13.85 and 16.50 seconds. Combined with
the preserved prior model, maximum work is 616.99 seconds and the existing
forecast is 1,019.54 seconds against 1,020. This narrow forecast margin is an
estimate; actual integrated-head hosted success remains required.

All ten task/repair source blobs and 146 branch blobs outside the six shared
integration paths are unchanged. Latest-dev diagnostic owners, privacy rules
and sinks are preserved except the previously reviewed Notes warning-owner
digest; the actual integrated inventory check passes. Both branches' lesson
entries survive. The [source conservation](2026-10-28-textual-followups-implementation/dev-sync/source-conservation.json)
and [diagnostic receipt](2026-10-28-textual-followups-implementation/dev-sync/diagnostic.json)
record these boundaries. All **251 targeted census/workflow/replay/queue/backlog
contracts pass** in one run, with zero failures/errors/skips. Scoped checker
lint/format and diff whitespace checks pass.

The [integration receipt map](2026-10-28-textual-followups-implementation/dev-sync/integration.json)
retains actual XML/commands and member hashes. The
[scoped Superpowers integration review](2026-10-28-textual-followups-implementation/dev-sync/review.md)
checks the shared paths and conflict resolution; it does not repeat the approved
product review. Final pushed-head CI and feedback results are recorded in the
PR description; the successful predecessor is not proof of the rebased result.


## 2026-10-31 backlog-clearance integration

Rebased on dev `dcdb2249294c60c5b660011e8d7b0f3c8babb28c`. The owner authorized targeted local tests and independent Superpowers review in place of the hosted CI/Qodo merge gates, and waived size limits. Earlier hosted measurements above remain historical evidence, not a qualification of this head.

All four product blobs match reviewed source `188fe0fc92dfd64ddb70f3fc890919640f77d036`. Integration retains the current YAML tags/timestamp contract, exact output/source/atomicity assertions, all current-dev workflow targets and guards. The census is 1,242 unique entries: all 1,198 dev selections plus 44 source-only additions, with the source's 1,189-entry order intact. The 32-shard/17-concurrent model and timeouts are unchanged; 53 newer selections have no source-model measurement, so no complete current-capacity forecast pass is claimed.

The metadata campaign yielded 246 passes and two obsolete exact-target pins; those pins were corrected and their two checks passed. Native representative checks yielded nine passes and seven setup/oracle failures. Diagnosis corrected real source readiness via bounded Retry/publication, the legacy node's private bootstrap marker, exact Windows path comparison, current Context focus, and export-worker/notification ownership. The export boundary retry passed three cases; the legacy export content control passed. An intermediate helper edit used the Windows default encoding and changed fixture text; those link results are not final evidence. Restoring the exact original UTF-8 constants and rerunning all three link journeys passed, including actual in-app mouse navigation, browser dispatch and deleted-target refusal. No production admission guard, file protection, timeout, or behavior assertion was weakened.

The [integration index](2026-10-28-textual-followups-implementation/backlog-clearance-2026-10-31/integration.json), [raw selected receipts and independent reviews](2026-10-28-textual-followups-implementation/backlog-clearance-2026-10-31/local-receipts.tar.gz), and [partial capacity qualification](2026-10-28-textual-followups-implementation/backlog-clearance-2026-10-31/capacity-partial.json) retain commands, XML, source hashes and intermediate failures. Final independent integration review found no new Critical/Important issue; publication additionally checks exact-head conservation and resolved threads. No full sweep, live-provider replay, release or main promotion was performed.
