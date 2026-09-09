# SQLite acceptance closeout — 2026-09-08

Status: SQLite correction closeout approved; TASK-31942 is complete.
The narrow independent review supported the seven-AC/platform mapping and found
one missing paired actual Canvas-child measurement. Task16 supplied it with five
passing samples per arm; fix-only review verified that finding ADDRESSED, spec
compliance PASS and task quality PASS, with no new issues. Canvas V2 stays
disabled. No completed benchmark or broad review was repeated.

Reconciled checkout: `210831e1928177d73d278b6adfad1112a240ce81`.
Latest product change: `28a37eac8d2be04cdb13d8678ada9a48e9e8ca34` (imports only).
This is completion of existing Task7, not a new implementation or gate waiver.

ADR required: no new ADR. Existing [ADR125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md),
including its owner-approved static amendment, and ADR097 apply unchanged.

## Original acceptance criteria

| AC | Closeout assessment and supporting evidence |
| --- | --- |
| 1 — Causal ownership regression | Satisfied for the demonstrated lock defect. The approved design records an independent writer entering while the first connection remains in a transaction after raw/private SHM inspection, with normal SQLite as the control. Committed `Tests/DB/test_private_sqlite_lock_preservation.py:84` covers actual private opens in WAL/rollback modes and both thread contexts; `:243` retains isolated positive/negative raw-close controls. These run in the affected selection. Historical SIGBUS traces remain evidence of symptoms, not proof of every crash's exact interleaving. |
| 2 — Private ownership, concurrency, integrity and cleanup | Satisfied on the qualified macOS runtimes. The 37-file affected selection covers all 27 originally required files, including privacy, backup, restore, lifecycle, exclusive finalizers, installed helpers and the repaired owner inventory. The separate exact eleven-case macOS CI result resolves the previously excluded concurrency cases. [Affected evidence](2026-09-08-sqlite-final-local-qualification.md), [macOS evidence](2026-09-08-sqlite-macos-concurrency-evidence.md). |
| 3 — Targeted database and actual Canvas workflows | Satisfied for correction handback, not V2 admission. The affected run records 1844 passes; its local schema `live` case separately passes. The five-node actual-child run has four passes and a retained failed recovery case; the latter is subsequently corrected and passes the exact node after both reviewed harness fixes. The final shared-deadline run is 1 pass in 46.86s, exit 0. These are incremental results, not an invented five-pass final run. [Final local evidence](2026-09-08-sqlite-final-local-qualification.md), [final harness evidence](2026-09-08-canvas-startup-deadline-fix.md). |
| 4 — ADR, evidence and independent review | Satisfied before handback. ADR125 and the approved design record the boundaries and amendments. Whole-correction review at `41f144ab90` found I1/M1/M2; the single fix-only review at `3b5031012c` verifies all three addressed. Later inventory, maintenance, harness, CI and import changes have their own scoped reviews. The remaining Q1 readiness items are accounted for below; no repeated broad review is claimed. [Review/fix record](../../Canvas/V2_VERIFICATION.md#whole-correction-review-and-bounded-fix-wave--2026-09-08), [latest static review](2026-09-08-sqlite-no-new-static-debt.md). |
| 5 — Terminal proof loss and process-exit safety | Satisfied within the approved availability tradeoff. Actual-app idle/read/write and partial-setup ordinary/abrupt cases cover all fourteen combinations, foreign-cohort preservation, separately recovered original data, blocked exclusive lease during retention and release after exit. They are included in the later affected selection without the earlier source override. Repeated refusal, charged owner bounds and healthy siblings remain distinct from normal leak-free cleanup. This does not promise in-process release after terminal loss. [Affected/native evidence](2026-09-08-sqlite-final-local-qualification.md), [native contract](../specs/2026-09-07-sqlite-lock-safe-private-validation-design.md#lost-proof-explicit-terminal-quarantine). |
| 6 — Python floor and native capability refusal | Satisfied. Active metadata/build/CI floor is Python >=3.12; real-floor syntax, packaging metadata, policy and pre-initialization refusal tests are in the affected selection. The native preflight and actual-handle checks have real/proxy coverage and independent review. Python 3.11 historical detector re-derivation remains unavailable, not a supported-runtime gap. Fresh macOS CI used explicitly approved Python3.12.10/SQLite3.49.1; local evidence used Python3.12.11/SQLite3.49.1. No arbitrary native build is claimed supported. |
| 7 — Unchanged startup budgets and Canvas safety | Satisfied. Final import-only commit has three passing unchanged guards: 625/660 imported modules, 963/972 at UI readiness, 499/500 preload modules; LOC budgets also pass unchanged. First-use/ownership tests and actual Canvas correction cases remain covered as above. [Final source/budget evidence](2026-09-08-sqlite-no-new-static-debt.md). |

## Fresh read-only evidence verification

The initial reconciliation, before the Task16 attempt below, did not rerun a
suite, benchmark or concurrency control. Fresh offline verification established:

- The preserved affected manifest has 37 unique files and contains every one
  of the original Task7's 27 required files. Thirty-three are byte-identical to
  the affected-run checkpoint `55f74aa009`; four have only the reviewed import
  edits, with identical import bindings and non-import AST.
- Every product change since that checkpoint is exactly the Task15 import-only
  diff. Its covering tests and independent review already qualify that change;
  all 58 current Python sources match the immutable final static audit's hashes.
- The four archived macOS artifact files decode and match their saved SHA256
  hashes. JUnit contains exactly the eleven originally deselected parameter IDs,
  each once, no outcome children and zero skip/error/failure totals. The successful
  allocation/acquisition/disposal control and tested SHA match the archive.
  The repository concurrency test file is byte-identical to the CI-tested copy.
- Preserved output corroborates the affected-run, separately run local-schema,
  final repaired actual-browser node and latest startup-budget results. Results
  remain attached to their original commits and environments, not summed into
  a fictitious single run.
- The production catalog still has no default diagram profile. No Canvas
  admission file was changed by this closeout.

The compact audit is retained as `task-7-closeout-audit.json` in the existing
ignored SQLite SDD evidence directory. Historical reports and raw evidence are
preserved unchanged; later sections describe which earlier gaps were resolved.

## Q1 readiness disposition and limits

The completed review's remaining readiness constraint was an evidence checklist,
not another code defect. Each item now has a specific disposition:

- Inventory and baseline owner/compaction failures: repaired, scoped-reviewed,
  and included in the 1844-pass affected selection and later 343-pass import
  covering run.
- Eleven pre-body semaphore failures: qualified separately on fresh macOS CI.
  The unchanged local Mac's exhausted semaphore capacity is not declared fixed.
- Installed wheel/import closure and Python-floor checks: included in the
  affected selection; shared editable-package repair was explicitly authorized
  earlier and does not authorize another environment change.
- Benchmark work: paired app/TTS/helper measurements were completed and reported.
  Task16 now supplies the omitted actual Canvas-child comparison: ten passing
  invocations, current/baseline median36.834/34.186s (+2.648s/+7.75%), with full
  assertions and unchanged deadlines. This measures the complete workflow, not
  pure SQL latency; [method/results](2026-09-08-sqlite-canvas-workflow-benchmark.md).
  Separately, the earlier UI median
  increased 3.4%; repository-open median increased from
  13.214ms to 192.666ms. The requirement is measured cost under unchanged ceilings,
  not unchanged latency. Existing guards pass. Both benchmark logs have disclosed
  lifecycle errors and sampled high-water limitations, so they are not used as
  pristine lifecycle or exact kernel-peak proof; actual lifecycle tests provide
  the separate behavioral evidence. The later successful Task16 series supplies
  the actual-child comparison required by design lines491–497 and Task7; it does
  not erase the earlier logs or introduce a new threshold.
- Static debt: the explicitly owner-approved no-new-static-debt gate passes.
  Whole-file lint/format and inherited dependency warnings remain nonzero and
  disclosed; no general cleanup, suppression or pristine-output claim.
- Platform/optional coverage: the approved design's preserved contracts retain
  Windows' unverified ACL/locking posture, and Task7 explicitly requires reporting
  unavailable OS/interpreter evidence rather than claiming a pass. The two
  Windows-only cases are unavailable evidence, not a new requirement to establish
  Windows support before this POSIX correction can hand back. Linux, Windows,
  other native builds and frozen distributions remain unqualified here. The
  disabled TTS draft-options case is outside its current slice; the supported
  floor is 3.12, not the unavailable historical 3.11 control. The local-schema
  `live` case has its separate passing result. No platform or feature policy is
  expanded or silently qualified by these distinctions.

The review's hypothetical simultaneous reuse of one exception instance remains
outside the established sequential-attempt contract; no reachable producer was
identified. Its prior recorded limitation is retained, not newly dismissed.
Historical unsafe benchmark-import and unknown initial helper-PID evidence also
remains preserved; later isolation checks do not erase that incident.

## Task16 attempted measurement — stopped before browser launch

Both immutable archives were prepared, with only the two approved current
test-harness files overlaid onto baseline. Root independently checked every
extracted regular file and archive hash, and verified per-arm source resolution
using each archive's own tldw_profile_core source path.

The first current-arm invocation failed before Chromium launch: **1 failed,
1 inherited Requests warning in6.73s; exit1**. Root omitted the existing
TLDW_CANVAS_CHROMIUM_EXECUTABLE override. The installed binary exists, but
the temporary archive and pytest-isolated home remove the default user-home
cache fallback. This is an invocation configuration error, not a demonstrated
Canvas runtime regression. JUnit records exactly one failed case (zero errors
or skips); its raw0.483s call time is not a valid Canvas workflow measurement.

The fail-stop contract was honored: no subsequent/current retry or baseline
sample ran, and no aggregate is reported. Output, command/source identity, JUnit
and remaining test state are preserved in
`/private/tmp/task16-canvas-benchmark.MirPEY/evidence/current-1`; the full
preparation and execution record is `task-16-report.md` in this plan's SDD
directory. No lifecycle/trace file was produced before this failure; the
diagnostics directory is empty and owned test_data/TLS files were cleaned up.

The user then approved a fresh five-pair series with the existing explicit
browser executable path in both arms, preserving that failed attempt. All ten
series2 invocations passed without a retry, skip or discarded sample. The exact
node, source/harness inputs, assertions, deadlines and state-isolation contracts
are unchanged. Median/current36.834s versus baseline34.186s; maximum38.056s
versus35.673s. Each run emitted the inherited Requests warning. Independent
on-disk aggregation revalidated every exact case and output/diagnostic hash.
[Complete measurement and limits](2026-09-08-sqlite-canvas-workflow-benchmark.md).
No installation, download or product/harness source edit occurred. Measurement
is complete; the agreed fix-only review subsequently approved the evidence.

## Handoff

The independent closeout review found one Important requirement gap: the
retained benchmark driver measures app UI, repository open and fixed helpers,
but no actual Canvas child workflow against baseline. Root verified this from
the driver and the design. Task16 completed the measurement and the independent
fix-only review confirmed ADDRESSED, spec compliance PASS, task quality PASS and
SQLite closeout READY, with no new findings. Together with the previously
reviewed original AC mapping, this supports marking TASK-31942 Done under the
explicit existing static/platform limits above. The complete scoped review is
preserved as task-16-closeout-rereview.md in the SQLite SDD directory. No broad
code review or completed benchmark was repeated, and no unmet gate was waived.

Then resume **Task8 / TASK-31941** in the approved Canvas V2 Mermaid plan.
Its full candidate/admitted qualification, independent review and immutable
admission remain separate required work. The earlier 1346-pass/4-fail admitted
run stays failed, and the catalog stays disabled until that plan's gates pass.
This closeout does not push, rebase, open a PR or merge a branch.
