# Chunking Lab verification record

Branch: `codex/chunking-lab`, based on `origin/dev` commit
`1a82db60ce47890c6d2df9f918f80309c8608ea6`. Architecture:
[ADR-118](../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md).
[Workflow and limits](Chunking_Lab.md),
[implementation plan](superpowers/plans/2026-09-04-chunking-lab.md).

## Review and targeted evidence

All eight tasks received independent specification and quality reviews. Task8
screen integration is `c1b320d11b`; its reviewed edit-drain correction is
`cd3e13926a`. Whole-branch review at `462e5cc30e` found seven Important gaps and
five bounded refinements. The single final correction commit `5d0df113ff` received
scoped independent re-review: all twelve findings addressed, no new breakage in
that fix diff. The user-requested runtime-boundary and test-fixture follow-up
also passed independent task and final review. TASK-31428 is complete with AC1–15
verified; the final targeted gate is **473 passed, 2 known warnings in 106.29s**.
No merge or push
has been performed. The original working checkout was not used for implementation.

Verification used the existing Python3.12.11 virtual environment with Textual8.2.8
and Pydantic2.12.5. Only targeted selections were run, not the repository-wide
suite. Exact commands, outputs, intermediate failures and per-task review reports
are retained under the ignored, plan-owned directory
`.superpowers/sdd/2026-09-04-chunking-lab/` in this worktree.

- Headless tests cover faithful whole-pipeline execution, lossless drafts,
  immutable request/result capture, real SQLite checkpoints, bounded recovery
  transfer, write/CAS failures, subprocess admission/cancellation and canonical
  conflict-safe template saving. Each task's final amended-code selection is
  recorded separately; counts are not summed into a fictitious full-suite result.
- Task8 final affected UI gate:36 passed in45.17s. After the edit-drain review
  correction, the affected screen/recovery modules passed27 tests in40.21s.
  Deterministic RED3/GREEN3 proves an edit arriving during the consumer's final
  render reaches autonomous state, recovery export and the navigation checkpoint.
- Real isolated-profile scenarios include SIGKILL after a committed invalid draft
  and completed result, fresh recovery without rerunning, failed-write export and
  restore into another profile, and Cancel-button reaping of a child ignoring
  SIGTERM. Replacement rollback was tested using a real SQLite abort trigger;
  do not confuse that with a replacement-specific SIGKILL test.
- Two bounded visual rounds covered80×24,120×40,160×50. The narrow correction
  made paging/full-text inspection keyboard-reachable; a150-chunk regression
  selects the last chunk. Accepted screenshots predate later logic-only
  status/lineage/worker/drain fixes, which have amended-code tests.
- Scoped new-code Ruff, formatting, compilation and whitespace checks passed.
  Existing large-module diagnostics were compared separately, not bulk-repaired.

Final correction evidence: **344 passed, 1 existing Requests warning in 73.73s**
across 15 amended-feature and directly affected compatibility test modules.
`final-fix-report.md` retains exact commands and RED/GREEN chronology;
`final-fix-targeted.xml` is genuine JUnit output in the same ignored evidence directory.
Coverage includes create/update post-commit peer interleaving followed by a stale
Save; malformed known UI/draft import and mounted stored-checkpoint fallback;
historical result counts/text/configuration after restore and reopen; explicit
snapshot preview/cancel/replacement, including unreadable-store read-only inspection;
persisted current/Previous choice with failed/canceled/pending reruns; slow raw tag
entry/reopen/failed Save; builtin copy defaults; source/full-excerpt labels;
unsupported authored template export; and lazy screen teardown. New controls/dialogs
use keyboard, focus and geometry assertions at 80×24, 120×40 and 160×50. No additional
screenshot or visual refinement round was performed. Scoped Ruff and formatting
passed; legacy interop lint uses its existing narrowed rule selection.

This is targeted correction evidence, not a replacement for the original non-green
integration run or a new startup/platform qualification. Earlier task notes that say
In Progress/pending review are chronological records; tasks 31421–31427 have since
received their task-level reviews and remain Done. Final targeted acceptance is
recorded below; integration into dev still requires the user's choice.

## Final acceptance after the requested follow-up

The [bounded follow-up plan](superpowers/plans/2026-09-04-chunking-lab-runtime-boundary-followup.md)
resolved the import guards at commit `9c3f69bd98`: the runner now calls narrow
runtime-owned preprocessing and sanitation adapters. Vendor access stays inside
the existing runtime seam, while runner limits, prescan ordering, resource
accounting and actual full-pipeline execution remain unchanged. Neither the guard
allowlist nor the vendor tree was modified.

The first combined follow-up gate produced470 passes and one narrow workflow
failure. Instrumentation proved that deferred splash startup could push Chat over
the tests' manually mounted Lab. Commit `a89f14d6d3` corrects only the two local
Lab fixtures' initial-screen ownership and adds real callback RED2/GREEN2
regressions. Production startup, the shared app factory and splash timers remain
unchanged; this does not qualify automatic startup or general cold-start behavior.

At that final code/test commit, the combined feature/compatibility selection
passed **473 tests with zero failures, errors or skips, and 2 known warnings in
106.29s**. It is the earlier468-case selection plus three real runtime-adapter
tests and two fixture regressions. Both tasks and the complete follow-up received
independent review; the final reviewer found no issues and independently parsed
the genuine XML. Scoped lint, formatting, compilation and whole-branch whitespace
checks passed, with previously documented legacy runtime lint exclusions.

Exact command/output, the intermediate failed run, diagnosis and reviews are in
`.superpowers/sdd/2026-09-04-chunking-lab-runtime-boundary-followup/`:
`controller-final-verification.md`, `final-targeted.xml`,
`final-targeted-after-fixture.xml`, `lab-startup-race-diagnosis.md`, and
`final-review.md`. The two remaining warnings are Requests dependency
compatibility and vendored datetime deprecation; no dependencies were upgraded
or warnings suppressed. This is a targeted gate, not a full repository sweep.

## Historical controller gate before the follow-up: not green

At correction commit `5d0df113ff`, the combined targeted feature/compatibility
selection produced **466 passed, 2 failed, 9 warnings in 100.58s**. Both failures
are in `Tests/Chunking/test_template_runtime.py::TestEnumerationGuards`:

- `test_exactly_one_flat_mapper_in_production`
- `test_the_mapper_guard_can_see_what_it_guards`

An exact two-node rerun reproduced both failures in 1.02s. The import census
detects `lab_runner.py:225`, introduced by this branch's Task5 commit `1a26e51827`:
`_child_admission` then imported and used the vendored `TemplateProcessor` for bounded
preprocessing/resource admission. It does not construct a second flat-template
mapper, but the existing guard permits vendor-template consumers only in
`template_runtime` and `auto_selection`. This is a branch-introduced integration
gap, **not a proven baseline failure** and not waived by the clean final-fix review.
The runtime-seam/guard contract under ADR-078/ADR-118 still needed reconciliation
at that checkpoint; the later follow-up resolved it without widening the allowlist.

The final scoped reviewer recorded this outside its correction diff, without
waiving branch readiness. The original single final correction wave was exhausted;
no second wave was dispatched in that pass and no source fix was made by the
controller. TASK-31428 remained In Progress until the user requested continuation
and the separately reviewed follow-up above completed AC15. The failed evidence
and branch/worktree are preserved; no merge or push has occurred.

Genuine XML: `controller-final-targeted.xml` and
`controller-final-guard-repro.xml`; exact commands and diagnosis:
`controller-final-verification.md` in the plan-owned evidence directory. Fresh
controller Ruff checks passed on the new Lab modules/tests, and the entire branch
diff passed whitespace checks. The nine test warnings include the known Requests
and vendored datetime warnings plus seven `record_property`/JUnit xunit2 notices
from the resource fixtures; those reporting notices do not explain either failure.

## Non-green checks and startup qualification

The required seven-module integration selection produced294 passes and33 failures.
An exact failed-node replay on Task8 BASE reproduced30 failures: two incumbent
ingest progress-color assertions,27 existing navigation/probe/focus/startup
failures, and one missing `research_workspace` migration owner. The original run
did not request JUnit; its retained failure extract is explicitly labeled as a
transcript extract. BASE and later runs have genuine XML artifacts.

Three navigation tests passed that first BASE replay but failed the amended
selection: overlapping FIFO navigation, search→Library RAG, and Study Escape.
A controller-only, isolated instrumentation probe subsequently explained the
boundary: a seven-second enabled splash remained active when the tests exhausted
their150×20ms readiness polling, before any initial-screen push. With the sole
test-time intervention of disabling splash, the exact three current-code tests
passed in9.45s. On untouched BASE with enabled splash, Search and Study also
failed with the splash still active after6.84/6.51s; FIFO passed in a slower9.69s
run. Iteration-count polling is not a guarantee the splash has closed.

This is a bounded causal explanation, not a claim the original integration gate
was green or general cold-start performance is qualified. No production startup
or unrelated harness changes were made. Full probe commands, observations and
limits are in `task8-startup-diagnosis.md`, with
`task8-startup-no-splash.xml` and `task8-startup-base-instrumented.xml`.

Earlier selections also reproduced three pre-existing private-SQLite census
failures and two pre-existing CSS guards on their exact task BASE revisions.
Those unrelated owners/classes were left unchanged. Known Requests dependency
compatibility and vendored datetime deprecation warnings remain. This is not an
all-checks-green or cross-platform qualification.

## Platform, resource and privacy limits

Actual execution evidence is macOS/arm64 Python3.12. Windows preview is explicitly
refused; Linux has not been qualified here. macOS applied a61-second CPU limit but
rejected the attempted1GiB address-space limit. The32MiB working-payload admission
estimate is **not an RSS cap**: an admitted formatter fixture reached480,313,344
bytes peak RSS (about458MiB). The final follow-up run's formatter fixture reached
481,935,360 bytes (about460MiB), with the61-second CPU limit recorded as applied.
These are measured fixtures, not universal memory guarantees, and
OS reaping latency is not a portable hard deadline.

Full samples/results are local recovery data and are included in exports. Private
permissions are not encryption; Clear is not secure erasure. One early Task1
review probe imported the app without temporary-profile isolation and read normal
configuration/ensured an existing `chat_dicts` directory. No prior fingerprint
exists, so non-mutation cannot be proven; no content write was observed. Later
probes used config, data and HOME isolation before app imports. The incident is
retained rather than claiming every probe was isolated.
