# Chunking Lab verification record

## Post-merge live UAT correction (2026-09-05)

Normal Library entry could remain in loading because the lazy worker ran before
Textual finished mounting and exited through its teardown guard. The local
follow-up defers dispatch until after refresh, retaining the existing guard and
coordinator boundary (ADR-118). A yielding-Mount regression failed before the
correction; final targeted UI/results/recovery tests passed66, scoped static
checks passed, and independent review found no issues. Real A/B authoring,
advanced configuration preservation, local template saving, reopening and
forced-process-exit recovery passed the bounded acceptance run. Two non-blocking
presentation findings and unexercised cases remain explicit in the
[UAT report](Chunking_Lab_UAT_2026-09-05.md). The user subsequently authorized a
separate follow-up PR against dev; earlier merge/CI evidence below does not cover
this new change. Publication verification is recorded in the follow-up plan.


Branch: `codex/chunking-lab`, originally based on `origin/dev` commit
`1a82db60ce47890c6d2df9f918f80309c8608ea6`, rebased onto
`93388ba69b7499c2bc3180fc26c82d7f341871a7` on 2026-09-05. Architecture:
[ADR-118](../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md).
[Workflow and limits](Chunking_Lab.md),
[implementation plan](superpowers/plans/2026-09-04-chunking-lab.md).

## PR 2416 integration and Qodo corrections (2026-09-05)

The user explicitly authorized rebasing onto latest dev, addressing PR feedback,
and merging after verification. This supersedes the earlier publication-only
authorization below. The 31-commit rebase completed at `e96ebdb23c`; additive ADR
and lesson conflicts retain both branches' content. The diagnostic inventory was
regenerated against the integrated sources after reviewing the same exception-
class-only cleanup diagnostic and explicit private export sink described below.
The pre-rebase head `cfd097b1aa` remains on backup branch
`codex/chunking-lab-pre-rebase-20260905`; historical evidence commit IDs remain valid.

All five Qodo findings on the original PR head are corrected:

- Recovery (comment 3940995304): raw size/count admission precedes full nested
  Pydantic validation; only validated sessions reach pruning/digest processing.
  Malformed structure and oversized pre-validation cases have regression tests.
- Record fields (3940995308): complete Google-style argument/return documentation.
- Comparison (3940995314): structured summary/distribution/budget/delta TypedDicts,
  including unavailable values and optional token deltas; no runtime change.
- Preview limits (3940995315): shared named ceilings for defaults and validation,
  without conflating independent IPC/working-memory policies.
- Template import (3940995316): regular-path admission plus nonblocking descriptor
  open and `fstat` closes the FIFO replacement race. Real FIFO/race tests verify
  rejection without session changes; ordinary imports preserve source permissions.

The affected ingest compositor tests initially reproduced two progress-color
failures (146 passed, 2 failed). Their local host loaded only the boot stylesheet,
omitting the lazy Library rules. Loading existing `APP_STYLESHEETS` fixes both
assertions (2 passed in 4.53s); production styling and shared harness are unchanged.
The two formerly failing diagnostic-architecture tests now pass on latest dev;
their source guards were corrected upstream, not weakened in this branch.

Final combined targeted feature, compatibility, route/ingest, diagnostic and
private-SQLite gate: **635 passed, zero failures/errors/skips, 5 warnings in
157.71s**. Genuine XML: `qodo-final-targeted.xml` in the follow-up evidence
directory. Warnings are the existing Requests version warning, vendored datetime
deprecation, and three unrelated invalid-escape SyntaxWarnings. Scoped Ruff,
format checks and whitespace checks pass; the legacy ingest module retains its
existing narrowed lint selection. Independent correction review found no
Critical, Important or Minor findings and independently ran recovery/comparison
(82 passed) and template import/export (3 passed). No full-suite sweep or broader
platform/startup qualification is claimed. Existing ADR-078/ADR-118 apply; no new
architecture decision is introduced. Current-head remote review and required CI
remain merge gates, not implied by these local results.

### Remote review and performance follow-up

Qodo reviewed published correction head `4cf786996657bb187840863cca97d697979f7850`:
[review summary](https://github.com/rmusser01/tldw_chatbook/pull/2416#issuecomment-5552646955)
reports zero remaining bugs/rule violations and marks all five findings resolved.
All five inline threads received fix/evidence replies and are resolved.
[Required CI run 33974600625](https://github.com/rmusser01/tldw_chatbook/actions/runs/33974600625)
passed both PR Fast Lane and Derived artifacts reproduce from their sources.
The six GGUF platform jobs, CSS guard and Backlog guard also passed. CodeRabbit's
success status denotes a skipped review on non-default dev, not a second approval.

The separate [Perf Guard run 33974600652](https://github.com/rmusser01/tldw_chatbook/actions/runs/33974600652)
failed one boot ratchet: 276 ancestor-scoped bare-type rules against a 274 ceiling
(14 passed, 1 failed, 3 optional-dependency skips). Dev's exact base had a green
Perf Guard run 33965632081. Local replay reproduced the feature's 276-rule failure;
this is a branch-introduced integration issue, not waived because the job is optional.
The fixed EditorRegion edit actions and Library Lab-entry actions now use their
existing unique button IDs as selector subjects. Declarations and DOM are unchanged;
the generated widget bundle was rebuilt. The unchanged exact guard then measured
274 and passed. The ceiling was not raised; zero remaining headroom is explicit.
This is a scoped selector-indexing correction, not broader startup qualification.

The combined exact Perf Guard selection plus full Lab screen/recovery-flow modules
passed **75 tests, zero failures/errors/skips, 10 warnings in 117.75s**. Genuine
XML: `pr-perf-correction.xml` in the follow-up evidence directory. This includes
the three viewport workflows and real computed-style equivalence guard. Warnings
remain unsuppressed: Requests compatibility, existing datetime/escape warnings,
boot-budget headroom/snapshot notices, joblib's semaphore-exhaustion serial fallback,
and one unchanged RAG startup `call_from_thread` callback warning. No dependency,
semaphore, startup or global styling repair is claimed by the selector correction.

### Final code acceptance

On `13441478984bf17cd28f62172bce4f59df0f85d9`, Qodo again reports zero remaining
issues and all five review threads are resolved. [Required CI run33975848478](https://github.com/rmusser01/tldw_chatbook/actions/runs/33975848478)
passed Fast Lane and Derived Artifacts, completing at 2026-09-05T16:02:57Z.
[Perf Guard run33975848601](https://github.com/rmusser01/tldw_chatbook/actions/runs/33975848601)
passed; the six GGUF jobs plus CSS and Backlog guards passed. The dedicated pinned-
workspace platform-evidence job was skipped by its own trigger policy; cubic was
neutral and CodeRabbit skipped review. No actionable comments remain. Fresh fetch
confirmed latest dev93388ba69b is an ancestor of the reviewed head.

TASK-31645 AC17 and all earlier criteria are accepted on the recorded local and
remote evidence. The final task/docs commit changes no production or test source;
its own current-head required CI and review are still checked before merge.
Historical failures, warnings, privacy incidents and platform/resource limits
above and below remain qualifications, not erased by this acceptance.

## Historical pre-push bookkeeping correction (2026-09-05)

The user selected push/create PR, then explicitly approved correcting task-ID
collisions and the diagnostic inventory before publication. No implementation
behavior changed in this correction. ADR-118 remains applicable; no new ADR is
required for task identities or regeneration of an existing inventory.

Fresh verification at `11232f8d3a`: **473 passed, 2 known warnings in 95.47s**;
genuine XML is `pre-push-targeted.xml` in the follow-up evidence directory below.
The full repository suite was not run. Initial preflight passed five of six
checks; only the production diagnostic inventory was stale.

Review against inventory pin commit `b7f8efde73` found exactly one added diagnostic
in `app.py`: `Chunking Lab writer cleanup failed: {}` interpolates only
`type(cleanup_error).__name__`, not exception text, sample content, a path, URL,
or secret. The new persistent sink is `_write_selected_file` in
`chunking_lab_screen.py`: an explicit user-selected template/recovery export,
with path validation, absolute-path requirement, explicit overwrite choice,
private opening and an identity precondition passed to the existing atomic
private writer. Export payloads intentionally include authored/sample/result
data as documented in the privacy limits; this is not a new diagnostic log.
The regenerated inventory changes only that app call count/digest, the export
sink row, and their aggregate counts. No logger, export behavior, or guard was
changed to make the inventory pass.

Remote `dev` owns unrelated tasks 31421–31424. Its 31421/31422 records have
created_date `2026-09-04 00:00`; 31423/31424 have `2026-09-05 01:07`.
Their add commits are `6c71826b11` and `4bf1187f8a`; the Lab chain was introduced
by `147d7476dd` with creation dates starting `2026-09-04 23:10`. Thus the usual
older-created-date keeper rule would split ownership across the two slices.
For this user-approved branch-only correction, the entire unpublished Lab chain
voluntarily relocates, avoiding edits to upstream tasks and preserving its
dependency order. A sweep of 333 local/remote refs and 65 worktrees observed a
maximum of 31637 before allocating these replacement IDs:

| Original Lab task | Current Lab task |
| --- | --- |
| 31421 | 31638 |
| 31422 | 31639 |
| 31423 | 31640 |
| 31424 | 31641 |
| 31425 | 31642 |
| 31426 | 31643 |
| 31427 | 31644 |
| 31428 | 31645 |

Task filenames, frontmatter, dependencies, plans, ADR references and lessons
use the current IDs. Each task retains a Renumbering provenance section.
Historical commits and ignored review artifacts retain their original IDs;
use this mapping when following that evidence. Upstream files are untouched.
Publication status is reported in the PR; no merge is authorized or implied.

After regeneration, all six `scripts/preflight.sh` checks passed. Targeted
`Tests/Architecture/test_derived_artifact_checkers.py`: 74 passed, one existing
Requests warning in 0.75s. All eight replacement records are unique, Done with
checked ACs, and have lower-numbered dependencies; no replacement ID exists on
the checked `origin/dev`. Whitespace checks passed; production and test sources
are unchanged by this bookkeeping correction.

The additional `Tests/Architecture/test_persistent_diagnostic_inventory.py` run
was **64 passed, 2 failed, 1 skipped, 10 warnings in 177.83s**, not green. The
regenerated-inventory/topology test passed. The failures are
`test_reviewed_diagnostic_changes_are_metadata_only` (three expected old Library
diagnostic labels are absent) and
`test_task_15743_exception_types_survive_loguru_forwarding` (the existing Console
activity-receipt diagnostic is not positional metadata). Both source assertions
were replayed against an exact `git archive` export of the feature BASE
`1a82db60ce` and produced identical complete failure lists. The test module itself
is unchanged from BASE. This is a two-assertion source replay, not a full BASE
pytest run. The skip concerns unavailable historical TASK-15743 commits; warnings
include existing Requests compatibility and invalid-escape syntax warnings.
No unrelated diagnostic repairs were made or guards suppressed. The PR must not
claim that every architecture check or the full repository suite is green.

## Review and targeted evidence

All eight tasks received independent specification and quality reviews. Task8
screen integration is `c1b320d11b`; its reviewed edit-drain correction is
`cd3e13926a`. Whole-branch review at `462e5cc30e` found seven Important gaps and
five bounded refinements. The single final correction commit `5d0df113ff` received
scoped independent re-review: all twelve findings addressed, no new breakage in
that fix diff. The user-requested runtime-boundary and test-fixture follow-up
also passed independent task and final review. TASK-31645 is complete with AC1–15
verified; the final targeted gate is **473 passed, 2 known warnings in 106.29s**.
At that acceptance checkpoint, no merge or push had been performed. The original
working checkout was not used for implementation.

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
In Progress/pending review are chronological records; tasks 31638–31644 have since
received their task-level reviews and remain Done. Final targeted acceptance is
recorded below; the subsequent publishing choice is recorded above.

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
controller. TASK-31645 remained In Progress until the user requested continuation
and the separately reviewed follow-up above completed AC15. The failed evidence
and branch/worktree were preserved; no merge or push had occurred at that checkpoint.

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
