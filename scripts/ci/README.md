# Development, nightly and release CI (TASK-41564 / ADR-366)

This policy is **prepared, not activated**. `CI_TIERED_ENABLED` must remain unset
until the rollout checks below pass. Execution uses GitHub-hosted runners. The owner
canceled the entire self-hosted CI effort and benchmark on 2026-11-17.

| Event | Checks | Blocking result |
| --- | --- | --- |
| PR into dev, opt-in unset | Existing artifact preflight and broad PR regression census | Existing Derived artifacts context |
| PR into dev, opt-in true | Preflight, CI contracts, core smoke, representative checks for changed Chat/UI/Widgets/Notes/CSS subsystems | Same context, fail-closed |
| High-risk or unclassified dev PR | Existing broad census; no test command removed | Same context |
| Protected dev push, opt-in true | Preflight, CI contracts and core smoke on the merged tree | Push result; does not rerun every PR matrix |
| Nightly, changed dev | Existing broad census, all smoke selections, Library shell matrix and Backup_Recovery owner suites | Nightly regression **workflow conclusion**, including Library shell |
| PR into main | Broad regression, comprehensive core/UI suite, native lease contracts, package build and installed-distribution checks | Stable Derived context and release-only qualification verdict |
| Protected main push / manual TestPyPI publication | Comprehensive tests and package checks on that exact commit before publishing | Both publishing jobs need successful qualification |

The small smoke tier targets 5–10 minutes including setup and queueing; this is
not a measured server guarantee. The exact combined light-job invocation passed
1,974 tests with one intentional child-test skip in 156.48 seconds locally. After
rebasing onto #3529, all mapped selections plus CI contracts passed 1,977 tests
with one intentional skip in 184.02 seconds locally. These are different test
sets, not a speed comparison. The
isolated smoke union passed 291 cases in 68.98 seconds. Measure the hosted job
from creation through the required aggregate before reporting an end-to-end number. It covers representative seams, not
all transitive effects of a changed module. Most coverage is intentionally nightly.

## Selection and failure behavior

`scripts/ci/tier_policy.py` is the single selector. It compares the PR base SHA to
the tested merge SHA with `git diff --no-renames --name-status -z`. New source
files, removals, moves, changes outside the explicit subsystem map, dependencies,
CI/shared fixtures, DB/backup/runtime boundaries, and authority/storage-related
path names retain broad coverage. Docs and backlog documents remain broad;
only task bookkeeping has an explicit exception, protected by artifact checks.
A malformed range fails preflight. Empty diffs select broad. Missing selected
tests, missing outputs, skipped expected jobs and cancellations cannot turn the
required aggregate green. No required workflow uses path filters. Required preflight executes the pinned
Ruff F gate and debt ratchet; the standalone lint workflow remains available.

For semantic high-risk changes whose filenames do not reveal their effect,
apply `ci:full` before the next push or reopen the PR after applying it. The label
forces broad coverage when the next normal PR event runs; applying a label alone
does not start another run. Do not dispatch a gate to satisfy PR protection.
Unknown/new subsystems stay broad until their smoke selection is reviewed and
qualified. A CI policy or test-harness change itself always takes the broad path.

## Nightly and immutable evidence

`regression-nightly.yml` is a small default-branch-owned dispatcher at 09:17 UTC
(01:17 PST / 02:17 PDT). It skips a dev SHA only if a matching `Nightly regression`
run succeeded or is still active. Failed/cancelled/skipped runs retry next night.
API errors fail the dispatcher. The actual verdict belongs to the dispatched
Derived Artifacts run, **not the dispatcher**. All its jobs use that run's
immutable checkout SHA. A concurrent dev push may advance the selected snapshot;
no job checks out mutable `dev` independently.

The old `nightly-deep.yml` remains disabled. This new nightly preserves the current
regression census, not the full Tests tree or the old deep cross-platform matrix.
Comprehensive core/UI plus the existing native lease checks run for releases;
other platform-specific qualification workflows remain separately available.
Smoke, regression and release jobs use hosted runners. The light job directly
selects `ubuntu-latest`, including fork PRs. Existing broad workflows select
their hosted defaults with `CI_TEST_RUNNER` and `CI_CONTROL_RUNNER` unset.

## Rollout (do not skip steps)

1. Rebase only this policy onto current dev and retain a successful hosted full
   PR gate as the timing baseline. The earlier #3529 result is historical
   evidence from the canceled local-runner topology and does not measure the
   current hosted rollout. The canceled #3532 benchmark and #3537 migration
   impose no wait on light CI. Preserve current queue and review requirements.
2. Run affected workflow contracts and actionlint. Run one normal full PR gate
   for this policy head. Keep `CI_TIERED_ENABLED` unset throughout preparation.
3. Merge the policy into dev. Promote only `regression-nightly.yml` to default
   main in a dedicated reviewed bootstrap PR; a cron present only on dev does
   nothing. Check that the workflow is active in GitHub.
4. Dispatch the nightly workflow and inspect the **resulting** Nightly regression
   run on dev. With opt-in still off it runs broad regression, Library shell and
   every light selection, proving both coverage paths before cutover. Retain SHA,
   job conclusions, runner identities and wall time; don't treat dispatch success
   as test success. Check the next scheduled execution as well.
5. Qualify one exact release candidate through a safe PR or run
   `gh workflow run derived-artifacts.yml --ref <candidate-branch> -f release=true`.
   Pin and retain the resulting event SHA. This calls comprehensive tests and
   package/install checks, with no publication job in the workflow. Do not publish to PyPI/TestPyPI or merge a product release merely to test.
   Require `Derived artifacts reproduce from their sources` and the nested
   `Full release qualification` verdict on main, bound to GitHub Actions, with
   strict/latest-base and admin enforcement. Verify both exact emitted contexts
   on the main PR; the skipped release wrapper is not the verdict. The extra
   existing verdict prevents earlier dev-push smoke from serving as release
   evidence under the shared Derived name. Read/merge with existing
   rulesets rather than overwriting them. A read on 2026-11-13 found no legacy
   branch protection on main; its active `safe` ruleset only prevents deletion
   and force pushes. Recheck both protections and rulesets during activation.
   Qualify comprehensive tests and packaging before release. Ensure dev requires
   PRs and has no direct-push bypass; post-merge smoke relies on that boundary.
6. Before activation, repeat the live check of current heads, required-status
   association, resolved reviews, mapped/unknown/high-risk/fork routing,
   fail-closed missing/skipped/cancelled outcomes, nightly SHA/all jobs/active
   schedule/deduplication/retry, release SHA/tests/package dependencies,
   branch protection/admin enforcement and hosted runner selection.
   Enable `CI_TIERED_ENABLED=true` only after these checks pass. Verify normal
   mapped-source PR selection, broad fallback and post-merge smoke; compare
   measured wall time with the broad baseline. Clear the flag if cutover fails.

Clearing `CI_TIERED_ENABLED` restores broad dev PR gating and the former dev-push
Library shell cadence. Existing runs keep their original selection. Keep nightly
coverage active during rollback. Release publication remains fully qualified;
do not remove those dependencies to make a failed release publish.

Weekly or every-other-week release timing is a human decision. No timer promotes
dev, updates main, uploads a package, or grants auto-merge.
