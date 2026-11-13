# Development, nightly and release CI (TASK-40727 / ADR-360)

This policy is **prepared, not activated**. `CI_TIERED_ENABLED` must remain unset
until the rollout checks below pass. ADR-359 owns runner isolation and limits.

| Event | Checks | Blocking result |
| --- | --- | --- |
| PR into dev, opt-in unset | Existing artifact preflight and broad PR regression census | Existing Derived artifacts context |
| PR into dev, opt-in true | Preflight, CI contracts, core smoke, representative checks for changed Chat/UI/Widgets/Notes/CSS subsystems | Same context, fail-closed |
| High-risk or unclassified dev PR | Existing broad census; no test command removed | Same context |
| Protected dev push, opt-in true | Preflight, CI contracts and core smoke on the merged tree | Push result; does not rerun every PR matrix |
| Nightly, changed dev | Existing broad census, all smoke selections and Library shell matrix | Nightly regression **workflow conclusion**, including Library shell |
| PR into main | Broad regression, comprehensive core/UI suite, native lease contracts, package build and installed-distribution checks | Same stable required context |
| Protected main push / manual TestPyPI publication | Comprehensive tests and package checks on that exact commit before publishing | Both publishing jobs need successful qualification |

The small smoke tier targets 5–10 minutes including setup and queueing; this is
not a measured server guarantee. The exact combined light-job invocation passed
1,974 tests with one intentional child-test skip in 156.48 seconds locally. The
isolated smoke union passed 291 cases in 68.98 seconds. Qualify it under local-runner
load before reporting an end-to-end number. It covers representative seams, not
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
Bulk regression/release jobs keep hosted runners because their budgets exceed
our local worker lifetime. Local smoke/PR jobs keep fork fallback and VM isolation.

## Rollout (do not skip steps)

1. Let #3529 merge and retain its complete baseline. Rebase this separate policy
   PR onto that dev commit; preserve its queue/Qodo review requirements.
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
5. Prepare a dev-to-main promotion PR. Require `Derived artifacts reproduce from
   their sources` on main with strict/latest-base and admin enforcement, after
   verifying that context is emitted by the promotion PR. Read/merge with existing
   rulesets rather than overwriting them. A read on 2026-11-13 found no legacy
   branch protection on main; its active `safe` ruleset only prevents deletion
   and force pushes. Recheck both protections and rulesets during activation.
   Qualify comprehensive tests and packaging before release. Ensure dev requires
   PRs and has no direct-push bypass; post-merge smoke relies on that boundary.
6. Enable `CI_TIERED_ENABLED=true` only after those live checks pass. Compare
   ordinary mapped-source PR wall time with the broad baseline. Keep broad
   fallback until each additional subsystem is measured and reviewed.

Clearing `CI_TIERED_ENABLED` restores broad dev PR gating and the former dev-push
Library shell cadence. Existing runs keep their original selection. Keep nightly
coverage active during rollback. Release publication remains fully qualified;
do not remove those dependencies to make a failed release publish.

Weekly or every-other-week release timing is a human decision. No timer promotes
dev, updates main, uploads a package, or grants auto-merge.
