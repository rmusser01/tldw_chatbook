# ADR-366: Tiered development and release CI

Status: Accepted; activated by the owner on 2026-11-23 (`CI_TIERED_ENABLED=true`), selection amended by TASK-42521 and TASK-42626
Date: 2026-11-13
Task: TASK-41564
Amends: ADR-103
Execution: GitHub-hosted runners; self-hosted CI canceled by the owner on 2026-11-17

## Context

The owner requested a light development gate, nightly regression coverage and full qualification before releases. The previous fast lane grew to dozens of jobs, while the old serial nightly was disabled after repeatedly failing to complete. GitHub schedules only the default branch's workflow. Reducing PR coverage before replacing that nightly would leave an unobserved gap.

## Decision

Keep the stable required context and every existing broad regression command. Add a bounded smoke job and a reviewed subsystem smoke map. Enable selective gating only with `CI_TIERED_ENABLED=true`; default behavior remains broad on dev PRs. Always run artifact/Ruff preflight and the smoke job when selective policy is active. After merging, dev pushes run smoke on the merged tree without repeating PR regression coverage; activation requires PR-only protected dev updates. Unknown subsystems, new source files, deleted/renamed inputs, shared configuration, dependency/CI/test harness changes, migrations, recovery and authority/security changes select broad regression. The map is deliberately finite (Chat, UI, Widgets, Notes and CSS): absence means broad, never no tests. These are representative subsystem checks, not inferred dependency closure or every affected test. Explicit authority/storage-related path names retain broad coverage; review must also identify semantic high-risk changes whose paths do not reveal them. Task bookkeeping is covered by preflight and smoke; it does not by itself force a broad run. This explicitly supersedes ADR-103's rejection of all path-based selection, but never filters the required workflow itself.

A default-branch-owned nightly dispatcher requests the existing broad regression workflow on dev, including the Library shell matrix and Backup_Recovery owner suites previously repeated on every dev push. The dispatched run's immutable GitHub SHA is the checkout source for every job. Skip only an already-successful nightly run at the currently resolved dev SHA; failures retry the next night. A race with a new dev push can select a newer SHA, never mix SHAs within the run. Keep the former deep cross-platform nightly available but disabled; it is not the replacement's coverage claim. Nightly covers the preserved regression census, not every test in the repository.

PRs targeting main require broad regression plus the existing comprehensive Tests workflow and package build/install checks through reusable workflows. Publication also depends on successful comprehensive qualification and package checks on that exact main commit; neither a previous night's result nor another SHA is accepted. Weekly or fortnightly releases remain an owner decision, not automatic publishing on a timer.

Main protection requires both the stable Derived context and the existing nested `Full release qualification` verdict, bound to GitHub Actions. A dev-push smoke run emits the same Derived name; GitHub accepts eligible push checks, so that shared name alone does not explicitly distinguish release evidence. Verify the nested verdict's exact emitted context before configuring protection; the skipped reusable-workflow wrapper is not the verdict. This adds no jobs or runtime.

Use GitHub-hosted runners for smoke, regression and comprehensive qualification. The owner canceled the entire self-hosted CI effort and its benchmark on 2026-11-17. Neither is a prerequisite for this policy. The new smoke job directly selects a hosted runner; existing broad jobs use their hosted defaults with the runner variables unset.

Keep the reader closeout child's containment boundary installed before application imports. ADR-126's Linux native storage identity requires read-only kernel mount metadata: the exact own-process `mountinfo` file and `fdinfo` only for a currently identity-verified, tracked directory inside declared scratch. Treat these as bounded runtime resources, without granting a `/proc` subtree, foreign-process access, directory enumeration, descriptor contents or writes. Retain existing resolved-target checks and profile, checkout, network and process refusals; never bypass native qualification or admit startup before containment to make CI pass.

The installed psutil Linux import probes five capabilities with `stat`: own-process `smaps`, `smaps_rollup` and `io`, and the exact `policy0`/`cpu0/cpufreq` CPU-frequency paths. Resolve those fixed paths before installing the boundary and allow only their real `stat` results or errors on Linux. This grants no file contents, sibling paths, enumeration, other metadata APIs or writes, and does not authorize later process metrics reads.

Prometheus registers and samples its default process collector during import. On Linux, permit contents of only the fixed child's exact canonical `stat` and `limits` files through the existing runtime read-file capture; its other initialization accesses use the existing `/proc/stat` read and canonical `/dev/fd` enumeration. Keep the five psutil capability paths stat-only and deny foreign-process files, sibling files, process-directory enumeration and writes. Later resource sampling through `statm` remains outside this authority. Preserve real collector initialization after containment, without disabling metrics or importing application code before the boundary.

Expose the declared vendored `packages/tldw_profile_core/src` import root only after validating that its canonical path remains within checkout. Keep editable finders removed and application imports after containment; a missing source in a synthetic checkout is harmless. Derive runtime read authority from the original filtered interpreter paths, excluding the added vendor search path: checkout already authorizes its contents, and a later symlink swap must not create a foreign runtime read root. Initial escapes and later foreign reads remain refused by the existing containment receipt.

## Rollout and rollback

Keep the opt-in unset until live qualification finishes. Retain a successful hosted full PR run for the light-CI timing comparison. Merge this policy only after focused contracts and review; install the identical nightly dispatcher on default-branch main independently of unrelated product changes. Qualify one nightly and an exact release candidate through a safe PR or the explicit `release=true` Derived Artifacts dispatch without publishing or merging a product release merely to test. That dispatch calls comprehensive tests and package checks at its immutable event SHA and fails the aggregate unless both succeed. Require the stable context on main with strict protection and admin enforcement. Repeat the live routing, evidence, schedule, protection and runner checks before enabling the opt-in, then verify mapped PR selection, broad fallback and post-merge smoke. Unset it to restore broad dev PR gating. Never weaken protection to qualify the rollout.

## Alternatives and limits

Weekly-only regression delays attribution across many changes; nightly bounds that delay. Running the whole test tree per commit recreates the throughput problem. Automatically inferred import coverage is unsound for plugins, dynamic imports and data fixtures; use an explicit map with conservative fallback. The small initial map limits speedup until more subsystem smoke selections are measured and reviewed. Smoke success is not proof that unselected integrations work; that feedback moves to nightly. Five to ten minutes is a target, not a measured promise.

## Renumbering provenance

Renumbered from ADR-360 on 2026-11-18 after dev908e9840 brought in the older beautifulsoup4 decision (Date2026-10-11); this policy was drafted on 2026-11-13. The current sweep covered1090refs,393decision trees,1143distinct decision blobs and55worktrees. The highest filename/header claim was365, so366 follows the canonical highest-plus-one rule without backfilling. Policy semantics and the older merged decision remain unchanged. Evidence: /tmp/tldw-light-ci-policy-adr-reservation-sweep.json.

## Amendment: sanity-check selection (2026-11-23, TASK-42521)

**Why.** The owner enabled `CI_TIERED_ENABLED` on 2026-11-23. The intent was rapid dev merges after sanity checks, with the nightly full run on dev and the release run into main as the broad gates. The selection above defeated that intent. It treated any path outside a five-entry map as broad: every test file, `Docs/` page and `backlog/docs` file, and every path whose name contained one of 18 sensitive words. Replayed over the last 60 dev merges at b31e3e9241, only 5 would have run light; under this amendment 23 do. Meanwhile a starved runner pool held each 62-job PR run for hours.

**Decision.** A dev PR runs light unless it changes an input that can break CI itself, or the schema. These keep the broad run:
- `.github/`, `scripts/ci/` and `packages/`;
- `pyproject.toml`, `setup.py`, `setup.cfg` and `requirements*.txt`;
- `conftest.py`, `pytest.ini`, `.pytest.ini` and `tox.ini` at any depth;
- schema code: any path under `tldw_chatbook/` inside a `DB`, `db` or `migrations` directory, and any changed application module containing uppercase DDL (`CREATE TABLE/INDEX/TRIGGER/VIEW`, `ALTER TABLE`) or `user_version`;
- any non-`test_*.py` Python file under `Tests/`, such as the profile harness or shared rigs;
- non-Python data under `Tests/Architecture/` or `Tests/Performance/` (allowlists, budget snapshots);
- a test module imported, directly or transitively, by more than 8 other test files, or a deleted one that any test imports;
- the test files no PR lane runs whole (`test_library_shell.py`, `test_response_rules_restore.py`), files only push-to-dev or nightly jobs run, files a PR lane runs with `--deselect`ed nodes, and files whose lane nodes are spread over more than one lane step;
- more than 20 test files of the PR's own (changed, importing or data-reading), which would not fit the light job's 30-minute cap when run one after another;
- a deleted file under `tldw_chatbook/`, a type-changed or unmerged path, an empty change range, or the `ci:full` label.

Everything else runs light:
- preflight, `Tests/CI` and the smoke set;
- the PR's changed `test_*.py` files under `Tests/`, the test files importing them (one-line and parenthesized imports, transitively), and the test files that name a changed or deleted test data file;
- the smoke nodes of any touched subsystem;
- nothing more for docs and backlog changes.

"The PR lanes" are the jobs the required check needs, plus the UI census, never the push or nightly jobs. Census entries are read as whole lines and workflow steps as shell words. A file the PR lanes run only by node id runs exactly those nodes, never whole. As in the full run, a new test in such a file runs only once a lane lists it. A file selected whole is never narrowed to its own smoke nodes, because pytest collapses `f f::node` to `f::node`. `scripts/ci/run_light.py` runs each file in its own pytest process, because bootstrap_profile and sandboxed suites must not share one (TASK-32873). It uses the lanes' plugin autoload and `--timeout=180`, with console-p0-gate's install set. It reports every red file, and warns about a file that collects nothing. The light job's cap is 30 minutes.

Sensitive path names (security, storage, recovery, instructions and the rest) no longer force broad runs on the PR. They rely on review, the PR's own tests, the nightly run and the release run. This supersedes the "absence means broad" and sensitive-name rules above. Adding `ci:full` takes effect on the next push or reopen: the workflow does not run on `labeled`, and re-running jobs reuses the original event's labels. Unset the variable to restore broad dev PR gating.

**Cost accepted.** A regression in an integration the PR's own tests don't reach can land on dev. It is caught the next morning, when the 09:17 UTC nightly runs, or at release, and attributed across one day's merges.

## Amendment: capped samples instead of broad fallback (2026-11-26, TASK-42626)

**Why.** Under the amendment above, a dev PR still ran the full CI (about 63 jobs, 42 of them UI shards) when it touched a widely imported test file. PR #3769's preflight gave `Shared test module (9+ importers): Tests/Chat/test_console_dispatch_recovery.py`. Its full run took 3 h 26 min, because jobs waited up to about 2 h 50 min for runners on the account pool shared with tldw_server. The one-at-a-time merge queue (`scripts/merge_queue.py`) held every armed PR behind it for 4.5 h. A light-tier PR (#3518) went from rebase to merge in 21 minutes. #3769 itself would still run broad under this amendment, because it also changes schema inputs (`tldw_chatbook/DB/assistant_generation_state.py`).

**Decision.** This replaces three bullets in the broad list of the 2026-11-23 amendment above:
- "any non-`test_*.py` Python file under `Tests/`, such as the profile harness or shared rigs";
- "a test module imported, directly or transitively, by more than 8 other test files, or a deleted one that any test imports";
- "more than 20 test files of the PR's own (changed, importing or data-reading) ...".

Those cases now run the light tier. The test files that depend on the change are the ones that import a changed or deleted test module or helper under `Tests/`, directly or through other modules, or that read a changed or deleted data file there (or import a helper that reads it).
- **Under the cap:** with at most 8 dependents the light job can run, all of them run, whether a PR lane runs them or not, as under the amendment above.
- **Over the cap:** the change is sampled. Only dependents that a PR lane runs are eligible, up to 8, which is a subset of what the full run would have run. On 2026-11-26, only 1,131 of the 5,872 `test_*.py` files were in a PR lane. 4,707 were in none, and the nightly run uses the same lanes, so those run only at release and their health on the PR's platform is unknown. A sample drawn from them could fail on a pre-existing red and evict an unrelated PR, so they are skipped, and the reason says how many.
- **Budget:** a PR's sampled files share a budget of 8 in all, handed out in turn. Its own changed test files, then the under-the-cap dependents, take the 20 slots first, and the total never exceeds 20. A real 20-file selection measured 37 min locally against the light job's 30-minute cap.

What counts as an importer:
- Imports are resolved by module path with `ast`, as pytest's prepend import mode loads them: `import Tests.a.mod`, `from Tests.a import mod`, `from Tests.a.mod import X`, relative imports, and bare imports of a module in the importer's own directory. Imports inside functions count, and so do the imports of a child script that a test embeds as a string. When such a script is a template that does not parse as a whole, its import lines are parsed one by one.
- A file that imports anything in a package, or lives in it, loads the package's `__init__.py`.
- A file that does not parse counts as an importer, which errs toward more tests.
- A data file's readers are the files that name it.

How the sample is drawn:
- The search runs breadth first over the Tests tree in sorted path order, so direct importers come first and the same diff and tree give the same targets on every machine.
- It follows every dependent, but counts only test files the light job can run that a PR lane runs. It stops at the ninth of those, or when there are no more dependents.
- `Tests/CI` runs whole anyway, so it never counts. Neither do files the light job cannot run: a file no PR lane runs whole, one a lane runs with deselected nodes, one split across lane steps, or one only push or nightly jobs run.
- The PR's own changed test files take the first of the 20 slots, by path. Then come the under-the-cap dependents, and then the samples within their budget of 8. Both are handed out in turn: the first file of each module, then the second, so no single module takes them all.

The full run is kept only where it reaches what light cannot. A change goes broad when it has a dependent that a PR lane (or a push or nightly job) runs but the light job cannot (heavy, deselected, split or push-only), and nothing laned would run in its place: `Only the full run reaches the laned dependents of <path>: <files>`. When no PR lane runs any dependent, the plan stays light and says so: `no PR lane runs a dependent of <path>; only the release run covers its N test files`. The full run would not run them either.

The plan's reason says, for each module or data file that was sampled, cut or skipped, how many test files ran of how many were found. Examples:
- `sampled 6 of 9+ importers of <path> a PR lane runs`, where the `+` means the search stopped at the cap;
- `skipping N no PR lane runs`;
- `skipping N the light job cannot run`.

It also says when a cap applied: `8 sampled files in all`, or `cut to 20 test files`, naming any of the PR's own test files the cut left out.

These stay broad, unchanged:
- `.github/`, `scripts/ci/`, `packages/` and the build files;
- `conftest.py`, `pytest.ini`, `.pytest.ini`, `tox.ini` and `setup.cfg` at any depth;
- schema inputs, and schema code (see the narrowing below);
- guard and budget data under `Tests/Architecture/` and `Tests/Performance/`;
- the PR's own changed test files that no lane runs whole, or that a lane deselects nodes from, splits across steps or runs only on push or nightly;
- removed application files, structural statuses, an empty change range and `ci:full`.

Anything a `conftest.py` loads also stays broad (`Loaded by conftest.py: <path>`). It runs at the start of every suite below that conftest, exactly as an edit to the conftest would. The set is derived, not listed by hand: the imports of every `Tests/**/conftest.py`, resolved as above and followed transitively. It includes the profile isolation harness (`Tests/private_profile.py`, `Tests/real_profile_guard.py`), `Tests/network_guard.py`, the shared doubles and Textual harness, the `__init__.py` of every package with a conftest, and `Tests/Chat/test_console_skill_substitution.py`, a test module a conftest imports. Deleting a path a conftest still loads goes broad too, and so does a data file that a conftest or a file it loads reads.

Schema code is narrowed: `user_version` counts only in the `PRAGMA user_version` form, as every real use in `tldw_chatbook/` writes it. A Python name `user_version` no longer sends a module broad. The old pattern did that to `tldw_chatbook/Chat/console_dispatch_repository.py` in #3769, which has no DDL.

**Cost accepted.** A break in a dependent the sample did not pick, or in one no PR lane runs, can land on dev. The 09:17 UTC nightly or the release run catches it, attributed across one day's merges. `ci:full` remains the opt-in for a PR that should run everything.
