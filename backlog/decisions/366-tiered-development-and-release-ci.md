# ADR-366: Tiered development and release CI

Status: Proposed; activation requires live qualification
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

## Rollout and rollback

Keep the opt-in unset until live qualification finishes. Retain a successful hosted full PR run for the light-CI timing comparison. Merge this policy only after focused contracts and review; install the identical nightly dispatcher on default-branch main independently of unrelated product changes. Qualify one nightly and an exact release candidate through a safe PR or the explicit `release=true` Derived Artifacts dispatch without publishing or merging a product release merely to test. That dispatch calls comprehensive tests and package checks at its immutable event SHA and fails the aggregate unless both succeed. Require the stable context on main with strict protection and admin enforcement. Repeat the live routing, evidence, schedule, protection and runner checks before enabling the opt-in, then verify mapped PR selection, broad fallback and post-merge smoke. Unset it to restore broad dev PR gating. Never weaken protection to qualify the rollout.

## Alternatives and limits

Weekly-only regression delays attribution across many changes; nightly bounds that delay. Running the whole test tree per commit recreates the throughput problem. Automatically inferred import coverage is unsound for plugins, dynamic imports and data fixtures; use an explicit map with conservative fallback. The small initial map limits speedup until more subsystem smoke selections are measured and reviewed. Smoke success is not proof that unselected integrations work; that feedback moves to nightly. Five to ten minutes is a target, not a measured promise.

## Renumbering provenance

Renumbered from ADR-360 on 2026-11-18 after dev908e9840 brought in the older beautifulsoup4 decision (Date2026-10-11); this policy was drafted on 2026-11-13. The current sweep covered1090refs,393decision trees,1143distinct decision blobs and55worktrees. The highest filename/header claim was365, so366 follows the canonical highest-plus-one rule without backfilling. Policy semantics and the older merged decision remain unchanged. Evidence: /tmp/tldw-light-ci-policy-adr-reservation-sweep.json.
