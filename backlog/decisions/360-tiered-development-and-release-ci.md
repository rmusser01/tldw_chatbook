# ADR-360: Tiered development and release CI

Status: Proposed; activation requires live qualification
Date: 2026-11-13
Task: TASK-40727
Amends: ADR-103; preserves ADR-359 runner isolation

## Context

The owner requested a light development gate, nightly regression coverage and full qualification before releases. The previous fast lane grew to dozens of jobs, while the old serial nightly was disabled after repeatedly failing to complete. GitHub schedules only the default branch's workflow. Reducing PR coverage before replacing that nightly would leave an unobserved gap.

## Decision

Keep the stable required context and every existing broad regression command. Add a bounded smoke job and a reviewed subsystem smoke map. Enable selective gating only with `CI_TIERED_ENABLED=true`; default behavior remains broad on dev PRs. Always run artifact/Ruff preflight and the smoke job when selective policy is active. After merging, dev pushes run smoke on the merged tree without repeating PR regression coverage; activation requires PR-only protected dev updates. Unknown subsystems, new source files, deleted/renamed inputs, shared configuration, dependency/CI/test harness changes, migrations, recovery and authority/security changes select broad regression. The map is deliberately finite (Chat, UI, Widgets, Notes and CSS): absence means broad, never no tests. These are representative subsystem checks, not inferred dependency closure or every affected test. Explicit authority/storage-related path names retain broad coverage; review must also identify semantic high-risk changes whose paths do not reveal them. Task bookkeeping is covered by preflight and smoke; it does not by itself force a broad run. This explicitly supersedes ADR-103's rejection of all path-based selection, but never filters the required workflow itself.

A default-branch-owned nightly dispatcher requests the existing broad regression workflow on dev, including the Library shell matrix previously repeated on every dev push. The dispatched run's immutable GitHub SHA is the checkout source for every job. Skip only an already-successful nightly run at the currently resolved dev SHA; failures retry the next night. A race with a new dev push can select a newer SHA, never mix SHAs within the run. Keep the former deep cross-platform nightly available but disabled; it is not the replacement's coverage claim. Nightly covers the preserved regression census, not every test in the repository.

PRs targeting main require broad regression plus the existing comprehensive Tests workflow and package build/install checks through reusable workflows. Publication also depends on successful comprehensive qualification and package checks on that exact main commit; neither a previous night's result nor another SHA is accepted. Weekly or fortnightly releases remain an owner decision, not automatic publishing on a timer.

Use the existing hosted bulk runners for overnight and comprehensive jobs: their 120-minute budgets exceed the disposable local workers' 45-minute deadline. Preserve same-repository local PR routing and hosted fork fallback. Do not increase host resource limits.

## Rollout and rollback

Keep the opt-in unset while #3529 establishes its baseline. Merge this policy only after focused contracts and review; install the identical nightly dispatcher on default-branch main independently of unrelated product changes. Qualify one nightly and a promotion candidate, require the stable context on main with strict protection, then enable the opt-in. Unset it to restore broad dev PR gating. Never weaken protection to qualify the rollout.

## Alternatives and limits

Weekly-only regression delays attribution across many changes; nightly bounds that delay. Running the whole test tree per commit recreates the throughput problem. Automatically inferred import coverage is unsound for plugins, dynamic imports and data fixtures; use an explicit map with conservative fallback. The small initial map limits speedup until more subsystem smoke selections are measured and reviewed. Smoke success is not proof that unselected integrations work; that feedback moves to nightly. Five to ten minutes is a target, not a measured promise.
