# Canvas PR #2537: latest-dev integration

Status: implementation and review qualified. Final documentation-head CI and
normal protected PR merge remain; no merge is claimed here.

## Scope and identity

The user authorized rebasing PR #2537 onto latest `dev`, addressing Qodo
findings, verifying the resulting head and merging after required checks.
No new Canvas capability or SQLite authority contract is introduced.
ADRs 124 and 125 continue to govern this direct implementation follow-up.
Backlog follow-up: TASK-32161; completed SQLite prerequisite: TASK-32160
(renumbered from conflicting TASK-31942 with provenance in its task file).

- Published review head: `feaa10790816b4ffb3a3190d7ab3877b1cd30d6a`.
- Rebase target: `2e3389e694e93592a1c66e5c3416bf29a1057d6c` (`dev`).
- Rebased implementation head: `ec133ecca7eef1ff771ffce65ca29b41d339f5ee`.
- All 91 feature commits retained. The import conflict preserves upstream
  `save_setting_to_cli_config`; independent documentation additions survive.
- Evidence root: `/private/tmp/canvas-pr2537-integration.JWVMtm`.

## Upstream-owner qualification and baseline attribution

The 123-case owner selection covers hidden-run Canvas authority, schema
migrations, trace compaction, receipt bootstrap, runtime lifetime, Buddy/TTS
ownership, continuation isolation, and startup/preimport guards.
Result: **120 passed, 3 failed, 6 warnings, 70.03s**, exit 1 (`owner.log`,
`owner.xml`, exact selection in `owner-integration-invocation.json`).

This is not represented as an entirely passing gate. All three failures were
reproduced against an immutable `git archive` of the exact new `dev` target,
with the same Python 3.12.11 environment and repository-isolated pytest fixtures:
**4 passed, 3 failed, 1 warning, 11.54s**, exit 1 (`dev.log`, `dev.xml`).
The baseline source is retained at
`/private/tmp/canvas-pr2537-dev-baseline.kYlow1`.

| Check | Unmodified dev | Rebased PR |
| --- | --- | --- |
| v68 upgrade source-pin test | Expects 69; current DB is 70 | Same failure |
| v69 fresh/upgrade schema parity | Builds current 70 instead of historical 69 | Same failure |
| Screen-preimport own modules | 505 / 500 limit | 504 / 500 limit |

The two failing tests in
`Tests/DB/test_chachanotes_v69_trace_source_pin_migration.py` are byte-identical
between target and rebased PR. The dedicated historical migration rollback,
retry and malformed-predecessor checks pass on both. The preload breach is
existing upstream debt; the PR removes one module from that phase. Per ADR-097,
no limit was raised and no over-budget snapshot was blessed. These unrelated
baseline failures are disclosed, not silently repaired or omitted from results.

The owner gate's other measured startup limits pass: boot import **626/660**,
UI-ready **965/973**. Existing requests-dependency/syntax and headroom warnings
remain visible. No full-repository sweep, package installation, host repair,
runtime network expansion or developer-profile access was performed.

## SQLite covering gate

The fresh 12-file affected selection passed: **721 passed, 2 Windows-only
skips, 1 existing Requests dependency warning, 266.79s**, actual exit 0
(`sqlite.log`, `sqlite.xml`). It includes connection quiescence, private SQLite
functional/interop/lock preservation/process/protocol checks, owner inventory
and SQL validation, installed helper distribution, and TTS helper lifecycle,
SQLite policy and proof. The actual ordinary/abrupt app-exit lock-cohort cases
ran. These are current rebase integration results, separate from the earlier
full prerequisite qualification and from the new focused helper-unit additions.

## Canvas covering gate

The fresh Canvas selection passed: **1,387 passed, 2 optional Firefox/WebKit
skips, 1 existing Requests dependency warning, 849.96s**, actual exit 0
(`canvas.log`, `canvas.xml`). Mandatory Chromium ran. Selection: `Tests/Canvas`,
Console Canvas controller, Canvas tool provider, Console message actions,
Chatbook Canvas round-trip, Canvas gateway distribution, served control spawn
and kill-switch tests. Required CI workflow contracts are covered separately
by the Fast Lane run, not silently counted in this selection.

This run exercised real parent/all-child restart, native and served zero-egress
adversarial cases, confirmed-unsent repair, native auto-open/hot reload,
TLS/two-child reconnect flows and byte-exact packaged closures. Product/test
bytes were frozen throughout. Subsequent final authoring test additions and
docstring/formatter-only changes receive a separate focused run.

## Required Mermaid rebuild path

The new checker was run without the offline-cache override, exercising actual
acquisition from the declared public URLs. All hash/size-pinned inputs were
accepted and all **6 generated outputs** reproduced exactly, actual exit 0
(`online-mermaid-rebuild.log`). This is build-time source acquisition only;
the Canvas zero-egress runtime and admitted asset bytes remain unchanged.

## Local Fast Lane qualification limit

The exact required PR Fast Lane selection plus the new private-helper unit file
completed: **783 passed, 6 failed, 1 existing Requests dependency warning,
361.78s**, exit 1 (`fast-lane.log`, `fast-lane.xml`). All six failures occur in
`Tests/Model_Artifacts/test_operation_leases_process.py` while allocating the
test's `multiprocessing.Event`/`SemLock`: errno 28, before lease/application
behavior runs. These test and owner files are unchanged from the rebase target.

Running that exact six-test file on immutable `dev` reproduced all six
allocation failures in **1.55s**, exit 1 (`dev-leases.log`, `dev-leases.xml`).
This is the previously diagnosed local semaphore exhaustion, not evidence of
six passing concurrency scenarios. No tests were disabled, no host resources
were removed, and no process cleanup or machine restart was attempted. The
required hosted Fast Lane must still pass on the published head before merge.

The updated local preflight separately passed all **7** checks, actual exit 0
(`final-preflight.log`). No inventory, generated-asset or startup guard was
weakened to obtain those results.

## Qodo and independent review closeout

Fix commit: `27cf82559a26d9ceb0425d32fbb54609006c9664`. The independent
scoped re-review passed with no Critical, Important or Minor findings. Final
authoring coverage (including admitted, revoked and unknown profiles through a
real run coordinator) passed all **4** tests; final scoped Ruff and formatting
checks passed for all **7** changed Python files. The agent's final combined
focused selection also passed **46** tests. The only changes after the large
Canvas gate were test coverage, formatter reflow and the public docstring.

Qodo's exact-head review reports zero bugs. All five threads received evidence
replies and were resolved:

- [3965119518: retained-profile guidance](https://github.com/rmusser01/tldw_chatbook/pull/2537#discussion_r3965480373).
- [3965119495: public helper documentation](https://github.com/rmusser01/tldw_chatbook/pull/2537#discussion_r3965479487).
- [3965119500: immutable JavaScript ABI disposition](https://github.com/rmusser01/tldw_chatbook/pull/2537#discussion_r3965479478).
- [3965119508: required Mermaid reproduction](https://github.com/rmusser01/tldw_chatbook/pull/2537#discussion_r3965479537).
- [3965119485: direct helper-unit coverage](https://github.com/rmusser01/tldw_chatbook/pull/2537#discussion_r3965479508).

The residual helper item in Qodo's summary cited only old integration tests and
overlooked the new unit file. Independent review verified that coverage directly.
The casing-only request was technically declined under ADR-124: both qualified
worker/renderer pairs use the same JavaScript ABI; changing those bytes requires
a new profile, not an in-place style edit. No runtime bytes changed.

## Hosted CI and final publication

[Required Derived Artifacts run 34320576418](https://github.com/rmusser01/tldw_chatbook/actions/runs/34320576418)
passed on exact code head `27cf82559a26d9ceb0425d32fbb54609006c9664`. Both
**PR Fast Lane** and **Derived artifacts reproduce from their sources** succeeded.
The six locally semaphore-blocked process cases remained enabled in the hosted
Fast Lane. CSS, Backlog ID, UI latency and all six GGUF platform evidence jobs
also passed. CodeRabbit skipped actual review; its success status is not claimed
as review approval.

TASK-32161's implementation/review acceptance is complete. This documentation
closeout changes no product or test behavior. Its published head must still
receive current-head required checks before the user-authorized normal merge;
no stale green result or admin bypass is permitted. The actual merge result is
recorded on the PR and in the retained SDD progress ledger when it occurs.
