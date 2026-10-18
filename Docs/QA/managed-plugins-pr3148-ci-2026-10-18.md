# PR #3148: remaining CI repairs, 2026-10-18

Source base: `40d61b3ebfd1d05045a5308536e6365777c9b0ce`. First repair code:
`223b59b7616f559e46044f62a703eb61e45c1190`, on dev
`b07fc0425288fb51d17cde5f745051720a7c9db6`.
Existing ADR-097/107/125/126/162/163 govern these routine repairs; no new
storage contract, platform support, dependency, permission, or schema was added.

**Latest local repair:** `ede1a24f48675e25ec7b2e56d26548ec29330642`.
First repair publication `18e8d0adae0e7f6f14ae4d965b7590958563b705` was verified
through GitHub's branch and PR records, preserving draft status and auto-merge off.
Its native macOS and Linux qualification passed; Windows exposed another layer,
repaired below. Total selected local accounting is **1,403 passing outcomes and
five genuine skips** across the named completed phases; this is not a count of
unique tests or a full-suite claim. Second publication/native renewal is pending.


## Observed CI

[Managed qualification run](https://github.com/rmusser01/tldw_chatbook/actions/runs/39144662200)
at the source base failed on all three native runners:

| Runner | Actual result |
| --- | --- |
| macOS | 767 passed, 2 failed, 3 skipped |
| Linux | 456 passed, 27 failed, 289 skipped |
| Windows | 199 passed, 56 failed, 292 skipped, 225 setup errors |

Original logs, complete JUnit, execution identities, collection records and
verified artifact ZIP digests are preserved in
`.superpowers/pr3148-ci-20261018-02/`. Skips do not qualify platform support.
Perf Guard and all three GGUF jobs passed at that base. Derived Artifacts was
still running when these repairs began; its subsequent outcomes require an
explicit observation, not an assumption of success.

## Repairs

- The full-pytest workflow used the strict hook-child bootstrap before pytest
  installed its own guard. Pytest then captured patched socket methods, blocking
  its owned loopback exemptions and Windows asyncio socketpair fallback. Install
  the existing canonical pytest profile/network guards before product imports.
  Preserve private environment selection, null keyring, exact checkout/data-root
  assertions, all 21 test-file selections and the strict hook-child helper.
- Apply existing native prerequisites only to tests that actually consume managed
  runtime, descriptor-relative package capture or vendor command normalization.
  Keep pure metadata/parser/refusal controls active. Split the mixed Codex
  capture-only assertions from the Darwin vendor-command success tail. The shared
  owned MCP fixture refuses an unqualified managed runtime before opening peers.
- Route the notes startup SELECT through existing
  `NotesDeviceStateStore._connect(read_only=True, must_exist=True)`, preserving
  missing-file, no-schema, read-only, error and close behavior. Private path/mode
  checks now apply. Correct its stale side-effect-free URI documentation; the
  module remains exactly 3,573 lines without raising its budget.
- Add the missing expected Dreams backup call, already covered by registered
  owner `recovery.operations.dreams` and ledger B40. Keep raw-call allowlists,
  scanners, duplicate checks and ownership policies unchanged.
- Preserve the ToolPack import test's empty-before and exact seven-after imports.
  POSIX still requires real successful attachment. Native Windows checks the
  observed closed state: no service/root, `composition_unavailable`, reconciliation
  `not_run`, no active profile guard and unchanged bootstrap guard. The original
  caught Windows exception was not exposed by CI; unsupported directory fsync is
  a source inference. This test qualifies deferral/closed authority only, and its
  generic failure code cannot distinguish every future preattachment error.

## Targeted verification

No full suite or dependency installation ran. The unchanged private runner uses
fresh preselected profiles, a null keyring, actual network/profile guards and
frozen Git archives; all completed phases check source hashes and reject
unexpected outputs.

| Phase | Result |
| --- | --- |
| `ci01-workflow-contracts`, base | 985 passed, 1 intentional child skip |
| `ci02-sqlite-baseline`, base | Both actual inventory failures reproduced |
| `ci03-workflow-guard-red` | Invalid harness attempt: profile lifetime refusal before intended behavior; retained separately |
| `ci04-workflow-guard-red`, fixture freeze03 | Genuine RED: literal workflow prefix refuses real marked loopback with `TestNetworkRefused` |
| `ci05-db-startup-green`, fixture freeze03 | 35 passed, including inventories/ownership, notes boot/read-only controls, startup/attachment and exact size budget |
| `ci06-ci-network-green`, repaired freeze04 | 1,162 passed, 2 expected skips; full Tests/CI, network guard, v2 execution and legacy run hooks; new regression GREEN |
| `ci07-platform-green`, repaired freeze04 | 128 passed, 3 genuine native-refusal-control skips; all failed selectors and all five helper consumers renewed |

Freezes and full argv/outputs/case records are retained under
`.superpowers/sdd/2026-10-17-managed-plugins-dev-rebase/`. Freeze03 and freeze04
have identical repaired Python source; only the workflow changes between them.
The workflow regression executes its literal bootstrap, real asyncio and the
existing marked numeric-loopback test, then checks the selected data-root identity.
The two ci06 skips are the intentional deadline-test child and the native Windows
socketpair fallback control on Darwin.

All four direct static gates return zero: timestamp writers, markup interpolation,
persistent diagnostic inventory and Textual worker contract. Per-file Ruff
code/message multiplicities match exact base bytes: no new findings; the inventory,
startup and wiring files retain 15, 1 and 29 existing findings respectively.
The six new/changed workflow/plugin test files are Ruff/Black clean. The changed
startup function and new connection statement pass bounded Python-3.12 Black
checks. The expected Dreams tuple follows its neighboring census rows; no
whole-file formatting claim is made for the inventory, startup or wiring modules.

## Review and publication

Independent requesting-code-review at repaired code found **0 Critical, 0 Important,
0 Minor** issues. Report: `.superpowers/pr3148-ci-20261018-02/review/ci-repairs-223b59b761-review.md`
(SHA-256 `a2d410dda10538c88f3153928881d6be510956ee2b744ad1fe72912f248f7962`).
The reviewer inspected connected owners and original native traces but reran no
runtime tests. The platform phase completed successfully after the review; repaired-head native
CI remains a separate requirement. Publication must be verified against the observed remote
head; `protected-repair-blobs.json` pins all ten reviewed code/workflow artifacts.
The PR remains draft with auto-merge off. No merge was requested or performed.

Native Linux/Windows repair outcomes require their actual new CI executions.
Local Darwin results do not establish native Windows receipt durability, keychain,
reboot, original vendor hosts, arbitrary remote containment or a full-suite result.

Latest prepublication observation: the required Derived Artifacts run has no
completed failures; its general fast lane and UI shards 2 and 6 are still running.
All other 12 observed lanes are successful. This is not an aggregate green claim.


## Second repair: Windows failures exposed after bootstrap recovery

At first publication, actual run
[39147530966](https://github.com/rmusser01/tldw_chatbook/actions/runs/39147530966)
reported macOS **770 passed / 3 skipped**, Linux **460 passed / 313 skipped**,
and Windows **291 passed / 424 skipped / 16 failed / 42 setup errors**.
All three JUnit artifacts and ZIP digests are retained. Artifact execution revision
`a3f15087eae98ec161d47ede0ee11dfec98ab617` is recorded separately from the PR head;
GitHub's PR merge checkout and the branch revision are distinct identities.
The old-head required Derived aggregate passed, including all ten UI shards.
The first-repair GGUF and UI-latency jobs also passed; its Derived run was queued.

- All 42 setup errors share the source-service autouse acquisition import, which
  reaches native storage before `source_stack` can apply its existing prerequisite.
  Use the same early no-op as the Git fixture on unsupported managed platforms.
  All 30 defining source tests require `source_stack`; that fixture remains the
  skip owner, and all test bodies and qualified-host sanitation are unchanged.
- Ordinary MCP health checks import `ConnectionOwnership`, whose eager `root_ref`
  import loaded native plugin cleanup. Move that same import into the existing
  non-null data-root reservation branch. Preserve every matching, currentness,
  reservation and authority statement. No fallback native flags, new API, support
  predicate, or permission changes. The first Windows trace proves this import
  exception; fourteen generic pipeline/audit failures are source-supported shared
  cause inferences until their actual native renewal.
- The shutdown test's final `os.kill(pid, 0)` is a nonportable PID probe. Query
  `psutil.pid_exists` using the existing base dependency instead. Preserve the real
  child, nonblocking close, one-second blocked result, denied new work and repeated
  close. Darwin passes do not establish Windows child termination.

| Additional phase | Actual result |
| --- | --- |
| `ci08-mcp-import-red`, tests freeze06 | 1 genuine failure: guarded fresh import/health calls pass, then native cleanup is unexpectedly loaded |
| `ci09-source-shutdown-green`, tests freeze06 | 44 passed: all source-service cases and two real shutdown/pipe controls |
| `ci10-mcp-import-green`, runtime freeze07 | 34 passed: regression, all affected MCP functions and initializer refusals, restricted/typed/unified controls, connection identity, real owned stdio/HTTP with data roots |

All completed phases have unchanged source inputs and zero unexpected outputs.
The tests are byte-identical between these RED/GREEN freezes; only the MCP import
location and task metadata change. No native OS was simulated.

The initial freeze05 failed with ENOSPC during extraction before running tests.
Only this task's completed extracted source copies were removed after verifying
all input hashes and their retained Git archives/manifests; the incomplete copy
was removed separately. Archives, manifests, results and failure records remain.
The successful retry is freeze06. See `recoverable-source-cleanup.json` and
`freeze05-prelaunch-failure.json`; the extraction failure is not a test result.

At repaired code `ede1a24f48`, all four direct static gates return zero again.
Per-file Ruff parity covers all 12 changed Python files and introduces no finding;
the same 45 inherited findings remain in the three previously documented files.
The MCP module and source test are whole-file Ruff/Black clean; the legacy
test is Ruff clean and its changed function passes bounded Black. The earlier
whole-file legacy formatting claim overstated the receipt, which checked only
lines1127–1149. Current baseline and repaired file retain three identical
unrelated Black26.5.1 hunks. The new startup test passes bounded Python-3.12
formatting. No module cap changed.
`protected-round2-repair-blobs.json` pins all 13 code/workflow artifacts.

Second independent review: **0 Critical / 0 Important / 0 Minor**, report
`.superpowers/pr3148-ci-20261018-02/review/ci-round2-ede1-review.md`, SHA-256
`c069ca1a8732f20e672fc2ced0e7a3b087cab1b150cba743201067c38142549b`.
The reviewer independently verified exact diff/AST scope, real RED, both completed
GREEN phases and frozen source equality. It reran no tests and makes no native
Windows-success or merge-readiness claim. Existing ADR-148/162/163 apply; no new ADR.

## Second published repair: native results and remaining diagnosis

Published head `2ab43a146fc7f2cf14bec23de6760e64c90cab23`, qualification run39150110865, actual merge checkout `a73741b7240ca6aa81ef3701741c5e87375cfd6c`: macOS771PASS3SKIP, Linux461PASS313SKIP, Windows306PASS466SKIP2FAIL0setupERROR. Prior source fixture, native import and portable shutdown failures are genuinely renewed green. Windows retains only initializer[success] original5s and admitted AgentService original10s outer TimeoutError. Their cancellation traces do not identify the running worker stage; a late cleanup disconnect is not evidence of the original cause. No product repair or deadline increase is justified yet.

Backlog plan111 authorizes one temporary, bounded, one-shot test observer at4s/9s in the existing selected MCP test file. It preserves original deadlines, assertions, cleanup, peers and profile/network/native guards. It copies only code objects and line numbers, drops all frame references before formatting/I/O, and emits at most8stacks x20frames /32KiB without source, locals, payloads or private paths. The existing diagnostics plugin was rejected because it retains frames while formatting source, matching the documented TASK33620.15 SQLite sampler trap. The observer will be removed after native diagnosis. This is diagnostic publication, not a fix or merge-readiness claim.

Raw second-run logs/artifacts and exact identities remain in `.superpowers/pr3148-ci-20261018-02/`; macOS artifact12145089569 ZIP SHA25645033d77b6f751758d477a6061489343524c6082ad0c17798a3b324c519dcdc9. Full suite, platform support expansion and merge remain out of scope. AC21 stays open pending diagnosis, repair and renewed native results.

Diagnostic `871d5664d762686015d47869a7d233aa63fd08b9` was published with exact2ab lease and verified by direct Git ref and REST PR, remaining draft/auto-merge off. Independent scoped review0C/0I/0M report SHA256`2f0237ef22d21713572434f97ce515f68f817e04db986954cb7e9bcd65592191`. Exact-helper stdlib privacy/frame-lifetime/cap/cancellation selfcheck passes; four direct static gates return0; changed test Ruff/Black clean. Frozen-source ci11 affected cases/control18PASS5.22s, inputs unchanged/0unexpected. All13 prior reviewed repair blobs unchanged.

Local freeze initially refused before runtime because external origin/dev advanced to6db2e5b535 while CI ran. The original qualification runner was retained by SHA212c59e9; only its accepted-base ancestry selection was pinned tob07 and the current remote tip separately recorded. All archive/source/profile/network/native/SQLite safeguards remain. This local evidence applies to the exact accepted-base PR source, not latest-dev integration. New hosted run39152582452 will record its actual GitHub merge checkout. The diagnostic is still pending native observation; both timeout causes remain unknown.

## Native observation and bounded test repairs

Third run39152582452 executes actual merge checkout22702d65fe9853ba81cd90af3c9a2e5c8dcc033e: Linux461PASS313SKIP; Windows306PASS466SKIP2FAIL0setupERROR. Windows artifact12146147314 ZIP SHA2565a6b785ad259727941e080c64fcf3fe785b2a0116d6ffbb734f49f388b4e9c98. The initializer succeeds under its original5s watchdog; total testcase7.092s includes setup/cleanup. Passing-test stderr is not retained, so absence of a captured4s snapshot proves no duration and does not establish the historical timeout cause.

The retained AgentService snapshot at9015ms shows the owner loop idle and the worker in normal calculator STEP_TOOL_CALL persistence: agent_runtime3245 -> observe_step8459 -> AgentRuns_DB.insert_steps_at_indices2172 -> guarded registry open -> Windows GetVolumeInformationByHandleW440, before registry flock. Required MCP pre-hook acceptance precedes this step. The other admission worker is in its normal idle wait. This identifies reached work, not syscall duration or eventual completion. The whole-turn10s test watchdog includes persistence plus a hook allowed10active seconds: that test budget is too short for valid integration work. Plan113 changes only the outer watchdog to30s and initializer success to15s, preserving five refusal-case5s bounds and all semantic assertions. The temporary sampler is removed. Product active/wall/cleanup budgets, authority and storage checks are untouched.

The other actual Windows failure, legacy test_notify_cwd_override, reads an empty payload file. The shared fixture creates the final file before writing JSON; its notify consumer treats existence as completion. The producer now closes a sibling temporary JSON file and atomically replaces the final payload. The original5s poll and cwd assertion remain; all four existing helper callers are covered by targeted renewal. No product synchronization API or close/cancellation workaround is introduced. Native failures are retained as RED; local targeted GREEN and native renewal are pending. Legacy changed-helper Ruff/Black pass; Black26.5.1 whole-file baseline/current retain the same three unrelated hunks.

Final test repair source861bf46a43ec5cb0f2e85a1d6ccb75b078ebae81: frozen ci12 **25PASS11.29s**, input manifest unchanged/0unexpected outputs, covering18MCP cases and7legacy fixture/notification/shutdown controls. Fresh independent scoped review **0C/0I/0M**, report `.superpowers/pr3148-ci-20261018-02/review/ci-round3-final-review.md`, SHA256`cd63f66b40a6c251864d9e43b58618ec66a0af4f08eae977fc4c4d209f0c37fe`. Both exact test-source hashes and observer removal are verified. All production, scripts and security-inventory inputs remain identical to the prior four passing static-gate run; applicability is recorded in round3-final-applicability.json. Ruff passes; MCP file and legacy changed helper pass Black; the three whole-legacy inherited formatting hunks remain. Native renewal/publication remain pending at this receipt; no full suite or merge claim.

## Remaining typed-transport watchdog

Publishedbcf14b3019 native Windows run39155307353/job117757190741 (actualmergeb161dc119f1fb4c4521b7e45a1df66c71f8172cc) is307PASS466SKIP1FAIL0ERROR. Artifact12146623595 ZIP SHA256ecbde6487b07c115c6b8b701fc47137eda310a579461ed738dd4bfa956aae103. All previous initializer, AgentService and atomic-payload fixes now pass with their original semantic assertions. The remaining actual failure is typed transport-result boundary payload0/stdio, whose five-second outer lifecycle watchdog cancels the same default ten-active-second hook pipeline.

Plan114 aligns this boundary's outer watchdog to15s across its ten real stdio/HTTP normalization cases, preserving payload and acceptance/refusal assertions, execution counters, terminal cleanup and product deadlines. All other wait sites were classified: one-second cancellation controls and five initializer preflight refusals remain unchanged; other five-second waits are existing v2-command-gated post/custody controls. No further tracing, platform skip or production change. Full modified MCP module is the targeted renewal.

Process deviation: the Backlog114 CLI was invoked before the source edit but yielded a running session; the edit occurred2.238s before the plan file actually persisted. Root then awaited completion and verified the exact accepted plan before any freeze/runtime/commit/publication. Source stayed inside the predeclared one-literal/comment scope. The write-order receipt is plan114-write-order-deviation.json; subsequent dependent CLI work waits for completion.
