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
The MCP module and source/legacy test changes are whole-file Ruff/Black clean;
the new startup test passes bounded Python-3.12 formatting. No module cap changed.
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
