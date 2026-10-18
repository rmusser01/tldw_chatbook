# PR #3148: remaining CI repairs, 2026-10-18

Source base: `40d61b3ebfd1d05045a5308536e6365777c9b0ce`. Repaired code:
`223b59b7616f559e46044f62a703eb61e45c1190`, on dev
`b07fc0425288fb51d17cde5f745051720a7c9db6`.
Existing ADR-097/107/125/126/162/163 govern these routine repairs; no new
storage contract, platform support, dependency, permission, or schema was added.

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
