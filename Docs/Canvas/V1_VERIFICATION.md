# Canvas V1 verification record

Date: 2026-09-05. Isolated branch: `codex/canvas-v1`.
Architecture: [ADR-121](../../backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md).
Plan: [Canvas implementation](../superpowers/plans/2026-09-03-chatbook-canvas-implementation.md).

## Authorized integration follow-up — 2026-09-05

The user has now authorized latest-dev rebase, a PR against dev, addressing Qodo
feedback, merging after gates pass, then V2 design. The historical scope limits
below describe their original checkpoints; they do not revoke that authorization.
No full repository sweep or expanded Canvas privileges were authorized.

TASK-31732 fixes the descriptor warning's proven Chatbook export/import cause.
`d94c40755` gives each operation-local conversation database a `finally` close;
two private helpers retain the original processing bodies unchanged. No database
quiescence policy, sentinel threshold or GC fixture changed. New real SQLite
tests preserve another same-file observer, verify exact operation handles close,
and repeat success three times plus test post-construction setup failure. Dedicated
malformed/cancellation lifetime cases were not added: the unconditional finally
covers those exits structurally and existing round-trip controls cover behavior.

- Unchanged ten-module diagnostic: **970 passed, 2 warnings, 176.27s**;
  plugin session12→235 (+223; regular+224, character-1), default sentinel+210.
- Forced-GC thinking-only control: **20 passed, 1 warning, 2.17s**, regular+83.
  GC did not release these operation-owned handles.
- Corrected lifetime RED: **4 failed, 1 warning, 1.70s**, all registrycount2
  rather than observerbaseline1. Earlier fixture/Row failures are not credited.
- GREEN lifetime plus thinking controls: **24 passed, 1 warning, 2.25s**,
  plugin12→12, zero category/module growth under default GC cadence.
- Original ten-module selection after repair: **970 passed, 1 warning, 172.49s**.
  Default FD sentinel no longer warns; plugin12→152 (+140; regular+141,
  character-1), an 83-descriptor reduction. Remaining module totals are
  controller+90, Settings+24, store+23, provider+3, thinking-13, otherfive0.
  This is not proof of globally zero retained descriptors or a full-suite result.
  Residual aggregate retention in other test lifetimes remains a qualification,
  not a reason to raise the sentinel or broaden this ownership repair.
- Independent static review of `6226f8a18..d94c40755`: **PASS** for spec/quality;
  no actionable defect. Reviewer performed no app imports or executable probes.

The diagnostic command is the ten-module command recorded below with
`PYTHONPATH=.superpowers/sdd/2026-09-03-chatbook-canvas-implementation` and
`-p fd_probe` added. The removable ignored plugin reports only resource categories
and parameter-free test names around complete teardown; it is not shipped runtime
instrumentation. All executable checks use repository pytest pre-import isolation.
The one retained warning is the existing Requests dependency-version mismatch.

TASK-31741 owns integration artifact guards. The diagnostic statement audit against
pin revision `b55d6b12f01461295d07b5938ca3c3cb28c7939e` covered ten changed owners:
agent runtime/bridge, Canvas controller, chat store/runtime, importer, chat screen,
Settings, served host and app. Four new Canvas exception/traceback handlers in
runtime disposal and config reconciliation were changed to fixed content-free
messages after **4 expected privacy failures (1.44s)**; GREEN **4 passed (1.58s)**.
The first privacy attempt used the wrong logger capture for Settings and is not
credited for those two cases. Existing generic review-hook logging semantics are
unchanged; importer Canvas validation uses fixed error codes, other legacy import
errors remain bounded as before. No new log sink was added. The regenerated path
candidate count changes partly because the FD repair names two new helper scopes;
unchanged legacy diagnostic bodies were not silently removed.

All six Canvas indexes now have census entries. Four read indexes have explicit
no-`sqlite_stat1` query plans in the real repository fixture; two composite UNIQUE
indexes enforce foreign-key parent/hint ownership rather than read optimization.
The isolated plan case passed; the four-module repository/Canvas privacy/Settings/
runtime selection passed **62 tests, 1 warning, 18.32s**.

The 110 feature commits were rebased onto `7aa0487907ba4166b5797149a03de8e86e173c7e`;
the recoverable `codex/canvas-v1-pre-rebase-20260905` ref retains
`ff2f18d81bdedb3c4a450784c7f5623d49eabdb4`. Conflict resolution preserves dev's
endpoint policy and Persona Buddy mounting as well as Canvas lifecycle startup.
The diagnostic inventory was regenerated from the combined source. All repository
preflight checks passed on the rebased tree, including 276 indexed declarations,
276 census rows, 61 pinned plans, 108 allowlisted DB tables, and 3,325 unique,
portable task paths. Follow-up `b65a69309` removes duplicated imports and updates
four test doubles to current selection-generation and meeting-owner contracts.
Independent static review passed; no production guard was relaxed.

- Initial restricted-executor selection: **88 failed, 1,429 passed, 1 skipped**.
  Of the failures, 84 were numeric loopback-bind permission denials; four were
  the corrected fixture mismatches. Unchanged gateway tests pass all **72** with
  local-listener permission. This was not a missing-marker problem.
- Definitive affected selection at `b65a69309`, with local-listener permission:
  **1,517 passed, 1 skipped, 7 warnings, 236.52s**. The optional runtime archive
  rebuild lacks its configured cache. Warnings are the existing Requests version
  mismatch and six legacy invalid-escape warnings; no FD-growth warning recurred.
- The native, served and zero-egress browser files: **89 passed, 2 skipped,
  1 warning, 171.82s**. Mandatory Chromium passed; optional Firefox/WebKit are not
  installed. This is targeted evidence, not a full-suite or all-browser claim.
- Additional startup guards exposed an integration blocker: **990 modules at
  first paint versus the unchanged 972 ratchet**. An untouched tracked-source
  archive of the exact dev base passes at **972**, reproduced twice. A diagnostic
  wrapper around the repository's isolated pytest census observes 18 additional
  residents and none removed: 16 Canvas-related modules plus the existing
  scheduler heartbeat/emergency-stop first-tick modules. The latter are not
  candidates for weakening or deferral.
- The other startup measurements pass before repair: app import **646/660**,
  CSS **787,764/804,000 bytes**, pre-import pass **496/500 modules**,
  **365,641/378,740 LOC**, largest route **112,023/123,319 LOC**.

Startup follow-up under ADR-097 (thresholds and snapshots unchanged):

- Group A (`0afa56dec`) defers Chatbook archive engines behind existing public
  exports and operation patch seams. Valid import-closure RED:2 failed/3 passed;
  final seven-module archive selection: **87 passed, 1 warning, 7.58s**.
  First paint990→984, removing six archive-related residents with no additions.
- Group B (`81f288e8f`) avoids native served-control construction and binds the
  existing runtime's stable settlement listener without creating a browser
  authority until first publication/open. First-create delivery, concurrent
  first-use identity, rebinding, disable and disposal controls pass: **41 tests**.
  First paint984→978, removing five browser modules and one timing-dependent
  scheduler heartbeat resident. No scheduler safety path was changed.
- Canvas CSS (`e6ba6becf`) adds exact control classes to two existing rule
  subjects, with declarations unchanged and generated CSS matching. The new
  computed-style controls first failed only on the intended missing classes;
  final **8 passed, 1 warning, 1.97s**, at80×24 and120×40. Broad ancestor subjects
  return276→274. Ruff, format and bundle generation pass. This is computed-style
  evidence, not screenshot equivalence across every theme.
- Group C (`6f7719fdf`) shares strict execution-flag normalization without
  constructing remote policy for a bool-only read; full Settings admission
  validation remains. Compiler imports move to actual compilation/error use,
  preserving keyword limits, injected compilers and exact exception ordering.
  Valid RED: **2 failed, 54 passed, 5.83s**. Expanded final verification after
  signature/ordering refinements: **139 passed, 1 warning, 22.57s**.
  First paint978→975: compiler/web_auth absent and the emergency-stop first-tick
  resident absent by timing only, not a guaranteed product saving.
- Independent static reviews pass for A, B, C and the CSS correction. New/tiny
  owned-file lint/format checks pass; unchanged legacy lint counts remain
  qualified (config160, message controller33, message actions7→6).
- Browser requalification after A/B/C at6f7719fdf: **89 passed, 2 skipped,
  1 warning, 166.01s** across the same three native/served/zero-egress files.
  Mandatory Chromium passes; Firefox/WebKit remain not installed.
- Post-startup preflight identifies only one diagnostic digest change in app.py.
  The statement audit against0512c3f68 reports387→387 calls, one moved/re-indented
  statement, zero added/removed. It is the unchanged fixed invalid-spawn warning
  nested under Group B's served-environment presence check. The inventory was
  regenerated after that review; all five other preflight categories passed.
- Local PR Fast Lane selection: **749 passed, 6 failed, 1 warning, 237.15s**.
  All six process-lease failures occur at multiprocessing `SemLock` creation
  with OS errno28, before exercising lease behavior. The untouched exact-dev
  archive reproduces all six: **6 failed, 1 warning, 0.88s**. The volume reports
  289GiB available; this is not evidence that disk files need deletion.
  No OS limits, semaphore objects, tests or guards were changed. Protected CI
  must still pass; this local run is explicitly not a green Fast Lane claim.

Group D restores the startup ratchet by deferring three user-open-only dialogs
and the local Chatbook conflict enum to their existing operation entrypoints.
The first mixed run was5 failed/4 passed: two expected closure failures and three
preexisting positive-fixture failures. Untouched dev reproduced the latter three
in12.83s. Supplying the actual workspace registry to two real-SQLite fixtures and
the current route keyword to the offline gateway yields the valid pre-product
RED: **2 failed, 7 passed, 1 warning, 12.17s**. No production persistence or
provider gate changed. Final expanded startup/dialog/archive controls:
**32 passed, 1 warning, 17.61s**; independent static review passes.

The measured census975→971 sheds exactly conflict_resolver, prompt_queue_modal,
review_notes_modal and side_chat_modal, with no additions. The complete six-file
startup selection passes **19 tests, 6 warnings, 28.27s**: import638/660,
first paint971/972, broad CSS274/274, CSS787819/804000bytes and pre-import496/500
with unchanged LOC totals above. Warnings include low-headroom diagnostics,
Requests and the host joblib SemLock limitation; none are hidden. UI latency and
stall-persistence controls pass **9 tests, 3 warnings, 22.35s**. These are scoped
checks, not a full repository sweep. The one-module first-paint headroom remains
tight and protected CI must independently verify its platform.

TASK-31741 remains in progress for final latest-base/preflight/PR integration.
No budget/snapshot increase, scheduler safety change, PR, push or merge has
occurred at this checkpoint. Latest fetched dev isd9d5763d6; its delta from7aa
is AudioTap code/tests plus schedules and snapshot follow-up documentation.

Final pre-PR rebase: **970c86da4d74fbccd17c9bf77eec625f5fceedd5** on
devd9d5763d6. All116 feature commits replayed without conflict; range-diff marks
every pair equal. Recovery branch`codex/canvas-v1-before-final-dev-20260905`
retains48b69db58. The only tree delta from that recovery tip is dev's13-file
AudioTap/documentation change (+403/-12), not a Canvas implementation change.
On this exact candidate, preflight passes all categories (now3332 task paths)
and the six startup files plus Canvas first-use tests pass **36 tests,
6 warnings, 36.52s**, with the same638/660 and971/972 module measurements.
The subsequent PR preparation edits are documentation/task bookkeeping only.
Qodo feedback and protected CI are not yet received and remain merge gates.

## Earlier V1 checkpoints

This is targeted evidence, **not full-suite, release, or integration approval**.
Latest scoped rereview (`648530ac6..03cd979df`) closes both remaining I2 DOM
cases with spec and quality gates passing and no new Critical/Important
fix-diff issue identified. Those final-review implementation findings are
closed. TASK-31232's final retry scoped review also passes both gates.
The six-baseline repair resolved its original six failures; the subsequent
user-authorized retry correction now passes all 970 directly affected tests,
including both newly identified retry failures. AC9 is satisfied by the
incremental targeted evidence reconciled below, not a newly run full matrix.
Details below preserve
earlier findings chronologically rather than representing every historical
finding as still open.
The user authorized one DOM-only correction and scoped rereview of those two
I2 cases. Implementation `981b1f8c1` and targeted evidence are recorded below;
the scoped verdict passed. This does not waive AC9's six characterized
baseline failures or authorize a full sweep.
The user subsequently approved repair of those six baseline failures. Diagnosis
reproduced all six in one isolated pytest run (6 failed, 1 warning, 5.94 s).
The original six repaired cases passed independent review. The user then
authorized the two additional retry repairs recorded below. The scope remains
those causes and directly affected regressions, not a full repository sweep.

During that diagnosis, a direct Library descriptor import outside pytest
repeated the earlier isolation mistake: it loaded the ambient config and logged
ensuring the same chat-dictionaries directory. The command printed public tool
names and config section names, not config values or credentials. No pre-command
snapshot establishes whether the directory was created. Worker executable probes
were stopped, no ambient cleanup occurred, and the user was informed. Further
worker work is static inspection/edits only; the coordinator runs all tests through
the repository's pre-import isolated config/data fixtures.
Independent Task 7.4 review found five Important evidence/fixture gaps. Fix
commit `0724726a0c` adds strict corpus outcomes, persisted Console completion,
create-triggered native opening, exact card/replacement-session coverage and
setup rollback, plus the selection repair described below. Independent
round-1 rereview closed four findings and found no new
Critical/Important breakage in the intent fence. It left the durable exact-
revision recovery evidence open; test-only fix `4ce7fc756f` now supplies that
evidence. Independent round-2 rereview closed that final finding with no new
Critical/Important breakage. The subsequent whole-branch review requires the
corrections recorded below; the clean task gate does not close these new gaps.
TASK-31232 remains
In Progress; its requirement that every selected suite passes is not satisfied
by the baseline characterization below. No full repository sweep was authorized.

## User-authorized retry correction

Implementation: `5bba89d3a` (two Python paths). The static-only worker used
`--no-verify`; root rechecked the configured hooks directory and found no
installed `pre-commit` hook. No hook pass is claimed.

The subsequent user approval covers the two retry cases and directly affected
regressions. Planning commit `61a27e887` precedes the correction; ADR-121 applies.
Fresh exact-two RED: **2 failed, 1 warning, 1.17 s**.

Source-free test observations established that attempt 1 is discarded correctly
and attempt 2 registers a distinct run, but its tool invocation returns
`invalid_scope` before finish. The controller accepts the retry request yet marks
the assistant failed; accepted alone is not completion evidence. The durable
scope contains three path nodes: two saved messages with valid ownership,
deletion state and parent chain, plus one native-only SYSTEM failure notice.
The service correctly rejects that unsaved ID. This is a path-projection defect,
not legitimate committed-stage cleanup or lost settlement bookkeeping.

The correction is limited to excluding native-only SYSTEM notices from durable
tool scope. Temporary paths, saved system messages, missing user/assistant
origin rejection, quota admission and service validation must remain intact.
All diagnostics were root-run under isolated pytest. Their intermediate results
were 2 failed/1 warning/1.54 s and single failures/1 warning in 0.78 s, 0.98 s,
0.93 s, 1.07 s and 0.91 s while improving only the source-free observations.
An intervening 1.14 s diagnostic failed on a nonexistent fixture lookup method;
it yielded no product evidence.

Definitive RED/control run: **3 failed, 4 passed, 1 warning, 3.15 s**. The
new durable-native-SYSTEM case failed at the actual tool rejection; both retry
cases failed at the strengthened terminal-completion assertion. Persisted
SYSTEM inclusion, temporary native-path preservation, and missing USER/ASSISTANT
`invalid_scope` controls passed. All diagnostic scaffolding was removed.
The first focused GREEN was **11 passed, 1 warning, 4.11 s**; root then removed
an unnecessary candidate change to temporary alias handling before final checks.
The final production change is only the six-line durable-native-SYSTEM guard;
the existing persisted-ID-or-native fallback is unchanged.

Final affected coverage: **970 passed, 2 warnings, 193.75 s**. This includes the
original six baseline repairs, both retry failures, all five new scope cases,
atomic rollback/restart assertions and surrounding Console/Canvas behavior.

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no \
  Tests/Chatbooks/test_chatbook_thinking_round_trip.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/UI/test_settings_raw_cli.py \
  Tests/Chat/test_console_trace_privacy_owners.py \
  Tests/Chat/test_console_semantic_writer_routing.py \
  Tests/Chat/test_console_canvas_controller.py \
  Tests/Canvas/test_service.py Tests/Canvas/test_staging.py \
  Tests/Agents/test_canvas_tool_provider.py
```

The warnings are the known Requests mismatch and descriptor growth of 207
(start 25, end 232, limit 200). The earlier six-module run recorded growth of
204; neither warning has been causally attributed, and this correction does
not claim to fix resource cleanup. No browser, network, permission or runtime
boundary changed. Prior live/browser/packaging evidence remains separately
qualified below; test counts across overlapping runs must not be summed.
After two formatting-only line joins in the new test, the final focused command
passed **11 tests, 1 warning, 4.88 s**:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no \
  Tests/Chat/test_console_chat_controller.py::test_canvas_scope_projects_only_native_system_rows_by_durability \
  Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry
```

Final two-path Ruff `(code, message)` Counters match `b125832cb`: 172/172 for
the controller and 17/17 for its test, zero added/removed findings. No formatter
diff overlaps changed ranges; inherited whole-file debt was not rewritten.
Both-file `compileall` and `git diff --check` returned zero. The worker remains
static-only; all executable tests and static orchestration were root-run without
application imports. Independent static review of `b125832cb..8388ae696` passes
both spec and quality gates, with no blocking retry-code regression. The
descriptor-growth warning remains unresolved and is not a resource-clean claim.

## Final acceptance reconciliation

Root reconciled every previously failing selected node with its correction and
later passing run, reviewed the completed TASK-31226–31231 acceptance records,
and cross-checked the design's failure, verification, configuration and final
boundary decisions. TASK-31232 is complete under the targeted-only test policy.
This is incremental evidence across reviewed commits; unaffected passing gates
were not rerun wholesale after this six-line scope correction.

| Previously selected slice | Closing evidence |
| --- | --- |
| Core/migrations/Chatbooks/Web Server: 3 failures | Schema assertion corrected and six-node schema/census slice rechecked; both Thinking cases now pass in the final 970-test run. Remaining original cases had passed. |
| Console/UI: 6 failures, plus 2 subsequently identified retries | Two census cases corrected and rechecked; the four baseline cases and both retry cases now pass in the final 970-test run. Remaining original cases had passed. |
| Agents and packaging | Prior 380-test Agents and two-test packaging gates retained; Canvas provider/composition covered again by the final affected run. Runtime asset and vendor checks remain recorded below. |
| Native/served/browser/archive | Prior outer-path and zero-egress gates, scoped selection fixes and final DOM/assets corrections retained with their recorded optional skips and cleanup limitations. This correction changes no browser/runtime/archive code. |

The final self-review confirms existing ADR-121 still governs all boundaries:
closed virtual runtime, local branch-aware immutable history, atomic staged
turns/promotion, same-origin authenticated served routing, confirmed unsent-draft
bridge, and inert archives. No scope expansion or new ADR was needed. Sync
remains deferred in TASK-31003; bundled libraries, multi-file projects and
privileged external capabilities are not V1 deliverables.

Completion does **not** certify a full repository sweep, pristine dependency/
resource output, clean dependency installation, every browser engine, or
unobserved cleanup. Retained warnings, static debt and earlier isolation/cleanup
incidents remain part of this record. No merge, push, release or worktree cleanup
is authorized or performed by this acceptance reconciliation.

## User-authorized six-baseline repair evidence

Implementation commit: `11ea68221`. The worker used `--no-verify` to honor its
post-incident no-interpreter restriction. Root subsequently checked the configured
hooks directory and found no `pre-commit` hook installed. The validation evidence
below and independent scoped review are explicit; no hook pass is claimed.

The scoped repairs preserve existing contracts rather than restoring obsolete
fixtures: direct mutation guards remain intact, the later ADR-097 soft-delete
envelope retention supersedes ADR-090's older clearing expectation, MCP filtering
is still source-scoped, the strict promotion fake accepts its explicit trace
boundary, and Settings waits on actual destination readiness without bypassing
its save/navigation vetoes. Two product corrections use already-frozen privacy
values for an attachment-only send with no preparation, and omit `_thinking`
from an exported tombstone that the importer would otherwise reject. Stored
semantic bytes, graph identity and separately governed `_private` remain intact.

The obsolete `test_durable_soft_delete_clears_thinking_from_tombstone` is replaced
by `test_soft_deleted_thinking_owner_round_trips_as_an_importable_tombstone`.
Its genuine product RED was **1 failed, 1 warning, 0.73 s**, at the exported
tombstone `_thinking` assertion after retention/history controls passed.
An intermediate six-case run had **1 failed, 5 passed, 1 warning, 12.78 s**:
the corruption fixture supplied a cursor instead of `cursor.connection` to the
authorization helper. Correcting the fixture left the production guard intact.
The final exact six-case run passed **6 tests, 1 warning, 11.83 s**, including
the complete Settings save/handoff/dirty-navigation behavior (9.32 s call).

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no \
  Tests/Chatbooks/test_chatbook_thinking_round_trip.py::test_chatbook_export_blocks_opaque_future_thinking_with_upgrade_copy \
  Tests/Chatbooks/test_chatbook_thinking_round_trip.py::test_soft_deleted_thinking_owner_round_trips_as_an_importable_tombstone \
  Tests/Chat/test_console_chat_controller.py::test_image_only_draft_is_sendable \
  Tests/Chat/test_console_chat_controller.py::test_compose_mcp_provider_excludes_console_shadowed_builtin_names \
  Tests/Chat/test_console_chat_store.py::test_first_persist_context_failure_does_not_force_atomic_promotion_legacy_path \
  Tests/UI/test_settings_raw_cli.py::test_pending_raw_cli_save_vetoes_real_navigation_until_arrival
```

The final directly affected six-module check was **737 passed, 2 failed,
2 warnings, 140.19 s**. All six repaired cases passed in this broader run.

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no \
  Tests/Chatbooks/test_chatbook_thinking_round_trip.py \
  Tests/Chat/test_console_chat_controller.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/UI/test_settings_raw_cli.py \
  Tests/Chat/test_console_trace_privacy_owners.py \
  Tests/Chat/test_console_semantic_writer_routing.py
```

Both failures are parameter cases of
`Tests/Chat/test_console_chat_controller.py::test_real_canvas_controller_allows_exact_failed_assistant_retry`:
`[False-True-False-False]` and `[True-True-False-False]`. Each expects a committed
settlement but receives `None`. An exact two-ID rerun on the repaired tree gave
**2 failed, 1 warning, 1.11 s**. The same two IDs on a clean, isolated checkout
of pre-repair `c8a1211e5768d1ce099b5168271ec1cc750ed21f` gave **2 failed,
2 warnings, 2.28 s** at the same assertion. These are pre-existing relative to
the six-baseline repair, not claimed to predate the entire Canvas branch.
No additional repair is made or cause beyond that failing assertion claimed.

The baseline used `/private/tmp/canvas-six-baseline.h0zDgq` with the same venv
Python by absolute path and repository pytest isolation. Root verified its
exact HEAD and clean status, then removed only that owned comparison worktree
with ordinary `git worktree remove`. Its source remains recoverable from the
commit; the implementation worktree and evidence were not removed.

Warnings: the known Requests dependency mismatch; broader-run descriptor growth
of 204 (start 12, end 216, limit 200), not causally attributed; and a cold-import
SyntaxWarning in the baseline's unchanged `Tools/patch_tool_impls.py`. The broad
run is not a full repository sweep or a clean selected-suite claim.

Final static checks: root compared Ruff `(path, code, message)` Counters across
the six changed Python files against `c8a1211e5`: **269/269, zero added/removed**.
No formatter diff overlaps changed ranges after normalizing the new constructor;
four files retain pre-existing whole-file format debt. `compileall` for all six
and `git diff --check` returned zero. No application imports were used for these
static checks. AC9 remains unchecked pending the newly identified retry failures.

Independent static scoped review of `85e7f36f4..610adcbaf` passed the spec and
quality gates for all six repairs, with no new Critical/Important fix-diff
regression identified. One nonblocking prose remnant remains at
`Chat/console_chat_controller.py:496`: the adjacent comment still mentions an
obsolete “18-tool contract.” It is recorded, not treated as functional failure
or silently fixed outside the reviewed diff. The two newly identified retry
failures remain unresolved; TASK-31232 is not Done. No full sweep or integration
is authorized by the scoped review.

## DOM-only correction evidence (`981b1f8c1`)

Under existing ADR-121, reconstruction now skips already-present descendants
while attaching them to rebuilt parents. Explicit option selection is restored
after subtree structure, followed by explicit select value. No facade API,
permission, runtime fallback or quota was added or relaxed.

Definitive actual-Chromium RED before production changes: **4 failed, 1 warning,
14.23 s**. Non-first and empty/no-match select values were independently
parameterized and both reopened as `first`; new-issued and existing-live child
cases failed reattachment. An earlier `live` parameter was skipped by repository
policy; another run's first assertion masked the empty case. Neither is used
in place of the definitive four-case RED. The matching GREEN was **4 passed,
1 warning, 4.08 s**.

Final browser command (owned wrapper `/tmp/canvas_dom_final_browser_probe.zsh`):

```text
../../.venv/bin/python -m pytest -q \
  Tests/Canvas/browser/test_canvas_zero_egress.py::test_dom_move_detach_and_reinsert_preserve_virtual_identity_and_bounds \
  Tests/Canvas/browser/test_canvas_zero_egress.py::test_detached_select_reconstruction_restores_explicit_and_default_state \
  Tests/Canvas/browser/test_canvas_zero_egress.py::test_detached_subtree_reconstruction_reuses_present_descendants
```

Result: **5 passed, 1 warning, 7.51 s**. The latter two nodes are the definitive
RED/GREEN command. Assertions cover actual value/selected state, a same-select
compatible option/value override, untouched defaults, exact native identity of
the moved existing child, new-child uniqueness, prior twenty detach/reinsert
cycles, listeners, cycle refusal, patch-limit refusal and generated-egress
observations. This is not a new conflicting-override chronology contract or
another broad adversarial qualification.

The wrapper captured live pytest `99035`, driver `99116`, Chromium
`99142/99146/99147/99155`, and the existing owned profile
`/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/playwright_chromiumdev_profile-zfXGdx`.
All captured PIDs and that profile were absent after completion. Only owned
descendants were inspected; no ambient cleanup occurred. Earlier incomplete
cleanup captures remain qualified in their historical entries.

`../../.venv/bin/python -m pytest -q Tests/Canvas/test_runtime_assets.py`:
**20 passed, 1 skipped, 1 warning, 0.81 s**. The skip requires
`TLDW_CANVAS_RUNTIME_ARCHIVE_DIR`; no archive download was attempted. The worker
is 55,488 bytes, SHA-256
`23ed3fc4fdf9109a4207d6aeebf6de26c198261c1ae980f7df7a6d6f4d9aae3c`.
Worker syntax and vendor-manifest verification passed. The single changed
Python test file passes whole-file Ruff lint and format checks; root repeated
these checks plus syntax, manifest and committed-range whitespace inspection.
This does not replace earlier inherited lint-debt qualifications elsewhere.

The pass touches only the runtime worker, its integrity manifest and the browser
test file. The review package starts at the actual previous-reviewed HEAD
`648530ac6` and ends at `03cd979df`. The static-only scoped rereview closed both
I2 cases: per-descendant presence preserves necessary attachments, and deferred
selection patches follow complete child structure. Spec and quality gates pass
with no new Critical/Important fix-diff issue identified. The reviewer did not
rerun tests or import application modules. Earlier authorship and evidence
qualifications remain recorded below; this is not another broad review.
AC10 is checked. AC9 independently remains open for the six baseline failures.
No baseline suites or full sweep rerun, no task Done or integration approval.

## Whole-branch review (`facd1e0fb0`): needs fixes

The full shared-base-to-HEAD review identified five Important gaps: admission
quotas bypassed by the production Console staging owner; DOM move/reinsert
patches losing renderer identity; ordinary continuation content rewritten with
Canvas disabled; transcript action discovery/dispatch compiling synchronously;
and settlement publication resetting a historical pin. Minor corrections cover
asset-limit argument validation and explicit Close versus pane Hide wording.
The consolidated correction wave is committed as `c875bad60f`; scoped rereview
of `facd1e0fb0..a7bcc6b094` still requires fixes in I1, I2 and I4. No Critical
finding was established; this is not a security certification or merge approval.

The reviewer authored Task 7.2a and the Task 7.4 continuation. Separate
author-independent task reviews exist, but the final review is not wholly
author-independent. Two narrow probes confirmed staging 11 Canvases despite
the 10 limit and the virtual runtime's remove/append sequence; renderer failure
and the remaining findings were established through static production paths,
not a fresh browser run. All named deferred inputs were triaged with existing
evidence limits retained.

The coordinator probe imported ambient configuration rather than using the
required isolated fixture. Its log reported loading the existing Chatbook
configuration and ensuring `/Users/macbook-dev/.local/share/tldw_cli/default_user/chat_dicts`.
Without a pre-probe snapshot, whether that directory was newly created is
unknown. Static bootstrap inspection permits directory/security setup effects;
no configuration-save evidence was observed. No database, provider or browser
was launched. Further executable review probes were stopped; ambient state was
not deleted or modified to conceal the deviation. Future tests must establish
owned configuration/data before importing application modules.

### Final correction-wave evidence (`c875bad60f`)

The implementation adds production-owner admission against existing durable
usage and concurrent staging, bounded DOM move/reinsert handling, unchanged
ordinary non-opt-in continuation records, deferred off-loop HTML compatibility
validation, pin-preserving update publication, safe-wire helper argument checks,
and accurate outer Hide/inner Close copy. The scoped verdict below qualifies
these implementation claims; test counts alone do not close the findings.

**Scoped rereview verdict: not ready to merge.** I3 (ordinary continuation
bytes), I5 (pin-preserving publication), M1 (helper limits) and M2 (Close/Hide)
are addressed. I1, I2 and I4 remain open:

- The production temporary owner still uses the durable 50 MiB aggregate
  ceiling, not ADR-121's additional **8 MiB temporary-session cap**. Count,
  revision and durable/concurrent admission fixes do not cover that default.
- Detached-node setters still emit native patches before reconstruction, so
  detach, edit while detached, then reinsert can reject the transaction.
  Reconstruction also restores only nonempty/true form properties after HTML
  attributes, losing current empty/false values. The latter is fix-introduced.
- A late `CanvasCompileError` can still reach the repair draft sink after
  Canvas is disabled. The new error-path checks cover mounted/runtime identity
  and active path, but not the enabled latch and exact captured source block.
  Successful imports retain their guard; the refused-import repair effect
  needs equivalent freshness checks. This is a fix-introduced error path.

These are static, code-supported rereview findings, not newly executed failed
regressions. The passing checks below do not cover these residual cases. The
new publication browser test was **native**; the served node verified Hide and
reopen, not a fresh served publication-while-pinned flow. Shared authority code
and static served wiring support I5's closure without overstating browser evidence.

The single final correction wave and scoped rereview allowed by the selected
workflow are exhausted. The residuals are required behavior, not waived or
parked as harmless. TASK-31232 remains In Progress with AC3, AC9 and AC10 open.
The user subsequently approved one additional focused correction pass for
I1/I2/I4. That pass is now authorized; the residual findings remain open until
new evidence and scoped rereview close them. Keep
the branch/worktree and this plan's review/recovery files intact; no merge,
push, full sweep or unrelated baseline repair is authorized.

```sh
../../.venv/bin/python -m pytest -q Tests/Canvas/test_limits.py Tests/Agents/test_provider_continuation_runtime.py Tests/Chat/test_console_canvas_controller.py Tests/Canvas/test_native_authority.py Tests/Canvas/test_service.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_message_controller.py::test_production_message_controller_prefills_canvas_repair_without_source_state Tests/Canvas/test_compiler_scheduling.py::test_chat_screen_html_import_yields_and_checks_view_before_apply
../../.venv/bin/python -m pytest -q Tests/Canvas/test_runtime_assets.py
```

Exit 0: **358 passed, 1 warning in 34.86s**, followed by **20 passed, 1 skipped,
1 warning in 1.18s**. The warnings are the known Requests dependency mismatch;
the asset skip requires the optional `TLDW_CANVAS_RUNTIME_ARCHIVE_DIR` cache.
No download, full sweep or unchanged long Console matrix was run.

Final browser regressions for DOM move/reinsert, real tool publication while
pinned, and outer Hide/reopen: **3 passed, 1 warning in 5.93s**. This is after
the final runtime edits and manifest refresh. Exact owned browser/driver PIDs
and profile were captured while alive; the child survivor set was empty and
the exact profile was removed afterward. Earlier fix-wave browser runs rely
on fixture cleanup, not independently captured PID/profile provenance.

```sh
../../.venv/bin/python -m pytest -q Tests/Canvas/browser/test_canvas_zero_egress.py::test_dom_move_detach_and_reinsert_preserve_virtual_identity_and_bounds Tests/Canvas/browser/test_canvas_native_flow.py::test_real_tool_publication_keeps_same_revision_pin_until_follow Tests/Canvas/browser/test_canvas_served_flow.py::test_owned_shell_starts_terminal_only_then_opens_and_reopens_canvas
```

The initial publication preview failed with `runtime_unavailable` because the
authored JavaScript edits did not yet match the manifest. Updating the exact
hash/byte records resolved that refusal; this was not a pinning failure. A
subsequent ambiguous Follow locator was narrowed to the intended control.
The strengthened DOM case separately failed when a detached subtree was
reconstructed and moved repeatedly in a later transaction; immediate virtual
renderer-presence bookkeeping corrected it before the final three-test run.

Manifest verification, three JavaScript syntax checks, changed-Python compileall
and diff whitespace checks exited 0. Normalized Ruff comparison across 19 changed
Python paths against `1a6e6c74fb` found zero added/removed findings; 49 changed
format ranges passed. This is unchanged inherited lint debt, not whole-repository
clean lint. The earlier six baseline failures and all review limitations above
remain outside these targeted passing results.

## Authorized residual correction (`1467bdf0a6`)

The additional pass implements session-incarnation-wide temporary 8 MiB
admission, virtual-only detached edits with bounded reconstruction of supported
explicit empty/false form properties, and original authority/owner/enabled/
source checks before compile-refusal repair. The static scoped rereview closes
I1 (temporary cap) and I4 (stale repair). It retains I2 for two cases not covered
by the passing browser test:

- `select.value` is restored before reconstructed options are appended, so a
  non-first or empty/no-match value can be replaced by native selection defaults.
  Child-dependent values must be restored after the child structure, with
  deliberate ordering relative to explicit option overrides.
- Reconstruction checks renderer presence only at its root, then recreates
  every descendant. A child newly created or still live in the reinsertion
  operation can therefore receive a duplicate create and be refused. Each
  descendant needs correct present/absent handling while retaining parent
  attachments and bounds.

These are static code-path residuals, not new executed failing tests. The
review found no separate new Critical/Important issue and did not reopen
I3/I5/M1/M2. One authorized additional pass and its scoped rereview are now
complete, but the feature is not ready to merge. An additional DOM-only pass
requires explicit direction; AC9's six baseline failures remain separate.

Final focused Python command:

```sh
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_canvas_controller.py::test_temporary_owner_enforces_exact_default_session_bytes_across_scopes Tests/Chat/test_console_canvas_controller.py::test_production_owner_counts_concurrent_bytes_and_abort_releases_them Tests/Chat/test_console_canvas_controller.py::test_temporary_import_and_rename_share_admission_without_partial_stage Tests/Chat/test_console_canvas_controller.py::test_durable_committed_stage_is_not_double_counted_after_persistence Tests/Canvas/test_staging.py::test_staging_uses_lower_durable_conversation_source_ceiling Tests/UI/test_console_message_controller.py::test_late_canvas_compile_refusal_cannot_replace_stale_draft Tests/UI/test_console_message_controller.py::test_production_message_controller_prefills_canvas_repair_without_source_state Tests/Canvas/test_compiler_scheduling.py::test_chat_screen_html_import_yields_and_checks_view_before_apply
```

Exit 0: **11 passed, 1 warning in 2.02s**. Coverage includes the actual default
8 MiB boundary across two temporary conversation scopes, concurrent reservation
release, durable accounting, and paused compiler refusal after disable/re-enable,
owner replacement or exact source change. Stale cases preserve the actual draft
and perform no composer load; legitimate repair remains covered.

```sh
../../.venv/bin/python -m pytest -q Tests/Canvas/browser/test_canvas_zero_egress.py::test_dom_move_detach_and_reinsert_preserve_virtual_identity_and_bounds
../../.venv/bin/python -m pytest -q Tests/Canvas/test_runtime_assets.py
```

Final browser exit 0: **1 passed, 1 warning in 3.45s**. The actual renderer
handles multi-event detach/edit/restructure/reinsert, current empty/false form
state, listener continuity and existing identity/cycle controls. Detached edits
also consume the existing 500-operation budget. The final wrapper captured
pytest 67173, driver 67257, Chromium 67276/67284/67285 and the exact profile
while alive; no captured process or profile remained afterward. Two earlier
passing wrappers missed Chromium/profile provenance and are not final cleanup
evidence. No ambient process was selected or terminated.

Runtime assets exit 0: **20 passed, 1 optional archive-cache skip, 1 warning in
1.24s**. Initial sandboxed browser and asset attempts failed at macOS launch/
loopback permission boundaries; these are not product REDs. The identical owned
checks outside that sandbox produced the results above. No download occurred.
Warnings are the known Requests dependency mismatch.

Product REDs were recorded separately: one extra byte admitted above the
temporary ceiling; detached setters addressing pruned renderer IDs; detached
mutations bypassing the operation budget; and three stale compile refusals
reaching repair. Intermediate corrections also fixed an internal non-iterable
null-prototype list and an option-element test using a checkbox-only assertion.
Those categories are not retrospectively assigned to the original defects.

Final worker syntax/manifest verification, compileall, changed-file/range lint
and format checks and diff whitespace checks passed. Root independently closed
the equal-lint-totals doubt without application imports or rerunning tests:
for all nine Python paths in `54fe0b160..1467bdf0a6`, compare Ruff JSON on each
exact revision's `git show` source through the same stdin filename. Counters of
`(path, code, message)` preserve multiplicity and ignore line shifts: **176 at
each revision, zero added and zero removed**. Exit 0. Four inherited whole-file
format-debt files remain; this does not claim globally clean lint/formatting.

## Earlier exact-selection diagnosis

The stricter exact-reopen test subsequently reproduced a product defect: the
child selects the historical root, but the existing served renderer remains on
its old branch. The committed gateway/shell/parent repair carries exact
selection epochs and distinguishes passive synchronization from explicit
selection. Fresh affected-path coverage is recorded separately below.

Before the successful fix runs, deterministic delayed-command
tests additionally reproduced an unfenced child navigation after an explicit
different-revision or same-revision pin (2 failed, 1 warning, 2.01s). ADR-121's
selection-intent amendment records the accepted pre-mutation generation/epoch
fence, implemented in `0724726a0c`. This code-proven race has not
been established as the cause of every intermittent live-card failure; initial
direct-TLS startup had insufficient retained state for causal attribution.
The final focused and covering runs passed both live cases. The actual card
fixture now acknowledges completed handler/pinned state rather than a queued
button press. Do not retroactively assign every earlier transient to this fix.

During diagnosis, a retained-browser-response probe timed out at 300 seconds
and teardown was interrupted at 332 seconds. It is inconclusive evidence, not
a demonstrated product timeout. The test/child/Playwright-driver processes
exited and the owned pytest data/certificate paths were absent. The run did
not capture the Chromium PID/profile, so its browser-process cleanup could
not be independently verified from exact provenance. Unattributed or ambient
processes were not killed; do not generalize the normal-cleanup claim below
to that interrupted diagnostic run.

## Revision and execution context

The actual shared merge base is `e4652f9d379639bd39c13e3b2d005269da0e16d6`.
Nonbrowser matrix runs used product HEAD `fa0f6fcb82`; subsequent Task 7.4 commit
`f41d8ca22a` changes tests and documentation only. Commands below run from the
Canvas worktree with `../../.venv/bin/python` as Python. No merge, rebase, push,
PR, ambient browser interaction, or paid/live provider call is implied.

## Selection-fix verification (`0724726a0c`)

Full affected command:

```sh
../../.venv/bin/python -m pytest Tests/Canvas/test_control_protocol.py Tests/Canvas/test_capabilities.py Tests/Canvas/test_gateway.py Tests/Web_Server/test_canvas_control_spawn.py Tests/Canvas/browser/test_canvas_native_flow.py Tests/Canvas/browser/test_canvas_served_flow.py Tests/Canvas/browser/test_canvas_zero_egress.py Tests/Web_Server/test_canvas_kill_switch.py -q --tb=short --show-capture=no
```

Exit 0: **193 passed, 2 skipped, 1 warning in 161.00s**. Actual Console 19.92s;
direct TLS 7.18s; proxy 7.21s; strict served/native corpora 30.05s/17.69s.
This run precedes a final error-code split that reserves `selection_refused`
for validated freshness mismatch and keeps actual navigation exceptions generic
and fail-closed. Its negative first failed on an incorrectly classified
ValueError (1 failed, 1 warning, 2.06s), then the final-split command passed:

```sh
../../.venv/bin/python -m pytest Tests/Canvas/browser/test_canvas_served_flow.py::test_authority_navigation_failure_is_not_selection_freshness Tests/Canvas/browser/test_canvas_served_flow.py::test_proxy_preserves_only_exact_navigation_freshness_refusal Tests/Canvas/browser/test_canvas_served_flow.py::test_queued_follow_cannot_overwrite_later_exact_pin Tests/Canvas/browser/test_canvas_native_flow.py::test_stale_navigation_is_discarded_but_unknown_failure_closes Tests/Canvas/browser/test_canvas_native_flow.py::test_old_failed_navigation_cannot_close_newer_valid_selection Tests/Canvas/browser/test_canvas_native_flow.py::test_shell_retries_only_validated_stale_selection Tests/Canvas/test_gateway.py::test_original_browser_epoch_is_checked_before_authority_mutation -q --tb=short --show-capture=no
```

Exit 0: **17 passed, 1 warning in 5.88s**. This is the final handler/proxy/browser
error-path coverage; do not describe the 193-test run as post-split execution
or add overlapping counts as distinct tests. RequestsDependencyWarning remains;
the two skips are missing Firefox/WebKit. Final Ruff checks pass on 12 changed
Python files, four whole-file formatter checks pass, other changed Python hunks
were formatted without rewriting inherited debt, and JS syntax/diff checks pass.

The served recovery case explicitly closes the old transport, waits for
production unbind, then opens an authenticated fresh child: Connected, old URL
404, other browser unaffected. Its new temporary root is not restoration of
old temporary IDs. **Automatic reconnect is not demonstrated.** Durable manual
recovery is covered separately below. The final normal run captured its exact browser/driver/
child PIDs and profile and verified their absence after cleanup; this does not
close the interrupted-run provenance gap above.

## Durable manual recovery (`4ce7fc756f`)

The actual served Console flow now creates and saves two revisions, closes the
old control/transport, and starts a fresh authenticated AppService child against
the same owned database. A normal saved-conversation load and real persisted
transcript-card open restore the original conversation, Canvas and root revision
IDs and digest. The shell is Connected and shows pinned v1 despite saved v2;
the old capability URL returns 404 and the replacement provider has zero calls.
This is explicit manual recovery, not automatic resume or temporary-history revival.

```sh
../../.venv/bin/python -m pytest Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_scripted_gateway_emits_create_then_stable_update Tests/Canvas/browser/test_canvas_served_flow.py::test_live_stack_setup_failure_rolls_back_owned_resources Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_child_abnormal_exit_cleans_owned_state -q --tb=short --show-capture=no
```

Exit 0: **6 passed, 1 RequestsDependencyWarning in 44.87s**. The amended actual
flow took 30.78s; abnormal cleanup took 11.60s. Ruff and whole-file formatter
checks on both changed test files and `git diff --check` pass. No product edits.
Captured owned browser/driver/child processes and browser profile were absent
afterward; the test also asserts both services and owned DB/data/certs/sites
are cleaned. This does not repair the earlier interrupted-run provenance gap.

Focused iteration first exposed a missing saved-load adapter (49.55s), then
composer focus settling (15.02s), then a fixture whose services were constructed
with no DB before attaching its real file DB (33.29s). The helper now runs the
existing production service-wiring method after that assignment, and waits
within the existing focus budget using real Escape/focus observation, without
replaying Send. The focused recovery then passed in 42.53s before the final
covering run. These failures establish fixture gaps, not production defects.

## Original browser and archive slice (`f41d8ca22a`)

```sh
../../.venv/bin/python -m pytest Tests/Canvas/browser/test_canvas_native_flow.py Tests/Canvas/browser/test_canvas_served_flow.py Tests/Canvas/browser/test_canvas_zero_egress.py Tests/Chatbooks/test_chatbook_canvas_round_trip.py::test_canvas_v3_whole_graph_round_trips_atomically_as_new -q --tb=short
```

Exit 0: **64 passed, 2 skipped, 1 warning in 135.70s**. Firefox and WebKit were
not installed; Chromium was exercised. The warning is the known Requests
dependency mismatch. This covers:

- Native browser lifecycle, actual temporary-history promotion/destruction,
  confirmed unsent draft, passive download, revision/history and branch behavior.
- Actual served `TldwCli`/Console Send, synthetic provider tool rounds, durable
  finalization, create/update and hot reload. Direct and progressive-disclosure
  fixture contracts are checked; these are not live-provider samples.
- A separate minimal Textual child using production AppService/authentication,
  TLS/WSS/private control and Canvas routes, with two real children/browser
  profiles. Direct TLS and a real TLS reverse proxy to a plain loopback backend
  are separate cases. Copied/guessed URLs, control loss and reconnect are covered.
- Forty-one canonical adversarial cases plus a same-origin canary through each
  product route. Admission refusals are distinguished from guest execution.
  Held execution acknowledgments require an exact completed startup request
  census and fixed worker bootstrap; subsequent guest traffic is forbidden.
  Independent listener and recorder-negative tests supplement browser events.
- Archive export, removal of the disposable source DB, atomic graph import,
  and separate browser reopen of the exact imported branch/revision.
- Owned subprocess/file cleanup after actual child kill and injected cleanup
  failure. No forced hung-AppService timeout was separately demonstrated.

An earlier combined run had 61 passes, two failures and two skips. One failure
was a sync-Playwright fixture retaining an event loop across an async test;
function-scoped fixtures corrected that overlap. The other was an intermittent
rapid-load disconnect with exact revision match but no acknowledgment. Narrow
and final covering runs passed, but **no causal product fix for that disconnect
or an earlier bridge 409 is established**. The paired diagnostic calls
`served_canvas_state`, which can synchronize selection and is not a pure
observer. The browser-only corpus does not add that diagnostic call.

## Broader targeted matrix

| Slice | Result | Qualification |
| --- | --- | --- |
| Packaging | 2 passed, 1 warning; 4.81s | Disposable wheel inputs/assets/imports, not a clean dependency installation |
| Affected Agents modules | 380 passed, 1 warning; 9.89s | Five modules, not all repository tests |
| Canvas/core/migrations/Chatbooks/Web Server | 1019 passed, 3 failed, 2 skipped, 2 warnings; 76.36s | One in-scope stale schema assertion subsequently corrected; two baseline failures |
| Affected Console/UI modules | 1441 passed, 6 failed, 11 warnings; 1212.25s | Two in-scope census assertions subsequently corrected; four baseline failures |
| Corrected schema/census slice | 6 passed, 4 warnings; 38.56s | Genuine v64 migration and bidirectional inventory assertions retained; warning categories not retained |

Exact commands:

```sh
../../.venv/bin/python -m pytest Tests/Packaging/test_canvas_gateway_distribution.py -q
../../.venv/bin/python -m pytest Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service.py Tests/Agents/test_canvas_tool_provider.py Tests/Agents/test_provider_continuation_runtime.py Tests/Agents/test_tool_record_projection.py -q
../../.venv/bin/python -m pytest Tests/Canvas --ignore=Tests/Canvas/browser Tests/ChaChaNotesDB/test_canvas_migration.py Tests/ChaChaNotesDB/test_index_census.py Tests/DB/test_chachanotes_v65_trace_compaction_migration.py Tests/Chatbooks Tests/Web_Server -q
../../.venv/bin/python -m pytest Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_canvas_controller.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_message_actions.py Tests/Chat/test_console_personal_context_snapshot.py Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_console_semantic_mutation_inventory.py Tests/Chat/test_message_metadata.py Tests/UI/test_console_canvas_card.py Tests/UI/test_console_message_controller.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_settings_canvas.py Tests/UI/test_settings_raw_cli.py -q
../../.venv/bin/python -m pytest Tests/DB/test_chachanotes_v65_trace_compaction_migration.py Tests/Chat/test_console_semantic_mutation_inventory.py::test_live_sql_mutation_sites_are_classified Tests/Chat/test_console_semantic_mutation_inventory.py::test_public_mutation_boundary_calls_are_classified Tests/Chat/test_console_semantic_mutation_inventory.py::test_inventory_document_exists_and_names_the_contract Tests/Chat/test_console_semantic_mutation_inventory.py::test_document_exact_census_is_bidirectionally_synchronized -q
```

The core skips are unavailable pinned-archive cache and an opt-in slow performance
case. Core/Console warnings include descriptor growth (1104/267 respectively),
which was not causally baselined, plus the known Requests mismatch and existing
source-escape warnings during AST scans. Do not count overlapping slices as
distinct tests or call the selected matrix green.

## Six reproduced baseline failures

Each of these IDs failed both on the branch and in a disposable archive of the
exact shared merge base, using the same venv. The two Thinking tests failed on
baseline in 1.35s; a six-ID Console/census comparison gave six branch failures
in 5.80s versus four baseline failures and two baseline passes in 8.77s.
The two passing baseline census IDs are the in-scope corrections above.

| Failing ID | Matching cause |
| --- | --- |
| `Tests/Chatbooks/test_chatbook_thinking_round_trip.py::test_chatbook_export_blocks_opaque_future_thinking_with_upgrade_copy` | Fixture directly updates a protected message; semantic-mutation authorization rejects it |
| `Tests/Chatbooks/test_chatbook_thinking_round_trip.py::test_durable_soft_delete_clears_thinking_from_tombstone` | Assertion expects NULL; tombstone retains Thinking JSON |
| `Tests/Chat/test_console_chat_controller.py::test_image_only_draft_is_sendable` | Missing preparation at `capture_mode` read |
| `Tests/Chat/test_console_chat_controller.py::test_compose_mcp_provider_excludes_console_shadowed_builtin_names` | Expected 29 MCP names, observed 26 |
| `Tests/Chat/test_console_chat_store.py::test_first_persist_context_failure_does_not_force_atomic_promotion_legacy_path` | Legacy fake rejects `trace_boundary` keyword |
| `Tests/UI/test_settings_raw_cli.py::test_pending_raw_cli_save_vetoes_real_navigation_until_arrival` | Fixed wait expires before Settings mounts |

Isolated comparisons used `-q --tb=short --show-capture=no`. The exact owned
baseline directory `/private/tmp/canvas-final-baseline.0PD4DO` was validated and
removed; its source is recoverable from the recorded commit. These failures
are not authorization to fix unrelated behavior. The Settings case overlaps
an earlier 29-failure baseline characterization; it is not a new distinct set.

## Static checks and remaining boundaries

Task 7.4's eight owned Python files outside the semantic inventory pass Ruff.
The six schema/census checks were rerun solely to resolve the review's missing
warning-category question: **6 passed, 4 warnings in 37.87s**. The retained
summary identifies RequestsDependencyWarning plus three SyntaxWarnings for
invalid escape sequences: `Tools/patch_tool_impls.py:32` and
`Utils/Splash_Screens/environmental/train_journey.py:31–32`. Both source files
are unchanged from the exact shared merge base. This characterizes the fresh
run, not a retroactive reconstruction of the earlier lost warning summary.
No unrelated warning cleanup was performed.

The inventory's base/current normalized `(path, code, message)` Counters are
equal at 34 findings each. Whole-file formatting passes for the five helper/
served/zero-egress files and archive test; native, inventory and schema tests
retain inherited whole-file formatting debt. All 31 changed-hunk format checks,
nine-file compileall and diff whitespace checks passed. Task 7.3's seven-file
lint comparison also has equal normalized Counters (103 each, no added/removed
findings), not merely equal counts. None of this claims globally clean lint.

Runtime quotas are synthetic, single-host measurements. Gross browser RSS is
not guest heap, and a recursion engine trap is not proof that the configured
stack cap caused it. Embedded generated QuickJS bytes are reviewed through
manifest, provenance, integrity and reproduction checks, not claimed manual
inspection of every byte. See the [runtime profile](V1_RUNTIME_COMPATIBILITY.md).
No ambient secrets, source/payload/capability logs, screenshots, recordings,
TLS keys, or disposable browser state are intentionally retained by these tests.
V2 libraries, V3 VFS, elevated capabilities and TASK-31003 sync remain deferred.
