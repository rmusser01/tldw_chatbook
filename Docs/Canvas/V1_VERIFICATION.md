# Canvas V1 verification record

Date: 2026-09-05. Isolated branch: `codex/canvas-v1`.
Architecture: [ADR-115](../../backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md).
Plan: [Canvas implementation](../superpowers/plans/2026-09-03-chatbook-canvas-implementation.md).

This is targeted evidence, **not full-suite, release, or integration approval**.
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
  ceiling, not ADR-115's additional **8 MiB temporary-session cap**. Count,
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
An additional focused correction pass requires explicit user direction. Keep
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

The stricter exact-reopen test subsequently reproduced a product defect: the
child selects the historical root, but the existing served renderer remains on
its old branch. The committed gateway/shell/parent repair carries exact
selection epochs and distinguishes passive synchronization from explicit
selection. Fresh affected-path coverage is recorded separately below.

Before the successful fix runs, deterministic delayed-command
tests additionally reproduced an unfenced child navigation after an explicit
different-revision or same-revision pin (2 failed, 1 warning, 2.01s). ADR-115's
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
