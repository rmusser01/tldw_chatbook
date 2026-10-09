# Incremental Send speed: first implementation slice

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Work inline with no subagents, as requested. Steps use checkbox syntax for tracking.

**Goal:** Advance immediate terminal feedback and fast Send dispatch without weakening long-term stability; this first slice removes needless stock MCP catalog preparation and measures the remaining work.

**Architecture:** Add one early return inside the existing source-qualified controller composition. Preserve the complete hook reconciliation path and all current worker, chain, history, permission, and resource ordering. Use original operation counts and matched native timings to select the next incremental edits.

**Tech Stack:** Existing Python >=3.12, Textual 8.x, asyncio, SQLite/private storage, pytest, and Ruff. No new dependency or schema.

**Spec:** [Approved revised design](../specs/2026-10-05-incremental-send-speed-design.md).

**Status:** Approved for inline execution on 2026-10-06. Task 1 test preparation complete; integration and production implementation remain pending.

## Global constraints

- The terminal must visibly acknowledge Send within 100 ms across normal, enabled-feature, approval-wait, and refused/error paths, and remain responsive to input while preparation runs.
- Ordinary chat targets less than one second of application overhead before provider-adapter entry.
- Received/preparing feedback must not claim durable acceptance before commit. Stable persistence, fresh consent, ownership, cancellation, and recovery are mandatory throughout.
- Preserve the existing enabled tool set, hook consent, capture, persistence, cancellation, recovery, and resource ownership.
- Native admission proofs cannot cross an await or worker boundary.
- Keep chain creation, accounting, admission, and callback ordering unchanged in this pass.
- Keep history as an awaited postcommit effect in this first pass.
- No permission verdict cache, global cache manager, longer TTL, new dependency, weaker guard, or raised performance ceiling.
- No full suite without explicit approval. Run directly affected regressions and scoped lint/format checks.
- Keep both active checkouts untouched. After plan approval, use an isolated managed worktree based on integrated current development/fix sources, rather than the frozen evidence directory.
- Serialize native app runs with other app tests. Never kill or interfere with another task's processes.
- Missing host evidence leaves acceptance unverified; a partial improvement does not close the overall Send-performance objective.

ADR required: no for the stock MCP early return and diagnostic/test additions that retain existing interfaces.

ADR path: existing `backlog/decisions/126-complete-local-backup-and-recovery.md` and `134-fleet-admission-and-automatic-work-budgets.md`; retain `197-console-hook-configuration-review.md` for the unchanged hook path.

Reason: no new data-sharing, worker, storage, or consent interface. A later hook admission interface requires an ADR-197 amendment before product code; any new snapshot/worker lifetime requires its applicable existing ADR amendment.

## Review focus

1. Empty versus unset maximum: an exact empty frozen set skips stock work; `None` preserves normal discovery/composition.
2. Previously nonempty live inspector state: empty live composition publishes `(None, None)` while disposable preview leaves the previous live values untouched.
3. Plugin/custom composition: a supplied plugin maximum, including `{}`, preserves validation and ownership checks; custom factories/services retain their ordinary callbacks.
4. Source/owner drift and cancellation: changed producer functions, receiver/service identity, loop ownership, and Stop/close retain original refusal, drain, and retirement behavior.
5. Hook disable/remove transitions: the current reconciliation still rotates queued epochs and removes grants; final launch checks are not substituted for observing those transitions.

## Task 1: Skip the qualified stock empty MCP maximum

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_controller.py`, `_compose_mcp_provider`, currently lines 15603–15847 in the reviewed main source. Re-resolve lines on the integrated execution base.
- Extend: `Tests/MCP/test_console_snapshot_source_contracts.py`, reusing its `snapshot_case` fixture from `Tests/Chat/test_console_async_mcp_snapshot.py` and the existing original-code observers.
- Extend: `Tests/Plugins/test_native_components.py` only for the empty-bound plugin controls.
- Verify existing: `Tests/Agents/test_mcp_tool_provider.py`, `Tests/Chat/test_console_turn_execution_context.py`, `Tests/Chat/test_console_agent_swap.py`, `Tests/Chat/test_console_close_during_postcommit_owner.py`, and `Tests/Chat/test_console_close_during_durable_postcommit.py`.
- Copy/commit the approved spec and this plan into the implementation branch under `Docs/superpowers/{specs,plans}/`; keep raw evidence in the side artifact directories.

**Interfaces:** `_compose_mcp_provider` retains all existing parameters and its `MCPToolProvider | None` return. It consumes the existing `capture_standard_controller_composition(factory, service)` qualification and synchronous `require_composition_current()` check; its local `publish` helper retains `publish_counts` semantics. No new production interface.

- [ ] **Prepare the isolated execution base.** Use the worktree skill after approval, inspect attached worktrees and exact Git roots, and create/reuse a suitable managed checkout on an integrated current main-fix base. Verify an isolated interpreter imports that checkout, not another editable installation. Read scoped guidance, relevant lessons, and the corresponding Backlog task before changing code. Do not copy uncommitted sources from an active checkout as a substitute for integration. If no integrated base is available, complete unaffected test/measurement preparation in the side directory and report the dependency without modifying those checkouts.
- [ ] **Enroll the atomic Backlog task.** Find an existing matching open task or create one through the CLI after a fresh ID collision check. Set In Progress and add this plan before implementation. AC: zero stock catalog preparation for the qualified empty route; live/preview parity; preserved unset/nonempty/custom/plugin behavior; original ownership/cancellation controls; targeted checks and measured counts recorded. Record the ADR check and commit the reviewed design/plan in the isolated branch.
- [ ] **Record the original operation baseline.** On real private config/permission/catalog storage, observe original provider construction, catalog preparation, catalog source reads, and inspector results for empty, unset, and nonempty maxima. Also record first rendered acknowledgment and main-loop stalls from Enter and mouse Send on the integrated base before changing it. Use the existing passive/original-code observation pattern; do not replace protected stock callables to count them. Explicitly confirm that the stock composition qualification is available before interpreting its operation counts.
- [ ] **Write the failing stock test.** Add `test_empty_stock_mcp_maximum_skips_catalog_work(snapshot_case)`, with the actual service and controller. Reuse the existing original-count observer and extend it locally to count the original provider constructor/catalog body if necessary. Its required assertions are:

```python
provider = await case.controller._compose_mcp_provider(
    case.session.id, maximum_tool_ids=frozenset(), plugin_maximum=None
)
assert provider is None
assert case.app.console_mcp_tool_count is None
assert case.app.console_mcp_not_connected_count is None
assert provider_constructions == 0
assert catalog_preparations == 0
assert catalog_source_reads == 0
```

  The three counters are scalar observations of original code within that call, not substituted producer functions. Native source acquisition required merely to qualify the route is recorded separately and is not mislabeled as catalog work.
- [ ] **Write the transition and fallback tests.** Add `test_empty_mcp_preview_preserves_live_counts`, with prior live values `(3, 1)` and `publish_counts=False`; assert those values remain `(3, 1)` and no catalog work runs. Add `test_empty_mcp_live_clears_previous_counts`, starting from a real eligible nonempty composition and asserting a subsequent empty live result is `(None, None)`. Add `test_empty_mcp_maximum_keeps_ordinary_fallback` for `None`, a nonempty set, a custom factory/service, altered source/code/default/closure inputs, changed receiver/owner, and a non-exact empty-set subtype; preserve the ordinary route or original refusal for each case. Add `test_empty_mcp_maximum_preserves_plugin_validation` using the native plugin fixture for both empty and nonempty supplied plugin maxima, including an unavailable/changed plugin owner. Retain the existing kill-switch, nonempty definition-hash, disconnected-server, and loop-ownership controls.
- [ ] **Run the new tests on unchanged code.** Run `python -m pytest Tests/MCP/test_console_snapshot_source_contracts.py Tests/Plugins/test_native_components.py -k 'empty_mcp or empty_stock_mcp' -q -rs`. Require the stock counter case to fail because the original preparation runs. A setup/profile-selection failure or skipped stock case is not red evidence. Existing parity controls may already pass; retain them.
- [ ] **Add the single early return.** Immediately after the existing successful synchronous `require_composition_current()` check, before provider construction, skip only when `composition is not None`, `plugin_maximum is None`, and `maximum_tool_ids` is an exact empty `frozenset`. Publish `(None, None)` through the existing helper, then return `None`. Preserve all earlier source/owner checks, plugin/custom paths, and exception handling. Do not add another guard in the provider, another cache, or a new helper interface.
- [ ] **Run the affected regressions.** Run the two extended files and the existing verification files listed above, plus `Tests/Chat/test_console_async_mcp_snapshot.py`. Expect all new cases and directly affected existing controls to pass. Inspect skip reasons and reproduce any unrelated failure on the unchanged integrated base before attributing it. This is a targeted selection, not the full suite.
- [ ] **Check and review the change.** Run changed-file Ruff, changed-file/new-test formatting checks, and `git diff --check`. Record any preexisting unrelated whole-file debt rather than mass-formatting the controller. Re-run original operation counts and check live/preview transitions. Review the branch diff and scope: one production guard, its regressions, task records, and documentation.
- [ ] **Commit the atomic deliverable.** Complete task AC and implementation notes with source base, exact count changes, validation, limitations, and existing ADR links. Set Done via CLI only when the repository DoD is satisfied. Commit with `perf(console): skip stock MCP composition for an empty maximum`. This task's completion does not imply the wider timing target is achieved.

## Task 2: Measure the slice and qualify subsequent incremental reductions

**Files:**

- Existing measurement: `Tests/Performance/test_console_native_pause_probe.py` and the existing private-profile/native-custody runner.
- Existing controls: `Tests/Agents/test_hook_permissions.py`, `Tests/Chat/test_prompt_history.py`, and `Tests/Agents/test_local_tool_provider.py`.
- Read for attribution: `Agents/hook_permissions.py`, `Agents/local_tool_provider.py`, `Chat/prompt_history.py`, and the original controller/agent worker callbacks.
- Create: `Docs/Development/2026-10-06-incremental-send-speed-first-slice.md` for concise verification findings in the implementation branch; raw captures remain in this task's side directory.
- Measurement-only additions, if needed, remain alongside the existing probe/runner and never alter product UI or protected guards.

**Interfaces:** Terminal acknowledgment and input responsiveness are independent requirements; do not treat a fast provider-entry number as evidence for either. The comparison report records source revision/hashes, host/interpreter/dependencies, private configuration shape, baseline/candidate sample identity, cold/warm classification, Send-to-actual-provider seconds, first-rendered-ack seconds, actionable input-loop evidence, or an explicit unavailable reason, original operation counts, heartbeat stalls, functional outcomes, cleanup qualification, and observation limits. Keep only scalar evidence; do not retain frames, Tasks, receiver/argument objects, prompt bodies, or credentials.

- [ ] **Enroll the measurement task.** Create/find the atomic Backlog record, set In Progress, and add this plan plus existing ADR-126/134/197 links. Its deliverable is a truthful comparison and ranked candidates, not a claim that every target or platform passed.
- [ ] **Verify the observer before using timing results.** Run the existing heartbeat synchronous-stall control and passive original-source controls separately from timings. Preserve original assertions, guards, deadlines, source hashes, and native identity/pipe cleanup. Confirm Send timing starts at the actual Enter or mouse action and reaches the actual provider-adapter seam. When adding rendered acknowledgment observation, pin `test_send_ack_observer_does_not_accept_status_assignment`: changing status without a completed rendered frame must leave its acknowledgment timestamp unset. A test/harness render completion is app-side render evidence; actual native terminal acknowledgment needs its corresponding native output observation. Cover normal Send, enabled tools/hooks, a held approval, and a refused/error Send; keep a painted acknowledgment visible while the original preparation/approval remains pending and use the existing mounted/input controls to verify keys and Stop remain actionable. Do not block on config, disk, or provider work before acknowledging receipt. Mark an unsupported observation unavailable rather than substituting a status update or task submission.
- [ ] **Run matched Windows comparisons.** Use an unchanged integrated baseline and candidate with matching interpreter, dependencies, instrumentation, private config shape, and filesystem placement. Alternate two baseline/candidate pairs, each with three Sends; distinguish first Send after restart from later Sends. Use the existing native private-profile runner to execute `python -m pytest Tests/Performance/test_console_native_pause_probe.py::test_native_console_pause_probe -q -rs`. Keep `TLDW_PAUSE_PROBE_RESULT` inside that runner's admitted private temporary root; copy the completed evidence out only through the existing cleanup path. Do not weaken the original 15-second or native-work assertions if they remain red.
- [ ] **Qualify every run's outcomes and retirement.** Require three provider calls, three complete traces, three reply links, and zero dispatch checkpoints. Record unchanged/source-current qualification, observer overflow or unknown exits, process-tree emptiness, native identity release, and retired pumps. Treat forced termination, shared-profile admission failure, source drift, or missing cleanup evidence as an unqualified run. Keep successful functional outcomes distinct from a failing legacy latency assertion or missing product/native-terminal qualification.
- [ ] **Measure enabled paths separately.** Use private configurations for enabled hooks, stock MCP tools, plugin-owned MCP tools, and local roots/permissions. Report the configuration and original controls for each; a fast empty bound proves only its own path. Repeat matched comparisons on Linux and macOS when available, with the same observer and source qualification. Missing hosts remain explicit, not inferred from Windows.
- [ ] **Count hook reconciliation effects without changing admission.** Observe original config/store open, lock, read, reconcile, and write operations for no section, empty section, master disable, and enabled v1/v2. Include approved -> disabled -> reenabled and removed -> readded named/legacy definitions. Run the original `test_disable_reenable_retains_consent_but_retires_queued_epoch` plus the existing cross-owner revoke/notification controls. Record which original call must rotate/delete each token/grant. No new hook query or store shortcut is implemented in this task.
- [ ] **Attribute local, worker, and history work.** Count `_default_specs` construction by admitted root, distinguishing static definitions from dynamic gates/descriptions/services and root-bound closures. Check IDs/order/schema/routing parity using the existing local-provider controls. Identify callback actor and repeated database opens within the existing finite worker/borrower ownership, retaining foreign transactions and positive retirement. Measure prompt-history cold load, warm append, duplicate suppression, and cap rewrite separately; `_load_locked` already avoids repeat successful loads. Run the relevant existing history cancellation/order/failure controls. Do not combine nested timing intervals into predicted savings.
- [ ] **Write and review the report.** List every qualified individual sample and independent target result: `ack_seconds <= 0.100` and `provider_seconds < 1.000`. Missing acknowledgment evidence is unverified, never a pass. Prefer proven count reductions when elapsed distributions overlap. Rank only existing-lifetime duplication that was actually observed; name exact candidate functions and the required negative controls. Retain current reads if their second consumer lacks valid ownership. Record the remaining gap and any required ADR amendment for the next concrete plan. Worker/chain/history scheduling remains option 2.
- [ ] **Finalize verified records.** Update measurement-task AC/notes and self-review the branch, report, source drift, targeted checks, and ADR links. Keep the overall Send-performance objective open if thresholds or host/native evidence remain unmet. Further product edits require a concrete reviewed plan within the approved incremental design; do not silently introduce the withdrawn hook shortcut or broader dispatch restructuring.

## Plan self-review

- Spec coverage: the safe MCP change has Task 1; hook reconciliation and local/worker/history attribution have Task 2. UI/startup edits remain after integration of the separately owned main fixes, with no competing screen/widget edits in this slice.
- Review focus: unset/empty and inspector transitions are new stock tests; plugin/custom and source drift are new controls plus existing source-contract tests; cancellation/close and hook epochs retain their original regressions.
- The only production change is inside `_compose_mcp_provider`; no new authority or worker interface is produced or consumed.
- Original source-qualified observers and private-profile fixtures are reused. Native timing and diagnostic attribution remain separate measurements.
- Completion remains governed by long-term stability, immediate rendered feedback/input responsiveness, and sub-second dispatch; a catalog-count improvement alone cannot fulfill the user's goal.
- This plan's side artifact does not modify either active checkout. Execution-base setup, Backlog changes, product code, tests, commits, and native runs start only after plan review.
