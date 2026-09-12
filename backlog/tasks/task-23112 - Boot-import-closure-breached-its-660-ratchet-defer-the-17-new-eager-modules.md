---
id: TASK-23112
title: Boot import closure breached its 660 ratchet -- defer the 17 new eager modules
status: Done
assignee: []
created_date: '2026-08-28'
updated_date: '2026-08-28 23:33'
labels:
  - performance
  - startup
dependencies:
  - task-23029
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`test_app_import_own_module_count_stays_at_the_post_diet_size` is RED on
pristine dev (`b5eaa9cf64`, 2026-08-28): **666** own modules after `import
tldw_chatbook.app` against the **660** ratchet. Under ADR-097 (boot budgets
are ratchets) the constant does not rise -- the cost defers or is shed.

Vs the last in-budget state (`c6218918d1`, 657 modules): 17 modules added,
8 removed (TASK-23023's Research_Workspace diet). The added edges, traced
with an import-parent recorder (TASK-23029):

1. **`Chat/chat_persistence_service.py`** (+912 lines since the pin) gained
   module-scope imports worth ~12 boot modules: `Chat.attachment_core`
   (drags `Utils.file_handlers`), `Chat.console_chat_fork` (drags the
   `Event_Handlers.Chat_Events` package + `chat_image_events`),
   `Chat.library_activity` (drags `Chat.trajectory` + `Utils.log_sanitizer`),
   `Video_Generation.video_metadata` (drags `video_formats` + the package).
   This is the highest-yield single fix.
2. **`app.py`** gained a module-scope `Chat.console_raw_cli` edge, dragging
   `Tools.raw_cli_executor` and `Agents.run_log` (3 modules).
3. **`Chat.console_runtime`** now eagerly imports `Chat.thinking_blocks`
   (1 module).
4. `Widgets.splash_screen` -> `Widgets.pausable_progress` (TASK-23022) and
   `tldw_chatbook/__init__` -> `Utils.tiktoken_runtime` (ADR-093) look like
   genuine boot-path needs; verify rather than assume.

Beware the known traps: a deferral changes WHICH objects the build binds
(lessons, TASK-21108), a lazy facade protects nothing consumers import
directly (TASK-21200), and tests that patch moved names disconnect silently
(TASK-19830).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_app_import_own_module_count_stays_at_the_post_diet_size` passes
  on dev with `MAX_TLDW_MODULE_COUNT` still 660 (no exception-ledger entry)
- [x] #2 The deferred imports still resolve on their real use paths (targeted
  tests per moved edge, per the closure-guard house pattern in
  `Tests/Packaging/`)
- [x] #3 The `boot_import_modules.txt` snapshot is re-pinned via
  `scripts/update_boot_budget_snapshots.py` once the count is back under
  budget (the script refuses while over budget)
- [x] #4 If reality lands well under 660, apply ADR-097's tightening convention
  (lower to measured + 30) -- assessed and correctly NOT applied: 646 measured,
  and `646 + 30 = 676` is above 660, so "tightening" would raise the ratchet.
  The reduction (20 modules) is also under the guard's 30-module standard slack,
  which is the threshold the convention is written against.

## Evidence

TASK-23029's implementation notes carry the full trace and the guard's new
breach message, which names all 17 modules.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-derive the trace with an import-parent recorder (`sys.meta_path` finder
   that names the module whose body triggered each import) rather than reading
   the diff; record the baseline count.
2. For each named edge, find where the imported symbols are actually used, and
   whether the enclosing function runs at import time or during
   `TldwCli.__init__` (TASK-22223 / TASK-21200 traps).
3. Defer the highest-yield edge first; **re-measure after each deferral** so
   every claimed saving is a measured delta, not an inferred one.
4. Verify the two edges the trace called "genuine boot-path needs" instead of
   assuming.
5. Write one subprocess-isolated closure guard per moved edge in
   `Tests/Packaging/`, each with an anti-vacuity check and a real-use-path
   proof; mutation-test every guard (re-add the eager import, watch it go red).
6. Re-pin `boot_import_modules.txt`, re-run all four boot budgets, run the
   affected suites, and confirm any red reproduces on pristine sources before
   attributing it to this change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaid ADR-097's standing import-weight breach by deferral only: **666 -> 646**
own modules against the unchanged **660** ratchet (headroom 14), no
exception-ledger row. Two edges moved; every number below is a measured
re-run of the tracer, not an estimate.

**1. `Chat/Chat_Functions.py` -> `chat_persistence_service` (-18 modules,
666 -> 648).** `ChatPersistenceService` had exactly one use site,
`save_chat_history_to_db_wrapper`, and one module-scope import; `app.py`
reaches the module through `Library.library_local_rag_search_service ->
library_rag_service -> library_rag_state -> library_rag_answer_service ->
Chat_Functions`. Moving the import into the function took the whole subtree
off the boot path -- `attachment_core`, `console_chat_fork` (+
`Event_Handlers.Chat_Events` + `chat_image_events`), `video_metadata` (+
`video_formats` + the package), `console_context_repository`,
`console_dispatch_repository`/`_checkpoint`, `console_library_policy_repository`,
`console_prefill`, `console_roleplay_metadata`/`_identity`, `message_metadata`,
`console_context_policy`, `Utils.file_handlers`. This is the consumer-side fix
the TASK-21200 lesson warns about, so the two questions it says to ask were
answered with evidence rather than reasoning: the enclosing function never runs
at import or during `TldwCli.__init__`, and the module was already resident at
`_ui_ready` before this change (`console_runtime.ensure_console_runtime`
imports it during mount), so nothing was *relocated* into first paint -- the
`_ui_ready` census is byte-identical at 968 with and without the change.

**2. `Chat/console_raw_cli.py` -> `Tools.raw_cli_executor` (-2 modules,
648 -> 646).** `app.py` imports `console_raw_cli` at module scope and
constructs `RawCliRuntime` inside `TldwCli.__init__`, so an import-only
deferral would have been the exact half-fix TASK-21200 records (guard green,
boot unchanged). Both halves moved: a lazy `_raw_cli_executor()` module
accessor replaces the module-scope import (with `TYPE_CHECKING` for the three
annotation-only names and a string forward reference in the `RawCliEventSink`
alias, which is an assignment and would otherwise still resolve eagerly), and
the default `RawShellExecutor` is now built by `RawCliRuntime.
_executor_or_default()` on the first `execute()` instead of in `__init__`.
The dedicated construction guard fails on the half-fix while the closure guard
passes -- verified by mutation. Only 2 modules moved rather than the traced 6
because `Tools`, `Tools.tool_executor`, `Agents` and `Agents.run_log_format`
are also pulled by the pre-existing `app -> UI.Tools_Settings_Window ->
Agents.local_tool_provider -> Agents.tool_catalog` chain.

**Two traced edges were refuted by measurement and deliberately left alone.**
The import-parent tracer records only the FIRST importer, which makes an edge
look load-bearing when a second boot-path importer keeps the module resident
anyway. `Chat.console_runtime -> Chat.thinking_blocks` buys **0**:
`Chat_Functions.py:98` imports `thinking_blocks` at module scope too.
`Chat.library_activity` (with `Chat.trajectory` and `Utils.log_sanitizer`) is
also imported by `Agents/library_tool_provider.py`, reached through the
pre-existing `UI.Tools_Settings_Window -> Agents.local_tool_provider ->
Agents.tool_catalog -> Agents.library_rag_tool_provider` chain, so deferring it
at `chat_persistence_service` bought nothing and deferring it at
`library_tool_provider` is a different (deprecated-window) chain, out of this
task's scope. The two edges the trace called "genuine" were verified, not
assumed: `tldw_chatbook/__init__` *calls* `install_tiktoken_runtime()` at
package scope (ADR-093 ordering requirement), and `Widgets/splash_screen.py`
yields `PausableProgressBar` from `compose()`. Both stay.

**Tightening convention: assessed, correctly not applied.** ADR-097 says to
lower the constant to `measured + standard slack` when a PR reduces the
measured value by more than that slack. The reduction here is 20 modules
(under the 30-module slack) and `646 + 30 = 676` is above 660, so lowering
would be raising. `MAX_TLDW_MODULE_COUNT` stays at 660.

**Trade-off accepted.** `Chat.console_raw_cli` itself (1 module) stays on the
boot path: removing it would mean making `app.raw_cli_runtime` lazy, and eight
consumers read it via `getattr(app, "raw_cli_runtime", None)` while tests
assign doubles to it -- a binding-semantics change (TASK-21108) out of
proportion to one stdlib-only module.

**Modified/added files.**
- `tldw_chatbook/Chat/Chat_Functions.py` -- import moved into
  `save_chat_history_to_db_wrapper`.
- `tldw_chatbook/Chat/console_raw_cli.py` -- lazy `_raw_cli_executor()`
  accessor, `TYPE_CHECKING` block, forward-referenced `RawCliEventSink`,
  `RawCliRuntime._executor_or_default()`.
- `Tests/Packaging/test_chat_persistence_import_closure.py` (new, 2 tests) and
  `Tests/Packaging/test_raw_cli_import_closure.py` (new, 3 tests) -- closure +
  construction + real-use-path guards, all mutation-tested.
- `Tests/Performance/boot_budget_snapshots/boot_import_modules.txt` -- re-pinned
  at 646 via `scripts/update_boot_budget_snapshots.py --only import-weight`.
- `Tests/Performance/test_app_import_weight.py` -- the standing-breach comment
  replaced by the repayment record.
- `backlog/decisions/097-boot-budget-ratchets.md` -- "Standing breach at
  adoption" marked REPAID with the measured deltas and the two refutations.

**Verification.** All four boot budgets green: boot-import-weight 646/660
(headroom 14, snapshot in sync), ui-ready-census 968/970, boot-css-bytes
854943/860000, preimport-payload 488/500 + 375925/380000 + 137783/145000 LOC.
`./scripts/preflight.sh` green. `Tests/Packaging/` 211 passed. Every new guard
was mutation-tested (re-add the eager import / restore the eager executor
construction; each goes red, and the construction guard is the one that catches
the import-only half-fix).

Suites run and their four pre-existing failures, each confirmed on a pristine
checkout before being dismissed:

* raw-CLI / agent-runs / console slice (18 files, 507 passed): two failures,
  reproduced with both changed files restored to `HEAD` content --
  `test_agent_runs_db.py::test_local_command_resume_projection_bounds_raw_rows_before_json_projection`
  (an SQL-shape assertion) and
  `test_console_runtime_ownership.py::test_attach_and_detach_cover_exactly_the_same_slot_set`
  (`'ChatScreen' object has no attribute '_library_activity'`).
* `Tests/Chat/test_chat_functions.py` + `Tests/ProductionApp/` (165 passed):
  two failures --
  `test_chat_root_state_removal.py::test_visible_console_stop_cancels_native_run_without_root_worker_state`
  and
  `test_retired_destination_root_state.py::test_registered_destinations_own_state_without_retired_root_mirrors`.
  Both reproduce with byte-identical assertion messages in a throwaway
  `git worktree` detached at the base commit `473e7c9298`.

Method note worth repeating: swapping `HEAD` file content in and out of the
working tree to establish "pre-existing" is fragile -- a run that outlives its
shell leaves the tree pristine, and restoring from a stale copy silently reverts
later edits (it did, twice, here). A detached `git worktree` at the base commit
is the safe form and is what settled the ProductionApp pair.
<!-- SECTION:NOTES:END -->
