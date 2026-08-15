# TASK-3070.4 Console Conversation Browser Consolidation Implementation Plan

> **For Codex:** Execute this plan in order with test-driven development. Do not run the full repository suite; the user explicitly limited verification to tests related to the modified browser/Workspace functionality.

**Goal:** Make `ConsoleWorkspaceController` the single owner of Console conversation-browser state and behavior while preserving the screen’s public compatibility names, Textual event entry point, rendering, resume, and collapse behavior.

**Architecture:** Move the 21 non-DOM browser methods and nine browser/cache defaults into the existing Workspace controller. Store only rich `ConsoleConversationBrowserInputRow` rows; expose legacy `ConsoleWorkspaceConversationRow` values through a lossless-enough compatibility projection with explicit defaults on writes. Keep the decorated Textual input handler on `ChatScreen` as a bounded delegate, and inject every non-Workspace dependency as a named late-bound callable rather than reaching through sibling controllers or querying the DOM.

**Tech stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, Ruff, stdlib AST checks.

**ADR required:** no
**ADR path:** N/A
**Reason:** the approved Wave 6 design already establishes the ownership and compatibility policy; this PR implements that existing decision without changing storage, provider, security, or dependency boundaries.

## Constraints and evidence rules

- Preserve the exact cancellation semantics of the query/token/timer lifecycle and stop superseded timers before replacing them.
- Do not mount Textual for new controller unit tests. Use small real data objects and injected fakes.
- Preserve browser-row object identity across controller boundaries; rebuild rows only for the documented legacy projection or an existing policy transformation.
- A RED test must fail for the intended missing ownership/projection behavior. Record any pre-existing focused failures and compare the identical command before attributing them.
- After every mutation probe, restore the production code before continuing.
- Do not broaden into unrelated Console cleanup or formatting.

## File map

- Modify `tldw_chatbook/UI/Console_Modules/workspace.py`: own canonical browser/cache state, moved browser behavior, dependency accessors, legacy row projection, and canonical refresh lifecycle.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: supply the Workspace controller’s new late-bound browser dependencies directly.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: remove moved state/method bodies, add nine read/write compatibility descriptors, and reduce the decorated search handler to a bounded delegate.
- Modify `Tests/UI/test_console_workspace_controller.py`: add no-mount controller characterization for query transitions, cancellation, rich/legacy row projections, and cache behavior; update older dual-state expectations.
- Modify `Tests/Architecture/test_console_wave6_inventory.py`: retain immutable-source arithmetic and enforce completed browser ownership, descriptors, delegate span/binding, and zero DOM access.
- Modify only if a directly affected assertion requires it: `Tests/UI/test_console_browser_search_echo.py`, `Tests/UI/test_console_rail_search_debounce.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_rail_sections.py`, `Tests/UI/test_console_controller_wiring.py`, `Tests/UI/test_console_moved_seam_guard.py`, `Tests/Workspaces/test_console_conversation_browser_state.py`, or `Tests/Workspaces/test_conversation_browser_subagents.py`.
- Modify `backlog/tasks/task-3070.4 - Consolidate-Console-conversation-browser-into-Workspace-controller.md`: close acceptance criteria and record implementation evidence only after all focused gates pass.
- Modify `Docs/security/production-diagnostic-inventory.json` only if the canonical non-write diagnostic check proves the moved logger call topology changed; regenerate once through the canonical writer, then immediately prove non-write equality.

## Task 1: Lock the canonical state and projection contract with RED tests

**Files:**
- Modify: `Tests/UI/test_console_workspace_controller.py`
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`

1. Before changing any test or production file, run and record both Task 5 focused commands exactly as written. This is the immutable pre-task baseline used for later attribution.
2. Add no-mount tests that construct a real `ConsoleWorkspaceController` with narrow fakes and assert these defaults exist before any screen compatibility read:
   - persisted cache `None`, key `None`, timestamp `0.0`;
   - query `""`, timer `None`, token `0`, rich rows `()`, total `None`, error `""`.
3. Add projection tests proving:
   - writing canonical rich rows yields legacy rows containing the same conversation id/title/status/selection;
   - writing legacy rows produces canonical rich rows with deterministic `row_key`, `conversation_id`, no native session, explicit workspace/default metadata, and the same display fields;
   - subsequent rich writes cannot change the stored tuple into the legacy runtime type.
4. Tighten the architecture test so the browser family must now be complete: all 21 M names exist only on `ConsoleWorkspaceController`, the decorated D name exists only on `ChatScreen`, its physical definition span is at most five lines excluding the decorator, all nine compatibility descriptors target `_workspace`, and every moved Workspace browser method is free of `query_one`/`query`.
5. Add two structural inventories:
   - no moved Workspace browser method or its new search-transition entry reaches sibling controller attributes such as `self._session`, `self._agent`, or `self._workspace`; every such dependency must be a named late-bound constructor callable;
   - `ChatScreen` contains no direct browser/cache-state writer outside the nine descriptor setters and the bounded decorated delegate. In particular, pin the current clear-button branch as a writer that must be routed through Workspace.
6. Run the exact RED nodes and confirm failure is caused by missing controller defaults/projections, current screen ownership, sibling reach-through, and the clear-button writer:

   ```bash
   project_python="$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)/.venv/bin/python"
   "$project_python" -B -m pytest -q \
     Tests/UI/test_console_workspace_controller.py \
     Tests/Architecture/test_console_wave6_inventory.py
   ```

7. Commit nothing yet. Preserve the baseline and RED output summaries in the task implementation notes later.

## Task 2: Move state and pure browser policies into Workspace

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

1. Add the nine canonical cache/browser defaults to `ConsoleWorkspaceController.__init__` under the newer `_console_*` names. Do not retain a second set of legacy scalar fields.
2. Add narrow constructor callables for dependencies used by the moved methods, including current store/session/conversation state, Workspace labels/records, local marks, persisted conversation listing, unseen/run-marker state, subagent-count projection, browser config/collapse preferences, timer scheduling, and final Workspace sync/focus. Prefer an existing Workspace-owned method or injected callable; do not access `self._session`, `self._agent`, `self._workspace`, or another sibling controller from the moved methods.
3. Move the pure identity/filter/star/merge/cache methods first. Replace static references to `ChatScreen` with `ConsoleWorkspaceController` or `self` so ownership is genuine.
4. Move native, membership, persisted, current-row, unseen-marker, and browser-state projection methods. Preserve row precedence, star eligibility, source metadata, run markers, queued counts, result caps, cache TTL, and error sanitization.
5. Add nine explicit read/write `ChatScreen` descriptors that target `_workspace`. Remove the corresponding `ChatScreen.__init__` assignments so descriptor writes cannot create shadow state.
6. Run the focused RED command from Task 1. Expect projection/default nodes to turn GREEN while ownership remains RED until lifecycle methods and the delegate move.

## Task 3: Collapse the duplicate search lifecycle and legacy writer

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_workspace_controller.py`

1. Move `_start_console_conversation_browser_search`, `_refresh_console_conversation_browser_search`, `_refresh_console_conversation_browser_after_selection`, and `_with_console_conversation_browser_state` into Workspace.
2. Make the canonical search lifecycle operate only on the newer canonical query/timer/token/rich-row/total/error fields.
3. Replace `_refresh_console_workspace_conversation_search` and `_refresh_console_workspace_conversation_search_after_selection` as independent writers. Route all legacy refresh consumers to the canonical rich pipeline or delete the obsolete implementation when no caller remains.
4. Replace the `console-workspace-conversation-search-clear` button branch’s direct timer/token/row writes with one Workspace-owned clear transition. That transition stops the one canonical timer, increments the one canonical token exactly once, clears query/rows/total/error, synchronizes the rail, and preserves focus behavior.
5. Implement legacy scalar aliases directly on Workspace and the legacy row projection property:
   - query/timer/token/total/error are direct aliases to canonical fields;
   - legacy rows are computed on read and converted to rich rows on write;
   - no legacy backing row tuple exists.
6. Preserve workspace-change resets, selection refresh, blank-query clearing, stale-token checks, timer stopping, local-first rendering, persisted merge, error copy, sync, and post-refresh focus.
7. Extend no-mount tests for stale token/query rejection, timer replacement, blank query, local-first/persisted-final phases, workspace reset, selection refresh, the clear-button transition’s exact single token bump, and proof that both legacy and newer names observe the same scalar state.
8. Update and run `test_console_workspace_conversation_search_clear_button_stops_pending_timer` against the real button path so it proves the branch delegates to Workspace and does not double-increment aliased tokens.
9. Run the focused Task 1 command plus the exact clear-button node to GREEN.

## Task 4: Replace the Textual handler with a bounded delegate and finish wiring

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`

1. Add one Workspace method that accepts only the plain query string and disabled boolean required by the search transition. The Textual `Input.Changed` object must not cross the controller boundary.
2. Reduce `on_console_workspace_conversation_search_changed` to a real `@on` entry point whose definition span is at most five physical lines excluding the decorator: the screen calls `event.stop()`, extracts the plain query/disabled values, and invokes the Workspace method.
3. Ensure the handler remains bound to `Input.Changed` and `#console-workspace-conversation-search`, and that disabled inputs, unchanged queries, timer cancellation, immediate in-memory filtering, and debounce scheduling remain unchanged.
4. Remove all 21 moved definitions from `ChatScreen`; update internal callers to `_workspace.<method>` or a direct late-bound callable. Do not leave one-line ordinary screen proxies for M methods.
5. Prove via AST that moved methods contain no sibling-controller reach-through and that the only remaining screen browser-state assignment nodes are the descriptor setters. Add wiring tests that replace each late-bound dependency after construction and observe the replacement at call time.
6. Run the architecture and no-mount command from Task 1. Expect all nodes GREEN.

## Task 5: Run related integration tests and mutation evidence

**Files:**
- Modify only directly affected focused tests if their old assertions encode the retired dual-state implementation.

1. Run the Workspace/browser-focused mounted and pure-state tests in checkpointed groups:

   ```bash
   project_python="$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)/.venv/bin/python"
   "$project_python" -B -m pytest -q \
     Tests/UI/test_console_browser_search_echo.py \
     Tests/UI/test_console_rail_search_debounce.py \
     Tests/UI/test_console_workspace_context_rail.py \
     Tests/UI/test_console_native_chat_flow.py

   "$project_python" -B -m pytest -q \
     Tests/UI/test_console_rail_sections.py \
     Tests/UI/test_console_controller_wiring.py \
     Tests/UI/test_console_moved_seam_guard.py \
     Tests/Workspaces/test_console_conversation_browser_state.py \
     Tests/Workspaces/test_conversation_browser_subagents.py
   ```

2. Compare any later failure to the exact pre-edit Task 1 baseline checkpoint. If no valid baseline exists for the node, create an isolated clean worktree at the recorded `origin/dev` implementation base and run the identical node there; never reverse hunks in the implementation worktree. Do not widen to the full suite.
3. Perform and restore targeted mutations, one at a time:
   - split legacy query storage from canonical query;
   - allow legacy row setter to store `ConsoleWorkspaceConversationRow` directly;
   - remove a stale-token guard;
   - omit timer stop before replacement;
   - put one moved method back on `ChatScreen` or add `query_one` to Workspace;
   - remove one descriptor setter or redirect it to screen shadow state.
4. Each mutation must make its named focused test fail for the intended reason. Restore after every probe and rerun the affected node GREEN.

## Task 6: Static, diagnostics, scope, and task closeout

**Files:**
- Modify: `backlog/tasks/task-3070.4 - Consolidate-Console-conversation-browser-into-Workspace-controller.md`
- Modify only if proven necessary: `Docs/security/production-diagnostic-inventory.json`

1. Run targeted formatting/lint without changing unrelated files:

   ```bash
   project_python="$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)/.venv/bin/python"
   "$project_python" -m ruff format --check \
     tldw_chatbook/UI/Console_Modules/workspace.py \
     tldw_chatbook/UI/Console_Modules/wiring.py \
     tldw_chatbook/UI/Screens/chat_screen.py \
     Tests/UI/test_console_workspace_controller.py \
     Tests/UI/test_console_native_chat_flow.py \
     Tests/UI/test_console_controller_wiring.py \
     Tests/Architecture/test_console_wave6_inventory.py
   "$project_python" -m ruff check \
     tldw_chatbook/UI/Console_Modules/workspace.py \
     tldw_chatbook/UI/Console_Modules/wiring.py \
     tldw_chatbook/UI/Screens/chat_screen.py \
     Tests/UI/test_console_workspace_controller.py \
     Tests/UI/test_console_native_chat_flow.py \
     Tests/UI/test_console_controller_wiring.py \
     Tests/Architecture/test_console_wave6_inventory.py
   ```

   Append every other optional Python test file actually changed under Task 5 to both exact Ruff commands; no changed Python file may be omitted from the format/lint gate.

2. Compile modified production modules into a validated temporary pycache root, remove only that exact root, and prove it absent:

   ```bash
   set -euo pipefail
   project_python="$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)/.venv/bin/python"
   cache_root="$(mktemp -d /private/tmp/task-3070-4-pycache.XXXXXX)"
   test -n "$cache_root" || exit 2
   case "$cache_root" in
     /private/tmp/task-3070-4-pycache.*) ;;
     *) exit 2 ;;
   esac
   test -d "$cache_root" || exit 2
   test ! -L "$cache_root" || exit 2
   test "$(stat -f %u "$cache_root")" -eq "$(id -u)" || exit 2
   PYTHONPYCACHEPREFIX="$cache_root" "$project_python" -m py_compile \
     tldw_chatbook/UI/Console_Modules/workspace.py \
     tldw_chatbook/UI/Console_Modules/wiring.py \
     tldw_chatbook/UI/Screens/chat_screen.py
   test -z "$(find -P "$cache_root" -type l -print -quit)" || exit 2
   rm -rf -- "$cache_root"
   test ! -e "$cache_root" || exit 2
   test ! -L "$cache_root" || exit 2
   ```

3. Run the exact production-diagnostic evidence sequence:

   ```bash
   project_python="$(cd "$(git rev-parse --git-common-dir)/.." && pwd -P)/.venv/bin/python"
   "$project_python" scripts/check_persistent_diagnostic_inventory.py
   "$project_python" -B -m pytest -q \
     Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged \
     Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
   ```

   If and only if the non-write command reports expected moved-call owner/digest drift, inspect the affected source diagnostics for fixed labels, metadata-only fields, and unchanged sink topology. Then regenerate exactly once, review the generated JSON diff, and prove equality immediately:

   ```bash
   "$project_python" scripts/check_persistent_diagnostic_inventory.py --write
   git diff -- Docs/security/production-diagnostic-inventory.json
   "$project_python" scripts/check_persistent_diagnostic_inventory.py
   "$project_python" -B -m pytest -q \
     Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged \
     Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
   ```

   Never hand-edit generated call digests and never accept unrelated diagnostic drift.
4. Run `git diff --check`, inspect the exact changed-file list, scan the diff for secrets/private paths and media/build artifacts, and confirm no unrelated files changed.
5. Self-review for controller boundary violations, duplicate state, stale call sites, exception/privacy changes, hidden DOM access, and unnecessary abstraction. The simplest acceptable result is one extended Workspace controller, one canonical row tuple, one legacy projection, and one bounded screen delegate.
6. Rerun all Task 1 and Task 5 focused commands. Do not run the full suite.
7. Update the task: check every AC, add concise Implementation Notes with RED/GREEN/mutation/static evidence and any documented deviations, then set status to Done only after all gates pass.
8. Commit the task implementation as one atomic PR branch, push, open the PR, address review comments, rebase onto the latest `origin/dev`, rerun the same focused gates, and merge only when checks and review are green.
