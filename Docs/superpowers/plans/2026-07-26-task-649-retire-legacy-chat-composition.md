# Legacy Chat Composition Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the unreachable `ChatWindow`/`ChatWindowEnhanced` production composition and its exclusive support surface while preserving the registered native Console route.

**Architecture:** `ChatScreen` composes only `ConsoleSessionSurface`. A checked-in import/reachability manifest classifies every legacy import, helper, style, and test before deletion. Shared app-independent helpers remain only when a live registered destination imports them; root-wired Chat handlers are explicitly deferred to TASK-650.

**Tech Stack:** Python 3.11+, Textual screens/widgets, import/AST analysis, pytest/pytest-asyncio, modular TCSS builder.

**Backlog:** [TASK-649](../../../backlog/tasks/task-649%20-%20Retire-the-unreachable-legacy-Chat-composition.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-648

**ADR required:** yes

**ADR path:** `backlog/decisions/033-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** The existing ADRs make native Console the production Chat destination and prohibit a second root/session owner.

---

## Execution and Test Boundary

Use the verified worktree environment. New mounted checks belong in
`Tests/ProductionApp/test_chat_composition_retirement.py` and construct the
normal `TldwCli`. Do not preserve a legacy widget by mounting it in a test
application. Delete tests whose only subject is the retired composition;
retain direct tests only for app-independent helpers with a proven live
consumer.

## File Structure

- Create `Docs/superpowers/reviews/2026-07-26-task-649-legacy-chat-reachability.md`: import/reachability disposition.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: remove the legacy field, constructor, branches, delegation, and diagnostics.
- Modify `tldw_chatbook/app.py`: remove the legacy checkbox branch that imports and queries the retired composition; defer unrelated root Chat state to TASK-650.
- Modify `tldw_chatbook/Chat/attachment_core.py`: remove the false claim that the retained app-independent helper still serves the deleted legacy composition.
- Delete `tldw_chatbook/UI/Chat_Window.py` and `tldw_chatbook/UI/Chat_Window_Enhanced.py`.
- Delete or prune `tldw_chatbook/UI/Chat_Modules/*`, `tldw_chatbook/Widgets/compact_model_bar.py`, and `tldw_chatbook/Utils/chat_diagnostics.py` only according to the manifest.
- Modify `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`, `chat_events_tabs.py`, `chat_events_worldbooks.py`, and `worker_events.py` only to remove imports/branches exclusive to the deleted composition; defer root-state deletion to TASK-650.
- Modify `tldw_chatbook/css/features/_chat.tcss` and rebuild `tldw_chatbook/css/tldw_cli_modular.tcss`.
- Modify `scripts/check_persistent_diagnostic_inventory.py` and regenerate
  `Docs/security/production-diagnostic-inventory.json` after reviewing the
  deleted diagnostic owners and proving persistent sink topology is unchanged.
- Delete legacy-only tests identified by the manifest.
- Prune stale deleted-module monkeypatches from the two product-maturity UI suites without using those simplified-app suites as TASK-649 verification.
- Create `Tests/ProductionApp/test_chat_composition_retirement.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-649 and Check In the Reachability Manifest

- [x] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 649 -s "In Progress"
backlog task edit 649 --plan $'ADR required: yes\nADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs select native Console as the only production Chat composition.\n\n1. Prove legacy import/reachability.\n2. Remove ChatScreen legacy branches.\n3. Delete exclusive modules, styles, and tests.\n4. Verify native Console behavior and structural absence.'
```

- [x] Generate the source/test inventory with:

```bash
rg -n "Chat_Window|ChatWindowEnhanced|ChatWindow\\b|chat_window" tldw_chatbook Tests
rg -n "UI\\.Chat_Modules|compact_model_bar|chat_diagnostics" tldw_chatbook Tests
rg -n "Chat_Window|ChatWindowEnhanced|ChatWindow\\b" tldw_chatbook/UI/Screens tldw_chatbook/UI/Navigation
```

- [x] In the manifest, classify every hit as:
  `delete-exclusive`, `prune-exclusive-branch`, `retain-live-shared`, or
  `defer-root-wired-to-TASK-650`. For every retained module, name its live
  production importer. The manifest must prove no registered screen imports
  either deleted composition.
- [x] Treat at least these surrogate suites as deletion candidates and verify
  their sole subject before removal:
  `test_chat_window_enhanced.py`,
  `test_chat_window_enhanced_integration.py`,
  `test_chat_window_enhanced_modules.py`,
  `test_chat_window_tooltips.py`,
  `test_chat_window_tooltips_fixed.py`,
  `test_send_stop_button.py`,
  `test_ui_example_best_practices.py`, and
  `test_legacy_attach_picker.py`.
- [x] Commit the manifest before deletion:

```bash
git add Docs/superpowers/reviews/2026-07-26-task-649-legacy-chat-reachability.md
git commit -m "docs(chat): prove legacy composition reachability (task-649)"
```

## Task 2: Write the Production-Route Failure First

- [x] Add a mounted production test that navigates to the registered `ChatScreen`, asserts the real `ConsoleSessionSurface` and composer are mounted, and asserts the screen exposes no `chat_window` field or legacy `#chat-window`.
- [x] Add structural tests that reject production imports or definitions of `ChatWindow` and `ChatWindowEnhanced`, while allowing historical prose only where explicitly documented.
- [x] Run:

```bash
pytest Tests/ProductionApp/test_chat_composition_retirement.py Tests/test_application_state_ownership.py -q
```

Expected: FAIL while the legacy classes/field/branches remain.

## Task 3: Remove the Dormant Branches from ChatScreen

- [x] Remove the imports, `self.chat_window`, `_ensure_chat_window()`, and every branch whose only receiver is that field, including legacy:
  provider/model refresh, shell sidebar delegation, diagnostics, tab-container
  fallback, save/restore, attachment transfer, settings extraction, empty-state
  hiding, resume synchronization, and button delegation.
- [x] Keep native Console store/session/surface behavior and production
  `ChatScreen` route construction unchanged. Do not add `LegacyChatState`, a
  compatibility property, or an adapter.
- [x] Replace any mixed helper with its native Console branch only when that
  branch has a live caller; otherwise delete the helper.
- [x] Run the production test:

```bash
pytest Tests/ProductionApp/test_chat_composition_retirement.py -q
```

Expected: PASS.

## Task 4: Delete Exclusive Modules, Styles, and Tests

- [x] Delete the two composition modules and every manifest entry marked
  `delete-exclusive`. Prune only the marked branches in shared event modules.
- [x] Remove `ChatWindowEnhanced` selectors from
  `css/features/_chat.tcss`, then rebuild the committed bundle:

```bash
python tldw_chatbook/css/build_css.py
```

- [x] Verify the rebuilt CSS contains no `ChatWindow` selector and still
  contains native Console selectors.
- [x] Delete legacy-only surrogate suites. Do not rewrite them around another
  widget shell; native behavior is covered by the production-app test.
- [x] Re-run the manifest searches and update the checked-in disposition with
  the final zero-hit/live-shared evidence.

## Task 5: Verify and Close TASK-649

- [x] Run:

```bash
pytest Tests/ProductionApp/test_chat_composition_retirement.py Tests/ProductionApp/test_provider_selection_ownership.py Tests/test_application_state_ownership.py -q
python -m compileall -q tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Event_Handlers/Chat_Events tldw_chatbook/Event_Handlers/worker_events.py
python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py tldw_chatbook/Event_Handlers/worker_events.py Tests/ProductionApp/test_chat_composition_retirement.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/Chat/attachment_core.py tldw_chatbook/Widgets/compact_model_bar.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py tldw_chatbook/Event_Handlers/worker_events.py scripts/check_persistent_diagnostic_inventory.py Tests/ProductionApp/test_chat_composition_retirement.py Tests/test_application_state_ownership.py Tests/test_remaining_diagnostic_sentinel_matrix.py Tests/test_smoke.py
git diff --check
```

- If the manifest deletes `UI/Chat_Modules`, omit that absent path. Do not
  mass-format the verified pre-task `chat_screen.py` or
  `Chat_Events/chat_events.py` baseline exceptions; both remain linted,
  compiled, behavior-tested, and diff-checked.

- [x] Commit the retirement using the exact manifest-approved file set:

```bash
git add Docs/superpowers/reviews/2026-07-26-task-649-legacy-chat-reachability.md Docs/security/production-diagnostic-inventory.json scripts/check_persistent_diagnostic_inventory.py tldw_chatbook/app.py tldw_chatbook/Chat/attachment_core.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/compact_model_bar.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py tldw_chatbook/Event_Handlers/worker_events.py tldw_chatbook/css/features/_chat.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/ProductionApp/test_chat_composition_retirement.py Tests/test_application_state_ownership.py Tests/test_smoke.py Tests/test_remaining_diagnostic_sentinel_matrix.py Tests/UI/test_product_maturity_phase1_core_loop.py Tests/UI/test_product_maturity_phase1_empty_setup_states.py
git commit -m "refactor(chat): retire dormant legacy composition (task-649)"
```

- Before committing, use `git status --short` and add each tracked deletion
  named by the checked-in manifest explicitly with `git add -u -- <path>`.
  Never stage the whole `tldw_chatbook/UI` or `Tests` trees.

- [x] Re-read TASK-649, add Implementation Notes containing the manifest,
  actual commands, counts, durations, modified/deleted files, ADRs, and
  deviations, check all acceptance criteria, then mark Done and commit its
  task file:

```bash
backlog task 649 --plain
backlog task edit 649 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-649 - Retire-the-unreachable-legacy-Chat-composition.md'
git commit -m "docs(backlog): close legacy Chat retirement (task-649)"
```
