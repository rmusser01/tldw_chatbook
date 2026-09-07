# TASK-649 Legacy Chat Reachability Manifest

Date: 2026-07-27

Status: verified post-deletion

Task: [TASK-649](../../../backlog/tasks/task-649%20-%20Retire-the-unreachable-legacy-Chat-composition.md)

ADRs:
[ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md),
[ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md)

## Reachability conclusion

The registered `chat` route resolves only to
`tldw_chatbook.UI.Screens.chat_screen.ChatScreen`. `ChatScreen.compose_content()`
mounts `ConsoleSessionSurface`; it never calls `_ensure_chat_window()`. The
repository contains exactly one `_ensure_chat_window` occurrence: its dormant
definition. No route-registry or navigation module imports `ChatWindow` or
`ChatWindowEnhanced`.

`ChatWindow` has no production importer. Outside the exclusive
`UI/Chat_Modules` package, `ChatWindowEnhanced` has four executable production
import sites, none reachable from registered Chat composition:

1. `ChatScreen` imports it for an uncalled constructor and dormant branches.
2. `TldwCli.on_checkbox_changed()` imports it inside a legacy Chat attach-button
   preference branch.
3. `chat_events.handle_chat_send_button_pressed()` imports it inside the
   legacy enhanced-chat attachment branch.
4. `CompactModelBar` imports it lazily as a legacy sidebar-toggle fallback.

The first three sites are removed or pruned in TASK-649. `CompactModelBar`
remains live because `ConsoleControlBar` imports and mounts it; only its
`ChatWindowEnhanced` fallback is pruned. Five exclusive `UI/Chat_Modules`
handlers also carry `TYPE_CHECKING` imports; those disappear with that package.

## Inventory commands

```text
rg -n "Chat_Window|ChatWindowEnhanced|ChatWindow\b|chat_window" tldw_chatbook Tests
rg -n "UI\.Chat_Modules|compact_model_bar|chat_diagnostics" tldw_chatbook Tests
rg -n "Chat_Window|ChatWindowEnhanced|ChatWindow\b" tldw_chatbook/UI/Screens tldw_chatbook/UI/Navigation
rg -n "_ensure_chat_window" tldw_chatbook Tests
rg -n "Chat_Modules|ChatInputHandler|ChatAttachmentHandler|ChatVoiceHandler|ChatSidebarHandler|ChatMessageManager" tldw_chatbook
rg -n "compact_model_bar|CompactModelBar" tldw_chatbook
rg -n "chat_shell_bar|ChatShellBar|ChatShellContext|ChatShellLabelResolver" tldw_chatbook
rg -n "chat_diagnostics|ChatDiagnostics|diagnose_chat_screen" tldw_chatbook
```

## Production disposition

| Path or symbol | Disposition | Evidence |
| --- | --- | --- |
| `UI/Chat_Window.py` / `ChatWindow` | `delete-exclusive` | No production importer; tests construct only the retired widget. |
| `UI/Chat_Window_Enhanced.py` / `ChatWindowEnhanced` | `delete-exclusive` | The registered route never constructs it; all four external production imports are dormant legacy branches classified below. |
| `UI/Chat_Modules/` | `delete-exclusive` | All handlers and message types are imported only by `ChatWindowEnhanced`; external hits are tests or prose in the extracted live `Chat/attachment_core.py`. |
| `Widgets/Chat_Widgets/chat_shell_bar.py` | `delete-exclusive` | Its only production importer is `ChatWindowEnhanced`. |
| `Utils/chat_diagnostics.py` | `delete-exclusive` | Its only production importer and receiver is the dormant `ChatScreen.chat_window` diagnostic branch. |
| `UI/Screens/chat_screen.py` `chat_window`, `_ensure_chat_window()`, and dependent branches | `prune-exclusive-branch` | `compose_content()` mounts native `ConsoleSessionSurface`; `_ensure_chat_window()` has no caller. |
| `app.py` enhanced Chat attach-button checkbox import/query | `prune-exclusive-branch` | The branch queries the unmounted `#chat-window`; the persisted preference write can remain until root Chat handlers are removed in TASK-650. |
| `Event_Handlers/Chat_Events/chat_events.py` enhanced-window attachment lookup and `#chat-window` fallbacks | `prune-exclusive-branch` | These branches can only find the deleted composition. The root-wired handler module itself is deferred to TASK-650. |
| `Event_Handlers/Chat_Events/chat_events_tabs.py` `#chat-window` tab-container fallbacks | `prune-exclusive-branch` | Native Console owns sessions and does not expose the legacy tab container. The remaining root-wired wrapper is deferred to TASK-650. |
| `Event_Handlers/Chat_Events/chat_events_worldbooks.py` `#chat-window` fallback | `prune-exclusive-branch` | The fallback can only target the deleted composition; the handler module is deferred to TASK-650. |
| `Event_Handlers/worker_events.py` `#chat-window` tab fallbacks | `prune-exclusive-branch` | These fallbacks are part of the legacy worker bridge; root worker-state deletion is deferred to TASK-650. |
| `Widgets/compact_model_bar.py` | `retain-live-shared` plus `prune-exclusive-branch` | `ConsoleControlBar` is a live native Console importer. Remove only the lazy `ChatWindowEnhanced` ancestor fallback. |
| `Widgets/Chat_Widgets/chat_tab_container.py`, `chat_tab_bar.py`, `chat_session.py`, legacy message widgets, and `ChatScreenState` | `defer-root-wired-to-TASK-650` | `ChatScreen` and root Chat handlers still reference these legacy session/state paths; TASK-650 owns their atomic removal with root state. |
| Remaining `Event_Handlers/Chat_Events` functions and root Chat registrations | `defer-root-wired-to-TASK-650` | TASK-650 removes the legacy root reactive, worker, watcher, and handler ownership in one slice. |
| `css/features/_chat.tcss` | `prune-exclusive-branch` | Remove the `ChatWindowEnhanced` type selector and retain shared Chat message/input rules until TASK-650 resolves their remaining consumers. Rebuild the committed CSS bundle. |

## TASK-650 follow-up disposition

TASK-650 completed every `defer-root-wired-to-TASK-650` item above:

- Deleted the root-only Chat event, sidebar, resize, tab, worldbook, streaming,
  sidebar-initializer, and worker-handler modules after confirming that no
  registered production route imported them.
- Reduced `Event_Handlers/worker_events.py` to the retained synchronous
  non-Console adapter used by CCP generation and media analysis. It rejects
  streaming requests; native Console owns streaming and cancellation.
- Moved the one live CCP branch-history renderer from the deleted Chat event
  module into `conv_char_events.py`.
- Removed the stale Chat-sidebar refresh callbacks from character and prompt
  ingestion. Their live CCP refreshes remain.
- Reduced `UI/Screens/chat_screen_state.py` to `TaskResumeState`; native
  Console session, transcript, settings, and rail snapshots are serialized by
  their actual owners in `ChatScreen`.
- Kept compatibility session models importable from their retained state/model
  modules. The superseded Chat widget-session and tab-container modules remain
  deleted on current `dev`; no registered route, app handler, or snapshot path
  constructs or drives the retired UI.
- Regenerated the reviewed production-diagnostic inventory after confirming
  the rebased tree contains 403 owners, 971 TASK-492 calls, 6,017 TASK-494
  calls, and four persistent sink files.

## Test disposition

The following files directly import or construct a retired composition and are
`delete-exclusive`:

- `Tests/Event_Handlers/Chat_Events/test_chat_streaming_textual.py`
- `Tests/UI/test_chat_approvals_and_resume.py`
- `Tests/UI/test_chat_first_run_orientation.py`
- `Tests/UI/test_chat_image_attachment.py`
- `Tests/UI/test_chat_shell_bar.py`
- `Tests/UI/test_chat_window_enhanced.py`
- `Tests/UI/test_chat_window_enhanced_integration.py`
- `Tests/UI/test_chat_window_enhanced_modules.py`
- `Tests/UI/test_chat_window_tooltips.py`
- `Tests/UI/test_chat_window_tooltips_fixed.py`
- `Tests/UI/test_legacy_attach_picker.py`
- `Tests/UI/test_send_stop_button.py`
- `Tests/UI/test_ui_example_best_practices.py`
- `Tests/integration/test_chat_image_integration_real.py`
- `Tests/integration/test_chat_tabs_integration.py`
- `Tests/unit/test_chat_image_unit.py`

These suites mount `ChatWindow`, `ChatWindowEnhanced`, `ChatShellBar`, or their
exclusive handlers in simplified/test applications. They are not rewritten
around another shell. Native behavior moves to the normal production
`TldwCli` route check.

Two mixed sentinels are `prune-exclusive-branch`:

- remove `TestUIComponents.test_chat_window_creation` from
  `Tests/test_smoke.py`;
- remove the `tldw_chatbook.UI.Chat_Window_Enhanced` diagnostic matrix row
  from `Tests/test_remaining_diagnostic_sentinel_matrix.py`.
- remove only the stale deleted-module monkeypatches from
  `Tests/UI/test_product_maturity_phase1_core_loop.py` and
  `Tests/UI/test_product_maturity_phase1_empty_setup_states.py`. These suites
  were not used as TASK-649 verification because they construct simplified
  applications; the production route check is the authorized replacement.

Tests that mention a local `chat_window` variable but primarily cover
`ChatScreenState`, handoff models, or root-wired tab handlers are
`defer-root-wired-to-TASK-650`; they do not justify retaining either
composition module. Native Console suites that assert legacy chrome is absent
are `retain-live-shared`, but are not an authorized TASK-649 gate when they use
a simplified application.

TASK-650 deleted the deferred root-state, root-handler, sidebar, tab-container,
worker-bridge, handoff, and rail suites that depended on mock, harness, or
simplified applications. It also removed the obsolete root-state assertion
from the mixed MCP approval suite. Their replacements are:

- exact AST ownership and deleted-module sentinels;
- direct retained-worker function and caller-contract tests; and
- normal production `TldwCli.run_test()` checks for native rail/session
  snapshots and visible Stop-button cancellation.

## Final evidence

- `rg` reports zero production Python or TCSS hits for `Chat_Window`,
  `ChatWindow`, `ChatWindowEnhanced`, `chat_window`, `#chat-window`,
  `ChatDiagnostics`, `chat_diagnostics`, `ChatShellBar`, `chat_shell_bar`, or
  `UI.Chat_Modules`.
- The structural ownership tests confirm zero production definitions/imports
  of `ChatWindow` and `ChatWindowEnhanced`, and zero `ChatScreen.chat_window`
  or `_ensure_chat_window` syntax.
- `CompactModelBar` remains live through native Console consumers:
  `Widgets/Console/console_control_bar.py` constructs it and
  `UI/Screens/chat_screen.py` queries/synchronizes that mounted instance. Its
  legacy ancestor fallback was removed.
- `python tldw_chatbook/css/build_css.py` rebuilt
  `css/tldw_cli_modular.tcss` from the pruned source (299,477 characters);
  the bundle contains native Console selectors and no legacy Chat selector.
- The generated persistent-diagnostic inventory now contains 416 owners,
  1,006 TASK-492 calls, 6,804 TASK-494 calls, and the same five persistent
  sink files. Relative to a fresh inventory generated from branch `HEAD`,
  TASK-649 removes exactly eight owner files, four TASK-492 calls, and 213
  TASK-494 calls. Sink-call digests are unchanged. Branch `HEAD` already had
  an inherited TASK-648 snapshot drift of one net TASK-494 call plus line/digest
  movement in `app.py`, `chat_screen.py`, and `settings_screen.py`; that
  pre-existing delta is reconciled in the same generated snapshot and is not
  attributed to TASK-649.
- The normal production `TldwCli` test mounts the registered `ChatScreen`,
  verifies `ConsoleSessionSurface` plus the real composer, clicks the real
  collapse and expand actions, navigates through the registered Settings
  route, and restores the native Console draft snapshot.
- Post-deletion checkpoint:
  `Tests/ProductionApp/test_chat_composition_retirement.py` plus
  `Tests/ProductionApp/test_provider_selection_ownership.py` plus
  `Tests/test_application_state_ownership.py` passed 34 tests in 85.91 seconds.
