# TASK-649 Legacy Chat Reachability Manifest

Date: 2026-07-27

Status: pre-deletion inventory

Task: [TASK-649](../../../backlog/tasks/task-649%20-%20Retire-the-unreachable-legacy-Chat-composition.md)

ADRs:
[ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md),
[ADR-026](../../../backlog/decisions/026-application-session-state-ownership.md)

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

Tests that mention a local `chat_window` variable but primarily cover
`ChatScreenState`, handoff models, or root-wired tab handlers are
`defer-root-wired-to-TASK-650`; they do not justify retaining either
composition module. Native Console suites that assert legacy chrome is absent
are `retain-live-shared`, but are not an authorized TASK-649 gate when they use
a simplified application.

## Final evidence

To be updated after deletion with:

- zero production definitions/imports of `ChatWindow` and
  `ChatWindowEnhanced`;
- zero `ChatScreen.chat_window` or `_ensure_chat_window` branches;
- the retained live importer list for `CompactModelBar`;
- rebuilt CSS evidence;
- normal production `TldwCli` route, action, and snapshot results.
