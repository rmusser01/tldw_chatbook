# Console System Prompt Chip — Design

Date: 2026-08-03
Status: Implemented

## Goal

Give users a one-click way to view and edit the current chat's system prompt from the
Console screen, via a clickable "System Prompt" chip in the control bar, placed between
the "Model: …" chip and the "Persona: …" chip.

## Background

The backend for this feature already exists:

- `ConsoleSystemPromptModal` (`tldw_chatbook/Widgets/Console/console_system_prompt_modal.py`)
  — modal editor with TextArea, Apply / Clear / Cancel, Escape binding. Dismisses with
  `str` (new prompt), `""` (clear), or `None` (cancel).
- `ChatScreen._open_console_system_prompt_editor()` (`tldw_chatbook/UI/Screens/chat_screen.py:9497`)
  — pushes the modal and applies the result.
- `ChatScreen._apply_console_session_system_prompt()` (`chat_screen.py:9468`) →
  `ConsoleChatStore.set_session_system_prompt()` (`tldw_chatbook/Chat/console_chat_store.py:892`)
  — normalizes blank → `None`, updates session settings, persists via
  `persistence.update_conversation_system_prompt` (DB column `conversations.system_prompt`).
- Existing entry points: command palette action and the `/system` slash command.

What is missing: a visible, clickable control in the console control bar.

## Design

### Components

1. **`ConsoleControlState.system_prompt_label`** (`tldw_chatbook/Chat/console_display_state.py`)
   - New derived label field, computed in `from_values()` (`console_display_state.py:300-327`):
     `"System Prompt"` when no prompt is set, `"System Prompt: set"` when one is active.
   - `from_values()` gains a parameter carrying the prompt-set state; its caller
     `_build_console_control_state` (`chat_screen.py:3088-3103`) already fetches the
     session settings and must stop discarding them (`provider, model, _settings = ...`)
     so it can pass the prompt-set state through.

2. **`ConsoleSystemPromptChip`** (`tldw_chatbook/Widgets/Console/console_control_bar.py`)
   - Focusable, clickable chip modeled on the existing interactive `ConsoleApprovalsChip`
     (`console_control_bar.py:92-111`).
   - Nested message class named `EditRequested` (i.e. `ConsoleSystemPromptChip.EditRequested`,
     matching the `<Chip>.<Verb>Requested` convention of `ConsoleApprovalsChip.ReviewRequested`),
     posted from both the Enter/Space key bindings and `_on_click`.
   - Inserted in `ConsoleControlBar.compose()` between the model chip (line ~320) and the
     persona chip (line ~321), id `console-system-prompt-chip`.
   - Add `"#console-system-prompt-chip": state.system_prompt_label` to the hardcoded
     selector→field `label_values` dict in `ConsoleControlBar.sync_state`
     (`console_control_bar.py:187-203`) — without this entry the label never refreshes
     (the `NoMatches` guard fails silently).

3. **`ChatScreen` handler** (`tldw_chatbook/UI/Screens/chat_screen.py`)
   - One `@on(ConsoleSystemPromptChip.EditRequested)` handler on `ChatScreen` (the editor
     is a screen method, so — unlike `ConsoleApprovalsChip.ReviewRequested`, which is
     handled on the bar — this handler belongs on the screen).
   - `_open_console_system_prompt_editor()` is `async`; the handler wraps it in
     `self.run_worker(...)` like the existing entry points (`chat_screen.py:1246`, `11453`).
   - `_apply_console_session_system_prompt` currently only calls
     `_sync_console_chat_core_state()` and `_sync_console_settings_summary()`, neither of
     which reaches `_sync_console_control_bar()`. Add a control-bar sync there so the chip
     label updates immediately after Apply/Clear instead of going stale until an unrelated
     sync fires.

### Behavior

- Click chip → `ConsoleSystemPromptModal` opens showing the current system prompt.
- Apply → saves to session state + persists per conversation; used in the next LLM request.
- Clear → removes the prompt (blank → `None`).
- Cancel / Escape → no change.
- Chip label reflects state after Apply/Clear.

### Rejected alternatives

- **Workbench action button** (`.console-control-action` / `WorkbenchActionRequested`):
  different styling and location in the bar; breaks the chip grouping.
- **Widget self-contained modal**: would duplicate apply/persist logic already in
  `ChatScreen`/`ConsoleChatStore`.

## Error handling

No new failure modes: the modal and apply path already exist and are in use. The chip
handler is a thin delegate; if no conversation is active the existing editor path's
behavior applies unchanged.

## Testing

- Extend existing console display-state tests: label derivation (set vs unset).
- Extend console control-bar tests: chip renders between model and persona chips;
  click posts the expected message.
- If a handler-level test pattern exists for `ChatScreen`, assert the handler invokes
  the editor entry point.

## ADR

ADR required: **no**. No schema, sync, persistence-policy, or boundary changes — this
wires a new UI trigger to an existing, already-persisted feature.
