# Collapsible Console Composer Design

Date: 2026-07-22
Status: User-approved

## Summary

Add a manual collapse control to the Console composer so a user reading a long
message can reclaim vertical transcript space without losing unsent work. The
expanded composer remains the existing `ConsoleComposerBar`; collapsed mode is
a one-row `Composer hidden` status bar with a distinct `Expand ▴` button.

The composer stays mounted in both modes. Draft content, paste segments,
attachments, and editing state therefore survive collapse and expansion in the
same widget instance. Collapse state is transient and Console-wide: it is shared
across conversation tabs and retained while navigating within the running app,
but it resets to expanded after restart.

## Context

The native Console composer currently occupies five to eight terminal rows
below the transcript. Those rows are useful while drafting, but become unused
chrome when the user is reading a long assistant response.

The existing implementation has several contracts this feature must preserve:

- `ConsoleComposerBar` owns the canonical draft, collapsed-paste segments,
  caret, pending-attachment presentation, and Send/Stop/Attach/Save actions.
- The composer is Console's keyboard home base. The screen-level Escape action
  returns focus to it after closer widget-level Escape handlers have declined
  the key.
- F6/Shift+F6 include the composer in the workbench pane cycle.
- The cursor blink timer runs only while the composer has focus.
- Setup-incomplete Console state makes the whole workbench inert.
- The approved [Console top-area layout design](2026-07-21-console-top-area-layout-design.md)
  places a separate one-row provider/model/status strip immediately above the
  composer. That strip is runtime context, not composer chrome, and remains
  visible while the composer is collapsed.

## Goals

- Reclaim four to seven transcript rows when the composer is not in use.
- Keep collapse and expansion visible, keyboard-reachable, and reversible.
- Preserve the active draft and pending attachments in memory.
- Preserve Console focus, transcript-selection, scrolling, setup, and run-control
  contracts.
- Avoid widget remounting, storage changes, and automatic layout shifts.

## Non-goals

- Persisting composer collapse state across app restarts.
- Per-tab collapse preferences.
- Automatically collapsing after send, during streaming, or for long responses.
- Adding a dedicated collapse keyboard shortcut or configuration setting.
- Serializing transient editor details such as caret and paste-token display
  state across a full screen recompose.
- Changing the separate status-chip strip, transcript, rails, or inspector.
- Refactoring unrelated composer action or input behavior.

## User Experience

### Expanded presentation

The existing left-side `Composer:` label becomes a compact, keyboard-focusable
`Composer ▾` button. It occupies the label's existing location rather than
crowding the fixed-width Send/Attach/Save action group. The button receives a
clear tooltip such as `Collapse composer for more transcript space`.

The remaining draft surface and actions retain their current behavior and
stable selector ids.

### Collapsed presentation

Collapsed mode is exactly one terminal row:

```text
Composer hidden · Generating · Draft retained                  Stop  Expand ▴
```

The left status is assembled from state, omitting inactive parts:

1. Always start with `Composer hidden`.
2. Append ` · Generating` while a Console run is active.
3. Append ` · Draft retained` when the canonical draft contains any content.
4. Append ` · Attachment retained` when pending attachment state exists.

The status never displays draft text or attachment filenames. It is a
single-line, non-focusable Static that ellipsizes before the right-side controls
can shrink.

`Expand ▴` is always visible and keyboard-focusable. A warning-styled `Stop`
appears immediately before it only while generation is active. Stop delegates
to the existing native run-control path, does not expand the composer, and
retains the existing stale-run warning behavior.

The planned provider/model/status strip remains visible immediately above this
row. The feature does not depend on that strip having landed: before or after
the top-area work, the composer owns only its own expanded/collapsed
presentation.

### Manual state changes

Only these actions change collapse state:

- Activate `Composer ▾` to collapse.
- Activate `Expand ▴` to expand and focus the draft.
- Press Escape while collapsed to expand and focus the draft.

Sending, receiving, streaming, stopping, changing tabs or workspaces, and
navigating away from Console do not change the state.

## Architecture

### Screen-owned transient state

`ChatScreen` owns one boolean initialized to `False`, conceptually
`_console_composer_collapsed`. It is not written to app configuration,
`ConsoleRailPreferences`, the database, or a session record.

Because the state belongs to the screen instance:

- all Console tabs share the same current layout;
- navigating away and back during the app run retains the layout;
- a new app/screen instance starts expanded;
- a fallback screen recompose can initialize the replacement composer in the
  same presentation.

### One stable composer widget

`ConsoleComposerBar` remains mounted and gains two always-composed child
containers:

- an expanded container containing the existing draft and action children;
- a collapsed container containing status, contextual Stop, and Expand.

New stable selectors:

- `#console-composer-expanded`
- `#console-composer-collapse`
- `#console-composer-collapsed`
- `#console-composer-collapsed-status`
- `#console-collapsed-stop-generation`
- `#console-composer-expand`

Existing child ids such as `#console-command-visible-text`,
`#console-send-message`, `#console-stop-generation`, and
`#console-attach-context` remain unchanged.

The widget accepts an initial `collapsed: bool = False` value, exposes an
idempotent `set_collapsed(collapsed: bool)` method, and provides a read-only
collapsed-state property. `ChatScreen.compose_content` passes the screen-owned
value into the constructor so a fallback recompose starts in the correct
presentation. The method:

- toggles the two presentation containers without mounting or removing them;
- adds/removes a stable collapsed-state CSS class;
- makes the composer root non-focusable while collapsed and focusable while
  expanded;
- applies collapsed or expanded geometry;
- synchronizes the collapsed status and contextual Stop control;
- updates cursor-timer state.

The new buttons use Textual's existing `Button.Pressed` messages.
`ChatScreen` handles their stable ids directly; no custom message classes or
parallel state model are needed.

### Geometry

Expanded sizing continues to use the current bounded one-to-four draft rows and
five-to-eight total composer rows.

Collapsed sizing is isolated from draft-row calculations:

- height, min-height, and max-height are all exactly one;
- padding is removed;
- the multi-row round border is removed because it cannot render correctly in
  one cell;
- background and text hierarchy provide separation from the transcript/status
  strip;
- status consumes `1fr`, while Stop and Expand retain fixed readable widths.

Draft mutations, run updates, resize events, or cursor refreshes must not
restore expanded root height while collapsed. On expansion, the widget
recomputes draft width and row count from current content rather than relying
on stale geometry.

The supported compact layout contract is 100×32. Smaller terminals remain
best-effort and do not justify hiding the Expand control.

## Input and Focus Behavior

### Collapse

On `Composer ▾`:

1. Stop the event from falling through to composer draft handling.
2. Capture transcript reading state.
3. Set the screen boolean and synchronize the mounted composer.
4. After the layout refresh, restore transcript reading state and focus the
   transcript.

The transcript's selected message id is not cleared.

### Expand

On `Expand ▴`:

1. Capture transcript reading state.
2. Clear the screen boolean and synchronize the mounted composer.
3. After layout refresh, restore transcript reading state.
4. Focus the composer root/draft surface, which resumes normal caret behavior.

### Escape

The current non-priority Escape binding remains unchanged for expanded mode so
transcript selection and modal dismissal keep their existing precedence.

Collapsed mode adds a distinct priority Escape action that is enabled only
while the composer is collapsed and the setup modal is not blocking. This
guarantees the agreed one-press expand-and-focus behavior even when the
transcript has a selected message. Expanding does not clear that selection.

### F6 pane cycling

The composer remains in `CONSOLE_FOCUS_REGISTRY`. Both
`_ensure_console_workbench_targets_focusable` and
`_focus_console_workbench_target` must use the same state-aware target resolver
for the composer pane:

- expanded mode returns only `#console-native-composer`;
- collapsed mode returns only `#console-composer-expand`.

This state-aware resolver is required because the current generic focusability
pass sets every displayed target's `can_focus` flag to `True`. Merely placing
Expand before the root in the static target tuple would still re-enable the
collapsed root and put a hidden draft surface back into Tab order.

### Hidden input guard

`ChatScreen._should_capture_console_input` must return `False` whenever the
composer is collapsed. It must also exempt the two toggle buttons when they are
focused so Enter activates the button instead of invoking the composer's
screen-level Send behavior.

Printable input, editing keys, Enter, and paste/drop events therefore never
alter or send a hidden draft. Transcript shortcuts keep their current meaning.

### Deferred-focus race guard

Collapse and expand update state synchronously, but focus must wait for Textual
layout. Every deferred focus callback carries or closes over its expected
collapsed value and rechecks the current screen state before acting. Rapid
Collapse → Escape/Expand cannot allow a stale collapse callback to move focus
back to the transcript.

## Transcript Reading Position

Viewport height changes make literal scroll-coordinate promises unreliable.
The feature preserves semantic reading position:

- If `ConsoleTranscript` is anchored at the tail, it stays anchored after
  collapse or expansion.
- If the reader released the anchor and scrolled upward, capture the current
  `scroll_y`, keep the anchor released, and restore that offset after layout,
  clamped only to the new valid range.
- Preserve `selected_message_id` throughout the transition.

This builds on the existing transcript tail-follow contract rather than adding
a second scroll model.

## State Preservation Boundaries

Within the same mounted composer instance, collapse/expand preserves:

- canonical draft text and segment boundaries;
- collapsed/expanded paste-token presentation;
- caret offset and draft selection;
- pending attachment state and label;
- current Send/Stop/Attach/Save action state.

A pending two-click `Unfurl?` confirmation resets on collapse, matching existing
click-away safety. The underlying paste segment and canonical content remain
unchanged, preventing an unexpectedly armed large-paste expansion later.

Tab switching continues to use the existing `ConsoleChatStore` behavior:

- the outgoing tab's canonical draft is saved to its session;
- the incoming tab's draft and pending attachments become active;
- the collapsed status updates to describe the active tab;
- existing caret placement on a reloaded tab is unchanged.

A full `ChatScreen` recompose is a remount boundary. The screen-owned collapsed
boolean and existing canonical session-draft recovery survive it, but transient
caret, selection, and paste-token display state are not newly serialized. The
collapse action itself never triggers a recompose.

## Run and Setup Behavior

The existing action-state synchronization also updates the collapsed status and
collapsed Stop visibility. Both expanded and collapsed Stop buttons delegate to
the same `_stop_console_generation_from_visible_action` path.

If the run completes between render and activation, the existing
`No active Console run to stop` warning is shown and the composer remains
collapsed.

The Console setup modal remains authoritative:

- collapse, expand, collapsed-only Escape, Stop, and hidden input are inert
  while it blocks the workbench;
- if setup becomes blocked while the composer is collapsed, the transient
  layout state is retained behind the modal;
- after setup resolves, workbench focus restoration targets the visible Expand
  button rather than the hidden composer root.

## Error Handling

- `set_collapsed` is idempotent, so duplicate button messages are harmless.
- Widget lookups used during layout/focus synchronization tolerate an unmounted
  or fallback-compose state using existing guarded query patterns.
- A missing transcript focus target leaves focus on the nearest visible
  workbench target instead of remounting the Console.
- Deferred callbacks verify expected state and become no-ops when stale.
- Collapsed status derivation uses presence booleans only; invalid or missing
  labels cannot leak into its text.
- No collapse failure discards or clears a draft or attachment.

## Testing

### Widget and state tests

- Expanded is the default presentation.
- `set_collapsed(True/False)` is idempotent and toggles only presentation.
- Collapsed root geometry is exactly one row; expansion restores bounded
  content-driven height.
- Draft, segment state, caret, selection, and pending attachments survive a
  collapse/expand round trip.
- `Unfurl?` confirmation resets without changing canonical pasted content.
- Status copy covers no retained work, draft, attachment, draft plus
  attachment, generating, and generating plus retained-work combinations.
- Contextual Stop appears only for active runs.
- Cursor timer remains paused while collapsed regardless of focus-event order,
  and resumes after expansion plus composer focus.
- Draft refresh and resize paths do not perform hidden draft rendering or
  change collapsed height.

### Mounted interaction tests

- Mouse and Tab/Enter can activate both toggle controls.
- Enter on either toggle never sends the draft.
- Collapsing moves focus to the transcript and preserves message selection.
- F6 targets Expand while collapsed and the composer root while expanded.
- The generic workbench focusability pass never re-enables the collapsed
  composer root or adds it to Tab order.
- One Escape expands and focuses even with an active transcript selection;
  expanded-mode Escape precedence remains unchanged.
- Printable keys, editing keys, Enter, paste, and dropped paths do not reach a
  collapsed composer.
- A rapid Collapse → Expand sequence ignores stale deferred focus work.
- An anchored transcript remains at the tail through both transitions.
- A manually scrolled transcript retains its reading offset, subject only to
  valid-range clamping.
- Collapsed Stop cancels an active run without expanding; stale Stop uses the
  existing warning.
- Tab and workspace changes retain the Console-wide collapsed layout while the
  status follows the active tab's draft/attachment state.
- Navigating away and back retains collapse; a new app instance starts
  expanded.
- Setup blocking makes all composer controls inert, and setup resolution while
  collapsed restores focus to Expand.
- Fallback recompose restores collapsed presentation and canonical session
  draft without claiming preservation of transient editor internals.

### Layout and regression verification

- Mounted geometry at 140×42 and the supported compact 100×32 size.
- In collapsed-with-draft and collapsed-while-generating states, status
  ellipsizes before Stop or Expand can clip.
- If the top-area status-strip work has landed, verify it remains visible and
  directly above the collapsed composer; otherwise verify the composer has no
  dependency on that widget.
- Update `css/components/_agentic_terminal.tcss`, regenerate
  `css/tldw_cli_modular.tcss` with the project CSS build script, and verify the
  generated bundle rather than editing it manually.
- Run focused composer, workbench-focus, transcript-tail-follow, and native
  Console flow suites, followed by the broader Console/UI regression set.
- Capture live Textual-web evidence at 140×42 and 100×32 for expanded,
  collapsed-with-draft, and collapsed-while-generating states.

## ADR Check

ADR required: no
ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`
Reason: This is a focused Console presentation feature that directly follows
ADR-011's existing stable-compose-tree, explicit widget-message, and
screen-owned orchestration boundaries. It introduces no storage, schema,
service, security, dependency, or cross-module contract decision.

## References

- [ADR-011: Chatbook Workbench UI System](../../../backlog/decisions/011-chatbook-workbench-ui-system.md)
- [Console dual-audience UX design](2026-07-02-console-dual-audience-ux-design.md)
- [Console top-area layout design](2026-07-21-console-top-area-layout-design.md)
