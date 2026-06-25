# Workbench Pane Focus Keyboard Convention

## Status

Accepted for `TASK-103`.

## Scope

This convention applies to destination-native workbench screens with multiple major panes, starting with Console, Notes, and Personas.

## Keys

- `F6`: move focus to the next major workbench pane.
- `Shift+F6`: move focus to the previous major workbench pane.
- `Tab`: remains local control traversal inside the current pane or active control group.
- `Ctrl+Left` and `Ctrl+Right`: reserved for text editing/navigation and must not be used for shell pane cycling.

## Focus Rules

- Pane cycling is an explicit user action, so it may leave a focused text input or text area.
- Passive screen updates, data refreshes, and mount-time work must not steal focus.
- Hidden, collapsed, unavailable, or disabled pane targets are skipped.
- Cycling wraps from the final available pane to the first available pane, and from the first available pane to the final available pane.
- Each pane owns a preferred focus target. If that target is hidden or unavailable, the next preferred target is used.

## Initial Pane Orders

Console:

1. Context rail
2. Transcript / Event Stream
3. Inspector rail
4. Composer

Notes:

1. Navigator
2. Editor
3. Inspector

Personas:

1. Library
2. Preview / Work Area
3. Inspector

## Rationale

`F6` and `Shift+F6` are standard pane/region traversal keys and avoid collisions with Textual `Input` and `TextArea` editing behavior. `Ctrl+Left` and `Ctrl+Right` remain available for word navigation in text controls, avoiding the inconsistent focus behavior that originally blocked the Personas workbench request.
