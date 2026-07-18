# Console Left Sidebar Usability & Legibility Redesign

## Context

The Console screen (`UI/Screens/chat_screen.py`) uses a left rail for session context, staged sources, model settings, and workspace details. The rail currently stacks four collapsible sections with minimal visual hierarchy:

- Section headers are one cell tall with no divider.
- Section bodies share the same indentation as the header.
- Status rows use a 10-column muted label and a value column, which feels cramped at the rail's minimum width.
- Model settings are compressed into two single-line statics that truncate aggressively.
- Workspace switching is hidden behind a single "Change workspace" button with no create action.

This design makes the rail hard to scan and obscures important state.

## Goals

- Improve scanability of the left rail at a glance.
- Make the four-section hierarchy (Session, Context, Model, Details) visually obvious.
- Keep the design text-only (no icons/glyphs).
- Allow long provider/model names to wrap instead of being ellipsized.
- Provide quick workspace switch/create affordances in the Session section.
- Avoid scope creep: do not build a new source-picker; reuse existing Library-to-Console handoff.

## Non-Goals

- Reorganize or remove any of the four sections.
- Add iconography or illustrations.
- Change the rail's resize/collapse behavior or persistence model.
- Build a new source-staging flow.

## Design

### 1. Rail Title

Rename the rail header from `Session & Context` to `Console context` to avoid naming the rail after one of its own sections.

### 2. Section Headers

Each `.console-rail-section-header` is restyled:

- `border-top: solid $ds-column-line`
- `height: auto; min-height: 2`
- `content-align: center middle` so the title and toggle button sit vertically centered
- Title: bold, left-aligned, `color: $ds-text-primary`
- Toggle button: bold, width 3, visible `:focus` style via `text-style: underline bold` plus `background: $ds-action-focus 30%` (Textual `outline` support is limited; use background/underline as a reliable fallback)
- The header container itself remains unfocusable; keyboard navigation uses the existing toggle button.

Implementation note: `ConsoleRailSectionHeader.__init__` currently sets inline `self.styles.height = 1; min_height = 1; max_height = 1`. Remove those inline constraints so the CSS rules above take effect.

### 3. Section Bodies

Each `.console-rail-section-body` is restyled:

- `padding: 0 1 1 1` (top right bottom left; right and bottom breathing room, no top padding because the header border already provides separation)
- No bottom border; the next section header's top border acts as the section divider, avoiding a double-rule when sections are collapsed.
- Content is indented by the body padding, so labels sit under the header.

### 4. Session Body

The Session section gets clearer workspace controls:

- A labeled status row:
  `Workspace   <active workspace name>`
- A second row below the workspace value contains two compact action buttons, aligned under the value column:
  - `[Switch]` — ID `#console-change-workspace`; opens the existing `ConsoleWorkspaceSwitcherModal`; disabled when `change_workspace_enabled` is False.
  - `[New]` — ID `#console-new-workspace`; creates a local-only workspace using `workspace_registry_service.create_workspace(name=..., description="Local workspace created from Console.")`, activates it, and runs the same post-switch sync flow used by `[Switch]` (`_sync_console_chat_core_state`, `_activate_console_session_for_workspace`, `_sync_console_workspace_context`, `run_worker(self._sync_native_console_chat_ui(), exclusive=True)`).
- The `[New]` button is disabled when `workspace_registry_service` is unavailable; the button row is still rendered so users can create a first workspace.
- On `WorkspaceRegistryServiceError` or any unexpected exception, notify the user "Workspace could not be created." and abort the activation/sync flow.
- A labeled row for the current conversation scope:
  `Scope   <conversation id / label>`
- The conversation browser sub-section keeps its own header (`#console-workspace-conversations-header`) and styles it to match the main section headers: bold title, top border, and the same height rules. No new collapse toggle is added to the browser header; existing section/group toggles inside the browser remain unchanged.

#### Shared workspace identity helper

`library_screen.py` currently contains `_next_local_workspace_identity()`. Move this helper to `Workspaces/registry_service.py` (e.g. `next_local_workspace_identity(registry_service)`) so both `LibraryScreen` and `ChatScreen` can call it. Update `LibraryScreen.create_local_workspace()` to use the shared helper.

### 5. Context Body

The Context (staged sources) tray becomes a first-class status panel:

- Header row: `Sources` label on the left, count badge on the right. The badge is a `Static` with id `#console-staged-context-count`, `text-align: right`, and `color: $ds-text-muted`. The badge value is `len(self.state.rows)` or the equivalent field on `ConsoleStagedContextState`.
- Each source renders as a `Vertical` pair with classes `.console-staged-source-name` and `.console-staged-source-status`:
  - Line 1 (`.console-staged-source-name`) shows `row.value` (the source name/identifier). Long values use `text-wrap: wrap; max-height: 2; overflow: hidden` and fall back to Python truncation with `…` if Textual cannot ellipsize wrapped text.
  - Line 2 (`.console-staged-source-status`) shows the normalized readiness state from `row.status`.
- Status normalization:
  - `ready`, `available`, `attached` → `$ds-status-ready`
  - `retrieving`, `running`, `stale` → `$ds-status-running`
  - `blocked`, `missing`, `unavailable` → `$ds-status-blocked`
  - Any other value → `$ds-text-muted`
- The readiness text itself remains readable; color is supplementary.
- Empty state: show the text "No sources attached. Stage sources from Library." and remove the existing dead `Attach` button.
- No new `[Add source]` button is added in this pass; staging continues through the existing Library → Console handoff.

### 6. Model Body

Model settings are split into horizontal labeled rows, composed directly in `chat_screen.py` instead of the current `build_console_model_section_lines()` two-line static. The existing `_sync_console_settings_summary()` method (which currently updates `#console-model-section-line1/line2`) must be updated to refresh the four new row widgets instead.

- `#console-model-section-provider` — label `Provider`, value from settings state
- `#console-model-section-model` — label `Model`, value from settings state
- `#console-model-section-temperature` — label `Temperature`, value from settings state
- `#console-model-section-max-tokens` — label `Max tokens`, value from settings state
- `#console-model-section-recovery` — shown only when model is unset/blocked

Each row is a `Horizontal` containing a muted 12-column label and a value that fills the remaining width.

- Provider and Model values use `text-wrap: wrap; max-height: 3; overflow: hidden`. The Python fallback truncates the value to the available value-column width (rail width − 12-column label − body padding − borders) multiplied by 3, then appends `…`. This fallback is applied at render time so it responds to rail resizing.
- Temperature and Max tokens remain single-line with `text-wrap: nowrap`.
- The `[Configure]` button (existing ID) is full-width with a top margin.
- When the model is unset/blocked, show `#console-model-section-recovery` in `$ds-status-blocked`, reusing the existing recovery string from the settings summary state, and place the `[Configure]` button immediately below it.

### 7. Details Body

The Details section uses the same header/body styling. Its existing rows are rendered as labeled pairs; no new widgets or data fields are added. Empty or not-configured states use `$ds-text-muted`.

### 8. Status Label Width

The fixed width for status labels is increased to `12` columns so that the longest label (`Temperature`, 11 chars) fits without clipping. The value column fills the remaining rail width but enforces a `min-width: 10` so it never collapses to a single character at the rail's minimum width.

Update the inline assignment in `ConsoleWorkspaceStatusPair.compose()` (`console_workspace_context.py:100-101`) from `10` to `12`, or move the width into the existing `.console-workspace-status-label` CSS rule and remove the inline assignment. Model-row labels use the same 12-column width.

### 9. Keyboard / Focus

- Section toggle buttons are reachable via Tab.
- Focused toggle buttons show a visible ring using `$ds-action-focus`.
- Standard Button `Space`/`Enter` behavior toggles the section.

### 10. ASCII Reference

```
┌─────────────────────────────┐
│ Console context           [<]│
├─────────────────────────────┤
│ ── Session ─────────────[−] │
│ Workspace   my-project       │
│             [Switch] [New]   │
│ Scope       conversation-12  │
│                              │
│ ── Conversations ─────────── │
│   3 conversations            │
│   [search         ] [Clear]  │
│   • Project intro            │
│                              │
│ ── Context ─────────────[−] │
│ Sources                    3 │
│   readme.md                  │
│   ready                      │
│   web-search-results         │
│   retrieving                 │
│                              │
│ ── Model ───────────────[−] │
│ Provider    llama_cpp        │
│ Model       unsloth/Llama-3.1│
│             -8B-Instruct     │
│ Temperature 0.7              │
│ Max tokens  4096             │
│ [Configure]                  │
│                              │
│ ── Details ─────────────[+] │
└─────────────────────────────┘
```

## Implementation Notes

- **Single-rule sections:** only the section header has a top border; the body has no bottom border. This avoids a double-rule when a section is collapsed and keeps the visual rhythm consistent.
- **Generated stylesheet guard:** if new classes are added (e.g. `.console-staged-source-status`), add them to `Tests/UI/test_console_persistent_rails.py` so the build fails if they are accidentally removed.
- **Compact button focus:** `.console-workspace-action` buttons are 1 cell tall. Use `outline` for the focus ring rather than `border` to avoid changing the button's layout size.

## Files to Modify

- `tldw_chatbook/UI/Screens/chat_screen.py` — compose Session/Context/Model/Details sections; wire `[Switch]` and `[New]` workspace actions.
- `tldw_chatbook/Widgets/Console/console_rail_section.py` — remove inline header height constraints; add keyboard handling if needed.
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` — render new workspace row and `[New]` button; reduce status-label width.
- `tldw_chatbook/Widgets/Console/console_staged_context.py` — render Context tray header and two-line source rows.
- `tldw_chatbook/Workspaces/registry_service.py` — add/move shared `next_local_workspace_identity()` helper.
- `tldw_chatbook/UI/Screens/library_screen.py` — adopt the shared helper.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — update rail header/body/label/toggle styles.
- `Tests/UI/test_console_*.py` and related UI tests — update selectors and add coverage.

## Testing

- Update existing UI tests that query `#console-active-workspace`, `#console-change-workspace`, `#console-staged-context-attach`, `#console-model-section-line1/line2`, and section headers.
- Add tests for:
  - `[Switch]` opens the workspace switcher modal.
  - `[New]` creates a local workspace, activates it, and refreshes the Session tray.
  - Provider/Model values wrap up to the max-height guard, then ellipsize.
  - `_sync_console_settings_summary` refreshes the new model-row widgets.
  - Section toggle button focus style is visible.
  - Section header borders render without clipping the title.
- Run the full `Tests/UI` gate before merging.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Header border clips title | Remove inline `height: 1` in `ConsoleRailSectionHeader`; use `height: auto; min-height: 2` in CSS; verify with snapshot tests. |
| Status labels clip or crowd values | Use a 12-column label width and a 10-column minimum value width. |
| Long provider/model names consume too much rail | Cap wrap to 3 lines; verify Textual ellipsis; fall back to Python truncation if needed. |
| Long source names consume too much rail | Cap wrap to 2 lines; fall back to Python truncation if needed. |
| `[New]` workspace identity logic diverges from Library | Move helper to shared service and update both call sites. |
| Existing UI tests break | Update selectors/counts; run full UI gate. |
| Color-only readiness states are inaccessible | Keep text label; color is supplementary. |

## ADR Check

Per `AGENTS.md`, long-lived UX/application-structure decisions require an Architecture Decision Record. This redesign changes the Console application's left-rail structure and introduces new cross-screen workspace-creation plumbing, so an ADR is required.

- **ADR required:** yes
- **ADR path:** `backlog/decisions/017-console-left-rail-usability.md`
- **Reason:** The change defines a persistent Console left-rail visual language and moves workspace-creation identity logic from one screen to a shared service, affecting future screen design.

## Decisions

- Text-only styling; no icons.
- Keep four sections; do not merge or rename them.
- Do not add a Console-native `[Add source]` button; reuse Library handoff.
- Model rows are horizontal label-value pairs (not stacked label-above-value).
- Conversation browser header is styled to match section headers but does not gain a new collapse toggle.
