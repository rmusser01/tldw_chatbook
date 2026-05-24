# Console Persistent Rails Design

Date: 2026-05-24
Status: Approved by user and spec review, pending user review before implementation planning
Primary Repo: `tldw_chatbook`
Scope: Console screen UX, layout state, rail affordances, persistence, and QA

## Summary

Console should remain a dense terminal-native workbench, but it should not show every operational panel at full weight on first start. The approved direction is a persistent rail workbench: left context rail open by default, right inspector rail collapsed by default, and both rails user-toggleable through visible narrow in-layout handles.

The goal is to keep first-start Console calm and task-forward while preserving the full agentic control surface for users who want it. Collapsed rails remain discoverable, show concise state badges, and never auto-open.

## Current Context

Console already has a strong implementation foundation:

- `ChatScreen.compose_content()` mounts a top status strip, left context rail, central transcript, right run inspector, and bottom native composer.
- `ConsoleStagedContextTray`, `ConsoleWorkspaceContextTray`, `ConsoleSessionSurface`, `ConsoleTranscript`, `ConsoleRunInspector`, and `ConsoleComposerBar` provide Console-owned widgets.
- `console_display_state.py` owns pure display-state contracts for staged context, control labels, inspector rows, and evidence.
- The native chat core handles composer draft preservation, blocked provider recovery, streaming, stop, transcript selection, message actions, and tab management.
- Existing tests and textual-web screenshots cover the restored Console layout and native chat core.

The main UX issue is hierarchy, not missing plumbing. In the empty/default state, the left rail, transcript guidance, right inspector, source readiness rows, status strips, and composer all have similar visual weight. This makes the interface feel like a fully loaded operations cockpit before the user has active work.

## Goals

- First-start Console shows left context open, right inspector collapsed, central transcript/composer primary.
- Users can collapse or reopen either side rail through visible narrow in-layout rail handles.
- Collapsed rail state persists per workspace/session.
- Collapsed rails show concise badges for important state without auto-opening.
- Composer, transcript, provider recovery, send/stop, tab, staged context, workspace context, and inspector business behavior remain unchanged.
- `ChatScreen.compose_content()` gets less layout branching through a small Console rail/layout seam.
- Actual textual-web screenshot QA verifies first-start and collapsed/open rail states.

## Non-Goals

- Do not redesign the full Console visual system.
- Do not replace the native chat core, provider gateway, transcript, composer, staged-context tray, workspace tray, or run inspector behavior.
- Do not add mode-based Console routing such as separate Chat, Review, Run, or Sources modes.
- Do not auto-open sidebars in response to provider blocked, failed run, tool-call, approval, or staged-context states.
- Do not remove existing compatibility seams unless the implementation naturally exposes a safe direct replacement.

## Approved Direction

The selected approach is a persistent rail workbench.

Rejected alternatives:

- Cosmetic collapse only: lower risk, but it only saves space and does not fix first-start hierarchy.
- Mode-based Console: potentially powerful, but it adds conceptual load and is premature while the current workbench structure is already sound.

The persistent rail approach fits the existing implementation because the Console is already composed as distinct left, center, and right regions.

## UX Model

### First Start

For a new workspace/session:

- Left rail: open.
- Right rail: collapsed.
- Center lane: transcript/event stream primary, wider than today because the inspector is collapsed.
- Composer: full-width under the entire workbench, aligned with the outer Console grid rather than scoped to the center lane.

The left rail remains open by default because it explains staged context, workspace authority, sync state, runtime state, and conversation scope.

The right rail is collapsed by default because inspector details are not required before there is an active run, selected message, pending approval, tool call, or staged item.

### Rail Handles

Collapsed rails render as narrow in-layout handles, not floating buttons.

- Left collapsed handle label: `Context`.
- Right collapsed handle label: `Inspector`.

Each handle is focusable and clickable. Activating a handle restores its rail. The handles are part of the Console grid so they remain discoverable and stable at supported terminal sizes.

### No Auto-Open

Rails never auto-open. If a rail is collapsed and its underlying state becomes important, the handle shows a concise badge. This preserves user control and avoids layout shifts during active work.

Example badges:

- `Context - 2 staged`
- `Context - workspace`
- `Inspector - blocked`
- `Inspector - failed`
- `Inspector - 1 approval`
- `Inspector - tools`

Badge copy should prefer one short status over multiple simultaneous labels. Badge priority is rail-specific.

Left `Context` badge priority:

1. Staged evidence/context.
2. Workspace/session status.
3. Empty or no badge.

Right `Inspector` badge priority:

1. Failed or blocked run/provider state.
2. Pending approval.
3. Ready tool call.
4. Source/artifact readiness.
5. Empty or no badge.

## Layout And Visual Hierarchy

The central transcript/composer lane owns the screen.

When the right rail is collapsed:

- The transcript region expands horizontally.
- The composer remains full-width under the complete workbench and does not become a center-lane-only composer.
- The right handle stays visible at the edge.

When the left rail is collapsed:

- Staged context and workspace context are hidden.
- The `Context` handle stays visible at the edge.
- The center transcript lane gains horizontal room.

When both rails are open:

- The current workbench density remains available for power users.
- Existing minimum widths continue to guard against broken panel rendering.

Visual changes should be restrained:

- Keep the terminal-native grid.
- Make collapsed rail handles quieter than full panels.
- Reduce equal-box pressure by avoiding full panel treatment for collapsed rails.
- Preserve readable panel headings but lower passive panel dominance.
- Make the composer/focus state more obvious than passive readiness panels.

## Code Architecture

Add a small Console rail/layout seam instead of embedding all collapse branching directly in `ChatScreen.compose_content()`.

### ConsoleRailState

Pure data for current layout.

Suggested fields:

```python
@dataclass(frozen=True)
class ConsoleRailState:
    left_open: bool
    right_open: bool
    left_label: str = "Context"
    right_label: str = "Inspector"
    left_badge: str = ""
    right_badge: str = ""
    persistence_key: str = ""
```

The exact shape can adapt to existing state patterns, but rail state should remain independent of Textual widgets.

### Rail State Builder

Add a pure builder such as `build_console_rail_state(...)`.

Inputs:

- Active workspace id.
- Active Console session id or conversation id.
- Stored layout preferences.
- Current staged-context display state.
- Current inspector display state.

Responsibilities:

- Apply default state for new workspace/session: left open, right collapsed.
- Restore stored user preference for known workspace/session.
- Derive short left/right badge text from existing display state.
- Keep no-auto-open behavior explicit.

### ConsoleRailHandle

Reusable narrow handle widget for collapsed rails.

Responsibilities:

- Render label and optional badge.
- Expose stable IDs for left/right handles.
- Be focusable and keyboard-activatable.
- Emit a simple intent or use existing button press routing to toggle rail state.

### Layout Composition

Use either a `ConsoleWorkbenchLayout` widget or focused compose helpers to keep `ChatScreen.compose_content()` smaller.

Preferred split:

- Compose Console header/status strip.
- Compose left rail or left rail handle.
- Compose central transcript lane.
- Compose right inspector rail or right rail handle.
- Compose native composer.

This keeps the current business behavior in `ChatScreen` while moving layout decisions into named units.

## Persistence

Rail state persists per workspace/session.

Default key shape should be deterministic and safe, for example:

```text
console_rail_state:<workspace_id>:<session_id>
```

If an active session id is unavailable, fall back to active workspace id and a stable global Console key. If workspace id is unavailable, fall back to `global`.

Use the repo's existing app/config/state pattern if there is already a suitable workspace/session preference store. If there is not, use a minimal Console config/state section that stores only rail booleans by safe key.

Persistence must be best-effort:

- Invalid stored values fall back to defaults.
- Missing workspace/session ids do not break rendering.
- Preference write failures should not block rail toggling.

## Behavior

- First Console open for a new workspace/session shows left rail open and right rail collapsed.
- Toggle left rail hides staged/workspace panels and shows the `Context` handle.
- Toggle right rail hides inspector/source readiness panels and shows the `Inspector` handle.
- Clicking or keyboard-activating a collapsed handle restores the rail.
- A visible rail can be collapsed through a small affordance in its panel header or edge.
- Collapsed badges update from existing display state without opening the rail.
- Provider blocked, failed run, pending approval, ready tool call, staged context, and evidence states remain discoverable through collapsed badges.
- Existing composer, send, stop, tab, transcript selection, provider recovery, and Settings routing behavior remains unchanged.

## Error And Recovery States

Provider setup recovery remains visible in the central transcript lane. Collapsing the inspector must not hide the primary recovery path.

If the inspector is collapsed while provider setup is blocked, the right handle can show `Inspector - blocked`, but the central recovery strip still shows the actionable `Open Settings` path.

If an approval or tool call appears while the inspector is collapsed, the right handle badge changes, but the user chooses whether to open it.

If staged context appears while the left rail is collapsed, the left handle badge changes, but the user chooses whether to open it.

If stored rail state cannot be loaded, Console uses the first-start default and should not show an error.

## Accessibility And Keyboard

- Collapsed handles must be reachable by tab.
- Handles must have readable labels in visible text, not icon-only indicators.
- Keyboard activation should use standard Button behavior or an equivalent Textual action.
- Badge text must not rely on color alone.
- Focus state must be visible on collapsed handles.
- The existing command palette may expose rail toggle commands, but visible handles remain the primary discoverability path.

## Testing

Add tests before implementation.

Required pure tests:

- New workspace/session defaults to left open and right collapsed.
- Stored per-workspace/session state restores both rail booleans.
- Invalid stored state falls back to defaults.
- Left badge summarizes staged context.
- Right badge prioritizes blocked or failed state over lower-priority readiness.
- Badge-relevant state changes do not auto-open rails.

Required mounted Console tests:

- First-start Console renders left rail open and right handle collapsed.
- Left collapse hides staged/workspace panels and renders `Context` handle.
- Right expand restores inspector/source readiness from `Inspector` handle.
- Right collapsed handle shows a badge for provider blocked state.
- Pending approval updates the collapsed right handle badge without opening the inspector.
- Staged context updates the collapsed left handle badge without opening the context rail.
- Rail state persists per workspace/session and changes when workspace/session changes.
- Central transcript width increases when either rail is collapsed.
- Composer remains full-width under the outer Console workbench across rail states.
- Collapsed handles are reachable through keyboard focus.
- Existing provider recovery `Open Settings`, composer send/blocked draft preservation, transcript selection, and tab controls still pass.

Required screenshot QA:

- First-start: left open, right collapsed.
- Right-collapsed actionable badge.
- Both rails open workbench.
- Left collapsed, right collapsed.

Screenshots must be actual textual-web/CDP captures, not generated mockups.

## Risks And Mitigations

Risk: rail persistence could introduce state that is hard to reset.

Mitigation: keep defaults deterministic, store only booleans, and make invalid data fall back silently.

Risk: collapsed badges could become a cramped status panel.

Mitigation: allow one short badge per handle and use a strict priority order.

Risk: `ChatScreen.compose_content()` becomes more complex.

Mitigation: introduce pure rail state plus a rail handle widget and extract layout composition into named helpers or a workbench layout widget.

Risk: users miss actionable inspector states when the right rail is collapsed.

Mitigation: keep central provider recovery visible, add concise collapsed badges, and ensure command palette/keyboard rail toggles are available.

## Acceptance Criteria

- [ ] First-start Console defaults to left context rail open and right inspector rail collapsed.
- [ ] Users can collapse and reopen both rails through visible in-layout handles.
- [ ] Rail state persists per workspace/session.
- [ ] Collapsed rails show concise state badges without auto-opening.
- [ ] Central transcript lane gains horizontal room when either rail is collapsed while the composer remains full-width under the outer Console workbench.
- [ ] Existing native chat, provider recovery, transcript, tab, staged context, workspace context, and inspector behaviors continue to pass.
- [ ] Focused unit and mounted tests cover defaults, toggles, persistence, badges, and no-auto-open behavior.
- [ ] Actual textual-web screenshot QA verifies the approved rail states.
