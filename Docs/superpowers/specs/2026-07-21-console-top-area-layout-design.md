# Console top-area layout — one-line header + relocated status pills

**Date**: 2026-07-21
**Status**: Approved design, pending implementation plan
**Base**: origin/dev @ `00dc60083`
**Scope anchor**: `tldw_chatbook/UI/Screens/chat_screen.py` Console shell compose

## Why

The Console screen's top area spends four rows before the chat: a three-line
identity header (`Console` / `Chat, source handoffs, live runs, and control
actions.` / `Ready`) and a two-row control bar (status pills + action
buttons). The header's three lines are mostly whitespace, and the status
pills (Provider / Model / Assistant / RAG / Sources / Tools / Approvals) are
more useful adjacent to the composer, where the user's attention sits when
sending. This reclaims vertical space up top and puts live status next to the
input.

## Decisions (user-approved)

1. **Header** collapses to a single full-width line: bold `Console`, an
   em-dash, the subtitle filling remaining width and ellipsizing when narrow,
   and the `Ready` status badge pinned to the far right.
2. **Status pills** move to a full-width strip directly above the composer
   (between the chat area and the composer), keeping their chip styling.
3. **Action row** (New tab / Settings / Attach context / Run Library RAG /
   Save Chatbook / Help) stays at the top, directly beneath the one-line
   header. Only the pills move.

Console-only. No other screen changes.

## Current structure (what changes)

`compose_content` in `chat_screen.py` builds `#console-shell` (Vertical) as:

```
DestinationHeader #console-workbench-header      3 stacked Statics (title/subtitle/status)
… hidden compat widgets …
ConsoleControlBar #console-control-bar (h=2)      #console-control-chip-row  (7 pills)
                                                  #console-control-action-row (action buttons)
                                                  … hidden -label statics, summary line …
Horizontal #console-workspace-grid                left rail | main column (transcript) | inspector
ConsoleComposerBar #console-native-composer       (framed)
ConsoleSetupModal
```

`DestinationHeader` is shared by ~10 screens (Evals, Media, Personas, …) —
its structure must not change. `ConsoleControlBar` and
`console_workbench_state.py` are Console-only.

## Target structure

```
DestinationHeader #console-workbench-header  + class console-header-inline   (1 line)
… hidden compat widgets …
ConsoleControlBar #console-control-bar (h=1)      #console-control-action-row  (+ hidden compat)
Horizontal #console-workspace-grid                left rail | main column | inspector
ConsoleStatusChips #console-status-chips (h=1)     the 7 pills, full width
ConsoleComposerBar #console-native-composer
ConsoleSetupModal
```

## 1. One-line header (CSS-scoped, shared widget untouched)

- At the Console compose site, add `console-header-inline` to the
  `DestinationHeader` classes (`classes="workbench-header console-header-inline"`).
  No change to `DestinationHeader`'s Python or to any other screen.
- New CSS in `_agentic_terminal.tcss`, scoped to `#console-workbench-header`
  / `.console-header-inline`:
  - container: `layout: horizontal; height: 1; min-height: 1;` (overrides the
    `.workbench-header` `min-height: 2` vertical stack).
  - `.workbench-header-title`: `width: auto;` (bold already).
  - `.workbench-header-subtitle`: `width: 1fr; text-wrap: nowrap;
    text-overflow: ellipsis;` so it fills and truncates with `…`.
  - `.workbench-header-status`: `width: auto;` right-aligned (the horizontal
    container places it last; the `1fr` subtitle pushes it right). Keeps its
    `ds-status-badge` styling and the state-driven color classes that
    `sync_state`/`_sync_status_classes` manage.
- **Em-dash**: `DestinationHeader.compose` has a fixed three children, so the
  separator can't be a new element. Prepend `"— "` to the subtitle in
  `console_workbench_state.py` (Console-only state — `subtitle="— Chat,
  source handoffs, live runs, and control actions."`). The header renders
  inline in Console only, so the leading dash never appears stacked.
- Density: `.density-compact .workbench-header-subtitle` already exists; the
  inline rules must win at equal-or-higher specificity for both densities —
  scope via `#console-workbench-header.console-header-inline …` where needed.

## 2. Relocate the status pills

- **New widget** `Widgets/Console/console_status_chips.py`:
  `ConsoleStatusChips(Horizontal)`, one job — render the seven pills and
  expose `sync_state(state: ConsoleControlState)`.
  - The chip builders (`_chip`, `ConsoleChip`/`ConsoleApprovalsChip` usage)
    and the chip label + emphasis (`console-chip-dim`/`console-chip-alert`)
    sync **move here** from `ConsoleControlBar`.
  - Chip ids are **unchanged** (`#console-provider-chip` … `#console-approvals-chip`),
    so existing global `console.query_one("#console-…-chip")` lookups and
    tests keep resolving.
  - Root id `#console-status-chips`; height 1; full width.
- **`ConsoleControlBar`** keeps: `#console-control-action-row`, the hidden
  `-label` compatibility statics, and the summary line. Its chip row
  (`#console-control-chip-row`) and chip-sync code are removed.
  `CONSOLE_CONTROL_BAR_HEIGHT` 2 → 1 (used only inside the widget).
- **Compose**: header → `_compact_console_workbench_widget(ConsoleControlBar,
  height=1)` → `#console-workspace-grid` → `ConsoleStatusChips` directly
  before the composer's `_frame_console_region(ConsoleComposerBar…)`. The
  chips strip is **unframed** (height 1, full width) so it reads as flush
  against the composer's top border — matching how the pills rendered
  unframed inside the old control bar.
- **Sync**: `_sync_console_control_bar` still builds `control_state` and calls
  `control_bar.sync_state(control_state, actions=…)`; it additionally does a
  guarded `self.query_one("#console-status-chips", ConsoleStatusChips)
  .sync_state(control_state)` (same `NoMatches`-tolerant pattern used
  throughout the file). No other call sites change — every readiness update
  already routes through `_sync_console_control_bar`.

## CSS bundle

Edit `css/components/_agentic_terminal.tcss`, then regenerate
`css/tldw_cli_modular.tcss` with `build_css.py`. Never hand-edit the bundle.

## Testing

- **Unit**: `ConsoleStatusChips.sync_state` updates the seven labels and flips
  `console-chip-dim`/`console-chip-alert` from a `ConsoleControlState`
  snapshot (sources/tools/approvals active vs zero).
- **Update the coupling test**: `test_console_workbench_contract.py` (~line
  208-228) walks `control_bar.walk_children()` and asserts it contains both
  chip text ("Provider:", "Model:") and action text ("Settings", "Library
  RAG"). Split: chip text now asserted over `#console-status-chips`, action
  text over `#console-control-bar`.
- **DOM order / placement mount test**: header → action row → workspace grid →
  `#console-status-chips` → composer; assert `#console-status-chips.region.y`
  is below the workspace grid and above `#console-native-composer`.
- **Header mount test**: `#console-workbench-header` renders on a single row
  (`region.height == 1`) with the class applied; title, subtitle, and the
  `Ready` badge all present; badge sits to the right of the subtitle.
- **Regression sweep** `Tests/UI` + `Tests/Chat`, judged against the known
  dev-tip baseline (scheduling / shell-snapshot pre-existing failures + the
  missing-`pytest-mock` env gap in a fresh venv).
- **Live verification** (tmux/SVG, wide + narrow): one-line header ellipsizes
  the subtitle when narrow with `Ready` still visible at the right; the pills
  strip sits directly above the composer; the action row stays at the top.

## Out of scope

- Any change to `DestinationHeader`'s structure or to other screens' headers.
- Changing chip content, chip ids, or the action set.
- The composer, rails, transcript, or inspector layout.
