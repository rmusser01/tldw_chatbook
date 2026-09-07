# Console Auto-Speak Ownership and Header Controls Design

**Date:** 2026-08-23
**Tasks:** TASK-3070.10, TASK-21201
**Status:** Approved

## Goal

Finish the Wave 6 auto-speak ownership extraction without changing speech behavior, then relocate the Console's Speak replies and Hands-free switches beside the Workbench status without sacrificing narrow-terminal or keyboard usability.

## Scope and sequencing

The work remains two atomic changes:

1. TASK-3070.10 moves auto-speak policy ownership into the existing `ConsoleHandsFreeController`. It does not change visible layout or speech behavior.
2. TASK-21201 relocates the two switches and reclaims the former always-present speech row. It depends on TASK-3070.10 and does not change destination, queue, retry, resume, or speech policy.

The changes should be reviewed and merged independently. This keeps the controller extraction compliant with the Wave 6 no-passenger rule and makes any visual regression easy to isolate.

## Ownership design

`ChatScreen` retains the three decorated Textual message entry points required by the Wave 6 contract. Each entry point stops the event and delegates to `ConsoleHandsFreeController` within the physical-line budget.

`ConsoleHandsFreeController` becomes the owner of:

- auto-speak destination resolution;
- enable, resume, and retry requests;
- coordination of auto-speak presentation state through an injected callback.

TASK-3070.10 also removes the controller's pre-existing DOM reach in
`_sync_hands_free_switch`. A second injected presentation callback mirrors the
Hands-free session state. Both speech presentation edges therefore obey the
same no-DOM controller boundary before the later widget relocation begins.

The controller must not query the Textual DOM or reach through `ChatScreen` to a sibling coordinator. `console_wiring.py` supplies named, late-bound callables for the auto-speak coordinator operations and the presentation callback. The existing coordinator continues to own persistence, queueing, retry payloads, and speech execution.

The control flow is:

```text
Console widget message
  -> bounded ChatScreen @on delegate
  -> ConsoleHandsFreeController
  -> injected ConsoleAutoSpeakCoordinator callable
  -> injected presentation callback when state changes
```

Destination lookup follows the same controller-owned path and preserves the current fallback to `None` when no resolver is available.

## Header layout design

The visible order is:

```text
Console  — subtitle…   Speak replies [switch]   Hands-free [switch]   Ready
```

`Ready` (or the current Running/Blocked state) remains the rightmost element. The speech-control group has intrinsic width and a two-cell separation from the status. The subtitle is the only flexible child, has `min-width: 0`, uses no wrapping, and ellipsizes before any fixed child can shrink or move. The supported minimum remains 60 columns.

The shared `DestinationHeader` receives one optional `before_status` widget. This is deliberately a single explicit seam, not a factory, registry, or generalized slot system. Console supplies a focused `ConsoleSpeechControls` widget through that seam. The widget preserves the existing switch IDs, names, tooltips, event semantics, and programmatic-sync guards.

The existing Retry speech and Resume auto-speak buttons stay out of the header. They remain in a recovery-only control-bar row and appear only when speech is paused. The control bar is one row during normal operation and grows to two rows only while recovery actions are visible.

`ConsoleControlBar` is the single authority for that dynamic height. Its sync
method updates the row visibility and the bar's height together. TASK-21201
removes or relaxes every competing fixed-height owner: the
`CONSOLE_CONTROL_BAR_HEIGHT` constant and constructor assignments, the framed
height supplied by `ChatScreen.compose_content`, and the CSS
`height`/`min-height`/`max-height` rules. Tests cover both 1 -> 2 and 2 -> 1
transitions so hidden buttons cannot leave a blank row behind.

## Compact height and focus mode

Compact-height mode currently hides the whole Workbench header. After relocation, compact mode keeps the one-row header visible while the normal control bar shrinks from two rows to one. The total normal vertical budget is therefore unchanged, including at 60x18.

Focus mode remains intentionally chrome-free: it hides the whole header, so status and both speech controls disappear together. No duplicate controls are mounted.

The obsolete compact status stand-in is removed once compact mode no longer hides the header. Focus mode continues to rely on its existing persistent footer status.

## Keyboard and interaction behavior

The header joins the same logical Tab/F6 region as the Console control bar and composer. Specifically, `console-workbench-header` is added to the first `CONSOLE_TAB_REGIONS` tuple and maps to `console-native-composer` in `CONSOLE_FOCUS_PANE_FOR_WIDGET`. This preserves the bounded Console keyboard tour after the switches move out of `#console-control-bar`.

Programmatic state synchronization must not emit user-intent messages. User changes remain optimistic only through the existing request path; rejected changes repaint to the authoritative state exactly as before.

## Verification

TASK-3070.10 uses strict characterization-first TDD:

- isolated no-mount controller policy tests;
- focused auto-speak coordinator and speech integration tests;
- architecture/delegate inventory checks;
- targeted Ruff and format checks;
- diagnostic inventory regeneration and comparison.

TASK-21201 uses strict geometry-first TDD with the consolidated production stylesheet:

- 60-column and representative wide header geometry;
- Ready/Running/Blocked right-edge and same-row assertions;
- subtitle-first truncation assertions;
- compact-height transcript/composer budget assertions;
- Retry/Resume recovery visibility and reachability;
- Tab/F6 navigation from both switches;
- focused switch interaction and state-sync tests;
- generated CSS bundle parity.

No local full-suite run is permitted for either task. GitHub Actions remains the broad regression gate.

## ADR decision

**ADR required:** no

**ADR path:** N/A

**Reason:** TASK-3070.10 directly implements the controller boundary already approved by the Wave 6 decomposition design. TASK-21201 is routine Console UI layout polish that preserves existing state ownership, event contracts, and application structure.
