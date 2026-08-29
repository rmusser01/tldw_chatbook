# Console Selection Click-Target Stability Design

**Date:** 2026-08-29
**Status:** Revised after three review iterations; awaiting user approval

## Problem

A plain click on a different transcript row while the floating text-selection
menu is open can select the wrong message. The failure is deterministic in the
existing Console interaction test: the pointer presses the body of message
`m2`, the menu closes, and message `m1` remains selected.

Textual resolves each mouse event at the same screen coordinates. During
`ConsoleTranscript.on_mouse_down`, menu removal, transcript anchoring, and
selection-highlight cleanup can all change row geometry before the matching
`MouseUp` and `Click`. The later Click can therefore resolve to a different row.
When MouseDown and MouseUp resolve to different widgets, real Textual may emit
no Click at all. `pilot.click` unconditionally injects a Click and therefore
masks this second failure mode. The wrong-target failure reproduces in the
focused pilot test, and a raw App MouseDown/MouseUp probe reproduces the
missing-Click path.

## Goals

- A plain left click activates the selectable row under the initial press when
  closing the text-selection menu would otherwise move the rows.
- The floating menu and prior text highlight are dismissed without changing
  which initially pressed message a plain click activates.
- A real drag still replaces the old menu with the new selection menu without
  duplicate IDs.
- Negative-space, protected-control, right-button, and menu-interior presses
  retain their current behavior.
- Cancelling an in-progress row press with Escape leaves no menu, text
  highlight, latched row, active selection-manager state, or mouse capture.

## Non-goals

- A general mouse gesture state machine, click manager, or menu manager.
- Changing menu geometry, transcript anchoring, drag thresholds, or protected
  control semantics.
- Changing keyboard message-navigation behavior.
- Fixing unrelated baseline warnings or pytest temporary-directory cleanup
  warnings.

## Design

Classify and remember the press before dismissing selection UI:

1. Resolve `press_control` with the existing live-terminal-aware helper.
2. Resolve a selection row only for a left-button press.
3. Preserve the existing menu-ancestor guard.
4. For a selectable-row left press, retain that row in the existing
   `_selection_origin_row` field before menu/highlight cleanup can move the DOM,
   perform the existing cleanup while the previous manager selection can still
   identify its highlight, and only then begin the normal drag. Other presses
   keep their current dismissal and non-drag behavior.

The existing field already owns the active drag's origin widget. Resolve the
interaction at MouseUp, before clearing it:

- An empty drag is a plain click. `on_mouse_up` reads the origin row's immutable
  message ID, clears `_selection_origin_row`, and immediately calls the existing
  validated message toggle. It deliberately leaves the manager's
  `just_finished` flag armed, so a Click synthesized by Textual or injected by
  the pilot is swallowed by the existing row/transcript suppression guards
  instead of toggling the message a second time. If no Click is emitted, the
  action has already completed and no row latch remains.
- A non-empty drag posts `TranscriptTextSelected`. The existing asynchronous
  remount boundary uses the manager's finished `TextSelection`, clears the
  widget latch as it does today, awaits removal of every attached menu, and
  mounts the replacement.

Menu and prior-highlight cleanup remains at mouse-down, where it can still
change geometry; the latched message identity makes that geometry change
irrelevant to activation. Existing Click handlers keep their suppression-first
behavior and require no pointer-target replay helper. Keyboard callers keep
using the existing selection methods unchanged.

No new field, helper, gesture token, timeout, DOM query, or manager is
introduced.

## Failure Handling

Menu-interior presses never arm a transcript drag, latch a transcript row, or
dismiss the menu before its button Click. Protected controls and non-left
presses remain non-selectable and keep immediate outside-dismissal behavior. If
the latched row is detached by reflow or the target message is independently
removed before activation, its immutable message ID remains available and the
existing model lookup makes the toggle a safe no-op. Existing Textual errors
continue to propagate.

Escape completes the existing cancellation by releasing this transcript's mouse
capture as well as removing the menu, clearing the highlight, cancelling the
selection manager, and dropping `_selection_origin_row`. Empty MouseUp always
clears the latch whether or not a Click follows. The existing next-press handling
consumes an old suppression flag on a non-selectable press; a new selectable
cycle finishes and commits only its own origin row. Row-removal reconciliation
and non-empty MouseUp keep their existing cleanup behavior.

## Ordered Dependency

The pruning-safe remount correction in
`2026-08-29-console-selection-menu-remount-race-design.md` is a prerequisite in
the same ordered Console slice. Current code still calls
`_attached_selection_menus()` from `_text_selected`, which excludes attached
`_pruning` menus. Implement and verify the public awaited query-removal change
first; only then does a real drag safely replace either a fully attached or
already-pruning old menu.

## Verification

Use the existing deterministic regression as the primary outcome proof:

1. drag over `m1` to open a selection menu;
2. issue a normal `pilot.click` on `m2`'s body;
3. assert the menu is gone and `selected_message_id == "m2"`.

Because `pilot.click` always injects a Click, add a raw App-event regression that
forwards only MouseDown and MouseUp at fixed screen coordinates while menu
cleanup shifts the hit-test target. Assert the initially pressed message is
selected even though no Click is emitted. Follow it with another raw press cycle
on a different message and assert the old message is neither replayed nor
retained.

The ordinary `pilot.click` regression is also the exact-once control: MouseUp
commits `m2`, the injected Click must be suppressed, and the final state remains
`m2` rather than double-toggling back to no selection.

Add a focused public-API lifecycle assertion using `pilot.mouse_down`. Before
cancellation, assert the manager is active, the transcript owns mouse capture,
the origin latch still names the initially pressed message, and the old
menu/highlight cleanup has completed. Then cancel with Escape and assert the
menu, highlight, manager state, selection origin, and mouse capture are all
clear. This pins complete interruption cleanup and cannot pass without
exercising the origin row.

Add two layout-sensitive targeting regressions:

- click the same row whose text was selected and assert that row is toggled,
  with no menu or orphan highlight; and
- open a Markdown or diff selection whose highlight strip affects layout, click
  a different selectable row, and assert the initially pressed row is toggled,
  with no menu or stale strip.

Add direct branch controls for the reordered classification:

- a right-button press on a selectable row dismisses the menu immediately and
  never arms a drag; and
- a synthetic press routed directly through `ConsoleTranscript.on_mouse_down`
  with a selection-menu descendant as its control keeps the menu mounted, never
  latches a row, and never arms a drag. Direct routing is required because a
  normal screen-mounted menu press is intercepted before reaching the
  transcript and would not exercise this guard.

Retain and run the focused controls for negative-space dismissal, protected
controls, selection-menu actions, genuine drags, consecutive menu remounts, and
keyboard message selection. Add a genuine-drag replacement assertion that
exercises the completed pruning-safe remount fix rather than assuming it already
exists. The pre-change baseline is explicitly one failure in
`test_menu_open_row_body_click_dismisses_menu_and_toggles`; the other selected
Console, CSS, Evals, runtime-policy, and private-SQLite cases passed.

## Delivery and ADR Check

This is a fourth atomic corrective task. It touches the same transcript method
as the menu-remount task, so implementation must apply the remount prerequisite
first and keep both changes in one ordered Console slice while preserving
separate acceptance criteria and tests.

ADR required: no
ADR path: N/A
Reason: this corrects event ordering within the existing Console selection and
Textual ownership model; it adds no persistent state, cross-module boundary, or
long-lived interaction policy.
