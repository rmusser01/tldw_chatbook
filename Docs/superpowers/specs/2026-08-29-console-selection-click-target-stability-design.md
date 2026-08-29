# Console Selection Click-Target Stability Design

**Date:** 2026-08-29
**Status:** Approved in conversation; pending independent review

## Problem

A plain click on a different transcript row while the floating text-selection
menu is open can select the wrong message. The failure is deterministic in the
existing Console interaction test: the pointer presses the body of message
`m2`, the menu closes, and message `m1` remains selected.

Textual resolves each mouse event at the same screen coordinates. The current
`ConsoleTranscript.on_mouse_down` removes the screen-mounted menu before the
matching `MouseUp` and `Click`. Removing that menu refreshes layout and transcript
anchoring, so the rows can move between events and the later Click can resolve to
a different row. The behavior is not an intermittent test artifact; the focused
test reproduces it in isolation.

## Goals

- A plain left click activates the selectable row under the initial press when
  closing the text-selection menu would otherwise move the rows.
- The floating menu and text highlight are dismissed before message-selection
  state is committed.
- A real drag still replaces the old menu with the new selection menu without
  duplicate IDs.
- Negative-space, protected-control, right-button, and menu-interior presses
  retain their current behavior.
- Keyboard message selection does not leave a stale text-selection menu open.

## Non-goals

- Latching and replaying arbitrary pointer targets across unrelated application
  recomposition or streaming updates.
- A general mouse gesture state machine, click manager, or menu manager.
- Changing menu geometry, transcript anchoring, drag thresholds, or protected
  control semantics.
- Fixing unrelated baseline warnings or pytest temporary-directory cleanup
  warnings.

## Design

Classify the press before dismissing selection UI:

1. Resolve `press_control` with the existing live-terminal-aware helper.
2. Resolve a selection row only for a left-button press.
3. Preserve the existing menu-ancestor guard.
4. If the press is a selectable-row left press, keep the old menu attached and
   begin the normal drag. Otherwise, an outside press continues to dismiss the
   menu immediately.

Keeping the menu attached preserves transcript geometry through target
resolution. The interaction then follows an existing path:

- An empty drag is a plain click. After Textual has resolved that Click to the
  stable row, the shared `toggle_message_selection` boundary dismisses the text
  selection/menu and then toggles the validated message ID. Removing the menu
  inside the already-dispatched handler cannot retarget that Click.
- A non-empty drag posts `TranscriptTextSelected`. The existing asynchronous
  remount boundary awaits removal of every attached menu before mounting the
  replacement.

Putting plain-click cleanup at `toggle_message_selection` covers both row-owned
and capture-routed Click handlers without duplicate code. Its one non-pointer
caller, keyboard Enter on the current message selection, also folds an open
text-selection menu; message selection and floating text selection should not
remain active together.

No row ID, gesture token, timeout, new DOM query, or additional persistent state
is introduced.

## Failure Handling

Menu-interior presses never arm a transcript drag or dismiss the menu before its
button Click. Protected controls and non-left presses remain non-selectable and
therefore keep immediate outside-dismissal behavior. If the target message was
removed independently before activation, the existing message-ID validation
makes the toggle a safe no-op. Existing Textual errors continue to propagate.

An interrupted press can leave the ordinary drag state active, exactly as it can
today; the existing Escape, next-press, reconciliation, and unmount cancellation
paths remain authoritative. No new pending-click state requires cleanup.

## Verification

Use the existing deterministic regression as the primary outcome proof:

1. drag over `m1` to open a selection menu;
2. issue a normal `pilot.click` on `m2`'s body;
3. assert the menu is gone and `selected_message_id == "m2"`.

Add a focused public-API lifecycle assertion using `pilot.mouse_down`: a left
press on a selectable row while the menu is open keeps that menu mounted until
the interaction resolves. Complete or cancel the gesture through public pilot
events so the test leaves no capture or selection state behind. This prevents
the outcome regression from passing through an unrelated retargeting change.

Retain and run the focused controls for negative-space dismissal, protected
controls, selection-menu actions, genuine drags, consecutive menu remounts, and
keyboard message selection. The pre-change baseline is explicitly one failure
in `test_menu_open_row_body_click_dismisses_menu_and_toggles`; the other selected
Console, CSS, Evals, runtime-policy, and private-SQLite cases passed.

## Delivery and ADR Check

This is a fourth atomic corrective task. It may touch the same transcript method
as the menu-remount task, so implementation should keep the changes in one
ordered Console slice while preserving separate acceptance criteria and tests.

ADR required: no
ADR path: N/A
Reason: this corrects event ordering within the existing Console selection and
Textual ownership model; it adds no persistent state, cross-module boundary, or
long-lived interaction policy.
