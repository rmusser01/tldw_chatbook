---
id: TASK-14903
title: Text-selection MouseDown crash during Library relayout kills the app
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 17:20'
labels:
  - library
  - stability
dependencies: []
priority: high
---

## Description

Observed once during task-4023's live verification (2026-08-10, 170x24 terminal,
isolated profile sdd_hat3): a mouse click landing on the Search/RAG canvas's
`#library-rag-query-quiet-line` Static ~1s after a 50→24-row terminal resize
raised an unhandled exception inside Textual's text-selection machinery and
terminated the whole app:

```
File .../textual/screen.py:1914, in <text-selection begin>
    event.screen_offset - container.region.offset,
AttributeError: 'NoneType' object has no attribute 'region'
locals: container = None
        content_widget = Static(id='library-rag-query-quiet-line', classes='library-rag-quiet-line')
        event = MouseDown(x=42, y=11, button=1)
```

Not deterministically reproducible (three replay attempts, including at the
task-2 base c3bfddc0b, did not trigger it) — the window appears to be a click
dispatched while the clicked Static's ancestor chain is being replaced by a
recompose/relayout, so the selection container resolves to None. Nothing in the
Library screen's own code is on the stack; this is a Textual 8.x race the app
still owns the consequences of (an app-killing click). Worth a guarded
reproduction attempt (synthetic MouseDown into a screen mid-recompose), then
either an upstream report + pinned workaround (e.g. a defensive guard via an
App-level exception handler for this signature) or a Textual version bump that
fixes it.

## Acceptance Criteria

- [x] The crash signature is reproduced in a test or conclusively attributed with a written analysis
- [x] A click during Library recompose can no longer terminate the application (guard, upstream fix, or pinned Textual bump — whichever lands first)

## Implementation Plan

1. Read Textual 8.2.8 `screen.py` around the crash line (1914) and trace the None
   source: `Screen._forward_event` MouseDown branch resolves the clicked widget via
   `get_widget_and_offset_at` → the COMPOSITOR's cached `layers_visible` map (built at
   the last reflow). A widget pruned mid-recompose (parent = None) stays resolvable in
   the stale map until the next layout pass; for a detached widget `Widget.region`
   returns NULL_REGION (NoScreen/NoWidget swallowed), so `get_widget_and_offset_at`
   clamps into its `x < 0 or y < 0` branch and returns a NON-None offset →
   `_forward_event` takes the content path, `container = widget.parent` = None →
   `container.region` → AttributeError → propagates out of `App.on_event` → app dies.
2. RED: pilot test that reproduces the crash deterministically — mount a Static, let
   the compositor map build, `await static.remove()` WITHOUT a pause (prune done,
   reflow not yet run — the mid-recompose window), assert the stale map still resolves
   the detached widget, then drive a MouseDown through `App.on_event` (the exact
   dispatcher seam) and assert the AttributeError signature.
3. Guard: a `TextSelectionCrashGuard` mixin (new module) added to `TldwCli`'s bases,
   overriding `on_event` to wrap `super().on_event(event)` in a signature-checked
   except: AttributeError AND event is MouseDown AND the RAISING frame is Textual's
   `screen.py::_forward_event` AND that frame's `container` local is None. On match:
   log the drop via loguru (no user data), reset the screen's `_select_state` (what
   Textual's own not-selectable branch does), drop the event. Anything else re-raises.
   Rejected seams: an `on_mouse_down` handler cannot fire (the crash happens during
   forwarding, before handler dispatch); `BaseAppScreen._forward_event` misses modals
   and non-BaseAppScreen screens; `_handle_exception` is past the point of recovery.
4. GREEN: guard test (same crash state, mixin app survives + logs) plus a pin that a
   normal MouseDown on a healthy Static still creates `_select_state` (text selection
   still works through the guard).
5. Check newer Textual 8.x for an upstream fix (installed 8.2.8, pin `>=8.0.0,<9`);
   record findings; NO dependency bump in this task either way.
6. Upstream-report text into the task notes; backlog hygiene; commit.

## Implementation Notes

The crash is now reproduced DETERMINISTICALLY (not just attributed) and the app
survives it behind a signature-exact guard.

**Root cause (Textual 8.2.8, read from source).** `Screen._forward_event`'s
MouseDown branch begins text selection by resolving the clicked widget through
`get_widget_and_offset_at`, which reads the compositor's cached
`layers_visible` map — a snapshot rebuilt only at the next reflow. A widget
pruned mid-recompose (`parent is None`) stays resolvable in the stale map. For
a detached widget, `Widget.region` swallows `NoScreen`/`NoWidget` into
`NULL_REGION` (widget.py:2291-2294), which forces `get_widget_and_offset_at`
into its `x < 0 or y < 0` clamp branch (compositor path, screen.py:919-927) and
returns a **non-None** offset — so `_forward_event` takes the content path,
`container = content_widget.parent` = `None` (screen.py:1900-1904), and
screen.py:1914 dereferences it. The AttributeError propagates out of
`App.on_event` → `_handle_exception` → app terminates. The live one-shot window
("~1s after a 50→24 resize") is exactly prune-done/reflow-pending; the pilot
reproduction forces it every time: `await static.remove()` with no pause, then
a MouseDown driven through `App.on_event`.

**Fix shipped: guard (no upstream fix exists).** PyPI's latest Textual is
8.2.8 — the exact version installed — so no fixed 8.x patch release exists
inside the `>=8.0.0,<9` pin; no dependency bump was made. New mixin
`TextSelectionCrashGuard` (`tldw_chatbook/Utils/text_selection_crash_guard.py`)
wraps `App.on_event` — the one dispatcher call every forwarded input event
passes through, and the only viable seam: the crash fires during FORWARDING
(before handler dispatch, so no `on_mouse_down` can see it),
`BaseAppScreen._forward_event` would miss modals/non-BaseAppScreen screens, and
`_handle_exception` is past recovery. A caught `AttributeError` is re-raised
unless ALL of: event is `MouseDown`, raising frame is Textual's
`screen.py::_forward_event`, and that frame's `container` local is `None`. On
match it resets `screen._select_state` (Textual's own not-selectable branch),
logs the drop via loguru (widget repr + coords, no user data), and drops the
event. The predicate fails OPEN: if a future Textual renames the seam, crashes
propagate loudly again and the pinned reproduction test breaks alongside.

**Tests** (`Tests/App/test_text_selection_crash_guard.py`, 8 tests): pinned
vanilla-Textual reproduction (fails ⇒ upstream fixed ⇒ guard retirable); guard
survives the crash click + logs + app stays interactive; normal MouseDown still
starts selection through the guard (feature-not-killed pin); lookalike
AttributeError from a non-Textual frame re-raises; matcher rejects
non-MouseDown/non-AttributeError/no-traceback; TldwCli MRO wiring pin.
Mutation-checked: neutering the matcher makes the survive test fail at
screen.py:1914. Battery: Tests/App + smoke + recompose-guard = 123 passed
(includes booting the real TldwCli with the mixin under run_test).

**Modified/added files.**
- `tldw_chatbook/Utils/text_selection_crash_guard.py` (new)
- `tldw_chatbook/app.py` (mixin import + TldwCli base list)
- `Tests/App/test_text_selection_crash_guard.py` (new)

**Upstream report (ready to submit to Textualize/textual).**

> **Title:** Clicking a widget that was removed since the last reflow crashes
> the app: `AttributeError: 'NoneType' object has no attribute 'region'` in
> `Screen._forward_event` (text-selection begin)
>
> **Version:** textual 8.2.8 (latest at time of writing), Python 3.12, macOS.
>
> A `MouseDown` landing on a widget that has been removed from the DOM after
> the last reflow (e.g. mid-`recompose()`, or any remove+remount) raises an
> unhandled `AttributeError` and terminates the application:
>
> ```
> File .../textual/screen.py:1914, in _forward_event
>     event.screen_offset - container.region.offset,
> AttributeError: 'NoneType' object has no attribute 'region'
> locals: container = None
> ```
>
> Mechanism: the selection-begin block resolves the click via
> `get_widget_and_offset_at`, which reads the compositor's cached map, so the
> pruned widget is still resolvable. For a detached widget `Widget.region`
> returns `NULL_REGION` (`NoScreen`/`NoWidget` swallowed), which pushes
> `get_widget_and_offset_at` into its `x < 0 or y < 0` clamp branch and
> returns a **non-None** offset; `_forward_event` then takes the content path
> and dereferences `content_widget.parent`, which is `None`.
>
> Deterministic reproduction (pytest + run_test):
>
> ```python
> class ProbeApp(App[None]):
>     def compose(self) -> ComposeResult:
>         yield Static("The quick brown fox", id="line")
>
> async def test_crash():
>     app = ProbeApp()
>     async with app.run_test(size=(60, 10)) as pilot:
>         await pilot.pause()
>         static = app.query_one("#line", Static)
>         await static.remove()          # prune done, reflow still pending
>         assert static.parent is None
>         event = events.MouseDown(widget=None, x=3, y=0, delta_x=0,
>                                  delta_y=0, button=1, shift=False,
>                                  meta=False, ctrl=False)
>         await app.on_event(event)      # AttributeError, screen.py:1914
> ```
>
> Related: #5629 (closed) is the same removal race surfacing at a different
> line (`NoScreen` in dom.py). Note `SelectState.is_attached_to_dom()` already
> defends against widgets detaching *after* selection begins — the begin path
> just misses the already-detached case. Suggested fix: in the MouseDown
> branch, treat `content_widget.parent is None` like the not-selectable case
> (`self._select_state = None`), and/or have `get_widget_and_offset_at`
> return `(None, None)` for a widget whose `parent` is `None`.
