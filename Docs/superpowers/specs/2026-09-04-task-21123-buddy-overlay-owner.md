# TASK-21123: App-owned Persona Buddy presentation

The approved scope is an ownership refactor preserving the current pet-only appearance,
controls, geometry, modal behavior, persistence, and unavailable-state fences.
Base: dev 68f9d865fa. TASK-21122 and the import half of TASK-21123 have merged.

## Decision

One lazy app-owned coordinator holds the current disposable view, its screen,
generation, and reconciliation lock. It mounts the existing widget on the active
primary BaseAppScreen, preserving Textual screen-overlay rendering and input routing.
The controller remains independent of Textual and never owns a screen or widget.

Textual's screen_change_signal covers push, switch, pop, and mode changes.
BaseAppScreen posts a generic ContentsRebuilt message after its existing recompose
and mouse-capture cleanup finish. The app consumes that notification only for the
current screen. This retains recompose awareness without per-screen Buddy state,
workers, or Buddy-specific mount/resume/recompose hooks.

Controller generation changes invoke an optional notification callback whose only
production action is posting a content-free message to the app. Textual post_message
is thread-safe; no UI work runs under the controller lock. The app coalesces requests
behind one reconciliation worker and re-reads current state after await boundaries.
An entirely disabled Buddy with no view to retire starts no reconciliation worker
and imports no widget, controller, visual runtime, or PIL as a consequence of navigation.

## Lifecycle

- Primary navigation retires the previous view and mounts one view on the current screen.
- A modal leaves the underlying view suspended; its currentness predicate rejects
  interaction and resolution commits until its owning screen becomes active again.
  Dismissal resumes resolution. Modal/splash/authentication surfaces never host Buddy.
- Recompose removes the old view normally; ContentsRebuilt triggers replacement.
  Queued notifications for old screens never mount into a newer screen's ownership.
- Disable/close removes a retained view even while its screen is covered.
- Exact current screen, view object, generation, controller, and visual identity gate
  unavailable confirmation; an old view cannot hide a replacement.
- Reconciliation cancellation cleans only its own candidate. Coalescing must not
  repeatedly cancel an in-progress mount or a geometry flush.
- Shutdown closes presentation admission, drains pending geometry before controller
  shutdown, and prevents queued notifications from mounting another view.
- Workbench affordances still refresh after reconciliation, including when no view
  is desired. No default selection, visibility, rendering, or keyboard changes.

## Alternatives

Keeping all lifecycle state in BaseAppScreen leaves the duplicated ownership in place.
Reacting only to screen changes loses Buddy after recompose. Marking the widget as a
private Textual system child relies on an internal retention convention and changes
its lifecycle. The generic post-rebuild notification uses the existing teardown flow.

## Verification

Extend the current real-CSS lifecycle harness to bind the production owner.
Pin recompose replacement, navigation, modal resume, stale delayed mount/cancellation,
unavailable recovery, disabled zero-worker/import behavior, worker-thread generation
notifications, coalescing, and geometry durability across shutdown.
Retain the existing widget/UAT tests for unchanged appearance and controls.
Run targeted suites and static/derived-artifact checks; no full repository sweep.

ADR required: yes, amend ADR-074 with the presentation ownership clarification.
No schema, dependency, provider, or persistence-format change.
