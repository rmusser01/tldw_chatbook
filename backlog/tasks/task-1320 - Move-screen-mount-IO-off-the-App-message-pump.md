---
id: TASK-1320
title: Move screen mount I/O off the App message pump
status: In Progress
assignee: []
labels:
  - performance
  - navigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Screen navigation freezes the entire app for as long as a screen's mount work
takes, because mount I/O is awaited on the App's own message pump.

`handle_screen_navigation` is an `@on` handler on the App. Textual awaits a
screen's mount inside `switch_screen`, and a widget's `on_mount` is awaited as
part of that mount, so any I/O awaited there blocks the App pump: no clicks, no
bindings, no further navigation until it finishes. This was measured directly --
during a 3-second mount the App handled 0 of 5 posted messages, and it holds for
child widgets, not just the Screen subclass.

Two acute cases were bounded in the "stop screen navigation from freezing the
whole app" change (the outgoing flush, and a 300s connect timeout). This task
covers the remaining structural cause: roughly 20 `on_mount` handlers across
`UI/` and `Widgets/` await real work -- DB reads, server fetches, audio-device
enumeration -- with `MCPWorkbench.on_mount` -> `reload()` -> server call the
worst offender, since it is reached by the destination users reported as
freezing.

The fix is to stop awaiting I/O during mount: mount immediately with a loading
state and move the fetch into a worker that populates the view when it returns.
This cannot be done by wrapping mount in a timeout -- cancelling a half-mounted
screen leaves a partially composed widget tree -- so each handler has to be
converted deliberately.

Scoped out of the original fix because `MCPWorkbench` alone has ~233 test
references, and converting it changes mount-completion timing that those tests
depend on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening a destination whose backing service is unreachable leaves the app responsive to clicks, keys and further navigation throughout
- [ ] #2 `MCPWorkbench` mounts immediately and shows a loading state while its readiness data is fetched
- [ ] #3 A regression test proves the App pump keeps handling messages while a screen's mount work is in flight
- [ ] #4 Mount-path fetch failures surface in the destination as a recoverable error, not a silent empty view
- [ ] #5 An inventory records every `on_mount` that still awaits I/O, with each either converted or explicitly justified
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every `on_mount` awaiting I/O and classify by navigation reachability
   (AC #5). Result: 10 handlers await I/O-shaped work; 4 are reachable while a
   destination mounts -- `mcp_workbench`, `personas_screen`, `study_screen`,
   `Chatbooks_Window_Improved`. The rest are modals/on-demand `push_screen`
   surfaces or have no Screen mounting them at all; `Chatbooks_Window.py` (the
   pre-"Improved" copy) has no importers and is dead.
2. Write the load-bearing regression test first (AC #3): a destination whose
   backing service is slow must not stop the App pump handling messages. Prove
   it fails against current code.
3. Convert `MCPWorkbench` (AC #2): `on_mount` becomes synchronous and schedules
   `reload()` in a worker; the canvas mounts immediately in a loading state.
   Follow the three-state precedent set by TASK-1020 (loading / empty /
   populated) so an in-flight load is never rendered as "empty".
4. Convert the remaining three navigation-reachable handlers the same way.
5. Make mount-path fetch failures resolve to a visible, recoverable error rather
   than a silent empty view (AC #4).
6. Record the inventory and the justification for each unconverted handler.

ADR required: no
Reason: no policy, framework or storage contract changes -- this moves existing
work off the message pump behind the existing worker seam.
<!-- SECTION:PLAN:END -->
