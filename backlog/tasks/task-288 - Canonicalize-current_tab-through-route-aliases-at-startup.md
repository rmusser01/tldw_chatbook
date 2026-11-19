---
id: TASK-288
title: Canonicalize current_tab through route aliases at startup
status: Done
assignee: []
created_date: '2026-07-17 14:56'
labels:
  - ui
  - navigation
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Startup can mount an aliased route's target screen while leaving app.current_tab set to the raw configured route id: _push_initial_screen sets the canonical tab, then _post_mount_setup and the deferred _set_initial_tab overwrite it with the raw _resolve_initial_shell_route() value. Affects every legacy alias (notes/prompts/ccp/tools_settings/subscription and now research from task-255) — e.g. default_tab=notes mounts LibraryScreen while current_tab stays 'notes', violating the canonical-tab invariant until the first real navigation re-canonicalizes in handle_screen_navigation. Currently low-impact because watch_current_tab early-returns under _use_screen_navigation, and the raw value is pinned by existing tests (test_returning_user_initial_route_preserves_configured_default) — fixing means deliberately changing that pinned behavior for all aliases at once. Surfaced by Qodo on PR #653 (comment 3599800204) and deferred there with evidence as a pre-existing cross-alias property.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After startup with any aliased default route current_tab equals the canonical destination of the alias and matches the mounted screen
- [x] #2 The tests pinning raw-route preservation are updated to pin canonical behavior for every legacy alias
- [x] #3 Startup with a canonical (non-aliased) route is unchanged
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

_push_initial_screen sets current_tab to the RESOLVED canonical tab (app.py:4297 'self.current_tab = resolved_tab') and applies alias-carried navigation context on boot via _LEGACY_ROUTE_LIBRARY_NAV_CONTEXT keyed on the pre-resolution route (app.py:4273-4294); the boot ordering the race exploited is now documented-guaranteed ('BOTH boot paths run _push_initial_screen() to completion first', app.py:541-546, TASK-21113).
