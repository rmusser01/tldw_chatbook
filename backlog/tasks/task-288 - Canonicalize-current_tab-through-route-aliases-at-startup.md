---
id: TASK-288
title: Canonicalize current_tab through route aliases at startup
status: To Do
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
- [ ] #1 After startup with any aliased default route current_tab equals the canonical destination of the alias and matches the mounted screen
- [ ] #2 The tests pinning raw-route preservation are updated to pin canonical behavior for every legacy alias
- [ ] #3 Startup with a canonical (non-aliased) route is unchanged
<!-- AC:END -->
