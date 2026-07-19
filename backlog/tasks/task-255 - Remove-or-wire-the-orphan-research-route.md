---
id: TASK-255
title: Remove or wire the orphan research route
status: Done
assignee: []
created_date: '2026-07-12 14:12'
labels:
  - cleanup
  - ui
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The research route is registered in UI/Navigation/screen_registry.py:90 and resolves to ResearchScreen, but no shell destination, legacy-route mapping or navigation call ever targets it. A registered screen with no entry point is dead weight and confuses route audits. Either give it a real navigation entry or remove the registration. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The research route is reachable via shell navigation or its registration is removed
- [x] #2 Route inventory and screen registry are consistent with each other
<!-- AC:END -->

## Implementation Plan

1. Verify no shell destination, legacy-route mapping, or navigation call targets "research" (grep NavigateToScreen("research"), TAB_RESEARCH usage, shell_destinations.py).
2. Remove the "research" ScreenRoute registration from UI/Navigation/screen_registry.py.
3. Add a "research" -> "library" compatibility alias in _SCREEN_ALIASES, mirroring the retired "notes"/"prompts"/"skills" routes, so the command-palette "Research" entry (TAB_RESEARCH is still in ALL_TABS) and any saved startup config resolve to Library instead of silently failing — this matches route_inventory.py's existing research -> library owner mapping (AC #2).
4. Delete UI/Screens/research_screen.py (registry was its only production consumer) and trim its two ResearchScreen tests from Tests/UI/test_research_screen.py, keeping the ResearchWindow/ResearchController tests (those modules stay).
5. Add a regression test in Tests/UI/test_screen_navigation.py asserting resolve_screen_target("research") resolves to LibraryScreen, mirroring the notes/prompts/skills tests.
6. Run registry/navigation/route test suites and import smoke test.

## Implementation Notes

Removed the orphan "research" screen registration and retired the route id to a Library compatibility alias, following the established "notes"/"prompts"/"skills" retirement pattern.

- Verification (audit confirmed on this checkout): no `NavigateToScreen("research")` call anywhere, no research entry in `UI/Navigation/shell_destinations.py`, and the only production consumer of `ResearchScreen` was the registry itself. The one indirect entry point is the command palette: `TAB_RESEARCH` is still in `ALL_TABS`, so `TabNavigationProvider.switch_tab` can post `NavigateToScreen("research")` — bare removal would have made that command (and any saved startup config using "research") dead-end in `handle_screen_navigation`'s "Unknown screen requested" branch.
- `UI/Navigation/screen_registry.py`: removed the `"research"` ScreenRoute; added `"research": "library"` to `_SCREEN_ALIASES` so the route id resolves to Library — exactly matching `UI/Workbench/route_inventory.py`'s pre-existing `research -> library` owner mapping (AC #2). Registry diff touches research lines only.
- Deleted `UI/Screens/research_screen.py` (registry was its only production consumer).
- `Tests/UI/test_research_screen.py`: dropped the two `ResearchScreen` tests + import; kept all `ResearchWindow`/`ResearchController` coverage.
- `Tests/UI/test_screen_navigation.py`: added `test_research_route_resolves_to_library_screen`, mirroring the notes/prompts/skills alias tests.
- `Tests/UI/test_tab_links_navigation.py`: `_expected_current_tab` now maps `TAB_RESEARCH -> "library"` (legacy TabLinks still renders a Research link because `TAB_RESEARCH` remains in `ALL_TABS`; clicking it lands on Library).
- Deliberately NOT touched (separate, larger decision): `UI/Research_Window.py` and `UI/Research_Modules/` are now fully orphaned from navigation (only their own tests exercise them); `TAB_RESEARCH` in `Constants.py`/`app.py` command palette stays, since removing it ripples through ALL_TABS consumers.
- Tests: `pytest Tests/UI -k "registry or navigation or route"` → 222 passed + the two touched failures fixed; remaining failures `test_master_shell_navigation_order_and_labels` and `test_master_shell_destination_order_matches_spec` verified pre-existing on clean origin/dev (fail identically with changes stashed). `import tldw_chatbook.app` smoke test OK.
