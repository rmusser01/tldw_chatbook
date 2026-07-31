---
id: TASK-1564
title: 'Settings: keyboard promises match reality (footer bindings, setting-level filter)'
status: Done
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P2]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique finding (P2): the static footer advertises "t test category" on
categories that answer with "No test action is available for this Settings
category yet." (toast contradicts chrome); RAG appends extra accelerators
only in its own footer. The "/" filter matches category names+descriptions
(verified live: "console" -> 6 matches, names its Enter target) but cannot
find an individual setting, even though the Scope Inspector already knows
every owned TOML key per category.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Footer bindings reflect the active category (no advertised key that no-ops).
- [x] #2 "/" filter also indexes owned config keys / setting labels and opens the owning category.
- [x] #3 Filter status still names the Enter target for the top match.
<!-- AC:END -->

## Implementation Notes

- `_footer_shortcut_entries()` (extracted, unit-tested) now derives the hints:
  `t test category` is advertised ONLY for the six categories whose test
  action does something (PROVIDERS_MODELS, DIAGNOSTICS, STORAGE,
  PRIVACY_SECURITY, APPEARANCE, LIBRARY_RAG -- everywhere else the key
  answers with the "No test action" toast); RAG accelerators appended only on
  RAG (pre-existing); "Esc, "-prefixed while a text field owns focus
  (task-1560's honesty rule).
- The "/" filter gained rank tier 2: each category's
  `owns_config_sections` keys are searchable, so "paste_collapse_threshold"
  surfaces Console Behavior directly (live-verified: "1 match | Enter opens
  Console Behavior"). The status line still names the Enter target.
