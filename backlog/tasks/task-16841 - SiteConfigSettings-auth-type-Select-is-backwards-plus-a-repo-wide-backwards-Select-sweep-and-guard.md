---
id: TASK-16841
title: 'SiteConfigSettings auth-type Select is backwards; repo-wide backwards-Select sweep and guard'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-15991 review (PR #1701) found a sixth instance of the backwards-`Select` bug
class, still live at dev `ee741cf10`: `UI/SiteConfigSettings.py:241-249` composes
`#auth-type-select` as `("none", "None"), ("basic", "Basic Auth"), ...` — `(value, label)`
order, backwards against Textual's `(label, value)` contract. Its consumer
`display_config` sets `auth_select.value = config.auth_type or "none"`, which raises
`InvalidSelectValueError` (the Select's real values are the display labels) — swallowed
by the bare `try/except Exception: logger.error(...)` around the
`call_from_thread(self.display_config, config)` in `load_site_config`, so selecting a
site with a saved config silently truncates the config-display refresh. Severity is
capped only because `SiteConfigSettings` is itself nav-unreachable (same as
`ScraperBuilderWindow`, task-15991).

This is the bug class TASK-15772 (PR #1691, six sites across `UI/STTS_Window.py` +
`Widgets/TTS/`) and TASK-15991 (two sites in `ScraperBuilderWindow.py`) fixed piecemeal —
**three file families, always found by review rather than by tooling, always paired with
a broad `except` that swallows the crash**. Two of the found sites hid from a plain
`grep "Select("` sweep because the options arrive later via `.set_options()`. Fix the
auth-type Select, then do the systematic version: an AST-based sweep for
`(id_string, "Display Text")`-shaped option tuples covering `Select(options=...)`,
`set_options(...)`, and option-list constructions, classify every hit, and land a
permanent guard (Tests/Architecture — none exists for this class today) so the seventh
instance cannot ship.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `#auth-type-select` composes `(label, value)` and `display_config` restores a saved auth type without raising (born-red test)
- [ ] #2 A repo-wide sweep (AST-level, covering deferred `.set_options()` population) classifies every Select option construction; every backwards site found is fixed or justified in the notes
- [ ] #3 A permanent guard fails on a reintroduced backwards `(value, label)` options list (proven by temporary reintroduction)
<!-- AC:END -->
