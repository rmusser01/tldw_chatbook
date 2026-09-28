---
id: TASK-33075
title: Profile the slow repaint after a theme switch
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. A theme switch takes ~1.2-1.5 s to repaint with no progress cue. Cause unverified (likely the app-wide restyle). Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The repaint time after a theme switch on the Settings screen is measured and its dominant cost identified
- [x] #2 A fix or a follow-up task is recorded from the measurement
<!-- AC:END -->

## Implementation Plan

1. In-process `app.run_test(size=(211,44))` probe on a private profile: open Settings ▸ Theme, time `app.theme = X`, the picker's Try, and `use_theme(persist=True)` to the next settled refresh; wrap Textual's `refresh_css` / `Stylesheet.reparse` / `Stylesheet.update` / compositor render and this repo's picker rebuild, catalog build and config persist; cProfile a batch. Disable the test-only global CSS parse cache (`TLDW_TEST_CSS_CACHE=0`) so reparse costs match production.
2. Repeat on Library and MCP to see whether the cost follows DOM size.
3. Fix any contained repo-side cost with a test; record Textual-side costs and follow-ups in the notes.

## Implementation Notes

**Measured** (in-process `run_test(size=(211,44))`, private profile, Settings ▸ Theme, `TLDW_TEST_CSS_CACHE=0` so the test suite's global parse cache does not hide production reparse cost; host load average ~40-55, so absolute numbers run high -- read the shares). One `app.theme = X` to a settled screen: **~0.9-1.2 s** (critique measured 1.2-1.5 s live). Per switch:

| Cost | ms / switch | Owner |
|---|---|---|
| `Stylesheet.reparse` (whole ~830 KB bundle, 4,732 rules, fresh `Stylesheet` each time) | ~430 (40-60%) | Textual `App.refresh_css` |
| `Stylesheet.update` (app-wide restyle; per-property `setattr` -> `Widget.refresh` -> `_set_dirty`, ~96k calls over 4 switches) | ~260 | Textual |
| Compositor full repaint | ~150 | Textual |
| Picker catalog rebuild (`build_catalog`: ~91 colour systems regenerated) | ~45 (x2 on the picker's Try/Use) | this repo |
| `use_theme(persist=True)` config write + cache reload | ~140 (Use only) | this repo |

**Dominant cost: Textual's `refresh_css` -- the stylesheet reparse, not the DOM.** Library (116 nodes) and MCP (152) reparse exactly as long as Settings (273); only `Stylesheet.update` and the repaint scale with DOM size. Measurement trap: the test suite installs a process-global, variables-keyed parse cache (`Tests/UI/css_cache.py`), so under pytest a revisited theme reparses in ~2 ms and the reparse vanishes from any profile unless `TLDW_TEST_CSS_CACHE=0`.

**Fixed (repo side):** `theme_catalog._colours` now caches resolved colours per theme definition (keyed on `repr(theme)`, so an edited/re-registered definition resolves afresh). Warm `build_catalog` 207 ms -> 0.5 ms (probe under load); picker rebuild per switch ~44 -> ~22 ms, and the picker's Try path (two rebuilds) ~250 -> ~28 ms. Test: `Tests/Utils/test_theme_catalog.py::test_catalog_colours_resolve_once_per_theme`.

**Not fixed -- follow-ups (Textual is not patched here):** (1) a variables-keyed parse cache on the app's own `TieAwareStylesheet` (the production analogue of `Tests/UI/css_cache.py`) would make a *revisited* theme (Try -> Revert, cycling rows) skip the ~430 ms reparse; a first visit still pays it. (2) Move `use_theme`'s config persist off the UI thread (~140 ms on Use). (3) The picker's `_switch` rebuilds the catalog twice (explicit call + its own `theme_changed_signal` subscriber) -- now ~10 ms, low value. No progress cue added: the switch blocks the event loop inside `refresh_css`, so a cue could only paint if the switch were deferred a frame across every `use_theme` caller (palette too) -- not cheap.

Files: `tldw_chatbook/css/Themes/theme_catalog.py`, `Tests/Utils/test_theme_catalog.py`.
