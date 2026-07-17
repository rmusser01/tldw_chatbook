---
id: TASK-296
title: CSS dead-rule prune (+ optional deferred second bundle) per task-262 findings
status: To Do
assignee: []
created_date: '2026-07-17 23:40'
labels: [performance, startup, css]
dependencies: [task-262]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-262 investigation (Docs/Design/2026-07-17-css-split-investigation.md) rejected a per-screen CSS split (Textual's `_load_screen_css` does a full uncached `Stylesheet.reparse()` of every loaded source at each screen's first push, charging ~36-45ms warm per screen against a −38ms/−70ms startup win; `$ds-*` variables also don't cross file boundaries). It recommended instead: (1) a dead-CSS prune — 271 of 1,792 class/id tokens (15.1%) are referenced nowhere in the repo, concentrated in `_search-rag` and the largely-orphaned legacy `_embeddings` module, worth ~1.5-3k generated lines ≈ 15-30ms parse with zero mechanics risk (edit the `css/build_css.py` source modules, rebuild, verify byte-level rule removal + visual spot QA); and (2) ONLY if the startup budget still misses after the shipped diet (285/257/258): a two-phase load — boot bundle + one vars-prefixed deferred bundle applied via a single idle reparse after first paint (~70ms cold win, needs an early-navigation force-load guard + one-time full-app visual QA). Also fold in: remove the dead `CSS_PATH` on a `Container` at `Chat_Window_Enhanced.py:72` (no-op attribute, misleading).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dead selectors removed from the css source modules; rebuilt monolith drops the corresponding rules; parse-time delta measured
- [ ] #2 No visual regressions (spot QA on the screens whose modules were pruned)
- [ ] #3 Dead Chat_Window_Enhanced CSS_PATH attribute removed
- [ ] #4 Two-phase-load decision recorded (implement only if startup budget still misses; otherwise explicitly declined in notes)
<!-- AC:END -->
