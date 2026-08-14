---
id: TASK-15995
title: 'Screen-sheet CSS gap for CSS_PATH-overriding test harnesses'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - tests
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Before consolidation, Textual auto-registered a pushed screen's class-level CSS in `_load_screen_css` regardless of the harness's own `CSS_PATH`. After TASK-15450, the 7 `BUNDLED_SCREEN_CSS` modals get their CSS from the app-level sheets — `Tests/UI/consolidated_css.py` carries them (fixed in the PR's m8 round), but a harness subclass that declares its OWN `CSS_PATH` overrides that list, and its comment (~:10-16) claims this 'matches what those harnesses had before', which is inaccurate: such a harness pushing one of the 7 modals now mounts it with NO CSS where it used to get the class CSS automatically. 33 test modules combine ConsolidatedCSSApp with a CSS_PATH; currently latent (only `test_library_prompts_canvas.py` also pushes a consolidated modal, and all 49 modal-adjacent tests pass) — but it is a vacuous-pass trap for the next geometry-asserting test. Fix direction: make ConsolidatedCSSApp merge the screen sheets into subclass CSS_PATH declarations (e.g. via `__init_subclass__` or a get_css_path override), and correct the comment. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A ConsolidatedCSSApp subclass with its own CSS_PATH still loads the generated screen sheets when pushing a BUNDLED_SCREEN_CSS modal
- [ ] #2 A test pins that behavior (a modal pushed under a CSS_PATH-carrying harness has its styles applied)
- [ ] #3 The inaccurate comment is corrected
<!-- AC:END -->
