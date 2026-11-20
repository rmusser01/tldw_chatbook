---
id: TASK-42144
title: "The test CSS parse cache grows without bound and makes every later gc.collect() slow"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:33
labels:
- follow-up
- tests
- performance
priority: low
references:
- Tests/UI/css_cache.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the skillsrec stream (TASK-42073, round 18). `Tests/UI/css_cache.py` keeps a session-wide `_GLOBAL_PARSE_CACHE` (line 27) that is never bounded or reset. Its key includes the stylesheet's variables (lines 63-70), so each theme adds a full parse of the app stylesheet.

TASK-42073 measured it: after the theme-contrast node's 89 themes, the process held about 10.6M gc-tracked objects (373k before it), mostly parsed CSS (516k `RuleSet`, 1.1M `Selector`, 515k `Styles`). `Tests/conftest.py::cleanup_file_descriptors` runs a full `gc.collect()` after every test under an app-mounting directory, and each collect then took about 5.5 s. With `TLDW_TEST_CSS_CACHE=0`, that node's teardown fell from 5.8 s to 0.5 s, and the next Skills tests' teardowns from 5.3 s to 0.11 s.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cache is bounded, or reset where a theme sweep ends, so that after the theme-contrast node each of the next tests' teardowns in the same process takes under 1 s (TASK-42073 measured 0.11 s with the cache off, and 5.3-5.4 s with it on)
- [ ] #2 The notes compare those teardown times with the cache on, before and after the change, in one process
- [ ] #3 Repeated parses of the same stylesheet with the same variables still hit the cache, and a test pins it
<!-- AC:END -->
