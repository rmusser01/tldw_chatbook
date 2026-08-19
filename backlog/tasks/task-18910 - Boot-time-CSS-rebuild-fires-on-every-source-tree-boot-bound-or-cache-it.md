---
id: TASK-18910
title: Boot-time CSS rebuild fires on every source-tree boot - bound or cache it
status: To Do
assignee: []
created_date: '2026-08-19 16:31'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured 2026-08-19 (TASK-18908 spike): _generated_css_is_stale (app.py) reports stale=True on a freshly-cloned/pulled checkout because .tcss/BUNDLED_CSS-carrying files get newer mtimes than the generated sheets after any pull touching them; every boot then runs a synchronous subprocess build_css.py BEFORE the app starts. The staleness walk itself is cheap (0.2ms warm, ~1713 files) - the cost is the rebuild subprocess (Python interpreter spawn + full parse) on every boot for source-tree users, worst on Windows (Defender scans). Wheel installs are unaffected (_is_source_tree gate).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rebuild frequency measured on a realistic pull-edit-boot cycle (how many boots actually rebuild),Fix lands: either content-hash-based staleness (no rebuild when content unchanged) or mtime-normalization after rebuild, plus rebuild moved off the startup critical path or its cost measured and accepted explicitly in the task,Boot-to-first-frame on a source tree re-measured after the fix
<!-- AC:END -->
