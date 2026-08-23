---
id: TASK-21104
title: >-
  bs4 is imported unguarded at boot but beautifulsoup4 is extras-only - base install cannot import the app
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - packaging
  - startup
  - subscriptions
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21104).

`Subscriptions/monitoring_engine.py:37` does `from bs4 import BeautifulSoup` with no guard
(directly after a carefully guarded defusedxml import), and the module is eager at boot via
scheduler handlers <- app.py:429. `beautifulsoup4` exists only in
`[project.optional-dependencies]` (verified: four extras, none core). A base `pip install .`
therefore cannot `import tldw_chatbook.app` at all, and every configured install pays the
import at boot.

## Acceptance Criteria

- [ ] Decision made and implemented: either beautifulsoup4 moves to core dependencies, or the import is guarded with graceful degradation of the affected monitors (consistent with `optional_deps.py`)
- [ ] A test covers the import-closure of extras-gated packages so the next unguarded optional import is caught at PR time
- [ ] A base (no-extras) install can import tldw_chatbook.app
