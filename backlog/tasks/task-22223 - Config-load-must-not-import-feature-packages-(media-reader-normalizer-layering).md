---
id: TASK-22223
title: >-
  Config load must not import feature packages (media-reader normalizer layering)
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - architecture
  - startup
  - library
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22223).

`config.py:1802` (added 2026-08-24, `455d8d061`) imports
`Library.library_media_reader_state` inside `_load_settings_uncached`, whose comment
claims the import is lazy — but `load_settings()` runs at config-module import
(`config.py:7416`), so it fires on every boot inside the app import closure. Honest
scoping from this review: the incremental cost TODAY is small (the Library package was
already in the pin's import closure via `app.py:238`), so this is a layering finding, not
a milliseconds finding — config load acquiring feature-package imports is the mechanism by
which the next 100 ms lands silently.

## Acceptance Criteria

- [ ] Preference normalization happens without config.py importing the Library package (move the normalizer to a config-safe leaf module, or normalize at first read in the Library layer)
- [ ] A comment or guard at the site prevents the next feature from repeating the pattern
- [ ] No behavior change to media-reader preference handling
