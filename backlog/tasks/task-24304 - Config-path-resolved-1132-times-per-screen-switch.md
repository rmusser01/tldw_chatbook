---
id: TASK-24304
title: >-
  The effective config path is re-resolved 1,132 times per Console screen entry
status: Done
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - config
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`config._get_effective_config_path()` reads an environment variable and runs `lexical_path()`
path normalisation on every call. It is called 1,132 times during a single warm entry to the Chat
screen, costing 54.8 ms cumulative on dev `3a3383123e`, plus its share of every keystroke through the same
derivation machinery.

The function is pure with respect to the `TLDW_CONFIG_PATH` environment variable. This is the
smallest self-contained item in the review and it pays into both hot paths above it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Resolving the effective config path repeatedly with unchanged environment does not repeat the path-normalisation work
- [x] #2 A change to TLDW_CONFIG_PATH is still observed -- a test proves the memo does not pin a stale path
- [x] #3 The call is no longer a measurable term in a warm Console screen entry
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split the pure normalisation out of `_get_effective_config_path` and memoise it.
2. Key on BOTH `TLDW_CONFIG_PATH` and `HOME`.
3. Test that each key component invalidates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_get_effective_config_path` kept its live environment read; the
`expanduser` + `abspath` + `normpath` work behind it moved into an
`lru_cache`d helper.

`HOME` is part of the key, not just the override: `lexical_path` calls
`os.path.expanduser`, which reads `HOME` at call time, so an override
containing `~` resolves differently when `HOME` moves -- and tests move `HOME`
routinely. Keying on the override alone would have handed back a stale path.

**Measured on a warm Console screen entry: 1,132 lookups -> 1,132 cache hits ->
0 path normalisations.** The lookup count matches the figure from the review
exactly. Mutation-tested: removing the `lru_cache` reds the guard.

Files: `config.py`, `Tests/Config/test_effective_config_path_memo.py` (new).
<!-- SECTION:NOTES:END -->
