---
id: TASK-32634
title: >-
  Library browse location reads a legacy config key across the notes fence
status: To Do
assignee: []
created_date: '2026-09-15 10:15'
labels:
  - library
  - notes
  - config
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while working task-32604. `tldw_chatbook/UI/Library_Modules/library_browse_location.py:77`
reads `[notes] sync_directory`, which is a member of `_LEGACY_CONFIG_KEYS` —
the set the notes-sync fence exists to keep out of live reads. This is a
**dev-side** violation, not a wave-5 regression: it landed in `da358ee63d` and
is present on both `origin/dev` and `origin/main`. It fails at the wave-5 base
commit, so no branch in this wave introduced it and none can be blamed for it.

Filed separately so the fence's own guard can be tightened at the same time —
a key in `_LEGACY_CONFIG_KEYS` that a live read still reaches means the guard
is not enforcing what its name claims.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `library_browse_location.py` no longer reads `[notes] sync_directory`, and the behaviour it needed is sourced from whatever superseded that key.
- [ ] #2 A check fails if any live read reaches a `_LEGACY_CONFIG_KEYS` member — the fence enforces its own list rather than documenting it.
- [ ] #3 The check's RED is captured by restoring the line at `:77` (re-derive the line number; it will have moved).
<!-- AC:END -->
