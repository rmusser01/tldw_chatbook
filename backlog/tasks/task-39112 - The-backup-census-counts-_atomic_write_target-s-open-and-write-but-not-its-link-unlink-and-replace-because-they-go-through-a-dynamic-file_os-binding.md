---
id: TASK-39112
title: "The backup census counts _atomic_write_target's open and write but not its link, unlink and replace, because they go through a dynamic file_os binding"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 18:23
labels:
- follow-up
- backup
- architecture
- tests
priority: low
references:
- Tests/Architecture/test_backup_owner_inventory.py
- tldw_chatbook/Tools/local_tool_impls.py
- tldw_chatbook/Tools/remote_worker_bundle.py
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, with the census from round-11 branch fu/r11-backupinv (TASK-39053). Line numbers are dev's.

**The binding.** `_write_file_os()` returns `os`, or on Windows the `Utils/platform_files` shim (`Tools/local_tool_impls.py:797`). The generated `Tools/remote_worker_bundle.py` has the same helper, around `:2626`.

**What the census misses.**
- `_atomic_write_target` (`local_tool_impls.py:883`) publishes through `file_os.link`, `file_os.unlink`, `file_os.replace` and the Windows `replace_open_target`.
- A probe of the census on that symbol counts only `open` ×2 and `write` ×1.
- The bundle's copy is the same.

So the review sees the symbol, but not that it publishes or deletes files without clobbering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census counts the publication calls in both `_atomic_write_target`s, or the doc lists the dynamic binding as an owner-accepted limit; the decision is recorded
- [x] #2 A unit test with a helper-returned module (`f = helper(); f.replace(a, b)`) is red before the change and green after (unless the limit is accepted)
- [x] #3 No production file changes for this
<!-- AC:END -->

## Implementation Plan

1. Unit test first: `f = helper(); f.replace(a, b)` where `helper()` returns `os` is counted as `os.replace` (red on base).
2. Census: a function whose `return` yields an os binding is an os helper; a name assigned from a call to it is an os alias in that lexical scope; conditional expressions (`a if c else b`) contribute both arms.
3. Recompute; review the new `_atomic_write_target` (and sibling) rows in both modules. No production change.

## Implementation Notes

Counted, not accepted as a limit: the census in `Tests/Architecture/test_backup_owner_inventory.py` now treats a module function whose `return` yields an `os` binding (`return os`, or a name imported as `os` from the `platform_files` shim) as an os helper. A name assigned from a call to it is an `os` alias in the assigning function's lexical scope, so `file_os.link/unlink/replace` count as `os.*`. Both arms of a conditional expression count, which covers `replace = file_os.replace_open_target if ... else file_os.replace`; `replace_open_target` (the Windows `platform_files` shim's) is a new native seam counted as `os.replace_open_target`.

- Decision (AC#1): the publication calls are counted, so no dynamic-binding limit is listed. Both `_atomic_write_target`s (`Tools/local_tool_impls.py`, `Tools/remote_worker_bundle.py`) gained `os.link` 1, `os.unlink` 2, `os.replace` 1 and `os.replace_open_target` 1. They take the symbol's existing `external_input` / `external` disposition (rule 2), in the doc's new "Census vocabulary extension" section. No other module gained a row from the helper rule.
- Red/green (AC#2): `test_census_tracks_os_returned_by_a_helper` failed on the base scanner with `assert 0 == 2` (`f = helper(); f.replace(a, b)` plus the conditional alias), and passes now. Its `unrelated` function (`f = text(); f.replace("a", "b")`) pins the lexical scope.
- AC#3: no production file changed for this task; the change is in the test module and the doc.
- Ceiling (marked `ponytail:` in the scanner): aliases resolve in source order, so a nested function that aliases an outer name bound later in the source is missed.
- Files: `Tests/Architecture/test_backup_owner_inventory.py`, `backlog/docs/backup-recovery-owner-inventory.md`.
