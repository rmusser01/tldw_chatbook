---
id: TASK-39112
title: "The backup census counts _atomic_write_target's open and write but not its link, unlink and replace, because they go through a dynamic file_os binding"
status: To Do
created_date: 2026-11-07 02:03
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
- [ ] #1 The census counts the publication calls in both `_atomic_write_target`s, or the doc lists the dynamic binding as an owner-accepted limit; the decision is recorded
- [ ] #2 A unit test with a helper-returned module (`f = helper(); f.replace(a, b)`) is red before the change and green after (unless the limit is accepted)
- [ ] #3 No production file changes for this
<!-- AC:END -->
