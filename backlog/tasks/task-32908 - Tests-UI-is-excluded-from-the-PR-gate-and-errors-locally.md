---
id: TASK-32908
title: Tests/UI is excluded from the PR gate and errors locally, so nothing there blocks a merge
status: To Do
assignee: []
created_date: '2026-09-22 01:10'
labels:
  - tier2-review
  - review-testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by chasing a single stale assertion and asking why it had never failed anything.

`Tests/UI/` is **1,172 test files / 19,182 test functions**. It is executed by neither of the two places
that would catch a regression before it lands:

1. **The PR gate skips it by name.** `.github/workflows/test.yml:121`:
   ```
   pytest Tests --ignore=Tests/UI \
   ```
2. **It errors locally in a clean worktree.** The ADR-126 recovery gate raises
   `RecoveryRequired: raw_participant_not_installed` at fixture setup, so a developer running it gets errors
   before any assertion executes. Confirmed on `origin/dev 338a127501`.

The only job that does run it is `nightly-deep.yml`, which is `on: schedule` (cron `30 8 * * *`) with
`fail-fast: false`, and whose own header notes that scheduled workflows register only from `main`. Nothing
waits on it to merge. The repo's only **required** check on `dev` is `Derived artifacts`.

So a broken test in `Tests/UI/` blocks nothing, and the signal it does produce arrives the next day in a job
no one is gated on.

## The evidence that made this visible

`Tests/UI/test_console_library_tool_setting.py:229`:
```python
assert service._collections is app.local_library_collections_service
```
`LocalLibraryToolService` has **no** `_collections` attribute -- the single occurrence of that string in the
module is inside a comment at `:102`. The assertion cannot pass; it can only raise `AttributeError`. It has
been in the tree since `0577884cf2` and has never blocked anything.

It is a third copy of a parameter that `5dd1077df6` retired: the production call site
(`console_runtime.py:681`), the duplicate builder
(`UI/Console_Modules/library_activity.py:92 ConsoleLibraryActivityController.build_provider`), and this
assertion. That commit updated one of the three. The surviving production copy is **the P0 in TASK-32892** --
Console builds no Library tool provider at all on the default config.

That is the cost of this gap stated concretely: **a P0 shipped, and the test that would have caught it was in
a directory nothing runs.**

## What to decide

This is a judgement call about CI economics, not an obvious bug, so it needs an owner rather than a patch:

- Is `Tests/UI` excluded because it is slow, because it is flaky, or because the ADR-126 gate makes it
  unreliable in a clean checkout? The fix differs for each, and the third is fixable (see below).
- If it stays out of the PR gate, then it should stop being presented as a test suite -- a directory of
  19,182 assertions that gate nothing gives false assurance, which is worse than having none.
- If it comes back in, it needs to be green first, and that is its own project.

**A concrete lead on the third cause.** TASK-32893 found that at least one gated file is gated by accident:
`Tests/Media/test_local_media_reading_service.py` raises `RecoveryRequired` not from its fixture but because
a media-DB migration lazily loads config *inside* an active raw-participant scope. Forcing the load first
(importing `tldw_chatbook.Chunking` before invoking pytest) makes the whole file runnable locally. If that
generalises, a meaningful share of `Tests/UI` may be recoverable cheaply.

Source: tier-2 code review 2026-09-21, found while verifying TASK-32892.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reason `Tests/UI` is excluded from `test.yml` is established and written down
- [ ] #2 A decision is recorded: bring it into the gate, or stop treating it as a suite
- [ ] #3 The ADR-126 local-gate cause is measured -- how many of the 1,172 files are gated by accident rather than by design
- [ ] #4 The stale `_collections` assertion is removed or corrected regardless of the above
<!-- AC:END -->
