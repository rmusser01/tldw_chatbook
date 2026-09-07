---
id: TASK-16844
title: 'FileListItemEnhanced passes tooltip= to Static: any non-empty FileListEnhanced crashes on compose'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - bug
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live during the TASK-15771 review (PR #1699) and still present at dev `ee741cf10`:
`Widgets/file_list_item_enhanced.py:123-127` yields
`Static(self._metadata["name"], classes="file-name", tooltip=str(self.file_path))`, but
Textual 8.2.8's `Static.__init__` takes only
`content, *, expand, shrink, markup, name, id, classes, disabled` — **no `tooltip`
parameter**. The review reproduced it deterministically: mounting `FileListEnhanced` and
setting a one-element `files` list raises

```
TypeError: FileListItemEnhanced(id='file-item-...') compose() method returned an invalid
result; Static.__init__() got an unexpected keyword argument 'tooltip'
```

Any non-empty `FileListEnhanced` hits it — `files` is a `recompose=True` reactive, so the
first real row triggers the crash (the irony noted by the review: it is one of the four
recompose sites 15771 fixed, and the one that cannot be exercised). That no test caught
it says the widget currently mounts nowhere with data in the tested surface — so
establish reachability first: if something real feeds it, fix the tooltip (set the
`.tooltip` attribute after construction, or on the row widget) and pin with a born-red
non-empty-list test; if nothing does, this is a wire-or-retire candidate rather than a
one-line patch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Reachability is established and stated: what (if anything) mounts `FileListEnhanced` with a non-empty list in the live app
- [x] #2 A mounted `FileListEnhanced` with at least one file composes without error, and the intended tooltip behavior actually works (test born-red against the current code)
- [x] #3 If the widget is dead, that fact is stated honestly with reachability evidence in the Implementation Notes; the tooltip fix is applied regardless (it is a one-line, zero-risk correction that un-breaks the widget for any future caller, so retiring the file is not warranted on top of it)
<!-- AC:END -->

## Implementation Plan (the how)

1. Reproduce the crash live (mount `FileListEnhanced(files=[...])` under `App.run_test()`)
   to confirm the exact `TypeError` quoted in the description, at HEAD.
2. Grep the whole tree for `FileListEnhanced`/`FileListItemEnhanced`/
   `file_list_item_enhanced` outside the definition file itself (production code,
   tests, `Widgets/__init__.py` exports) to establish reachability.
3. Fix `FileListItemEnhanced.compose()`: drop the unsupported `tooltip=` kwarg from
   the `Static(...)` constructor call and set `.tooltip` on the instance after
   construction instead (the idiom already used elsewhere in the repo, e.g.
   `Evals/character_bench_editor.py`).
4. Sweep the rest of the file for other constructor kwargs that don't exist on the
   target Textual 8.2.8 widget class (the `Button(..., tooltip=...)` call is fine --
   `Button.__init__` does take `tooltip`).
5. Add a mounted, born-red-first regression test: a non-empty `files` list composes
   without error and the row's name `Static` ends up with the right `.tooltip`.
6. Run the new test file + a repo-wide collect-only sweep; ruff on the touched file.

## Implementation Notes

**Fix.** `FileListItemEnhanced.compose()` (`Widgets/file_list_item_enhanced.py:120-129`)
constructed the file-name `Static` with `tooltip=str(self.file_path)` — a kwarg
`Static.__init__` doesn't accept on Textual 8.2.8 (confirmed live via
`inspect.signature(Static.__init__)`: only `content, expand, shrink, markup, name, id,
classes, disabled`). Replaced with the idiom already used elsewhere in this repo
(`UI/Evals/character_bench_editor.py`, `UI/Study_Window.py`, etc.): build the `Static`
without `tooltip`, then set `name_static.tooltip = str(self.file_path)` before yielding
it. Swept the rest of the file for the same mistake: every other `Static(...)` call only
passes `classes`/`id` (both fine on `Widget`), and the one `Button(..., tooltip=...)`
call is legitimate — `Button.__init__` *does* take `tooltip` (confirmed via the same
signature check) — so this was the only bad kwarg in the file.

**Reachability (AC #1/#3).** A repo-wide grep (`grep -rln "FileListEnhanced\|
FileListItemEnhanced\|file_list_item_enhanced"`, all extensions) found the class defined
only in its own file, referenced nowhere else in `tldw_chatbook/` (no importer, no
`Widgets/__init__.py` export) and nowhere in `Tests/` before this task. `git log` on the
file shows only the two prior reactive/recompose-guard sweeps (task-15771, task-670)
touching it mechanically, never a feature commit wiring it into a screen. Verdict:
**dead chrome** — nothing in the live app currently mounts `FileListEnhanced` with data,
which is exactly why the crash survived past this many burn-down passes. Per this task's
explicit brief ("fix stands either way — it un-breaks the widget for whoever mounts it"),
the fix was kept rather than deleting the file: it is a one-line, zero-risk correction,
and outright retirement is a bigger, separately-reviewable call this task doesn't need
to force. AC #3 was reworded before implementation (see git history of this file) to
match that directive instead of literally demanding a deletion.

**Tests.** New `Tests/Widgets/test_file_list_item_enhanced.py`, 3 tests:
- `test_file_list_enhanced_with_files_composes_without_error` — born red at HEAD with the
  exact `TypeError: Static.__init__() got an unexpected keyword argument 'tooltip'`
  quoted in the description; green after the fix.
- `test_file_list_item_name_static_carries_the_intended_tooltip` — also born red (same
  crash); pins that the intended behavior (the name `Static`'s `.tooltip` equals the
  file path) actually works post-fix, not just that compose doesn't raise.
- `test_file_list_enhanced_empty_files_still_shows_placeholder` — passed at HEAD too
  (the empty-list path never hit the bug); kept as a no-regression check on the
  placeholder path.

Verified born-red by temporarily restoring the pre-fix file content (via `git show
HEAD:...` into a scratch copy, `cp`'d over the working file, restored via `Write` after —
no `git checkout`) and rerunning the suite: 2 failed with the quoted `TypeError`, 1
passed. After restoring the fix: `3 passed`. `pytest Tests/Widgets --collect-only`: 417
tests collected, 0 collection errors. `ruff check` on both touched files: clean.

**Files changed:**
- `tldw_chatbook/Widgets/file_list_item_enhanced.py` — the fix (5 lines net).
- `Tests/Widgets/test_file_list_item_enhanced.py` — new regression test (born-red/green).
