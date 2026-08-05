---
id: TASK-929
title: Audit remaining self.call_from_thread misuse outside Tools_Settings_Window
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 09:00'
updated_date: '2026-07-27 18:10'
labels:
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While fixing TASK-899 it emerged that all four database-maintenance workers called `self.call_from_thread(...)` on `ToolsSettingsWindow`, which is a `Container`. That method exists only on `App` — verified: `Widget` and `Container` both lack it. Every notification raised from those worker threads would therefore have raised `AttributeError` rather than notifying, which is why the failures were never seen.

All 39 call sites in that file now use `self.app.call_from_thread`, and a guard test asserts the bare form does not come back. That guard is file-scoped.

The same mistake is plausible anywhere a `Widget`/`Container` subclass runs a threaded worker, and it is invisible until the error path executes. Sweep the codebase for `self.call_from_thread` on non-`App` classes and fix or clear each.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every `self.call_from_thread` call site outside `App` subclasses is identified
- [x] #2 Each is fixed to use `self.app.call_from_thread` or confirmed to be on an `App`
- [x] #3 A repo-wide guard replaces or supplements the file-scoped one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep tldw_chatbook/ for self.call_from_thread( excluding self.app.call_from_thread(, map each hit to its enclosing top-level class, and verify with hasattr(cls, 'call_from_thread') style checks (not trust the pre-supplied table blindly).
2. Fix each genuinely-broken site by changing self.call_from_thread( to self.app.call_from_thread(.
3. For app.py specifically, verify the enclosing class per site before touching anything -- do not assume the pre-supplied table's TabDropdown attribution is correct.
4. Skim the notification text each newly-working call site will now surface; fix any broken placeholder/format issue found along the way, nothing more.
5. Add a repo-wide guard test (tokenize-based, comment/string immune) under Tests/ that fails on any bare self.call_from_thread( outside the documented App-safe exceptions, and prove it by temporarily reintroducing a bare call and reverting.
6. Run the targeted test files covering the six touched modules plus the existing Tools_Settings_Window guard test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed 31 bare `self.call_from_thread(` sites across six files: ScraperBuilderWindow.py (9, Screen), swarmui_widget.py (10, Container), multi_item_review_window.py (5, Container), CodeRepoCopyPasteWindow.py (4, ModalScreen), SiteConfigSettings.py (2, Container), media_viewer_panel.py (1, Container) -- all changed to `self.app.call_from_thread(`. Every message text produced by the now-working notifications was checked; all were properly formatted f-strings with no broken placeholders, so no message copy needed fixing.

`app.py`'s 9 sites were investigated and left untouched: 7 sit in `LibraryIngestQueueMixin`, a mixin only ever combined with `App` (`class TldwCli(LibraryIngestQueueMixin, App[None])` in production; two test harnesses do the same); Python resolves `self.call_from_thread` through the instance's full MRO, not the class whose body contains the call, so these are safe despite the mixin itself not subclassing App. The other 2 sit directly on `TldwCli(App)`. Empirically verified with `hasattr`/`issubclass`. The original triage table's "TabDropdown(Widget)" attribution for these 9 was incorrect (a backwards nearest-`^class` regex misattributed them); TabDropdown itself has zero call_from_thread sites.

Added `Tests/test_call_from_thread_guard.py`, a repo-wide backstop (the existing Tools_Settings_Window guard is file-scoped). It tokenizes every .py file under `tldw_chatbook/` (not a substring/regex scan) and looks for the exact NAME('self') OP('.') NAME('call_from_thread') OP('(') token sequence, dropping COMMENT and STRING tokens first -- this is immune by construction to a comment or docstring merely mentioning the pattern (the trap called out in the task), and it naturally does not match `self.app.call_from_thread(` since the token after `self` `.` there is `app`, not `call_from_thread`. `app.py` is allowlisted by relative path with a documented reason; a second test (`test_app_py_allowlisted_sites_are_still_safe`) parses app.py's AST and asserts every remaining bare site sits inside `LibraryIngestQueueMixin` or `TldwCli` specifically, so the allowlist can't silently become a hole for a future unsafe class in that file.

Revert-checked twice: (1) reintroduced a bare call in SiteConfigSettings.py, confirmed the guard failed listing exactly that file/line, reverted, confirmed pass; (2) reintroduced a bare call in CodeRepoCopyPasteWindow.py, confirmed the guard failed listing only that site (app.py's legitimate sites did not also fire), reverted, confirmed pass. Also verified a decoy comment containing the bad pattern does not trip the guard.

Ran targeted tests in the foreground: guard test (2 passed), CodeRepoCopyPasteWindow/multi_item_review/bulk_selection_tooltips/site_config_manager/integration (25 passed), Image_Generation suite (103 passed), UI media/settings tests (153 + 10 + 75 passed), Chat/ChaChaNotesDB (201 passed), Tests/UI/test_tools_settings_window.py (40 passed, 6 pre-existing chat_api_key failures per prior task note, 16 skipped -- its own file-scoped guard passed). Total: 609 passed, 0 regressions, across everything actually exercising the touched modules.

Tests/UI/test_console_native_chat_flow.py was in the initial keyword-match candidate list but only matches on the string literal "swarmui" used as image-generation backend metadata in unrelated console-chat tests; it does not import any of the six touched modules. It has 18 pre-existing failures (ChatScreen._task_resume_state AttributeError, provider-selection default mismatch, composer timeouts) verified unrelated to this change and left alone.

Files modified: tldw_chatbook/UI/ScraperBuilderWindow.py, tldw_chatbook/UI/CodeRepoCopyPasteWindow.py, tldw_chatbook/UI/SiteConfigSettings.py, tldw_chatbook/Widgets/Media_Creation/swarmui_widget.py, tldw_chatbook/Widgets/multi_item_review_window.py, tldw_chatbook/Widgets/Media/media_viewer_panel.py, Tests/test_call_from_thread_guard.py (new).
<!-- SECTION:NOTES:END -->
