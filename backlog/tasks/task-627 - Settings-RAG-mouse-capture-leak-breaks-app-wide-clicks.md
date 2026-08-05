---
id: TASK-627
title: Settings RAG mouse-capture leak breaks app-wide clicks
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:47'
updated_date: '2026-07-25 20:11'
labels:
  - ui
  - textual
  - bug
  - settings
  - rag
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Entering Settings > RAG and interacting with it (open F1 help, edit a field, save, toggle preview) leaves the app's mouse_captured pointer referencing a removed/recomposed widget, so every subsequent mouse click app-wide silently resolves to no widget and is swallowed. In the worst observed case this escalates into a full CPU-pegged hang with zero input working. This makes the RAG Settings category effectively unusable for mouse-driven terminal users and can render the whole app unresponsive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering and interacting with Settings > RAG (open/close help, edit a field, save, toggle preview) never leaves the app with a stale mouse_captured widget reference
- [x] #2 Mouse clicks on other tabs/screens keep working after visiting and interacting with the RAG settings category in the same session
- [x] #3 A regression test locks in click-delivery/capture-release after entering and interacting with the RAG settings category
- [x] #4 If a true headless repro of the click-swallowing symptom is not achievable, the strongest available proxy (capture-state assertion) is added and the coverage gap is documented honestly in the task notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read BaseAppScreen.refresh's existing guard (base_app_screen.py:46-90) and its own detailed docstring/history to understand what it already covers (releases App.mouse_captured synchronously whenever refresh(recompose=True) is CALLED).
2. Confirm SettingsScreen's recompose=True reactives (active_category, server_sync_workspace_handoff_rows, manual_sync_rows, theme_editor_modified) are all declared on SettingsScreen itself (not a sub-widget), so they DO route through the guarded BaseAppScreen.refresh (verified: SettingsScreen.refresh IS BaseAppScreen.refresh, no shadowing) -- ruling out the UAT's own "sub-widget reactive falls outside the guarded path" theory for this screen.
3. Confirm Textual's own push_screen/switch_screen/_replace_screen (pop_screen's teardown) already call capture_mouse(None) too, including for the F1 WorkbenchHelpPanel modal specifically -- ruling out F1 push/pop itself as the leak source (verified empirically: a headless repro driving enter-RAG + F1 help open/close via screen.action_show_workbench_help() never leaves a stale capture).
4. Read Textual 8.2.7's actual recompose scheduling: Widget.refresh(recompose=True) only sets a flag and does self.call_next(self._check_recompose) -- runs on a LATER message-loop iteration, not synchronously. This means BaseAppScreen.refresh's guard checks capture state at CALL time, not at the actual (deferred) teardown time.
5. Empirically confirm this is a real, exploitable gap: headless pilot script that sets active_category (triggering the guard, which correctly finds nothing captured) then IMMEDIATELY (same synchronous stack, before any await) captures a widget the pending recompose is about to remove -- confirmed the capture survives the recompose and stays stuck on the torn-down widget, and a subsequent real nav-bar click produces zero effect (no route change) -- matching the live UAT's exact symptom.
6. RED: add a region test (Tests/UI/test_settings_rag_profile_region.py) reproducing this exact race deterministically and asserting mouse_captured is released and a real nav-bar click still works. Confirm RED against the pre-fix code (git-stashed base_app_screen.py).
7. Fix at the true root: override BaseAppScreen.recompose() (the coroutine Textual's deferred _check_recompose actually calls to perform the teardown), releasing capture as its first synchronous statement before delegating to super().recompose() -- asyncio only yields at await points, so nothing can run between the release and the actual removal starting, closing the window entirely rather than narrowing it. Keep the existing refresh() guard too (harmless, complementary).
8. Rerun the new test (GREEN) plus the pre-existing, closely-related capture-leak regression tests (test_library_skills_canvas.py, test_library_selection_updates.py) to confirm no interference with the ALREADY-fixed, DIFFERENT trigger those cover, plus broader UI navigation suites (test_master_shell_navigation.py, test_destination_shells.py, test_destination_headers.py) given base_app_screen.py is shared by every screen, plus Tests/RAG/ and the full test_settings_rag_profile_region.py file for regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause DIVERGES from the task's own speculative hypotheses (sub-widget recompose=True reactive outside the guard, or pop_screen not releasing capture) -- both were checked and ruled out empirically:
- SettingsScreen's four recompose=True reactives (active_category, server_sync_workspace_handoff_rows, manual_sync_rows, theme_editor_modified) are all declared on SettingsScreen itself; `SettingsScreen.refresh IS BaseAppScreen.refresh` (verified directly, no shadowing) -- they DO route through the existing guard.
- Textual's push_screen/switch_screen/_replace_screen (what pop_screen's teardown uses) already call capture_mouse(None) -- including for the F1 WorkbenchHelpPanel modal specifically. A headless repro driving enter-RAG + F1 help open/close (screen.action_show_workbench_help()) never left a stale capture.

The REAL root cause: Textual 8.2.7's `Widget.refresh(recompose=True)` only *schedules* the actual teardown (`self.call_next(self._check_recompose)`) -- it runs on a LATER message-loop iteration, not synchronously. `BaseAppScreen.refresh`'s existing guard (originally added for a DIFFERENT trigger -- see `test_opening_skill_editor_does_not_break_tab_bar_click_activation`) releases capture at the moment `refresh()` is CALLED, but never re-checks before the deferred teardown actually fires. Empirically confirmed this is a real, exploitable window: a headless script that sets `active_category` (the guard correctly finds nothing captured -- a no-op) then IMMEDIATELY (same synchronous stack, no `await` yet) captures a widget the pending-but-not-yet-run recompose is about to remove showed the capture surviving the recompose, staying stuck on the torn-down widget -- and a subsequent REAL click on the top nav bar produced zero effect (`seen_routes == []`), reproducing the live UAT's exact "clicking '1 Home' did nothing" finding. In production this window is closeable by any genuine MouseDown on an Input/TextArea/ScrollBar arriving as a separately-timed message -- exactly the plausible mechanism the existing guard's own docstring already named for the ORIGINAL bug ("plausible over textual-serve's websocket transport, where down/up travel as independently-timed messages"), just landing in a narrower, later window than that original fix closed.

Fix: added `BaseAppScreen.recompose()` (overriding the actual coroutine Textual's deferred `_check_recompose` calls to perform the teardown, not just `refresh()`) that releases capture as its first synchronous statement, before delegating to `super().recompose()`. asyncio only yields control at `await` points, so nothing else in the event loop can run between this release and the real `remove()`/`mount_all()` teardown starting -- NARROWING the window (see the CODE-REVIEW UPDATE below: a residual, drain-internal window remained, closed by a separate sweep). The existing `refresh()` guard is left in place (harmless, complementary early release for the common non-racy case).

RED/GREEN: added test_recompose_releases_a_capture_that_lands_in_the_deferred_teardown_window to Tests/UI/test_settings_rag_profile_region.py -- opens Library/RAG (matching the reported trigger), sets active_category (scheduling a recompose), immediately captures a widget the recompose is about to tear down (the same call `Input._on_mouse_down` makes internally), lets the deferred recompose run, and asserts BOTH mouse_captured is released AND a real subsequent nav-bar click is actually delivered (a route change is observed). Verified RED by git-stashing just base_app_screen.py: failed with mouse_captured still stuck on the torn-down widget and `seen_routes == []` (no route change at all) -- the exact reported symptom. Restored the fix, confirmed GREEN.

This is a TRUE headless repro of the underlying mechanism (not merely a capture-state proxy) -- AC #4's fallback branch was not needed. The CPU-pegged 14-minute hang from live session 1 (stack sample showing recursive `_asyncio_Task___init__`/`task_eager_start` C-frames, no application-level frames, no `mouse_captured`/`capture_mouse` symbols anywhere in the trace) is almost certainly a SEPARATE bug (a reactive-watcher/recompose feedback loop under Python 3.12's eager-task-start semantics, per the UAT's own analysis) -- session 2 reproduced the pure click-swallowing symptom this task fixes with ZERO CPU spike, confirming the two are at least somewhat independent failure modes. The hang is NOT covered by this fix and would need its own separate investigation if still reproducible; flagging as a known follow-up, not silently claimed as fixed here.

Verification (initial fix): the new test -> 1 passed (both RED and GREEN states independently confirmed). Tests/UI/test_settings_rag_profile_region.py (full file) -> 119 passed. test_library_skills_canvas.py (full file, the pre-existing DIFFERENT-trigger capture-leak regression) -> 104 passed. test_library_selection_updates.py -> 6 passed. test_master_shell_navigation.py + test_destination_shells.py + test_destination_headers.py -> 129 passed, 1 unrelated pre-existing skip. Tests/RAG/ -> 562 passed, 8 skipped (same baseline as tasks 628/629).

Modified files (initial fix):
- tldw_chatbook/UI/Navigation/base_app_screen.py (new `recompose()` override)
- Tests/UI/test_settings_rag_profile_region.py (1 new regression test)

---

CODE-REVIEW UPDATE (post-Done, same task): a reviewer confirmed one Important finding at HEAD (bfdec604e) -- a RESIDUAL capture-leak window survives the `recompose()` override, one level deeper than the call_next-scheduling gap it closes. `BaseAppScreen.recompose()`'s pre-teardown release runs once, as the coroutine's first synchronous statement -- but `super().recompose()` then `await`s `query_children("*")...remove()`, and Textual lets each child's OWN message pump drain during that removal (a separate asyncio task per widget). A message ALREADY queued on a CHILD's pump BEFORE the pre-teardown release even ran (e.g. a MouseDown Screen._forward_event posted to an Input, not yet dispatched when the screen's recompose starts) can still be processed DURING the drain: Input._on_mouse_down calls capture_mouse() unconditionally, and neither Widget.capture_mouse nor App.capture_mouse have an attachment guard, so the widget is re-captured while mid-removal -- leaving App.mouse_captured pointing at a now-detached widget once recompose() returns. Reproduced deterministically (the reviewer's probe, adopted into the test suite) with `call_later` on the VICTIM's own pump, mechanism-equivalent to a forwarded MouseDown whose dispatch is still pending when the enclosing screen's teardown begins.

Fix: added a post-`super().recompose()` sweep in `BaseAppScreen.recompose()` -- once the ENTIRE recompose (removal AND remount) has finished, any still-captured widget that is NOT `is_attached` is by definition stale (nothing legitimately captured during remount would already be detached) and is released again. `is_attached` is what distinguishes this stale case from a widget a later, unrelated interaction has since legitimately captured (left alone).

CORRECTED OVERCLAIM: the original Implementation Notes and the `recompose()`/test docstrings said the fix "closes the window entirely." That was wrong -- the fix NARROWS the window to the teardown drain itself (the call_next-scheduling gap is closed; the drain-internal gap needed this separate sweep). Docstrings in base_app_screen.py and Tests/UI/test_settings_rag_profile_region.py have been corrected to say "narrows" and cross-reference the sweep/its own test. This Implementation Notes entry and .superpowers/sdd/task-627-629-report.md are corrected too.

RED/GREEN (review fix): adopted the reviewer's probe as test_post_recompose_sweep_releases_a_capture_dispatched_during_the_teardown_drain in Tests/UI/test_settings_rag_profile_region.py. Verified RED by git-stashing just the sweep addition in base_app_screen.py (kept the earlier recompose()-release fix): failed with the exact stale capture the reviewer found. Restored the sweep, confirmed GREEN.

Verification: Tests/UI/test_settings_rag_profile_region.py -> 120 passed (one transient, order-dependent failure in an unrelated footer test on one run -- passed both in isolation and on an immediate clean rerun of the full file; consistent with this suite's known pre-existing flakiness under load, e.g. the Theme-category timeout flake found during task-629, not a regression from this change). test_library_skills_canvas.py::test_opening_skill_editor_does_not_break_tab_bar_click_activation (the pre-existing, different-trigger capture-leak regression) -> still passes.

Modified files (this update):
- tldw_chatbook/UI/Navigation/base_app_screen.py (post-recompose stale-capture sweep; docstring overclaim corrected)
- Tests/UI/test_settings_rag_profile_region.py (1 new regression test; docstring overclaim corrected)
<!-- SECTION:NOTES:END -->
