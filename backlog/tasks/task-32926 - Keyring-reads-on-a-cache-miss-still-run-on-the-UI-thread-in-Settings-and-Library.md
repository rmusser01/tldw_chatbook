---
id: TASK-32926
title: Keyring reads on a cache miss still run on the UI thread in Settings and Library
status: Done
assignee:
- '@claude'
created_date: 2026-09-23 19:58
labels:
- performance
- linux
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32921/32922/32924 cache keyring reads so repeated UI-path lookups share one round trip, but a cache miss still reads the OS keyring synchronously on the Textual event loop. On Linux a locked gnome-keyring can block that read on an unlock prompt. Remaining UI-thread sites (Qodo review on #2820): Settings Privacy & Security builds skill-trust posture inside compose(); Settings sync scope and Library screen helpers resolve the active server context (auth token -> principal id) synchronously; the Image/Video Gen panels resolve backend secrets in compose().
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Settings Privacy & Security never reads the OS keyring on the UI thread; trust posture renders as pending and fills in from a worker
- [x] #2 Settings sync scope and Library server-scope resolution never read the OS keyring on the UI thread
- [x] #3 Image and Video Gen panels compose from configuration loaded off the UI thread, including after save, revert and Test
- [x] #4 A test with a keyring backend that blocks proves each surface stays responsive
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Every UI-thread path that could reach a keyring read now runs that read on a worker and renders an honest pending state until it finishes.

- **Privacy & Security**: `_settings_privacy_posture(read_skill_trust=False)` renders `Skill trust: checking…`. A new `_skill_trust_status_worker` (a thread worker in its own exclusive group) reads `overall_status()` and updates the `#settings-privacy-skill-trust` row. The Check Privacy worker still reads the status directly, which is fine because it already runs on a thread. `"checking"` was added to the closed status set, and `safe_skill_trust_status` is now public so the worker path can reuse it.
- **Settings sync scope**: `_run_manual_sync_once` and the Notes-adoption handler now resolve `_active_sync_scope` through `asyncio.to_thread`. The adoption tail became `_resolve_notes_adoption`, which runs as a worker (`run_worker`, exclusive, with a group). The workspace record is still read on the loop and passed in, so the thread never opens a WorkspaceDB connection. Scoping is unchanged: the principal still comes from the same auth token.
- **Library**: `_active_library_principal_id` was split out of `_active_library_sync_scope`, which gained a `resolve_principal` flag (default True, so other callers behave as before). The onboarding admission key's synchronous half omits the principal. `_resolve_library_onboarding_admission_key` fills it in off the loop, both when the evidence round starts and when it settles. `_apply_library_onboarding_evidence` now takes that settled key as an argument, so the fence still compares the full key including the principal.
- **Image and Video Gen**: the panels no longer contain a config loader. They compose from the `config=` and `cleared_key_sources=` they are given, and show `Loading … settings…` while `config` is None. The screen's `_image_gen_panel_load_worker` / `_video_gen_panel_load_worker` load the config (plus the after-Clear key sources for staged Clears) on a thread, then queue the Select mount-echo suppression and recompose. Category open, Save and Revert all go through this path. For Test, the effective-value and secret fallbacks now resolve inside the probe worker. For Clear, the key-source line shows `checking…` and a worker fills it in.
- Trade-off: the Gen panels and the skill-trust row now pay one extra paint while the worker loads. In return, a locked keyring no longer freezes the app.
- Tests: `Tests/UI/test_settings_keyring_off_ui_thread.py` (9 tests). Its fakes block on a `threading.Event` and count main-thread calls. The tests cover all four surfaces: pending state rendered, main-thread calls == 0, and the value arrives after release. Each new test failed before the fix for the expected reason. Existing Gen-panel, video and hub tests now wait for the loaded panel rather than just its container. The library-shell `capture_apply` forwards the new argument.
- Regression check: I ran the touched suites (hub, privacy, image/video gen, library shell and onboarding, trace maintenance, library support surface) on the branch and on a clean origin/dev worktree. Both runs finished at 1095 failed / 219 passed / 34 errors, with identical FAILED/ERROR name sets. The large red count is the known local ADR-126 `RecoveryRequired` fixture baseline. Because of that baseline, most of the existing Gen-panel tests cannot run locally, so the wait-helper edits to them are verified only by reading them; CI has to confirm them. `preflight.sh` exits 0.
- Qodo review (#2831) follow-ups:
  - `keyring_get` is now single-flight per key, so overlapping workers share one blocking lookup (one unlock prompt).
  - Each off-thread read carries an `object()` token, so a stale load, Clear lookup or Privacy status result is dropped. This matters because thread workers cannot be cancelled mid-read.
  - A config load that raises now logs only the exception type and shows "… settings could not be loaded" in the panel. Before, it exited the app via `exit_on_error`.
- Files: settings_screen.py, settings_privacy_security.py, library_screen.py, Library_Modules/screen_helpers.py, Widgets/settings_image_gen_panel.py, Widgets/settings_video_gen_panel.py, plus the tests above.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
