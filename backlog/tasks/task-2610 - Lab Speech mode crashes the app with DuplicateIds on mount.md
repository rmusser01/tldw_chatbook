---
id: TASK-2610
title: Lab ▸ Speech mode crashes the app with DuplicateIds on mount
status: Done
updated_date: '2026-08-06 20:30'
assignee:
  - '@claude'
created_date: '2026-08-06 18:00'
labels:
  - speech
  - crash
  - lab
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Navigating to Lab ▸ Speech crashes the running app with a Textual
`DuplicateIds: ... 'lab-speech-row-playground'` exception. Reproduced 100% of the time
during slice-3 live verification, and confirmed present on `origin/dev` at `a17f9d369`
(base of `feat/voice-profiles-slice3`) — this is NOT introduced by the voice-profiles
work.

The rail row ids built in `STTSScreen.compose_lab_rail`
(`tldw_chatbook/UI/Screens/stts_screen.py:202-219`) are unique per view key within a
single compose, so a duplicate-id collision means the screen (or its rail) is being
composed/mounted twice rather than that the ids themselves are wrong. The same collision
is the long-standing failure in
`Tests/UI/test_settings_speech_tts_panel.py::test_production_settings_actions_cross_the_pushed_screen_boundary`,
which has been dismissed as an unrelated pre-existing test failure across many tasks —
live driving showed it is not merely a test artifact but a real, user-facing crash.

Impact: the Speech Lab (TTS Playground, Voice Profiles, Studio preferences,
AudioBook/Podcast, Voice Cloning, Speech Recognition) is unreachable through the UI.
Voice profiles can still be created and assigned through other surfaces, but the
playground's "Save result as profile" — the primary creation path — is behind this crash.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Navigating to Lab ▸ Speech mounts the Speech screen without raising, in the running app
- [x] #2 The double-mount root cause is identified and fixed at its source (not by making the ids unique per mount, which would hide a duplicated mount)
- [x] #3 `Tests/UI/test_settings_speech_tts_panel.py::test_production_settings_actions_cross_the_pushed_screen_boundary` passes
- [x] #4 A regression test covers the navigation path that reproduces the crash
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce via the failing boundary test; capture both call stacks with instrumentation.
2. Root-cause from the stack diff; verify the Textual dispatch mechanism in the installed
   library source before accepting the theory.
3. Fix at source; mutation-check by reintroducing the defect; add a focused regression test.
4. Blast-radius suites + live tmux verification of the actual navigation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (evidence-first, not the double-mount-of-the-screen the description guessed):
Textual's `MessagePump._get_dispatch_methods` walks the MRO and invokes EVERY class's
`on_mount` for a single Mount event (verified in the installed library source,
message_pump.py:743-800). `STTSScreen.on_mount` also called `super().on_mount()`, so
`LabFrameScreen.on_mount` ran twice — the second `_populate_regions()` mounted a second
copy of the rail rows into the still-populated rail → `DuplicateIds` on
`lab-speech-row-playground`. Models/Evals never crashed because they define no
`on_mount` at all. Stack-diff proof: the two captured stacks are identical except one
frame — the first via `stts_screen.py on_mount` → super chain, the second dispatched
directly to `lab_frame.py on_mount`.

Fix: `STTSScreen` no longer defines `on_mount`. The footer-shortcut ordering the super()
call was (wrongly) protecting — Speech's combined hints must out-write the frame's plain
set in the last-writer-wins registration slot — is now owned by a polymorphic
`_lab_footer_registration()` hook on `LabFrameScreen`, overridden by `STTSScreen`.
`LabFrameScreen.on_mount` also drops its own `super().on_mount()` (same anti-pattern; it
only duplicated a log line), and `BaseAppScreen.on_mount`'s docstring now states the MRO
dispatch contract so the base handler stays idempotent.

Mutation-verified: reintroducing the `super().on_mount()` shape kills 4 tests with the
exact production crash. Regression test added
(`test_speech_screen_mounts_rail_rows_exactly_once`) pinning one-rail-per-mount AND the
footer-registration winner. Live-verified in the real TUI via tmux: F7 → `]`+Enter renders
the Speech Lab (previously a 100% fatal navigation), rail rows each present once,
Models→Speech round trip clean.

Investigation note: an initial control run against "unfixed dev" used the main checkout,
which another session had moved to a different branch — the concurrent-sessions trap; the
clean control was the Edit-based mutation in this worktree.

Files: `tldw_chatbook/UI/Screens/stts_screen.py`, `tldw_chatbook/UI/Screens/lab_frame.py`,
`tldw_chatbook/UI/Navigation/base_app_screen.py` (docstring),
`Tests/UI/test_stts_capability_state.py`. 137 blast-radius tests + collection sweep green;
ruff clean. Follow-up task-2710 files the repo-wide `super().on_mount()` audit (~20 latent
sites, harmless only while the base handler stays a log line).
<!-- SECTION:NOTES:END -->
