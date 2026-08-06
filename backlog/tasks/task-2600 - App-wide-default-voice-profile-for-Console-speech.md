---
id: TASK-2600
title: App-wide default voice profile for Console speech
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 19:09'
updated_date: '2026-08-06 19:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A saved voice profile can be named as the app-wide default voice, used for Console speech whenever no character-specific voice applies. The default is live-linked (editing the profile changes the default everywhere) and sits between a character's assigned voice and the raw global TTS axes in precedence, so a character voice still always wins. When the configured default cannot be used, speech refuses honestly and offers a one-tap fallback to the global voice rather than silently substituting a different voice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A saved voice profile can be set as the app-wide default voice from Settings > Speech & TTS > Global defaults, and the choice persists across restarts
- [x] #2 Editing the chosen default profile elsewhere changes what speaks everywhere it applies, without needing to reselect it
- [x] #3 Precedence is explicit request, then a character's assigned voice, then the default profile, then the global axes fields, then provider fallback -- a character voice always outranks the default profile
- [x] #4 When the default profile is deleted, its store is unavailable, or its stored id is malformed, Console speech refuses and offers a one-tap Use global voice confirmation naming the default voice specifically, never mislabeled as a character voice
- [x] #5 Settings never silently drops a saved-but-broken default profile selection; it keeps showing the saved id with an honest note distinguishing still-loading from confirmed-unavailable
- [x] #6 Deleting a voice profile that is the current app-wide default warns that it is the app default before deletion completes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a DEFAULT_PROFILE precedence rung between character and global in the effective-settings resolver (TTSDefaultProfileSelection, TTSSelectionSource.DEFAULT_PROFILE).
2. Persist [app_tts] default_profile_id through the Speech & TTS settings model (load/save/restore, diffed independently of the axes snapshot).
3. Add a "Default voice profile" selector to Settings > Speech & TTS > Global defaults, sourced from an impurely-injected profile list, honest about loading/unavailable/dangling-id states.
4. At speak time, resolve the default profile (sibling resolver reusing the character resolver's bounded-error mechanism) only when no character voice applies; refuse honestly with a one-tap "Use global voice?" override naming the correct domain on failure.
5. Warn when deleting a voice profile that is the current app-wide default.
6. Live-verify against a real provider, extend the user guide, file this task, run gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the app-wide default voice profile (voice-profiles slice 3, tasks 1-6, dev base a17f9d369):
resolver rung (TTSDefaultProfileSelection between character and global), [app_tts]
default_profile_id persistence through the settings model, a "Default voice profile"
selector at the top of Settings > Speech & TTS > Global defaults with honest
loading/unavailable/dangling-id states, a sibling fail-closed resolver
(default_profile_request_resolver.py) that refuses with a one-tap "Use global voice?"
override naming the actual failed domain, and a delete-time warning when removing the
profile that is the current app default. Full technical narrative and review history in
task-6-report.md. Task id assigned by scanning every worktree (`git worktree list
--porcelain` + `os.listdir`) and every remote ref (`git ls-tree` over
`refs/remotes/*`) for the true max `task-NNNN` in `backlog/`: 2560. Filed as 2600
(headroom over the CLI's own auto-assignment, which picked the stale/colliding-prone
2531 -- renamed after the same scan confirmed 2600 was free everywhere).

Live verification (real OpenAI account, no mocks): set Live-Verify-Echo (voice=echo) as
the default profile via the real Settings UI and saved; a real Console speak with no
character active resolved to voice=echo (confirmed via the in-app Logs screen's DEBUG
request line), not the global axis voice=alloy; deleting the profile and speaking again
produced a real refusal toast plus the real "Use global voice?" ConfirmationDialog with
copy naming the default voice, never "character"; a fresh app launch with the default
already saved resolved Settings straight to the real profile name with no false
"unavailable". Profile creation/deletion were done via the real production
TTSProfileService (not the Playground UI) because entering Lab > Speech mode crashes the
app with DuplicateIds -- confirmed pre-existing (reproduces identically on base commit
a17f9d369, unrelated code) and 100% reproducible (3/3, two navigation paths, two window
sizes) -- filed separately, not fixed here per scope.

Docs: Docs/User_Guide/openai-compatible-tts.md gained an "App-wide default voice
profile" section (concept, live-linked behavior, precedence, refuse+override, deletion
warning) plus an appended Verified-against line; index.md's row description updated.

Gates: ruff check clean on all 23 touched files except 5 pre-existing errors in
settings_screen.py (confirmed present at base a17f9d369, untouched by this branch's
diff there). ruff format --check: 10 files needed reformatting under this environment's
ruff version (5 pre-existing at base, 3 drifted from this branch's own edits, 2 new
files) -- reformatted all 10, confirmed AST-identical before/after (zero semantic
change). Targeted suite (10 touched test files): 329 passed, 1 failed
(test_production_settings_actions_cross_the_pushed_screen_boundary, DuplicateIds --
confirmed failing identically on base a17f9d369 via a disposable worktree, pre-existing
and unrelated). Repo-wide --collect-only: 31293 tests collected, 0 errors.
<!-- SECTION:NOTES:END -->
