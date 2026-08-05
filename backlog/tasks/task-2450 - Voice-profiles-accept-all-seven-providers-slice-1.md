---
id: TASK-2450
title: Voice profiles accept all seven providers (slice 1)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 04:47'
updated_date: '2026-08-05 04:56'
labels:
  - tts
dependencies:
  - TASK-1626
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reusable TTS generation profiles were audio.cpp-only end to end: the profile validation table, the character speech resolver, and the playground's save-as-profile eligibility all refused every other provider. This slice extends the closed provider set to all seven adapters the app already ships (audio_cpp plus the six legacy-bridge providers: openai, elevenlabs, kokoro, chatterbox, higgs, alltalk), so a profile built around any of them can be created, assigned to a character, and spoken through Console, while keeping the store safe for older builds and honest about what has and has not been verified for the newly admitted providers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Profile creation and edit validate model/voice/format/speed per a closed seven-provider table (audio_cpp exact WAV/1.0/empty-options; the six legacy providers free-text model/voice, a shared response-format catalog, speed 0.25-4.0, empty options this slice)
- [x] #2 The character speech resolver (CharacterTTSRequestResolution) accepts an assignment on any of the seven providers and speaks it instead of refusing with assignment_invalid
- [x] #3 The profile service's save-as-profile eligibility gate (_selection_is_profile_safe / create_from_artifact) accepts a legacy-provider selection carrying real request provenance
- [x] #4 The profile store schema is versioned to v2 as a downgrade fence; existing v1 stores upgrade in place on open and on restore-candidate validation, routed through the repository's EXCLUSIVE lease path
- [x] #5 Legacy-provider profiles classify as unverified (not falsely available or unavailable) everywhere availability is observed, and that classification no longer forces every legacy-only page through an audio.cpp capability probe
- [x] #6 Playground preset adoption reports a legacy preset as unverified rather than a false unavailable
- [x] #7 A live, real-network exercise of the shipped code (TTS Playground generate against the real OpenAI API, a profile created from that result, assigned to a character, and spoken through Console's speak action) is recorded, including any defects the live path surfaced that unit tests did not
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the audio.cpp-only profile validation with a per-provider table (formats, speed bounds); keep audio_cpp's exact contract (task 1).
2. Lift the character resolver's provider refusal and the profile-service save-eligibility/native-capability gates to the seven-provider set; decouple availability observation from an audio.cpp-only capability probe (tasks 2, 2b).
3. Version the profile store schema to v2 as a downgrade fence with an in-place v1 upgrade, routed through the EXCLUSIVE lease path (task 4).
4. Make playground preset adoption and legacy classification report 'unverified' honestly instead of a false 'unavailable' (task 5); characterize the real adapter registry's configuration_revision for all six legacy providers.
5. Live-verify the shipped path against a real provider, amend ADR-028, file backlog follow-ups, run gates (task 6).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented across tasks 1, 2, 2b, 3, 4, 5 (commits 6aa21f7f6..b0e650e44) and closed out by this task (task 6: live verification, ADR-028 amendment, backlog hygiene, pre-ship gates). Summary: profile_types.py's PROFILE_PROVIDER_FORMATS/PROFILE_PROVIDER_IDS closed the provider set to seven (audio_cpp + openai/elevenlabs/kokoro/chatterbox/higgs/alltalk) with a per-provider contract table; the character resolver and profile-service save-eligibility gate were lifted to the same set; two emergent construction-time pins (TTSRequestedSelectionSnapshot, PortableTTSProfile) were found and lifted in task 2b along with decoupling observe_availability from an audio.cpp-only capability probe; the profile store gained a v2 downgrade fence with an in-place v1 upgrade routed through the EXCLUSIVE lease path; playground preset adoption now reports legacy profiles as unverified instead of a false unavailable, and a real-registry characterization proved configuration_revision works for all six legacy providers.

**Task 6 (this task's own close-out work):**

*Live verification* (real tmux TUI, scratch TLDW_CONFIG_PATH profile, real OpenAI key from repo-root openai-api-key.txt, no mocks): confirmed live, end to end, that clicking a character message's Console speak action against a real openai-provider profile assignment makes a real network call to OpenAI's TTS endpoint and plays the result back (app log: `TTSBackendManager: Creating backend for ID: openai_official_tts-1` -> `OpenAITTSBackend: Successfully completed TTS generation` -> `TTS playback action: play`). The profile-creation and character-assignment steps of that chain were completed via an honest, clearly-labeled in-process substitute (real TTSService.generate_audio_stream + real TTSProfileService.create_from_artifact/set_assignment, same scratch store, no mocks) after live verification found the real UI affordances for both steps are currently broken by two PRE-EXISTING/emergent defects this task did not fix (Task 6's scope is verification+hygiene+gates, not further implementation): the Playground's only real Generate path (_generate_studio_effective) never attaches request provenance for a legacy provider, so 'Save result as profile' is unreachable through the live UI for any of the six providers this slice added (filed TASK-2452); and the Roleplay Voice & Speech assignment Select silently refuses any profile whose availability isn't exactly 'available', so no 'unverified' legacy profile can be assigned through the live UI (filed TASK-2453). Both were confirmed, live, to be a UI-wiring gap rather than a backend gap: calling the real service methods directly succeeded, and reloading the live app showed the resulting profile/assignment rendered correctly and honestly (availability column read 'Unverified'; the Roleplay status line read 'Unverified . Used by 1 character. Refresh or repair the profile; the assignment is preserved.').

*ADR-028 amendment*: added a dated 2026-08-04 amendment block recording the seven-provider closed set, the per-provider contract, the v2 downgrade fence, the interim 'unverified' availability state, and the design spec's four-pin undercount (six gates actually closed: P1 character resolver, P2 profile-service gate, the two emergent typed-pin fixes, the availability-coupling fix, and playground-adoption honesty) plus the two gaps TASK-2452/2453 found live and left open.

*Backlog hygiene*: scanned every local worktree's backlog/tasks + backlog/drafts (os.listdir + regex, numeric sort) and every remote ref's backlog/ tree; true max was 2390 (local worktree voice-control-v2) vs 2364 (remotes). Filed this task at 2450 with generous headroom, plus TASK-2451 (sample-persona follow-up, spec ruling 1), TASK-2452 and TASK-2453 (the two live-found defects above). The backlog CLI's own auto-assignment (v1.44.0) offered TASK-2321 for the first task -- collided with the true max exactly as the lessons file warns; renumbered before filing anything else.

*Gates*: `ruff check` on the 23 touched files (git diff --name-only ab9105c9d..HEAD): clean. `ruff format --check`: 10 files were NOT already formatted under this environment's ruff 0.15.22 (pure line-wrap style drift from whichever ruff version each of tasks 1-5 ran under; confirmed AST-identical before/after via `ast.dump` for every reformatted file, so this task applied `ruff format` to make the gate green -- zero semantic change). Targeted suites: the 12 touched test files, `943 passed, 2 skipped` (the 2 skips are the slice's own deliberate 'options re-enabled in a later slice' markers); full `Tests/TTS/`, `2222 passed, 15 skipped` (all skips are optional-dependency/hardware gates, e.g. Chatterbox/Higgs/Kokoro-ONNX not installed). Repo-wide `pytest --collect-only`: `30067 tests collected`, zero collection errors (one pre-existing, unrelated skip note for `Tests/Notes/test_notes_api_integration.py`).

Full trace, the live-verification transcript, and the ADR amendment text are in `.superpowers/sdd/2026-08-04-voice-profiles-slice1-gate-lifting/task-6-report.md`.

**Task 6b (controller ruling, 2026-08-05): the two live-found UI-reachability gaps are fixed in-slice, not follow-ups.** Task 6's live verification found the backend correct but the user-facing capability unreachable end to end -- shipping with TASK-2452/2453 as deferred follow-ups would have shipped dead code: no user could create a legacy voice profile through the real Playground, and even a profile created another way could not be assigned to a character through the real Roleplay UI. Both fixed with the same TDD + mutation discipline as tasks 1-5 (see TASK-2452 and TASK-2453's own Implementation Notes for the full detail):

- **TASK-2452**: `_generate_studio_effective` (the Playground's real, only Generate dispatch path) now attaches `TTSRequestedSelectionSnapshot` provenance for every provider, not only `audio_cpp`, via a new shared `_build_requested_selection` helper factored out of `_generate_legacy`'s existing shape (same defensive `logger.warning` swallow, so a construction failure degrades eligibility rather than breaking generation).
- **TASK-2453**: found and fixed TWO independent stale gates, not one -- the Roleplay Voice & Speech widget's own client-side `Select` gate, AND a separate screen-side gate in `personas_screen.py`'s assignment worker that fixing the widget alone does not reach. Both now accept `"unverified"` alongside `"available"`, refusing only genuine `"unavailable"`; the widget's Edit/Repair button label was also corrected so an unverified-but-working assignment is never presented as needing repair.

Live re-verified end to end against a real OpenAI account, a fresh tmux session, and a fresh scratch profile -- no in-process substitutes this time: real Playground Generate -> real "Save result as profile" click -> real modal -> "Voice profile saved." -> visible in the Voice Profiles library as Unverified -> real Roleplay assignment via the real Select (persistence confirmed by navigating away and back) -> real Console speak action -> real OpenAI TTS call -> real audio playback (`afplay`). Full trace in `.superpowers/sdd/2026-08-04-voice-profiles-slice1-gate-lifting/task-6b-report.md`.
<!-- SECTION:NOTES:END -->
