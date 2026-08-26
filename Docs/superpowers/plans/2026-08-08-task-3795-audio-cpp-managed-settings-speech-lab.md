# TASK-3795 — Managed audio.cpp Settings and Speech Lab Implementation Plan

**Goal:** Expose the completed managed audio.cpp runtime through the canonical
global Settings and Speech Lab surfaces without changing External-mode or
complete-WAV behavior.

**Architecture:** Extend the existing explicit audio.cpp Settings form and
Speech Playground rather than introducing a provider-form framework. Persist a
full two-mode audio.cpp Settings mapping, but project only the selected mode
into the adapter registry. Add one passive, coherent service observation for
the Speech Lab runtime card; lifecycle mutations continue to use the existing
service APIs. The card remains presentation-only, and the pane owns async
workers, stale-result fencing, notifications, and focus.

**Tech stack:** Python 3.11, Textual 8, Pydantic, pytest/pytest-asyncio, httpx
test transports.

## Governance

- ADR required: no new ADR.
- ADR path: `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`,
  `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`, and
  `backlog/decisions/040-speech-lab-current-result-and-auto-play.md`.
- Reason: these accepted ADRs already decide provider/process ownership,
  global-versus-Studio persistence, lifecycle UI ownership, complete-WAV
  playback, and the current-result hierarchy. This task implements the already
  approved Slice 5 and makes no new architectural choice.

## Scope guard

- Do not download, install, build, update, or verify audio.cpp.
- Do not generate, edit, or reinterpret `server.json` beyond the existing
  strict host/port launch validation.
- Do not add automatic restart, process adoption, multiple instances, arbitrary
  arguments, environment overrides, or a generic subprocess/provider UI
  framework.
- Do not move durable provider fields into Studio preferences or character TTS
  profiles.
- Keep saving, mounting, status observation, and diagnostics passive.

## Task 1: Preserve both modes in global Settings without changing runtime projection

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Test: `Tests/UI/test_settings_audio_cpp_experience_model.py`
- Test: `Tests/TTS/test_stts_settings_reconfiguration.py`

1. Replace the Slice 4 External-only expectations with failing tests for the
   explicit mode and managed fields.
2. Add round-trip tests proving a Managed save retains the dormant External
   origin, an External save retains previously stored managed values, and a
   legacy External mapping does not acquire unused managed keys.
3. Add active-mode-only validation tests: malformed dormant fields do not
   block the selected mode, but become field-specific failures when selected.
4. Define the explicit full audio.cpp global field inventory and load the
   stored dormant values without passing them to runtime validation.
5. Build one durable full mapping while continuing to pass the existing
   active-mode projection to `TTSService` publication.
6. Re-run the focused model and settings-publication tests.

## Task 2: Add side-effect-free managed save validation and draft detection

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Test: `Tests/UI/test_settings_audio_cpp_experience_model.py`
- Test: `Tests/UI/test_settings_speech_tts_panel.py`

1. Write failing tests for binary detection success/failure, path-picker draft
   updates, field-specific binary/JSON validation, and preservation of the
   previous draft when detection finds nothing.
2. Add a narrow validation adapter around
   `validate_audio_cpp_managed_launch()` that maps bounded existing failure
   codes to safe field copy without echoing paths, JSON, or exception strings.
3. Revalidate Managed artifacts on Save before posting the persistence event;
   assert no launch, probe, catalog, or synthesis seam is invoked.
4. Implement explicit `shutil.which("audiocpp_server")` detection only from
   the user action and fill the draft without saving.
5. Configure the existing file picker for an executable and JSON file without
   reading or mutating user artifacts from picker actions.
6. Re-run panel/model tests and mutation-test the no-launch and draft-preserve
   guards.

## Task 3: Render the explicit External/Managed Settings experience

**Files:**

- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Generate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_settings_speech_tts_panel.py`

1. Write Textual pilot tests for mode switching, focus stability, visible
   primary fields, Advanced lifecycle/common limits, trust/privacy/help copy,
   and Settings-to-Lab navigation.
2. Mount one mode selector plus stable External and Managed sections, updating
   display/disabled state in place so focused controls are not replaced by
   status refreshes.
3. Add the trust notice, detected-binary action, two path pickers, loopback and
   working-directory guidance, and lifecycle timing fields under Advanced.
4. Keep common timeouts and safety limits discoverable, and update Settings
   deep-link focus to the mode/active primary field.
5. Add responsive styles from the source partial, regenerate the modular CSS
   bundle, and assert real compositor geometry at supported widths.
6. Re-run focused Settings tests and the CSS integrity test.

## Task 4: Expose one coherent passive runtime observation

**Files:**

- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Test: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Test: `Tests/TTS/test_tts_registry_service.py`

1. Write failing tests for an observation containing saved/applied modes and
   provider generations plus the process snapshot, including staged Managed,
   staged External, crash, and closed-service states.
2. Add one immutable public `AudioCppRuntimeObservation` and an async
   `TTSService` snapshot method that reads existing registry/supervisor state
   under the established publication/admission ordering.
3. Prove the method performs no adapter materialization, launch, network I/O,
   catalog refresh, or configuration mutation and cannot publish a mixed
   saved/applied/process relation.
4. Keep full configured paths out of repr/log/error surfaces; expose only the
   exact read-only values needed by an explicit UI details disclosure.
5. Re-run focused service, ownership, lifecycle, and privacy tests.

## Task 5: Add the Speech Lab audio.cpp runtime card

**Files:**

- Add: `tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Generate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_speech_playground_pane_lifecycle.py`

1. Write pure projection tests for saved/applied/process generations, pending
   relations, process/capability labels, endpoint/catalog freshness, and every
   action label/enabled reason in ML-UX-011.
2. Write pilot tests proving the card is visible only for audio.cpp, links to
   canonical global Settings, mounts without launching, and updates existing
   controls in place without losing focus.
3. Implement one presentation widget with non-color status cues, concise
   primary status, a read-only details disclosure, and always-mounted lifecycle
   actions.
4. Keep the current result and playback actions above secondary diagnostics;
   label process shutdown distinctly from playback Stop.
5. Poll only the passive observation while audio.cpp is selected and reject
   stale results by observation/provider/process generations.
6. Regenerate CSS and assert 120x40, 80x24, and narrow-scroll geometry.

## Task 6: Wire deliberate lifecycle actions without blocking Textual

**Files:**

- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `tldw_chatbook/UI/Speech/speech_catalog_mixin.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playback_mixin.py`
- Test: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Test: `Tests/UI/test_speech_playground_pane.py`

1. Write failing interleaving tests for Start/Test, Test, Restart/Apply,
   Apply/Stop, Shutdown, Refresh, Generate, detached waiters, superseded
   observations, and operation failures.
2. Route the card actions only through `TTSService.start_and_test_audio_cpp()`,
   `restart_audio_cpp()`, and `shutdown_audio_cpp()`; reuse existing catalog
   refresh and generation paths.
3. Run accepted work in retained Textual workers, render Starting/Draining/
   Stopping immediately, disable incompatible controls with reasons, and keep
   focused buttons mounted.
4. Refresh card/catalog state after accepted completion or failure without
   synthesizing hidden audio or changing the current result.
5. Preserve External Test, Refresh, Generate, playback, and complete-WAV
   semantics.
6. Mutation-test busy/admission/stale-result guards and re-run focused pane and
   app-ownership tests.

## Task 7: Render bounded managed diagnostics safely

**Files:**

- Modify: `tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py`
- Test: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Test: `Tests/TTS/test_tts_logging_privacy.py`

1. Write tests for collapsed-by-default diagnostics, process generation,
   stdout/stderr labels, dropped-output count, sensitivity/restart copy, and
   inert open/close/scroll/copy interactions.
2. Render only the supervisor's already bounded and sanitized snapshot;
   neither log nor persist it, and never concatenate exception context or raw
   configuration values into primary errors.
3. Clear visible prior-generation lines when a new generation becomes active
   and preserve post-exit diagnostics until the next start.
4. Run exception-graph and caplog privacy scans with synthetic path, JSON,
   prompt, credential, and child-output markers.

## Task 8: Document setup, ownership, and recovery

**Files:**

- Modify: the existing Speech/TTS user guide selected by the documentation
  inventory
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Test: the relevant documentation/link checks

1. Add first-time setup steps for a user-provided prebuilt binary and existing
   `server.json`, including trust, loopback, working-directory, and no-edit
   ownership boundaries.
2. Explain lazy launch, one active child, saved versus applied generations,
   Restart & Apply, shutdown, crash recovery, diagnostics privacy, and switching
   to External mode.
3. Keep examples free of user-specific absolute paths, credentials, prompt
   text, raw diagnostics, or promises of automatic install/restart.
4. Run doc/link/privacy checks.

## Task 9: Verify the integrated automated surface

1. Run focused model, panel, pane, service, managed lifecycle, settings
   publication, playback, and privacy tests.
2. Run affected broader TTS, Settings, Console, Roleplay, and app-ownership
   suites.
3. Run Ruff, formatter check, mypy for changed modules, compileall, CSS build
   integrity, `git diff --check`, boundary/privacy scans, and cumulative
   `origin/dev...HEAD` inspection.
4. Record any unchanged dev-baseline failures from the identical command;
   do not waive new failures.

## Task 10: Perform real-binary first-time-user UAT and close the task

1. Use an isolated temporary Chatbook config/profile and user-provided
   `audiocpp_server` plus existing `server.json`; never modify those artifacts.
2. Prove Settings Save does not launch, Start/Test owns exactly one PID, the
   configured multi-model catalog appears, and a character-roleplay response
   produces a structurally valid playable WAV.
3. Ask the user for the sole subjective audible-playback confirmation; record
   only sanitized binary/version, state/generation, catalog IDs, WAV metadata,
   and cleanup facts.
4. Prove saved-while-running Restart & Apply, unexpected-exit recovery,
   explicit shutdown/lazy restart, External apply, and final app cleanup.
5. Check every acceptance criterion and DoD item, add concise Implementation
   Notes and any incident-backed reusable lesson, set TASK-3795 Done, then
   request final code review before branch integration.
