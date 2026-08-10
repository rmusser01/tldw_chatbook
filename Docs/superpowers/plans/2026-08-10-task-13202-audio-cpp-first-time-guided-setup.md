# TASK-13202 First-Time Guided audio.cpp Setup Implementation Plan

> **For Codex:** Execute this plan test-first in order. Keep Model Library,
> downloads, clone-profile authoring, new model-family recipes, and managed-lifecycle
> redesign out of this PR.

**Goal:** Let a first-time user select a separately installed audio.cpp server and
reviewed local model packages in Global Settings, then deliberately start the one
managed child and hear, replay, regenerate, and save a complete WAV in Speech Lab
without editing JSON.

**Architecture:** Extend the existing typed Guided settings and sealed recipe/scanner
foundation through the canonical Global Settings surface. Save remains a pure durable
configuration operation. Speech Lab owns deliberate runtime actions and projects its
single primary control from an immutable, path-safe runtime observation. The existing
adapter registry, managed supervisor, full-WAV response, captured-selection, and
Studio autoplay seams remain authoritative.

**Tech Stack:** Python 3.11, Textual 8, Pydantic 2, stdlib `asyncio`, existing TTS
adapter/supervisor services, pytest, and the modular Textual CSS build.

**ADR required:** no

**ADR path:** N/A (`backlog/decisions/039-global-and-studio-tts-settings-ownership.md`,
`backlog/decisions/040-speech-lab-current-result-and-auto-play.md`, and
`backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md` apply)

**Reason:** This task implements the approved guided onboarding/product seam without
changing settings ownership, Studio preference ownership, generated-configuration
ownership, or managed-lifecycle boundaries.

---

### Task 1: Full guided Settings round-trip and side-effect-free validation

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_speech_tts.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_guided_config.py` only if a missing bounded validation helper is proven
- Modify: `Tests/UI/test_settings_speech_tts_model.py`
- Modify: `Tests/TTS/test_stts_settings_reconfiguration.py`

- [x] Write failing tests that load, edit, validate, save, and reload External,
      manual `server.json`, and Guided settings with all dormant source values intact.
- [x] Add negative fixtures for invalid executable/package/default/backend tuples and
      a side-effect spy proving Save performs no launch, socket, HTTP, catalog,
      synthesis, generated-artifact, or model write.
- [x] Run the focused tests and confirm failures describe the missing guided projection.
- [x] Extend the canonical Settings inventory/default/load/validation path through
      `AudioCppSettingsConfig`, retaining the manual runtime projection unchanged.
- [x] Re-run focused tests and the guided foundation tests.

### Task 2: Guided Global Settings controls and package review

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss` if layout rules are required
- Modify: `Tests/UI/test_settings_speech_tts_panel.py`

- [x] Write failing mounted tests for the External / manual JSON / Guided hierarchy,
      detected-or-browsed server selection, directory browsing, bounded package scan,
      exact candidate review/removal, default model and backend selection, dormant
      values, validation recovery, keyboard focus, and narrow layout.
- [x] Add stale-scan/cancellation and unmount regressions; only the latest explicit
      root may update the visible draft.
- [x] Build the smallest review UI that exposes exact family/variant, task,
      evidence/compatibility, public model ID, path-safe package summary, lazy
      load/resident-memory truth, and recovery guidance without download/support
      overclaims.
- [x] Make successful Save announce `Configuration saved — ready to test` and route
      focus to Speech Lab's one dynamic primary action, never to Refresh.
- [x] Re-run Settings panel/model tests.

### Task 3: Immutable Speech Lab runtime observation and primary action

**Files:**
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`

- [x] Write failing pure and mounted tests for path-safe Guided observation fields
      and the exact labels/operations/reasons/tooltips/focus targets: Start & Generate
      Sample, Restart & Apply Settings, Retry Sample, Test Connection, and Shutdown.
- [x] Prove a click executes the exact visible immutable projection, including
      provider switches, newer observations, late lifecycle/catalog results, and
      failure recovery; retain only one restart action.
- [x] Extend the runtime observation with only the safe Guided facts needed to derive
      the primary action and render manual/External/Guided details truthfully.
- [x] Re-run lifecycle/runtime-card and managed integration tests.

### Task 4: Deliberate first sample and complete current-result experience

**Files:**
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playback_mixin.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `Tests/UI/test_speech_playground_pane.py`
- Modify: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Modify: relevant `Tests/TTS/test_stts_*generation*.py` and adapter tests

- [x] Write failing tests that the Guided primary deliberately starts/tests once,
      verifies the exact selected catalog model, and only then invokes the existing
      full-WAV generation path if provider/request/config fences still match.
- [x] Add current-result regressions for prominent Play/Pause, structural WAV status,
      duration, path-free provider/model/voice/config/process provenance, Generate
      again, Save WAV, last-good-result retention, and Studio-only optional autoplay.
- [x] Implement the combined action by composing existing lifecycle and synthesis
      seams; do not add streaming, history, comparison, or another player.
- [x] Re-run Speech Lab, playback, adapter, and STTS generation tests.

### Task 5: One-child multi-model and captured consumer selections

**Files:**
- Modify tests first: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Modify tests first: `Tests/TTS/test_console_audio_cpp_native.py`
- Modify tests first: the existing first-time Roleplay/character speech UAT test
- Modify production files only if a regression exposes a TASK-13202 gap

- [x] Add regressions proving multiple accepted models remain registered, selecting a
      second model reuses the same child, and visible copy warns loaded models may
      remain resident until explicit shutdown.
- [x] Prove Console uses the exact captured global selection and Roleplay uses the
      exact captured character override, with no passive-browse launch.
- [x] Prove provider switches and late lifecycle/catalog/generation completions cannot
      relabel, disable, or execute a stale visible action.
- [x] Re-run the focused managed, Console, and Roleplay suites.

### Task 6: Accessibility, responsive polish, and CSS bundle

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Modify: `tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playback_mixin.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: focused mounted UI tests

- [x] Verify keyboard-only traversal, explicit labels and non-color status, current
      disabled reasons, hidden-field expansion/focus, focus restoration, bounded
      scrollable diagnostics, and non-spamming live announcements.
- [x] Verify the supported narrow width keeps status, primary action, player, and
      recovery controls usable without clipped or duplicate actions.
- [x] Regenerate the modular CSS bundle and run bundle sync plus the Impeccable
      detector against the changed UI targets.

### Task 7: User documentation and pinned first-time UAT

**Files:**
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Create: `Docs/superpowers/qa/audio-cpp-guided-first-run-2026-08-10/README.md`
- Create: pinned macOS and Linux UAT evidence beneath that QA directory

- [x] Document that users install audio.cpp separately, then choose a server and
      reviewed package without editing JSON; distinguish Global Settings from Studio
      preferences and Guided from External/manual JSON.
- [x] Run the pinned first-time macOS journey with isolated Chatbook config/data and a
      task-owned child only; record exact app/server/model versions and never touch
      unrelated PID 84574.
- [x] Record package selection, no-side-effect Save, complete WAV, audible playback,
      multi-model reuse, restart/apply, crash recovery, shutdown, and unchanged
      External/manual behavior. If audible confirmation needs a human, pause only at
      the actual playback checkpoint.
- [x] Record a reproducible pinned Linux journey or CI/harness evidence for the same
      supported paths, clearly separating automated structural evidence from human
      audible evidence.

### Task 8: Verification, review, and task closeout

**Files:**
- Modify: this plan checklist
- Modify: `backlog/tasks/task-13202 - Deliver-the-first-time-guided-audio.cpp-setup-and-sample-flow.md`
- Modify: a relevant `backlog/docs/lessons-*.md` only if this task produces an evidenced reusable lesson

- [x] Run focused tests after each batch, then the complete affected TTS/UI suites,
      Ruff on changed Python files, CSS bundle sync, `git diff --check`, and any
      proportionate static/privacy checks.
- [x] Self-review every changed line against all nine acceptance criteria and the
      three governing ADRs; fix all confirmed issues before closeout.
- [x] Update this checklist, check every acceptance criterion, add concise
      Implementation Notes with exact verification/UAT evidence and ADR decision,
      and set TASK-13202 Done only when the repository Definition of Done is met.
