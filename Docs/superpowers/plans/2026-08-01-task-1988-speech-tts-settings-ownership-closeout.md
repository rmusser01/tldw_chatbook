# TASK-1988 — Speech & TTS Ownership Closeout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Prove the already-landed global, Studio, character, and runtime TTS
ownership model as one accessible, privacy-safe, fake-only journey, then leave
a mechanically complete release-evidence map for live UAT.

**Architecture:** Keep the accepted four-owner boundary unchanged. Exercise
the real Settings, Studio, navigation, resolver, adapter-registry, generation,
artifact, and Console playback seams with injected stores and provider fakes.
Add only the smallest product correction when a new closeout test exposes an
actual defect; otherwise this task adds verification and release evidence.

**Tech stack:** Python 3.11+, pytest/pytest-asyncio, Textual pilot tests,
existing TTS adapter/service fakes, TOML-backed configuration fixtures, and a
pinned structurally valid WAV fixture generated in memory by the test suite.

**Global constraints:** No network access, provider process, model download,
audio hardware, managed audio.cpp lifecycle, new provider, native migration of
a legacy provider, character-profile redesign, hidden discovery, automatic
speech, or incremental-streaming claim. Synthetic test text and non-secret
loopback/example endpoints only.

ADR required: yes
ADR path: `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`
Reason: TASK-1988 verifies the accepted global/Studio ownership, persistence,
precedence, revision, privacy, migration, and navigation boundaries. No new
ADR is needed because this closeout must not introduce a new architecture or
product boundary.

## Task 1: Establish the closeout and evidence contracts

**Files:**

- Create: `Tests/TTS/test_speech_tts_release_evidence.py`
- Create: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/release-evidence.md`

1. Add a failing parser test that extracts every IA, OWN, CFG, CAT, STATE,
   MIG, SEC, and A11Y requirement heading from the approved PRD.
2. Require the release-evidence document to contain exactly one row per
   requirement ID, at least one repository-relative automated test node or
   explicit UAT journey per row, and no missing, duplicate, or unknown IDs.
3. Seed the document from the focused TASK-1981 through TASK-1987 coverage and
   mark TASK-1989 live audible playback as pending rather than claiming it was
   automated.
4. Run:

   ```bash
   pytest Tests/TTS/test_speech_tts_release_evidence.py -q
   ```

## Task 2: Close the accessibility and narrow-layout gaps

**Files:**

- Create: `Tests/UI/test_speech_tts_settings_ownership_closeout.py`
- Modify if a test exposes a defect:
  `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Modify if a test exposes a defect:
  `tldw_chatbook/UI/Speech/speech_settings_pane.py`
- Modify if a test exposes a defect:
  `tldw_chatbook/css/tldw_cli_modular.tcss`

1. Add Textual pilot tests at normal and supported narrow sizes for both
   editors. Assert every interactive field/action has stable visible label or
   tooltip/help association, focus order follows the rendered order, state is
   readable without color, and disabled audio.cpp format/speed controls name
   the contract reason and recovery path.
2. Drive keyboard-only search, setup, validation recovery, Save, dirty-draft
   Cancel, Studio Save/Reset, and cross-surface navigation. Assert the status
   mechanism updates without stealing focus.
3. Assert primary controls remain inside the viewport width and are reachable
   through vertical scrolling with no horizontal-scroll dependency.
4. Implement only the label/help, focus, announcement, or responsive-layout
   corrections needed to make those tests pass.
5. Run:

   ```bash
   pytest Tests/UI/test_speech_tts_settings_ownership_closeout.py -q
   ```

## Task 3: Add the deterministic first-time fake journey

**Files:**

- Modify: `Tests/UI/test_speech_tts_settings_ownership_closeout.py`
- Reuse: `Tests/TTS/adapter_fakes.py`
- Reuse: `Tests/TTS/fixtures/audio_cpp_http_v1/`

1. Add a failing end-to-end Textual harness that searches Settings for
   audio.cpp, enters and locally saves a canonical fake external origin, and
   proves the setup phase invoked no provider operation.
2. Follow the bounded Settings-to-Lab target, explicitly run fake readiness,
   model, and voice refresh operations, then follow the return target and
   prove provider/exact-selection context is restored without automatic work.
3. Generate a small valid complete WAV response through the native async
   adapter interface and hand the resulting artifact to the existing Console
   or Roleplay completion/playback control. Validate RIFF/WAVE structure,
   immutable non-secret provenance, and the playback event/path handoff; do
   not assert audible output or incremental playback.
4. Run the closeout UI test file and the existing native Console/playground
   handoff tests.

## Task 4: Prove cross-owner, privacy, migration, and legacy invariants

**Files:**

- Create: `Tests/TTS/test_speech_tts_settings_ownership_hardening.py`
- Modify only if failures expose defects:
  `tldw_chatbook/TTS/studio_preferences.py`
- Modify only if failures expose defects:
  `tldw_chatbook/TTS/effective_settings.py`
- Modify only if failures expose defects:
  `tldw_chatbook/TTS/legacy_bridge.py`
- Modify only if failures expose defects: bounded Speech status/navigation or
  artifact-provenance modules under `tldw_chatbook/UI/Speech/` and
  `tldw_chatbook/Event_Handlers/STTS_Events/`

1. Add cross-owner tests proving Studio Save and Reset touch only
   `[speech_studio]`, trigger no adapter reconfiguration, inherit later global
   values after deletion, and leave character profile/assignment repositories
   unchanged through preview and assigned-roleplay resolution.
2. Use unique sentinel credentials, masked strings, environment values,
   synthesis text, response bodies, URL material, and exception text. Scan all
   produced stores, navigation/status/catalog values, diagnostics, metrics,
   migration results, caches, and artifact provenance and require every
   sentinel to be absent.
3. Add rollback tests for idempotent migration, field-level malformed-data
   isolation, legacy-key compatibility, an older global reader ignoring
   `[speech_studio]`, and an empty/disabled Studio read resolving identically
   to prior global behavior without rewriting the config.
4. Pin representative request shapes for OpenAI, ElevenLabs, Kokoro,
   Chatterbox, Higgs, and AllTalk, plus the rule that approximate catalogs
   cannot invalidate an exact value by omission.
5. Run:

   ```bash
   pytest Tests/TTS/test_speech_tts_settings_ownership_hardening.py -q
   ```

## Task 5: Exercise the cross-slice stale-result race

**Files:**

- Modify: `Tests/UI/test_speech_tts_settings_ownership_closeout.py`
- Modify: `Tests/TTS/test_speech_tts_settings_ownership_hardening.py`
- Modify production revision/status/artifact code only when a deterministic
  race test demonstrates stale publication.

1. Add controlled `asyncio.Event` barriers around global save and targeted
   reconfiguration, Studio save, catalog/voice refresh, navigation, native
   generation, and playback handoff.
2. Complete operations out of order and assert older configuration/catalog/
   model/request revisions cannot replace the current editor, status, exact
   selection, or artifact.
3. Assert a completed WAV remains playable after newer configuration, Studio,
   status, and navigation changes.
4. Repeat the focused race tests enough times to catch order dependence, then
   run the neighboring TASK-1984 through TASK-1987 race suites.

## Task 6: Finish the evidence record and verification gate

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/release-evidence.md`
- Modify: `backlog/tasks/task-1988 - Harden-Speech-and-TTS-settings-ownership-end-to-end.md`

1. Replace every provisional evidence row with the exact passing test node or
   TASK-1989 manual UAT journey. Record that CI used fakes only and distinguish
   complete-WAV/playback-handoff proof from pending human audible proof.
2. Run the new closeout suites, all focused TASK-1981 through TASK-1987 suites,
   legacy-provider and Console/playback regressions, Ruff, compileall, and
   `git diff --check`.
3. Self-review for PRD coverage, secret/synthesis-text leakage, managed-server
   scope creep, placeholders, path accuracy, and type/interface consistency.
4. Request an independent code review. Address every priority-zero/one issue
   and any priority-two issue that violates acceptance criteria.
5. Check all TASK-1988 acceptance criteria, add concise Implementation Notes
   with the ADR and verification commands, and mark the task Done only after
   the complete gate passes.
