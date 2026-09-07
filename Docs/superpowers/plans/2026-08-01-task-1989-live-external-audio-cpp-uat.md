# TASK-1989 Live External audio.cpp UAT Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to execute this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the live release gate for the external audio.cpp Settings,
Speech Studio, character-profile, and response-playback journeys without
making Chatbook responsible for the server process.

**Architecture:** Run the real Chatbook application through the repository's
`tldw-serve` browser surface with isolated first-run XDG config/data/cache roots.
Drive only visible UI actions, use the user's already-running external
`audiocpp_server`, and compare bounded config snapshots outside the UI to prove
scope isolation. Record headless complete-WAV evidence separately from the
human audible confirmation.

**Tech Stack:** Python 3.12 virtualenv, Textual, `tldw-serve`, Playwright CLI,
the native external audio.cpp HTTP adapter, TOML configuration, SQLite, WAV
structural inspection, and macOS `afplay` through Chatbook's existing playback
event path.

## Global Constraints

- Do not download, launch, adopt, restart, supervise, signal, or stop
  `audiocpp_server`; server stop/start checkpoints belong to the user.
- Do not expose or persist a binary path, `server.json`, bind address, model
  contents, or the user-supplied model path.
- Use only synthetic non-secret synthesis text, character names, endpoints,
  credentials, screenshots, and diagnostics.
- Keep complete WAV responses behind the existing asynchronous adapter
  interface; do not claim incremental streaming.
- Do not silently pass an unavailable legacy provider. Record it as unavailable
  and retain the deterministic request-shape evidence from TASK-1988.
- Preserve the user's normal Chatbook config and data by using task-specific
  temporary XDG roots.
- Stop only Chatbook/test harness processes created by this task. Never stop or
  signal the user-owned audio.cpp process.

ADR required: yes
ADR path: `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`
Reason: This UAT verifies the accepted external-only runtime boundary,
four-owner persistence model, precedence, exact-selection, credential,
navigation, and playback contracts. It makes no new architecture decision.

---

### Task 1: Establish an isolated, privacy-safe live harness

**Files:**

- Create: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`
- Runtime-only: `/tmp/tldw-task-1989-uat/`
- Create (privacy-reviewed evidence only): `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/screenshots/`

**Interfaces:**

- Consumes: user-owned external audio.cpp listener and the TASK-1988 release
  evidence contract.
- Produces: one isolated Chatbook process, one browser session, redacted
  baseline hashes, and an evidence ledger with no live secret/model path.

- [x] Verify the external process and listener read-only with `ps`, `lsof`, and
  safe GET requests. Record only canonical provider ID, loopback/remote class,
  HTTP status, and bounded catalog identifiers; do not record command lines or
  model paths in committed evidence.
- [x] Confirm `npx`, `tldw-serve`, and Playwright availability.
- [x] Create task-specific HOME/XDG roots and launch the real app with a
  synthetic environment-managed credential:

  ```bash
  /usr/bin/env \
    TLDW_CONFIG_PATH=/tmp/tldw-task-1989-uat/config.toml \
    XDG_CONFIG_HOME=/tmp/tldw-task-1989-uat/config \
    XDG_DATA_HOME=/tmp/tldw-task-1989-uat/data \
    XDG_CACHE_HOME=/tmp/tldw-task-1989-uat/cache \
    OPENAI_API_KEY=task-1989-synthetic-not-a-secret \
    PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/tts-slice4-portability \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve \
      --host 127.0.0.1 --port 8768
  ```

- [x] Open the browser-served app with Playwright, snapshot before every
  interaction, and re-snapshot after every state change. Stage raw captures
  under `/tmp/tldw-task-1989-uat/playwright/` until privacy review, then retain
  only approved evidence under
  `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/screenshots/`.
- [x] Record starting config/store hashes and prove the isolated config has no
  audio.cpp values or Studio overrides.

### Task 2: Execute UAT-01 first-time setup and audible response playback

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`

**Interfaces:**

- Consumes: isolated live app and the user-supplied external origin from Task 1.
- Produces: Saved/Not checked evidence, explicit Test/Refresh evidence, a valid
  complete WAV, response-control playback invocation, and human confirmation.

- [x] Start a timer at the Settings destination, use visible Settings search
  only, and reach `Speech & TTS` by searching `audio.cpp` within 60 seconds.
- [x] Select audio.cpp under `Configure Provider`, enter only the canonical
  origin, save, and record separate `Saved` and `Not checked` labels. Confirm
  the UI contains no managed-process control.
- [x] Use `Open Speech Lab`, invoke `Test` and `Refresh` explicitly, and record
  the resulting Ready/configuration/catalog revisions without raw response
  bodies.
- [x] Generate the synthetic line `Lantern check complete; the harbor is
  ready.` and verify the artifact is a complete RIFF/WAVE file before playback.
- [x] Create or use a synthetic assistant character response in Console or
  Roleplay, invoke that response's TTS control, then invoke its playback control.
- [x] Ask the user to confirm that the expected synthetic line was audible.
  Do not mark human playback passing from button state, WAV bytes, process
  creation, or headless handoff alone.

### Task 3: Execute UAT-02 offline save and user-controlled recovery

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`

**Interfaces:**

- Consumes: saved canonical audio.cpp origin.
- Produces: Saved+Unavailable evidence followed by Ready at the same persisted
  revision after user-owned recovery.

- [x] Ask the user to stop their external audio.cpp server; do not send any
  process signal or lifecycle request from Chatbook or the harness.
- [x] Explicitly Test in Lab and record `Saved` plus `Unavailable`, unchanged
  provider selection, no fallback generation, and unchanged configured origin.
- [x] Ask the user to start the same external server again.
- [x] Explicitly Test again and record `Ready` without editing or re-saving the
  origin.

### Task 4: Execute UAT-03 exact and dynamic catalog choices

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`

**Interfaces:**

- Consumes: accepted live model/voice catalog observations.
- Produces: exact round-trip, dynamic-mode persistence, and missing-exact
  no-substitution evidence.

- [x] Refresh models and voices, record whether the live catalog is genuinely
  multi-model, and select exact model/voice values only from accepted results.
- [x] Navigate Settings → Lab → Settings and verify exact case-sensitive values
  return unchanged.
- [x] Select `First available` plus `Server default`, save, generate once, and
  prove persisted config still contains the dynamic policy rather than the
  resolved ephemeral identifiers.
- [x] Stage a synthetic missing exact identifier in the isolated config/UI,
  remount, and verify it remains visible and blocks/substitutes nothing. Restore
  the valid selection through visible UI before continuing.

### Task 5: Execute UAT-04 and UAT-05 Studio persistence and reset-by-deletion

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`

**Interfaces:**

- Consumes: a valid live audio.cpp global selection and at least two
  representable exact/dynamic choices.
- Produces: Studio remount persistence, global/normal-request isolation, and
  reset-by-deletion/inheritance evidence.

- [x] Save a Studio-only selection, capture the `[speech_studio]`-only diff,
  leave/remount Studio, and verify the exact Studio choice returns.
- [x] Confirm global Settings and one normal non-Studio effective selection are
  unchanged.
- [x] Change and save a global default, verify the Studio override still wins,
  then use `Reset to Global`.
- [x] Change the global default again and verify Studio inherits it with no
  copied Studio selection value. Record the bounded TOML diff.

### Task 6: Execute UAT-06 and UAT-07 character precedence and preview safety

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`

**Interfaces:**

- Consumes: synthetic canonical characters and exact supported audio.cpp
  profiles created through current Roleplay/Profile UI.
- Produces: assigned-character precedence, unassigned global fallback, safe
  preview, explicit adoption, and store-isolation evidence.

- [x] Create two synthetic characters, create one exact audio.cpp TTS profile,
  and assign it to exactly one canonical character.
- [x] Generate/play an assigned character response and an unassigned response;
  record the non-secret source metadata proving profile vs global resolution.
- [x] Confirm Studio config hash/content is unchanged by both roleplay flows.
- [x] Open the profile in Studio, generate/play a preview, leave unadopted, and
  prove saved Studio preferences did not change.
- [x] Repeat, select `Adopt as Studio Preferences`, save, and prove only the
  Studio namespace changed.

### Task 7: Execute UAT-08 through UAT-10 compatibility and status journeys

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`

**Interfaces:**

- Consumes: the synthetic environment credential, retained legacy config
  fixtures, live external audio.cpp, and current optional-dependency state.
- Produces: credential-boundary, legacy compatibility, and independent-status
  evidence with unavailable live providers explicitly recorded.

- [x] Open OpenAI global setup and verify only `Environment` plus
  `OPENAI_API_KEY` are displayed. Ordinary Save must not write a local key;
  explicit local-fallback Clear must leave the process environment effective.
- [x] Visit OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk global
  forms; verify retained saved/init fields and Studio-supported tuning. Run
  available live smoke only when already configured; otherwise record
  `Unavailable/not live-tested` and cite the passing TASK-1988 request-shape
  fixture rather than claiming a live pass.
- [x] With absent unrelated local TTS/STT dependencies, verify their truthful
  independent rows while audio.cpp remains Ready, generates, and plays.

### Task 8: Privacy review, defect gate, and closeout

**Files:**

- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/release-evidence.md`
- Modify: `Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/live-uat.md`
- Modify: `backlog/tasks/task-1989 - Run-live-external-audio.cpp-Settings-Studio-and-roleplay-UAT.md`

**Interfaces:**

- Consumes: all UAT observations and human audible confirmation.
- Produces: final release evidence, reviewed findings, and a completed Backlog
  task only if every acceptance criterion is satisfied.

- [x] Inspect every screenshot, trace, config diff, log excerpt, and diagnostic
  for secrets, live model paths/contents, private text, raw URLs, and raw
  provider bodies. Delete unsafe captures; commit only synthetic/redacted
  evidence.
- [x] Record exact branch/commit, app/server origin classification, provider
  configuration/runtime/catalog revisions, WAV structure, playback-control
  action, and the user's audible confirmation separately.
- [x] Classify every finding. Fix P0/P1 in scope with a failing regression and
  rerun the affected live journey; reject a P1 only with technical evidence and
  explicit user approval. Defer a non-AC P2 only after creating a Backlog task.
- [x] Update the release-evidence rows from `UAT pending TASK-1989` to the exact
  result, preserving separate automated vs human evidence labels.
- [x] Run focused regressions, Ruff, compileall, evidence checks, and
  `git diff --check`; request independent review.
- [x] Check all TASK-1989 acceptance criteria, add concise Implementation Notes,
  mark Done through Backlog CLI, and commit the UAT evidence. Stop only the
  task-owned Chatbook/browser harness.
