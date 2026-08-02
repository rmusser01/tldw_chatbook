# TASK-1700 Speech Lab Current-result UX Implementation Plan

**Goal:** Make the complete-WAV result immediately playable and understandable,
with optional Studio-only auto-play, without changing adapter or provider
ownership.

ADR required: yes
ADR path: `backlog/decisions/040-speech-lab-current-result-and-auto-play.md`
Reason: The approved remediation adds persistent Studio state and establishes a
long-lived result interaction contract.

## 1. Studio auto-play persistence

- Add failing tests for off-by-default, sparse true round-trip, invalid-value
  recovery, and Reset to Global.
- Add the additive boolean to `StudioTTSPreferencesSnapshot`, parsing, and
  sparse serialization.
- Run the focused preference suite.

## 2. Studio preference control

- Add failing UI tests for visible Studio-only scope copy, load/save/dirty
  behavior, and reset.
- Add an explicit switch and consequence label to Speech Studio Preferences.
- Run the focused settings-pane suite.

## 3. Current-result interaction

- Add failing layout regressions that check the action strip against the result
  pane and viewport after an artifact is delivered at supported sizes.
- Add failing tests for known/unknown duration, temporary ownership copy,
  default focus, and opt-in auto-play.
- Replace the unpopulated take history with one current-result region while
  retaining the existing playback/export control IDs and async generation path.

## 4. Hierarchy and responsive polish

- Add failing checks for audio.cpp's hidden language cell, collapsed connection
  details, compact disclosures, and revised action labels.
- Move verbose status rows behind progressive disclosure and apply semantic,
  responsive CSS.
- Regenerate the modular CSS bundle using the repository's normal build path.

## 5. Verification and UAT

- Run focused TTS/Studio/UI suites, Ruff on changed Python, compileall, CSS and
  task/evidence checks, and `git diff --check`.
- Inspect the rendered Speech Lab at supported and narrow dimensions.
- Restart only the task-owned Chatbook harness, never the user-owned audio.cpp
  server, then repeat generation and audible playback UAT.
