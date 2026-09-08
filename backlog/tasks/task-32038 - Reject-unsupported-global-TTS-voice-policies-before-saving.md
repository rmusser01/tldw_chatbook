---
id: TASK-32038
title: Reject unsupported global TTS voice policies before saving
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:10'
updated_date: '2026-09-08 06:06'
labels:
  - tts
  - bug
  - settings
dependencies: []
references:
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Global Speech and TTS Settings currently allow legacy providers such as Kokoro to save a server-default voice policy that fails every automatic reply at effective selection. Provider changes can also carry that unsupported policy from audio.cpp.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings reject unsupported server-default voices with a field-specific recovery message before publishing preferences.
- [x] #2 Switching from audio.cpp to a legacy provider offers an exact voice and does not retain a foreign exact model or voice; returning to the saved provider restores its saved choices.
- [x] #3 Mounted Settings and real TTS admission tests cover Kokoro provider transitions, persisted invalid preferences and supported audio.cpp controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: Align the canonical Settings controls and save validation with the existing provider selection contract; no persistence or provider boundary change.

1. Reproduce unsupported saved Kokoro voice policies and audio.cpp-to-Kokoro transitions in mounted Settings tests and actual effective admission.
2. Reject unsupported voice omission before publication, offer exact legacy voices, and reset foreign model and voice choices on explicit provider changes while restoring the saved provider choices.
3. Verify supported audio.cpp policies, stale-value recovery, provider round trips and privacy-safe diagnostics; document the settings repair.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Canonical Speech & TTS Settings now reject server-default voice policies for legacy providers before publication, matching the existing resolver contract. Explicit provider changes choose that provider's canonical model and voice instead of retaining foreign IDs; returning to the saved provider restores its saved choices. Unsupported saved policies remain visible for deliberate correction. A valid repaired draft can replace invalid saved defaults and its status correctly reports Unsaved.
Ten initial regression cases failed before the production repair. Mounted Settings tests follow the resulting Kokoro defaults through real service admission twice, with only audio execution substituted. They cover both audio.cpp policy shapes, saved-provider round trips and repair/publication of invalid saved preferences. Independent review found the misleading repaired-draft status; its added assertion failed first, then passed after independent draft/baseline validation. Settings model/panel and Console autoplay gate: 320 passed. Final targeted recovery/status gate: 37 passed, 96 deselected. Existing audio.cpp capability policies remain covered. New tests Ruff/formatter clean; no introduced production Ruff diagnostics, changed ranges formatted and diff check clean.
No new ADR: routine enforcement of ADR-039. Files: settings_speech_tts.py, speech_tts_settings_panel.py and test_legacy_tts_voice_policy.py; updated developer/user TTS recovery guidance and an incident lesson. Changes also applied to the isolated newer dev worktree. No full suite, live provider, personal configuration or audible engine test was used.

Closeout ID audit across 495 refs and 12 worktrees found that the Petdex companion task had claimed TASK-32031 at 2026-09-08 05:09, before this TTS task at 05:10, and was committed as b4e460f75142b37ff3df277fd712bdb1812007f9 on codex/buddy-import-design. The earlier claimant keeps its ID. Renumbered only this TTS task and its scoped lesson reference to TASK-32038 after rechecking the global maximum (32037); preserved creation time, status and acceptance evidence. The unrelated Petdex task and checkout were untouched.
<!-- SECTION:NOTES:END -->
