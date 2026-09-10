---
id: TASK-32165
title: Qualify the PortAudio CoreAudio stop-deadlock repair on macOS ARM
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 07:18'
updated_date: '2026-09-09 08:40'
labels:
  - audio
  - tts
  - validation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish whether the pinned upstream PortAudio repair resolves the CoreAudio shutdown deadlock captured during real Higgs Console playback, without changing the user environment or claiming an unmerged library as the shipped default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two observed failures retain exact runtime/library identities and native stacks tied to the upstream lock inversion.
- [x] #2 An isolated pinned repair passes bounded real sink drain and Stop controls with every owned process and stream joined.
- [x] #3 Real Higgs Console playback, cancellation, successor recovery and full-content checks are rerun on the identified repaired library, with limitations preserved.
- [x] #4 Reproduction commands, source and binary hashes, upstream review status and deployment prerequisites are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 TTS ownership boundary remains unchanged.
Reason: Investigation and isolated upstream-library qualification only. Adopting a new default dependency/runtime requires a separate decision and follow-up.

1. Preserve both Higgs failure receipts and compare the exact loaded PortAudio binaries with the passing Kokoro environment. Read the pinned upstream #1174/#1175 diagnosis and diff.
2. Build the reviewed candidate source in a task-owned prefix, retain its license and hashes, and run bounded sink drain/Stop controls without changing system libraries.
3. After the exclusive audio slot is free, rerun real Higgs playback/cancellation/recovery using only the isolated candidate library; preserve initial failures and verify full content separately.
4. Record exact scope, limitations, upstream merge status and deployment prerequisites. Keep default runtime changes out of this qualification. Run targeted checks and inspect all owned cleanup.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved both original Higgs native deadlocks, exact loaded-library identities and native samples tied to PortAudio issue #1174. Built the pinned upstream PR #1175 candidate in a task-owned prefix without replacing system or default libraries. Corrected controls-02 passed all six production-sink drain/Stop/successor phases, all four full-drain callback PCM hashes matched, and live Stops returned in 0.116791 and 0.111009 seconds with 11.72 seconds queued. All stream/notify owners and worker processes exited. The earlier profile-isolation-limited control remains explicitly preserved.

Real Higgs on the exact candidate image passed full Lab/Console playback, active-generation Stop, successor recovery and clean process exit; all three complete clips matched ASR. Its 96.400-second model join is retained as a separate decoder-cancellation limitation. Source/patch/license/build hashes, corrected private-profile audit, sandbox-device prerequisite, upstream review status and deployment limits are documented in Docs/QA/tts-macos-burndown-2026-09-09/{portaudio,providers/higgs}/README.md. Existing ADR-023 applies; adopting a default repaired runtime remains separate work.
<!-- SECTION:NOTES:END -->
