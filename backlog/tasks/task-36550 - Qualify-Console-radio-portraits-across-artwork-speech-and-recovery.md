---
id: TASK-36550
title: Qualify Console radio portraits across artwork speech and recovery
status: Done
created_date: 2026-10-23 22:56
dependencies:
- TASK-36547
- TASK-36548
- TASK-36549
labels:
- console
- qa
- documentation
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-implementation-review.md
- Docs/superpowers/qa/console-radio-portraits/2026-10-23/qualification.md
- Docs/superpowers/reviews/2026-10-23-task-36550-review.md
updated_date: 2026-10-24 06:26
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the complete first-version portrait experience with imported and generated artwork, real speech output and profile recovery, recording the limits of actual device and provider coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A mounted editor-to-Console journey imports, previews, applies, reopens and removes a set; generation uses the user-selected real reference-image service and demonstrates reviewed per-slot replacement, with external generation distinguished from live in-app adapter coverage.
- [x] #2 Real TTS qualification separately covers streaming and supported WAV artifact output with onset, nonzero quiet pauses, continuous updates without test-only publication, final-syllable/device-tail completion, stop/replacement during pending output, no early sentence/listening handoff and conversation changes; hardware latency and untested players are reported explicitly.
- [x] #3 Known waveform and measured visible output prove the 100 ms onset/silence target on a qualified local path; timing-unavailable external playback retains truthful fallback without claiming synchronization.
- [x] #4 Full-profile backup/restore preserves frames and retires stale presentation owners; original artwork, ordinary export/sync formats and soft-delete/restore behavior remain unchanged.
- [x] #5 Targeted integration, static, migration packaging, design-token, CSS and Backlog checks pass or unrelated baseline failures are reproduced and reported; no full test sweep runs without explicit user request.
- [x] #6 User and architecture documentation describe preparation, six-request generation, profile-local storage, motion settings and capability fallback; evidence and each task's implementation notes match what was actually exercised.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Qualify the approved profile-local attachment and speech/UI contracts; no new provider boundary is part of this unit.

1. Read current qualification acceptance criteria and existing production-app, backup/recovery and speech probes.
2. Add an owned mounted editor-to-Console import/apply/reopen/remove journey and full-profile backup/restore attachment coverage, preserving normal export/sync boundaries.
3. Exercise the real available image-edit/TTS services and local device paths with recorded inputs, outputs, timing, cancellation/tail and ownership evidence. Inspect user-provided ComfyUI/MOSS capabilities first; keep unavailable paths explicitly pending.
4. Measure known waveform onset/silence against actual presentation paint and device timestamps; document measurement scope and hardware loopback limits.
5. Run only relevant targeted integration/static/packaging/style checks; reproduce unrelated baseline failures separately.
6. Update user/architecture guidance and dated evidence, review this unit, and mark Done only for acceptance criteria actually proved.
Live CoreAudio qualification exposed an early finished acknowledgement (~24 ms before final scheduled DAC sample). Before completing this unit, add deterministic tests with/without portrait observation; fence natural completion on both native acknowledgement and the validated final output interval, retaining immediate scoped Stop/replacement. Keep timing extraction bounded and optional amplitude analysis out of audio completion authority. Re-run affected sink/speech races and real PCM/WAV calibration, then review the corrective diff. ADR-242 already requires completed device tails; clarify the measured implementation implication in its notes.
User steering: use ChatGPT built-in image generation instead of the unreachable ComfyUI/MOSS server. Generate six individual reference-preserving poses from the packaged Samira neutral portrait plus an explicit selected replacement; save outputs/prompts non-destructively in the qualification workspace, inspect identity/alignment and actual rail-size preview, and import through the existing authoring path in an isolated profile. This qualifies real generated-artwork import/preview, not an in-app ChatGPT provider integration; existing adapter generation coverage and real TTS qualification remain separately stated.
User authorized local resizing of generated copies; retain raw hashes and prepare 1024x1024 copies. The mounted external ChatGPT-artwork import/replacement/preview/Apply/reopen/Remove journey now passes in a private profile; external generation remains distinct from in-app adapter qualification. User selected Pocket TTS and explicitly authorized installation. Install it in an isolated qualification environment, run the existing native /tts integration locally with a built-in voice, and qualify actual speech/playback/ownership paths without adding a provider or changing the user's default voice settings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Qualified the Console radio portrait implementation under ADR-242 using user-selected real ChatGPT reference edits, authorized 1024×1024 copies and actual Pocket TTS 3.3.0 speech. Preserved seven raw originals and exact prompts/hashes; six selected imports, one-slot replacement, mounted color/monochrome preview, Apply/reopen/Remove and unchanged source bytes pass. The user-approved external image workflow is distinguished from unexercised live in-app adapter generation.

Installed Pocket TTS in an ignored isolated environment without changing the app environment or user voice defaults. Actual canonical character-context admission, trusted Console events, native /tts WAV repair, local speakers and automatic portrait updates pass. Complete source/device WAVs are byte-identical and local tiny.en ASR retains the final word. Separate real HTTP chunk streaming exercises the existing PCM transport without adding a provider or claiming native Pocket PCM capability. Stop near a pending tail, replacement, conversation change and production sentence/listening-intent boundaries pass.

The earlier real CoreAudio probe exposed a finished ACK about 24 ms before scheduled output ended. The reviewed sink correction (already committed in 58ca4c1e06) fences natural completion on ACK plus a bounded valid final interval, independently of portrait analysis, preserving immediate scoped Stop. ADR-242 and the incident-backed live-verification lesson explain it.

Validation: previously recorded installed journey/recovery/export checks, 136 storage/authoring/ownership checks, 226 final sink/speech checks and three real PCM/WAV/afplay probes; new Pocket protocol/Console presentation gate: 26 passed. New opt-in probes pass Ruff lint/format; artifact hashes, dimensions, JSON and git diff checks pass. Real speech completion follows its final DAC interval by 34.770 ms; next sentence/listening intent follow the respective tails by 276.150/23.619 ms. No full sweep ran; unrelated baseline findings remain documented.

Core changes/evidence: Audio/streaming_sink.py and associated regression coverage in the earlier correction; production/recovery/export tests; two new opt-in QA probes, image/audio provenance and playable captures; final task review/plan and user/architecture guidance. No new production provider, dependency declaration, schema or default artwork was added.

Limits: no independent acoustic/human audition, microphone/STT hardware, other OS/device, native graphics-terminal rendering or real in-app image-adapter qualification. Hands-free capture-end input is simulated while actual dispatcher, token sequencer, synthesis/device completion and listening intents are exercised. Local ASR and full device submission are explicitly identified as automated evidence. No push, PR or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->


## Qualification checkpoint

Installed artwork journey, native full-profile restore/fresh open and original F9
control pass. Portable exports: 10 checks pass. Storage/authoring/ownership gate:
136 pass; final reviewed sink/speech gate: 226 pass. Real PCM/WAV/afplay probes:
three pass; exact waveform/terminal-flush scoring stays below 100 ms and verifies
completion after the scheduled tail. CoreAudio's early finished ACK exposed and
led to a scalar tail-fence correction independent of portrait analysis; ADR-242
and the live-verification lesson record it. See the linked dated evidence and
partial review for measurements, runtime hashes and baseline limits.

**Artwork checkpoint:** user-selected ChatGPT reference edits and explicitly
authorized resized copies pass the actual-app import/replacement/preview/Apply/
reopen/Remove journey in color and monochrome. Original files and source bytes remain
unchanged; the selected replacement preserves the other five slots. Actual 23×12
cell captures show stable framing, with subtle lip detail at 24×24 mosaic resolution.
Tracked prompts/hashes/observations and the opt-in probe are in the dated QA folder;
local deliverables are in `output/imagegen/console-radio-samira-2026-10-23/`.
No in-app ChatGPT provider or live adapter qualification is claimed.

**Final checkpoint:** user-requested Pocket TTS 3.3.0 is installed and real speech
qualification passes: actual Console WAV playback, a separate incremental transport
probe, full source/device PCM equality, local ASR final-word recovery, pending-tail
Stop, replacement, conversation retirement and production sentence/listening-intent
handoffs. New targeted regressions: 26 passed. Limits remain explicit: no independent
physical/acoustic audition, microphone/STT hardware, other platforms, native graphics
terminal or live in-app image-adapter coverage. See the final review and dated evidence.
