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
updated_date: 2026-10-24 13:58
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
User review rejected the selected image set: exposure varies and eye/mouth variants are not exact matched pairs. Reopen artwork qualification and its final evidence claim; retain the independently verified transport/storage/audio evidence. Inspect every selected full-size image, preserve v1, and prepare a true 2×3 matrix (eyes open/closed × mouth closed/small/open) from one original base. The user subsequently explicitly authorized local compositing; reuse the same generated mouth patch across both eye states and the same closed-eye patch across all mouth states, verify invariant pixels and pairing mechanically, show a labeled comparison, then rerun actual import/preview qualification. No production generation behavior change is part of this corrective artwork pass.
The user explicitly authorized local compositing. Build and inspect the paired-v2 matrix, assert unchanged original pixels outside expression patches and exact eye/mouth pairing, then run the actual mounted editor-to-Console import/replacement/apply/reopen/remove journey on these corrected assets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrected the rejected v1 artwork under ADR-242 after explicit user authorization for local compositing. One original base, shared mouth patches and one eyelid patch now produce an exact 2×3 matrix. Pixel assertions prove unchanged regions, identical mouths across blink pairs and identical eyelids across mouth states; six file hashes are distinct and all input originals remain unchanged. Full-resolution visual review checks lighting, alignment and patch edges.

The mounted actual-app corrected-set journey passes six imports, one selected replacement preserving the other five slots, exact final normalized matrix equality, six timer-painted Console previews, idle, Apply/reopen/Remove and source preservation. New compositor and opt-in probe pass Ruff lint/format; asset dimensions, 4 MiB/frame and 16 MiB/set bounds, hashes, JSON and diff checks pass. No production behavior or default artwork changes; live generation still requires visual review.

Preserved earlier installed/recovery/export, 136 storage/authoring/ownership, 226 sink/speech, three real PCM/WAV/afplay and 26 Pocket/Console regression results. Real Pocket TTS qualification and reviewed CoreAudio completion correction (58ca4c1e06) remain valid: full source/device equality, final-word ASR, scoped Stop/replacement/conversation retirement and sentence/listening-intent handoffs. The artwork-only corrective run makes no new speech claim.

Updated the plan, dated evidence, task review, user guidance and incident-backed live-verification lesson. Added compose_portrait_matrix.py, paired_artwork_probe.py, v1 rejection measurements, corrected manifest/observations and labeled comparison. V1 visual qualification is withdrawn; raw originals/historical probes remain intact. Existing ADR-242 governs this correction; no new ADR required.

Limits: external ChatGPT generation plus authorized local compositing does not qualify live in-app adapters; no native graphics-terminal, physical acoustic loopback, microphone/STT hardware or other-platform qualification. Facial differences remain subtle in the 24×24 mosaic. No full sweep, push, PR or merge.
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

**Artwork checkpoint (corrected 2026-10-24):** v1 was rejected for exposure drift
and unmatched mouth/blink pairs. The explicitly authorized `paired-v2` composites
now pass full-resolution matrix review and invariant-pixel/pairing assertions.
The actual-app import/replacement/preview/Apply/reopen/Remove journey passes on all
six corrected frames; five slots survive the selected replacement unchanged and
the final normalized draft exactly equals the corrected matrix. Source bytes and
raw generated files remain unchanged. See `paired-{manifest,observations}.json`,
`paired-review.png` and the v1 rejection measurements in the dated QA folder.
Final local imports: `output/imagegen/console-radio-samira-2026-10-23/paired-v2/selected/`.
Fine detail remains subtle at the fitted 24×24 mosaic resolution. No live in-app
adapter or automatic-generation consistency qualification is claimed.

**Final checkpoint:** user-requested Pocket TTS 3.3.0 is installed and real speech
qualification passes: actual Console WAV playback, a separate incremental transport
probe, full source/device PCM equality, local ASR final-word recovery, pending-tail
Stop, replacement, conversation retirement and production sentence/listening-intent
handoffs. New targeted regressions: 26 passed. Limits remain explicit: no independent
physical/acoustic audition, microphone/STT hardware, other platforms, native graphics
terminal or live in-app image-adapter coverage. See the final review and dated evidence.
