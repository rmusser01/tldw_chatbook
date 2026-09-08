---
id: TASK-31827
title: 'Meetings: MOSS/server diarizer backend behind the Diarizer seam'
status: In Progress
assignee: []
created_date: '2026-09-05 22:49'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a second Diarizer backend using MOSS-Transcribe-Diarize (0.9B, batch, CUDA-first, Apache-2.0), local when a CUDA GPU is present and server-hosted otherwise, selectable via the [meetings] diarizer_backend = server config reserved by the phase-2 design (Docs/superpowers/specs/2026-09-05-meeting-diarization-design.md). Needs its own design, including the off-device-audio privacy model, since the server backend sends meeting audio off the device unlike the local SpeechBrain backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design approved (torch-free ONNX backend spec 2026-09-07; server/MOSS specs follow as their own sub-projects)
- [x] #2 A base install (no torch) gets live speaker labels and a Stop pass through the sherpa-onnx engine, selected by `[meetings] diarizer_backend`
- [x] #3 Model files are fetched on first Start from pinned GitHub release URLs, hash-verified, into the user data dir; the meeting starts immediately and labels begin when the worker is ready; air-gapped placement works
- [x] #4 The SpeechBrain engine keeps working unchanged behind the `diarization` extra; every failure degrades to coarse labels with a visible reason
- [x] #5 Voiceprints are keyed by the active engine's model id; switching engines asks for re-enrollment instead of matching across vector spaces
- [x] #6 A measured bake-off (DER, live purity, RTF, embed latency, self-match separation vs ECAPA on VoxConverse + AMI) decides the default engine and the per-embedder thresholds
- [ ] #7 Server streaming backend behind the same seam (own spec)
- [ ] #8 MOSS post-meeting re-transcription (own spec)
<!-- AC:END -->

## Renumbering provenance

Renumbered from TASK-31742 to TASK-31827 on 2026-09-06 when `feat/meeting-diarization` was brought up to date with dev: TASK-31742 had already been taken on dev by a Canvas task (the older arrival keeps the id, per the TASK-19601 owner rule). No dependencies: entries or doc/code references pointed at the old id.

## Implementation Plan

1. Research the self-hosted options (Docs/Design/2026-09-07-self-hosted-diarization-options.md) → split into three sub-projects
2. Sub-project 1 — torch-free ONNX backend: worker engine flag → ONNX engine → downloader → backend warm-up → session overlap rule → owner resolution → rail → bake-off → go/no-go (9 SDD tasks)
3. Sub-project 2 — ServerDiarizer over tldw_server's streaming endpoint with the off-device-audio consent model (TASK-32014)
4. Sub-project 3 — MOSS-Transcribe-Diarize post-meeting re-transcription (TASK-32015)

## Implementation Notes (sub-project 1)

Branch `feat/meeting-onnx-diarizer` (head 19bf900e6). The bake-off FAILED the spec §7 go/no-go on one gate (self-match separation: titanet_small 0.640 vs the 0.684 gate) while beating ECAPA on DER (0.143 vs 0.371), purity (1.000 vs 0.941), RTF (0.038) and latency (12 ms); per the spec's fail branch AUTO_ORDER stays SpeechBrain-first, ONNX ships as the base-install fallback, and the measured per-embedder thresholds are pinned (report: Docs/STT_Evaluation/task-31827/report.md). Key decisions: engines are plug-ins of one worker loop (the
wire protocol did not change); reconciliation's map-and-mint step is shared; the downloader trusts only pinned hashes and
https GitHub release hops; the model fetch rides the backend's warm-up thread so Start never waits; `auto` is not sticky
(pin `diarizer_backend` to avoid a voiceprint re-enrollment when the torch extra comes or goes); the live and Stop
clustering thresholds are per engine. Reviews found and fixed two downloader Criticals (redirect scheme downgrade;
unbounded tar member) and one pre-existing restart bug in `wait_ready`. Follow-ups: TASK-32014 (server backend),
TASK-32015 (MOSS), TASK-32016 (Library ingest diarization via sherpa-onnx), TASK-32017 (CUDA providers).
