---
id: TASK-1630
title: 'Briefings phase 2b: audio synthesis, stitching, playback'
status: To Do
assignee: []
created_date: '2026-07-31 19:03'
labels:
  - watchlists
  - briefings
  - tts
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2a (`feat/briefings-phase-2`) shipped presets, script casting, the selection-mode picker,
and citations — see `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`, "Casting
and audio (phase 2)" section, and its "Phase 2a delivery notes". Audio was split out of that phase
because plan-time verification against the real TTS adapters ("the stitching path and any
conversion get verified against the real adapters at plan time rather than promised here", per the
spec) turned up five adapter-reality findings that make audio a materially different task from
script casting, not a same-shape extension of it:

1. `synthesize()` returns a byte-stream `TTSAudioResponse` the caller must drain and `aclose()`.
2. Legacy adapters (kokoro/openai/elevenlabs/chatterbox/higgs/alltalk) reject a plain
   `TTSRequest` -- per-call synthesis goes through `generate_audio_stream(OpenAISpeechRequest,
   internal_model_id)`.
3. `text_processing`'s chunking has zero live callers -- callers must chunk AND stitch themselves;
   there is no free ride from an existing pipeline.
4. The only existing stitcher (`_generate_legacy`,
   `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`) is naive byte-concat, which is wrong
   for WAV headers -- a real pydub decode-and-concat primitive must be written.
5. `Utils/private_paths` has no binary append/stream/move helper -- storage is either
   buffer-whole-then-`atomic_private_write_bytes`, or a new helper, and that choice has not been
   made.

This task is the audio half of spec #2 phase 2: turning a cast script (`briefing_scripts`, already
shipped) into stored, playable audio (`briefing_audio`, not yet built). The roster's
`voice_profile_id` field is recorded by phase 2a but inert until this lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-turn synthesis honors each speaker's voice profile from the script's roster snapshot (not the live preset -- snapshots protect existing artifacts per the spec's entity-model rule)
- [ ] #2 A real stitching primitive decode-and-concats audio (WAV-first, per the `_generate_legacy` precedent) rather than byte-concatenating containers
- [ ] #3 Duration is computed and recorded on the `briefing_audio` row
- [ ] #4 Audio is stored under the private data dir through `Utils/private_paths`, with the buffer-whole-vs-stream/append storage decision made explicit and justified (not left implicit)
- [ ] #5 In-app playback works through the existing audio player
- [ ] #6 A synthesis failure names the turn and speaker, keeps the script, and fails only the audio artifact -- never the script or the briefing
<!-- AC:END -->
