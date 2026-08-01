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
4. The only multi-segment stitcher (`AudioBookGenerator._combine_segments`,
   `tldw_chatbook/TTS/audiobook_generator.py:638`) is naive byte-concat, which is wrong for WAV
   headers -- a real pydub decode-and-concat primitive must be written. (The real decode-and-concat
   that exists is M4B-specific, buried in `AudioService.create_m4b_with_chapters`,
   `audio_service.py:340`.)
5. `Utils/private_paths` has no binary append/stream/move helper -- storage is either
   buffer-whole-then-`atomic_private_write_bytes`, or a new helper, and that choice has not been
   made.

**Re-verified against dev `8b7fa5eb6` (2026-07-31)** after ~2,000 lines of TTS character-assignment
work merged. Findings 1, 3, 4 hold (4's file attribution above is the corrected one -- an earlier
draft wrongly named `_generate_legacy`, which merely joins the chunks of ONE response). Corrections
and additions, all detailed in the spec's re-verification block:

- **Finding 2:** a public all-provider path now exists (`TTSService.synthesize_default`,
  `TTS_Generation.py:895`) that builds the legacy options itself -- but it is global-preferences
  only (one voice for everyone), so it cannot serve a roster. `character_request_resolver.py` is
  hard-fenced to `audio_cpp` and does not help legacy providers.
- **Finding 5:** `open_private_binary` exists but is READ-ONLY (`private_paths.py:864`); the write
  decision is unchanged.
- **New constraint:** per-speaker *exact* voices work only on `audio_cpp` (`synthesize_exact`
  refuses all other providers). Scope multi-voice rosters accordingly, or own the response
  validation for the raw `generate_audio_stream` path.
- **New surfaces this task must add:** a general decode-and-concat stitcher; a
  `TTSProfileService.get_profile(UUID)` passthrough (the repository has one at
  `profile_repository.py:1314`, the service does not expose it, so today a stored
  `voice_profile_id` can only be resolved by paging `list_profiles`); the `briefing_audio` table;
  and a public legacy internal-model-id builder (two private copies exist and they DISAGREE on
  kokoro/alltalk ids -- `stts_events.py:737` vs `request_admission.py:356`).
- **Transcribe, don't reinvent:** `TTSEventHandler._generate_tts` (`tts_events.py:731`) is a
  hardened per-message synthesize→drain→append→publish loop (response-contract validation, batched
  writes, `aclose()` in a `finally` that preserves the primary exception, cancellation cleanup,
  bounded error copy, metrics). Playback (`TTS/audio_player.py` + `tts_events.py:1568-1600` -- read
  the stop-guard docstring first) and cooldown/admission already exist.
- **Snapshot more than 2a records:** `briefing_scripts.roster_snapshot_json` is immutable and holds
  only `voice_profile_id`; the `briefing_audio` row must also snapshot the profile's `revision` and
  denormalized selection to stay self-interpreting. `profile_portability.py` is a shape precedent,
  not a reusable mechanism.
- **UI note:** `ArtifactsPane.selected_script` is `recompose=True`, so audio widgets hung off
  `#artifacts-script-detail` are rebuilt on every script selection -- playback state must live in
  the app-level player singleton, not the widget.

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
