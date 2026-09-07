---
id: TASK-1881
title: 'Fold the RIFF data-chunk offset into Pcm16WavInfo instead of a parallel walk'
status: To Do
assignee: []
created_date: '2026-08-02 12:00'
labels: [tts, audio, hygiene]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TTS/pcm_stream.py::_data_chunk_offset` re-walks RIFF chunks that
`audio_cpp_contract.validate_pcm16_wav` already walked, because `Pcm16WavInfo` does not expose the
data payload offset. Two implementations of the same walk must now change in lockstep (a comment
marks it). Compute the offset once inside the validator's existing loop and expose it on
`Pcm16WavInfo`; make `pcm_stream` consume it. Origin: streaming-sink Task-3 review follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `Pcm16WavInfo` carries the data payload offset computed during validation.
- [ ] #2 `pcm_stream` uses it; the parallel walk is deleted.
- [ ] #3 The trailing-chunk and word-alignment pins stay green unchanged.
<!-- AC:END -->
