---
id: TASK-868
title: Audio and video ingest crash in chunking after transcription
status: Done
assignee: []
created_date: '2026-07-27 01:48'
updated_date: '2026-07-27 01:49'
labels:
  - audio
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every audio or video ingest that reaches the chunking stage dies with a type error: the audio processor calls the chunking service with an options dictionary in the position where that service expects a chunk size, so a dict is compared against an integer. A second defect sits underneath it: the service returns chunk records carrying a text field, while the caller wraps each one as though it were a bare string, nesting the whole record inside another text key. Transcription itself is fine; the failure is entirely after it. This stayed hidden while no transcription engine was installed, because the path failed earlier for want of an engine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An audio file containing speech ingests successfully and stores its transcript
- [x] #2 Chunk records reach the media row as text rather than nested dictionaries
- [x] #3 Video ingest is covered by the same fix, since it shares the path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed and verified with real speech.

The audio processor called ChunkingService.chunk_text(text, chunk_options) -- passing an options dict positionally into a parameter the service declares as chunk_size: int. Its signature is flat keyword arguments (content, chunk_size, chunk_overlap, method), so the dict landed in an integer comparison and every audio or video ingest that got as far as chunking died with "'<=' not supported between instances of 'dict' and 'int'".

A second defect sat underneath: chunk_text returns records like {'text': ..., 'start_char': ..., 'word_count': ...}, but the caller wrapped each as {'text': chunk}, nesting the whole record under another text key. Fixing only the call would have written dictionaries into the media content.

Verified end to end rather than by absence of an exception: macOS 'say' synthesised a Gettysburg line, afconvert made it 16 kHz mono, and after the fix the pipeline returned media_id 1 with 82 characters reading '4-Score and 7 years ago our fathers brought forth on this continent to new nation.' Recognisably the input, so transcription, chunking and persistence all work.

Two notes on how this stayed hidden. It was masked by the missing engine: without faster-whisper or parakeet-mlx the path failed earlier, at transcription, so nothing reached chunking -- fixing packaging in task-839 is what made the crash reachable. And an earlier probe with a 440 Hz sine tone failed with 'No text could be extracted', which is task-677's empty-extraction guard behaving correctly on audio that contains no speech; that fixture could not distinguish a working transcriber from a broken one.

Same shape as the pagination defect in task-684.2: a caller assuming a signature the callee does not have. See backlog/docs/lessons-testing-evidence.md.

Files: Local_Ingestion/audio_processing.py.
<!-- SECTION:NOTES:END -->
