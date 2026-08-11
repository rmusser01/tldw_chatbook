# Linux arm64 guided first-run UAT

## Host and artifacts

- Run date: 2026-08-10.
- Host boundary: Docker 29.6.2, Ubuntu 24.04 image digest
  `sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`,
  Linux arm64.
- Chatbook implementation commit
  `3ad24a5180579d91924f8829d9953d48a5653589` was mounted read-only; its
  tracked working-tree diff was empty during the run.
- audio.cpp: official `release-0.5.1` at commit
  `238ab6a9e321c17de8e120559f57efeedaeb1345`, CPU-only custom
  `supertonic,pocket_tts` build, 72,972,128-byte AArch64 ELF, SHA-256
  `a41b68b227153f6e879307a158fd40a8cc23932f6ab8a26228b7e4ee2097cb1b`.
- Both model mounts were read-only and matched the hashes in the directory
  README before and after the run.

## Journey result

The production Textual Settings panel and real TTS service/supervisor executed
the same nine-step journey described in the README. The result matched macOS:

- Guided was distinct from External and manual JSON.
- Save was passive and returned the exact ready-to-test/handoff copy.
- One child exposed both exact models; selecting PocketTTS for its voice
  observation kept process generation 1 and launched no second child.
- Supertonic generated one complete WAV.
- Explicit restart/apply, forced-crash invalidation and deliberate recovery,
  and explicit shutdown passed.
- Manual JSON preserved Chatbook ownership; External preserved outside
  ownership and used only the configured origin.
- The final independent process table contained only the container's idle
  control process; no audio.cpp process remained.

## WAV evidence

- Container/codec: RIFF/WAVE PCM16.
- Channels/sample rate: mono, 44,100 Hz.
- Frames/duration: 218,910 frames, approximately 4.963946 seconds.
- Total bytes: 437,864.
- SHA-256:
  `0c098d84012b3b9e4ff83bd3186cedb31b92e905c0e45ee09ca10f6e5b78a3dc`.
- Structural validation passed through the current Chatbook adapter and the
  Python standard-library WAV reader. No audible claim is made for the
  headless Linux container.

## Gate status

Objective exact-commit Linux UAT passed. The independent post-journey process
table contained only `sleep infinity` and the `ps` check; the task-owned
container was then stopped.
