# macOS arm64 guided first-run UAT

## Host and artifacts

- Run date: 2026-08-10.
- Host: macOS 15.6 (`24G84`), arm64.
- Chatbook implementation commit:
  `3ad24a5180579d91924f8829d9953d48a5653589`; tracked working-tree diff was
  empty during the run.
- Homebrew audio.cpp: `0.5.1`, 14,784,368-byte executable, SHA-256
  `3de9bdb0fd1443110b73bdf5cc196e43ed9f143b47595b4fcd59e4a1ed18d467`.
- The reviewed model hashes are recorded in the directory README.
- Backend: Guided `Auto`, resolving to the reviewed portable CPU baseline.

## First-user observations

- The production Settings panel defaulted a first Managed selection to
  **Guided setup — no JSON editing**.
- Scanning the explicitly selected package root returned two reviewed entries:
  `supertonic-3-orig` and `pocket-tts-english-bf16`.
- The package review included exact family/variant, TTS/Clone tasks, pinned
  compatibility, safe local name, lazy/resident-memory warning, and recovery.
- PocketTTS GGUF was labeled **Reference: Required** and **voice setup
  required**. Supertonic was selected as the text-ready default.
- Save reported **Configuration saved — ready to test** and offered **Open
  Speech Lab & Hear a Sample**. Binary inode and model size/mtime snapshots
  were unchanged; no child or generated artifact existed after Save.

## Runtime observations

- The first deliberate operation launched one owned child and returned the
  exact two-model catalog.
- Supertonic produced one complete structurally valid WAV.
- A PocketTTS model-specific voice refresh remained on process generation 1;
  no second child launched. It returned zero built-in voices, consistent with
  the standalone GGUF's voice-required classification.
- Changing `guided_threads` to 2 staged a new generation. Explicit
  restart/apply reaped the first child and launched one replacement.
- Killing the replacement published unavailable state; the next deliberate
  operation launched one recovery child with the exact catalog.
- Explicit shutdown reaped the recovery child. All three task-owned PIDs had
  terminal return codes, and the generated-artifact root was empty.
- A pre-existing unrelated audio.cpp 0.4 PID was alive before and after the
  journey and was neither adopted, signaled, nor modified.
- The manual-json source launched and reaped its one child. External mode
  contacted only the configured origin, launched no managed child, and left
  the externally owned process alive when the Chatbook service closed.

## WAV evidence

- Container/codec: RIFF/WAVE PCM16.
- Channels/sample rate: mono, 44,100 Hz.
- Frames/duration: 218,910 frames, approximately 4.963946 seconds.
- Total/audio bytes: 437,864 / 437,820.
- SHA-256:
  `cc959d5389daed6996512e21a4d2faca096be05e66d3dcd1549342c65ab14235`.
- Structural validation: Python `wave`, Chatbook's native adapter, and macOS
  `afinfo` all passed.
- macOS `afplay` returned success for this exact WAV. The user confirmed this
  exact artifact was audible on 2026-08-10.

## Gate status

Objective exact-commit UAT and human audible playback passed. The macOS release
gate is closed.
