# TASK-13202 guided first-run UAT

## Status

Chatbook commit `3ad24a5180579d91924f8829d9953d48a5653589` passed the
complete first-user journey on macOS arm64 and provisioned Linux arm64 against
audio.cpp `release-0.5.1`. The objective results are recorded in the platform
files. The user confirmed the exact macOS WAV was audible, so the release gate
is closed.

## Shared evidence boundary

- Chatbook implementation commit:
  `3ad24a5180579d91924f8829d9953d48a5653589`.
- Tracked working-tree diff SHA-256 during both runs was the empty-input digest,
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- audio.cpp source contract: official `release-0.5.1` at commit
  `238ab6a9e321c17de8e120559f57efeedaeb1345`.
- Supertonic package SHA-256:
  `af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`.
- PocketTTS package SHA-256:
  `267e774a671138c4ebbc1d6d9d73af92f4a8e83a64b45b84f3457ac700ad0cc9`.
- No executable, model, generated configuration, or temporary private path is
  retained in this evidence.

## Journeys covered on both platforms

1. Isolated first-run Chatbook config/data and production Settings panel.
2. External / existing `server.json` / Guided source hierarchy.
3. Guided binary selection and bounded scan of one explicitly selected root.
4. Exact Supertonic and PocketTTS review with Supertonic chosen as the
   text-ready default and PocketTTS GGUF labeled voice-required.
5. Side-effect-free Save, exact ready-to-test copy, and Speech Lab handoff.
6. One deliberate start, two-model catalog, complete Supertonic WAV, and a
   PocketTTS model-specific voice observation in the same process generation.
7. Staged Guided setting, explicit restart/apply, forced child crash, later
   deliberate recovery, and explicit shutdown.
8. Existing user-JSON managed launch/reap and External configured-origin test
   without Chatbook adopting or stopping the externally owned child.
9. Empty generated-artifact root and no owned audio.cpp child after teardown.

The PocketTTS GGUF model did not synthesize during this task. Real 0.5.1
evidence showed that the standalone file needs separate voice material, so the
revision-2 recipe truthfully remains registered but requires voice setup. This
is not reported as a second text-ready sample.

## Platform evidence

- [macOS arm64](macos.md)
- [Linux arm64](linux.md)
- [TASK-13201 pinned POSIX foundation](../audio-cpp-guided-posix-2026-08-09/live-uat.md)
