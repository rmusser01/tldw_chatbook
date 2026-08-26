# TASK-13205 audio.cpp clone workflow live UAT

Date: 2026-08-11
Result: Passed

This artifact is intentionally sanitized. It contains no source or
materialization path, transcript, generation prompt, audio bytes, credential,
or child diagnostic body.

## Environment and identities

- Application revision: `bb334816a`. The audible journey ran from the working
  tree captured by this commit; its UAT-found one-row Speech Lab containment
  amendment was then locked by the real-width Pilot regression at the same
  revision.
- Host: macOS Darwin 24.6.0, arm64 (Apple Silicon).
- Server: Homebrew `audio-cpp` 0.5.1, SHA-256
  `3de9bdb0fd1443110b73bdf5cc196e43ed9f143b47595b4fcd59e4a1ed18d467`.
- Backend: CPU portability/testing backend.
- Recipe: `audio-cpp-0.5.1.pocket_tts.pocket_tts_english_bf16`, revision 2.
- Package variant/model: `pocket_tts_english_bf16` /
  `pocket-tts-english-bf16`.
- Model SHA-256:
  `267e774a671138c4ebbc1d6d9d73af92f4a8e83a64b45b84f3457ac700ad0cc9`.
- Generated `server.json` SHA-256 after canonical JSON ordering and replacement
  of the private model path with `<private-path>`:
  `42294368e6da2268ae13d2c09293433d226a1a99a04523595f09221df119a820`.

## Safe reference and result evidence

- Canonical reference ID: `5c7b85d4-d017-4e2a-883c-8f460bb1b959`.
- Reference SHA-256:
  `7a780deacc42386f1e605c8fe2954221cd2024003163681ca32efd5bb4b8060c`.
- Reference structure: 422,570 bytes; 4,791 ms; 44,100 Hz; mono;
  PCM signed 16-bit little-endian.
- The post-commit structural probe returned a complete WAV with SHA-256
  `1b3d10f9c1fb510a198be39ef994d5ff758337a1cd66661e3b2fab8484ab45c1`:
  111,404 bytes; 2,320 ms; 24,000 Hz; mono; PCM signed 16-bit
  little-endian.
- The structural probe used exactly one supported server process (PID 65261),
  which reached healthy with one configured model and was stopped after the
  response.

## User journey

1. Started from an isolated clean config, data directory, and profile store.
   Saving Guided Managed configuration did not launch audio.cpp.
2. Opened Speech Lab, selected the reviewed PocketTTS model, and used **Start &
   Set Up Voice**. One managed child became healthy.
3. Chose a bounded PCM16 WAV and supplied its exact transcript. The UI exposed
   local-plaintext/privacy copy without rendering the source path.
4. Used **Create Voice & Generate**. The result became the current complete-WAV
   artifact and remained playable. Human audible confirmation: passed.
5. Used **Save as Voice Profile**, reviewed the save, and chose the explicit
   Roleplay handoff. The saved profile revision was 2.
6. Roleplay displayed the new profile as a non-mutating suggestion. Explicitly
   selecting it persisted the assignment for local character ID 2. Reloading
   preserved the assignment and showed it as unverified while audio.cpp was
   stopped; passive profile/character browsing did not start a child.
7. A deterministic localhost OpenAI-compatible stub generated only the
   character-response text; it had no role in TTS and used no external account
   or secret. In Console, **Speak** on that real character response lazily
   started the compatible managed child and used the assigned exact clone
   profile. Human audible confirmation: passed.
8. The UAT exposed one usability defect: **Save as Voice Profile** was
   keyboard-reachable but clipped at the production pane width. The new
   100-column containment regression reproduced it; raising the split floor by
   one row made the complete action set visible without changing scroll
   ownership.

## Ownership and teardown

- Closing the live app emitted successful TTS-resource cleanup and released the
  handler-owned current result.
- The app, localhost text stub, and task-owned managed child were stopped.
- No recognized private request-materialization directory remained under the
  isolated audio.cpp runtime root.
- No task-owned audio.cpp process remained after teardown. The unrelated
  pre-existing system audio.cpp 0.4 process was still alive and was not touched.
- The separate post-commit structural probe was also stopped definitively.

This satisfies the clean-profile real-process UAT gate for transient audition,
exact save, explicit character assignment, lazy roleplay speech, two audible
playbacks, privacy, and definitive task-owned cleanup.
