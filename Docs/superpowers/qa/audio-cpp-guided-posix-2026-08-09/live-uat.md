# TASK-13201 guided audio.cpp POSIX engineering UAT

## Status

This is sanitized engineering evidence for TASK-13201. The native macOS run,
human audible check, provisioned Linux arm64 real-process gate, and exact
committed-revision reruns passed.

## Build and host

- Chatbook implementation commit:
  `29e4262d9d6a7abe107206bb4ac097e7c06a444e`, rebased onto `dev` commit
  `8ffded2e4f98114db3a7b9d340ac49c980ebf499`.
- Host: macOS 15.6 (`24G84`), arm64.
- Selected backend: `cpu` through Guided `Auto` and the intersection of both
  accepted recipe tuples.
- audio.cpp: Homebrew `audio-cpp 0.5.1`, 14,784,368-byte executable,
  SHA-256 `3de9bdb0fd1443110b73bdf5cc196e43ed9f143b47595b4fcd59e4a1ed18d467`.

### Provisioned Linux gate

- Run date: 2026-08-10.
- Host boundary: Docker 29.6.2, Linux arm64, Ubuntu 24.04 image digest
  `sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`.
- audio.cpp source: official `release-0.5.1` tag at commit
  `238ab6a9e321c17de8e120559f57efeedaeb1345`.
- Toolchain: GCC/G++ 13.3.0 and CMake 3.28.3.
- Build: CPU-only, portable CPU kernels, deployment specs, and the custom
  `supertonic,pocket_tts` model composite.
- Result: 72,972,128-byte AArch64 ELF executable, SHA-256
  `a41b68b227153f6e879307a158fd40a8cc23932f6ab8a26228b7e4ee2097cb1b`.

## Reviewed packages

- `audio-cpp-0.5.1.supertonic.supertonic_3_orig`, 454,072,836 bytes,
  SHA-256 `af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`.
- `audio-cpp-0.5.1.pocket_tts.pocket_tts_english_bf16`, 219,096,064 bytes,
  SHA-256 `267e774a671138c4ebbc1d6d9d73af92f4a8e83a64b45b84f3457ac700ad0cc9`.
- Both files occupied one selected package root. The scanner truthfully
  reported multiple exact candidates; each candidate was explicitly accepted
  by exact recipe and file identity.

No full executable, package, generated-artifact, or temporary path is retained
in this evidence.

## Procedure and observations

1. Built a full Guided Managed settings snapshot for both reviewed packages.
2. Materialized one owner-private generation directory and read-only generated
   `server.json` only at the deliberate catalog/Test boundary.
3. Started the Homebrew binary with direct no-shell `--config` argv and the
   generated directory as its deterministic working directory.
4. Observed one Running child with TTS capability `available`.
5. Observed the exact lazy catalog:
   - `uat-supertonic`: family `supertonic`, capabilities `tts`;
   - `uat-pocket-tts`: family `pocket_tts`, capabilities `tts`, `clone`.
6. The optional Supertonic voices endpoint returned a complete empty list; a
   speech request with no voice field correctly used the server default.
7. Generated one complete WAV from short synthetic roleplay-style narration.
8. Closed the adapter and supervisor, waited for definitive shutdown, and
   confirmed no owned generation, no retained generated artifact, and no owned
   `audiocpp_server` process remained.

### Linux lifecycle observations

The provisioned Linux run executed the same modified Chatbook adapter/service
boundary from a read-only working-tree mount and the reviewed package root from
a read-only model mount.

1. Guided `Auto` selected CPU and generated one owner-private lazy multi-model
   configuration.
2. The first deliberate operation launched one child and returned the exact
   typed `supertonic` and `pocket_tts` catalog described above.
3. First Supertonic synthesis returned a structurally valid complete PCM16
   WAV.
4. Saving a second guided generation while Running did not mutate the live
   child or artifact.
5. Deliberate replacement stopped the first child, retired its exact artifact,
   and launched one successor with the staged generation.
6. A forced unexpected exit invalidated the second process generation and
   removed its artifact before a later deliberate operation launched one
   recovery child.
7. Explicit shutdown reaped the recovery child. All three owned children had
   terminal return codes, no PID remained, the supervisor owned no generation
   or lifecycle task, and the private artifact root was empty.
8. An independent container process-table check found only the container's
   idle control process; both model SHA-256 digests still matched the reviewed
   values.

### Exact committed-revision reruns

Both platform journeys were rerun after the implementation commit was rebased
onto the latest `dev` and all PR review amendments were applied:

- Linux reproduced the exact catalog and WAV metadata below, kept the Running
  generation unchanged after staging, retired the prior generation on apply,
  invalidated the forced-crash generation, recovered on the next deliberate
  operation, and left no owned child or artifact.
- macOS exercised the same three-child lifecycle against the Homebrew 0.5.1
  binary: initial synthesis, staged replacement, forced crash, recovery, and
  explicit shutdown all passed with no owned child or artifact remaining.
- The independent macOS process scan also observed one unrelated pre-existing
  Homebrew audio.cpp 0.4 process. It predated the UAT and was neither adopted,
  stopped, nor modified; owned-child assertions used the three exact PIDs
  launched by the supervisor.

## WAV evidence

- Container/codec: RIFF/WAVE PCM16.
- Channels/sample rate: mono, 44,100 Hz.
- Frames/duration: 146,539 frames, approximately 3.322880 seconds.
- Total/audio bytes: 293,122 / 293,078.
- SHA-256: `a7b5fd967f5ec30af5438bab1fc3fb065cce2e0939579a152bbd59da2c5513b0`.
- Structural validation: passed both the Chatbook adapter validator and host
  audio-file inspection.
- Human audible confirmation: passed on 2026-08-09; the user confirmed hearing
  the retained sample generated by this exact run.

### Final committed-revision macOS WAV evidence

- Container/codec: RIFF/WAVE PCM16.
- Channels/sample rate: mono, 44,100 Hz.
- Frames/duration: 206,603 frames, approximately 4.684875 seconds.
- Total/audio bytes: 413,250 / 413,206.
- SHA-256: `325ad575e3a220a0657fc680032b6b2d6d6a66bb463eadb864c51636e79b5a6c`.
- Structural validation: passed through the real Chatbook native adapter and
  independent host `afinfo` inspection.
- Human audible confirmation: passed on 2026-08-10; the user confirmed hearing
  clear speech from the retained sample. The final review-fix commit reproduced
  that file byte-for-byte under both `cmp` and SHA-256, so the confirmed audible
  datum applies to the exact bytes generated by the commit above.

### Linux WAV evidence

- Container/codec: RIFF/WAVE PCM16.
- Channels/sample rate: mono, 44,100 Hz.
- Frames/duration: 206,603 frames, approximately 4.684875 seconds.
- Total bytes: 413,250.
- SHA-256: `bca686cb97a49f9322d065e8bca477ce509c12e8ccf007ce396583b21c45218a`.
- Structural validation: passed through the real Chatbook native adapter and
  an independent standard-library WAV inspection on Linux.

## Issue found during UAT

The first real-package attempt failed before launch because launch
revalidation required the discovery-level state to be `exact`. A folder with
both reviewed packages is discovery-level `ambiguous` even though each
explicitly accepted candidate retained an exact matching recipe, root,
configuration, and weight identity. The implementation now revalidates the
accepted candidate identity rather than the aggregate discovery label. A
same-root multi-model regression failed before this correction and passes
afterward.

## PR review amendments

- The public materializer now has complete Google-style `Args`, `Returns`, and
  `Raises` documentation.
- User-selected executable paths pass through the centralized arbitrary-path
  safety policy before regular-file/executable checks. The native Homebrew
  symlink remained accepted in the exact-commit macOS journey.
- Expected validation failures remain sanitized and quiet. Unexpected launch
  revalidation, generation cleanup, and artifact cleanup exceptions now retain
  only a bounded phase plus an allowlisted failure category; tests prove raw
  messages and custom exception type names cannot enter diagnostics.

## Automated evidence

- Exact committed revision, complete affected TTS/UI/privacy surface outside
  the sandbox: `1109 passed in 13.98s`.
- Generated-launch/supervisor/managed-integration subset: `144 passed` before
  the same-root amendment; the amended and hard-link ownership tests are
  included in the 1,109-test run.
- Ruff check: passed.
- Ruff format check: passed after formatting the two amended files.
- Scoped mypy for the nine changed core TTS/STTS source files: passed.
- Compileall for the changed TTS event and runtime packages: passed.

## Release-gate result

TASK-13201's automated, macOS, Linux, structural-WAV, audible-playback, process,
artifact, privacy, static-analysis, and exact-commit evidence gates are
satisfied. A future rebase or runtime-code amendment would require rerunning
the affected exact-revision gates.
