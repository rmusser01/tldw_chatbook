# TASK-13206 clone voice bundle portability UAT

Date: 2026-08-12
Result: Audible UAT passed — final independent review pending
Mounted UAT commit under test: `6eab86144`
Final automated code/test commit under test: `77a5e22b2`

This artifact contains sanitized engineering evidence only. It retains no
audio, transcript, source/bundle/staging path, checksum, credential, provider
configuration or origin, generated configuration, or private runtime value.

## Environment and safe identity

- Host: macOS 15.6, Darwin 24.6.0, arm64.
- Python: 3.12.11.
- Profile schema: v4.
- Ordinary reference-bearing wire: sanitized v2.
- Recipe: `audio-cpp-0.5.1.pocket_tts.pocket_tts_english_bf16`, revision 2.
- Model ID: `pocket-tts-english-bf16`.

Each launch used a fresh task-owned `HOME`, XDG config/data, application data,
profile store, model, generated-configuration, and runtime environment. All
environment variables were set before application imports. The developer
profile database was not opened, and the task-owned roots were removed after
the run.

## Partial service-layer setup

A preliminary isolated service-layer harness used the production repository,
wire encoder, and portability service to create a reference-bearing profile,
inspect sanitized v2, publish an acknowledged bundle, import it inactive into
an independent store, and reopen that store. It did not mount Chatbook, drive
the warning UI, inspect a real model root, create generated configuration,
launch audio.cpp, or establish the production dependency projection. Those
claims are intentionally excluded from this layer.

## Production-mounted Pilot journey

The mounted-UAT commit above was then exercised through mounted production
`TldwCli`, `STTSScreen`, `STTSWindow`, and `STTSProfileLibrary` instances with
the real app-owned profile repository and bundle portability service. The
production warning, export-choice, review, inactive-consent, library-detail,
unmount, and composite app-shutdown paths ran normally. Only external chooser
authority and non-launching capability observation were deterministic: the
file picker returned the task-owned transfer file, and the capability observer
reported the safe recipe/model above as catalog-visible while its exact
dependency was missing. No provider process, model package, network, or audio
generation was involved.

### Launch A

- Selecting explicit bundle export opened the production plaintext warning.
- Cancelling that warning left the destination unpublished.
- Repeating the action, checking acknowledgement, and continuing published the
  real bundle through the app-owned service.
- No character assignment was created.
- Composite app shutdown joined the portability owner; its sessions, admitted
  calls, owned calls, workers, and inspection reservations were empty, and its
  operation root contained no staging/output residue.

```text
launch=A-mounted warning_refused_before_publication=true
bundle_published_after_ack=true assignments=0 sessions_tasks=0 owned_residue=0
```

### Launch B and independent restart

- Import opened the production plaintext warning before the source chooser.
- After acknowledgement, production inspection showed the safe recipe/model
  facts and required explicit inactive consent before Create.
- Production library detail displayed `Needs compatible model` after import.
- An independent Chatbook restart reopened the profile and reproduced
  `Needs compatible model` using the same deterministic missing-dependency
  observation.
- Import/restart created no character assignment. Both app shutdowns left no
  portability sessions/tasks or staging/output residue.

```text
launch=B-mounted-import warning_before_picker=true
dependency_projection="Needs compatible model" assignments=0
sessions_tasks=0 owned_residue=0

launch=B-mounted-restart
dependency_projection="Needs compatible model"
recipe_id=audio-cpp-0.5.1.pocket_tts.pocket_tts_english_bf16
recipe_revision=2 model_id=pocket-tts-english-bf16
assignments=0 sessions_tasks=0 owned_residue=0
```

Cancellation/unmount races and transactional rollback/partial-row prevention
remain automated-test evidence; this mounted journey did not inject a
mid-worker cancellation or storage failure and does not relabel those tests as
live observations.

## Automated verification on the commit under test

- The complete planned 20-module scoped matrix finished with `2278 passed, 2
  skipped, 3 failed in 459.34s`. The three failures were exclusively the real
  managed-child tests: this execution sandbox denied loopback socket binding
  (`PermissionError`), and the two dependent cases reported no private
  audio.cpp loopback port. No app-ownership fake or UI timing failure remained.
- The same three exact real-child node IDs were then rerun on the parent host
  at revision `d3d60abcb`: `3 passed in 2.33s`. Together, the sandbox matrix
  and host rerun close the planned scoped automated matrix at `2281 passed, 2
  skipped`. This host result is process/integration evidence only; it did not
  provision the clone model or perform audible UAT.
- The focused old-reader, rollback, composite-shutdown, runtime-privacy,
  UI-privacy, and ownership regressions passed.
- The runtime collaborator regression additionally walked the full linked
  exception graph, rendered the public traceback, inspected production-frame
  locals, and captured logs plus adapter events/requests; its private value and
  private exception type were absent from every available surface.
- Ruff check, the planned Ruff format check, scoped mypy, normalized legacy
  mypy comparison, CSS bundle synchronization, and `git diff --check` passed.

## Exact real-model and audible UAT

The real-model run was performed on revision `3583343d1` using the production
`TTSService` Guided path. The installed audio.cpp 0.5.1 executable matched the
previously reviewed SHA-256
`3de9bdb0fd1443110b73bdf5cc196e43ed9f143b47595b4fcd59e4a1ed18d467`.
The official `audio-cpp/audio.cpp-gguf` PocketTTS English bf16 package was
219,096,064 bytes and matched the previously reviewed SHA-256
`267e774a671138c4ebbc1d6d9d73af92f4a8e83a64b45b84f3457ac700ad0cc9`.

- Dependency observation was `exact` for recipe
  `audio-cpp-0.5.1.pocket_tts.pocket_tts_english_bf16`, revision 2, model
  `pocket-tts-english-bf16`.
- Production generation returned a complete PCM16 mono WAV at 24,000 Hz:
  71,040 frames, 142,124 total bytes, and 2.96 seconds.
- The reference playback process exited 0, followed by generated playback
  exiting 0. Human audible confirmation: **yes**.
- Service shutdown left no owned audio.cpp process, private reference
  materialization, or generated configuration artifact.

The evidence intentionally retains no private path, transcript, audio bytes,
or audio checksum.

## Full-suite terminal result and exact-base classification

A complete repository run finished on feature revision `8821c3293`, based on
exact `origin/dev` revision `a4b16b1e2`, with:

```text
383 failed, 41064 passed, 241 skipped, 4 xfailed, 59 errors
6840.46s (1:54:00)
```

The raw terminal output SHA-256 was
`6ac513ac3b83a87f36751f568bcd26c297cfc3fcb7ff9f7e7de8f947dfb0aa63`;
the JUnit SHA-256 was
`8a976be39ab3be0473209d2d723b46fc9908bcf6b8935d155f21fb3e23a05cc1`.
JUnit contained 441 unique failure/error node IDs. Five overlapped files
changed by this feature: three were the sandbox-blocked real-child nodes and
passed on the host; two exposed stale test contracts and were corrected in
`8512b5537`, then passed both focused and full-file reruns. Exact-node execution
of the other 436 nodes on detached base `a4b16b1e2` produced `373 failed, 59
passed, 2 skipped, 25 errors in 193.69s`. The feature/base comparison left one
reference-free UAT fixture mismatch, corrected in `285399506`; its exact node
and complete file then passed. No base-green, feature-red node remained.

The consolidated 411 baseline-red node IDs are checked in beside this report
as `baseline-red-nodes.txt`. The SHA-256 of its node-only lines is
`2894b62c20d160c916354c6b2da90077b6c2df9ad6c774398c8acc89280f8f2f`.
This inventory includes 397 nodes directly red on the detached base plus 14
nodes whose sequential base rerun passed the call but reproduced the same
blocked-Hugging-Face-egress teardown error (`14 passed, 14 errors in 29.62s`).
The largest file-level families are OpenAI realtime sessions (34), feed server
(33), default-assistant migration (19), Library shell (15), Library file/notes
Git (15), media-state ownership (13), model stream fetch (12), model provision
fetch (10), and QwenCloud (10). The checked-in list retains every exact node,
including smaller families. It is follow-up evidence, not a claim that the
repository suite is green.

## Latest-dev rebase bridge and final feature gates

The final automated code/test tree `77a5e22b2` has clean merge-base
`f7fe006ca` and is 71 feature commits ahead. Every one of the 11 intervening
upstream commits from `a4b16b1e2..f7fe006ca` was inspected. Their net delta is
the library-ingest option parity feature: 20 files, including 10 test files.
The sole path overlap with Task 13206 is `tldw_chatbook/app.py`; upstream
changes are confined to library-ingest option projection and persistence,
while Task 13206 changes are confined to app-owned TTS service construction
and audio shutdown. There is no shared hunk or semantic contract.

- All 10 upstream-changed test files passed: `804 passed in 32.58s` (JUnit
  SHA-256
  `45cb06af28b35f2dbd3917e1d43e49791e8bd4205ef601de3035a3a68ce374b2`).
- The exact 20-module Task 13206 matrix produced `2278 passed, 2 skipped, 3
  failed in 361.67s` in the sandbox. The three failures were only the known
  loopback-denied real-child nodes; their exact host rerun passed `3 passed in
  2.05s`. Aggregate: `2281 passed, 2 skipped`.
- Changed-Python Ruff check passed; the planned plus final-review Ruff format
  set passed; the format-touched test file passed `56 passed`; scoped mypy
  passed all 17 sources; CSS bundle sync and `git diff --check` passed.
- The detached exact-merge-base normalized legacy mypy inventory contained 10
  diagnostics, the feature inventory contained 9, and the new-diagnostic set
  was empty.
- Relevant inventories produced `88 passed` and the two persistent-diagnostic
  census failures. Both exact nodes reproduced on detached `f7fe006ca`, so
  they are current-base failures rather than a Task 13206 regression.
- Task-ID uniqueness checked 524 refs and 86 worktrees with zero duplicate-ID
  violations.

The audible requirement and requested affected/scoped/static verification are
complete. The task remains In Progress and its acceptance criteria remain
unchecked until the fresh independent final review is accepted; the full-suite
baseline failures above are preserved for the separate follow-up task/PR.
