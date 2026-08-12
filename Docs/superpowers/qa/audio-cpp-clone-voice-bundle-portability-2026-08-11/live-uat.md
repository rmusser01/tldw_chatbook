# TASK-13206 clone voice bundle portability UAT

Date: 2026-08-12
Result: Partial — real model and human audible gate pending
Commit under test: `6eab86144`

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

The exact commit above was then exercised through mounted production
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
- The focused old-reader, rollback, composite-shutdown, runtime-privacy,
  UI-privacy, and ownership regressions passed.
- The runtime collaborator regression additionally walked the full linked
  exception graph, rendered the public traceback, inspected production-frame
  locals, and captured logs plus adapter events/requests; its private value and
  private exception type were absent from every available surface.
- Ruff check, the planned Ruff format check, scoped mypy, normalized legacy
  mypy comparison, CSS bundle synchronization, and `git diff --check` passed.

## Pending real-model and audible gate

No exact model package was provisioned inside Launch B, audio.cpp was not
launched, and no speech was generated or played. No human audible confirmation
occurred. The deterministic missing-dependency observer proves the production
inactive projection only; it is not evidence of real dependency discovery or
generation. Acceptance criterion 7 and TASK-13206 therefore remain open.

To complete the gate, use the same isolated two-launch procedure with the exact
recipe/model above actually pre-provisioned for Launch B. After restart,
configure/apply that dependency through the production UI, refresh until the
imported profile becomes available, generate and play speech with it, and have
a human confirm yes or no that the expected voice is audible. Record only the
yes/no result and safe metadata above.
