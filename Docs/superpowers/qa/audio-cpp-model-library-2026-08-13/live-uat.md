# TASK-13207 audio.cpp Model Library live UAT

Date: 2026-08-14

## Environment and identity

- Opt-in run used isolated HOME, XDG configuration/data, model-store, generated
  runtime, and application-data roots. Network and loopback access were enabled
  only for the live run.
- Official package: `audio-cpp-inflect-micro-v2-orig`, variant `orig`, pinned
  inventory commit `597048d9a920592808d7d4e2acd7b9c4596a143a`.
- Declared and downloaded size: 72,082,176 bytes. Verified SHA-256:
  `d4af1cb6a92cdd8be550e8e7c25805ece222ec0f8e75daf26fc00b4e04ef4b03`.
- Compatible host prerequisite: official audio.cpp 0.5.1 server. No private
  paths, submitted text, audio bytes, credentials, or generated configuration
  contents are recorded here.

## Results

- **PASS — curated install:** the real shared consent, downloader, size, and
  checksum path installed the exact package with activation disabled. No
  readiness marker, default-selection mutation, server launch, or socket
  appeared during install.
- **PASS — detached draft and return authority:** unrelated dirty Speech/TTS
  fields remained byte-for-byte equivalent in the detached snapshot. The exact
  installed root was rescanned under lease and merged as one unsaved package.
  A mounted slow-lease reproduction exposed a publish-before-ack recompose race;
  the acknowledgement ordering regression is now covered. The live Save action
  remained fenced in the final run, so the guided Save portion is **PARTIAL**.
- **PASS — server lifecycle and catalog:** a deliberate Managed Test/Start
  launched one owned child. Health and catalog checks passed and exposed exactly
  one `inflect_v2` TTS model. The runtime lease was busy while the child ran and
  available after stop.
- **PARTIAL — sample generation:** the real server returned structured HTTP 500
  for the pinned Inflect recipe on both automatic Metal selection and explicit
  CPU. Chatbook reported bounded `generation_failed`; no playable WAV existed.
  The host server emitted bounded structural diagnostics and no dropped lines.
  Health/catalog success plus identical CPU/Metal failure classifies this as a
  host binary/package generation incompatibility, not audible success.
- **PASS — removal authority:** removal was blocked while referenced, remained
  blocked without acknowledgement, and completed only after explicit resolution
  and acknowledgement. Runtime handles were released only after child stop and
  the final package fingerprint was rechecked before public service deletion.

## Automated and mutation evidence

- Final focused 17-file matrix: 1,285 passed and 3 suite-load scheduling
  failures; all three failed tests passed together in the immediate isolated
  rerun. The earlier sandbox-only run classified 18 loopback bind failures and
  1,267 passes.
- All five required mutations produced the named red test: provisioning without
  `activate=False`; stale-draft acceptance; runtime-handle release before child
  stop; bypassing public deletion; and skipping the removal fingerprint recheck.
  Restored batch: 5 passed.

## Release gate

UAT is **PARTIAL**. Do not mark TASK-13207 Done until a clean isolated run both
saves the returned Guided draft and produces a human-audible sample with the
exact supported server/package pair. The current evidence must not be read as
audible generation success.
