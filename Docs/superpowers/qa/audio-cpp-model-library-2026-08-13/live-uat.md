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
- Host server: official audio.cpp 0.5.1 at pinned runtime commit
  `238ab6a9e321c17de8e120559f57efeedaeb1345`. Its pinned Inflect guide requires
  eSpeak-ng and documents explicit library/data options when the default names
  are not loader- and data-search discoverable. The host had eSpeak-ng 1.52.0,
  a dynamic library, and English data installed, but installation did not prove
  discoverability: a basename-only dynamic-load probe failed, and the server
  binary exposed neither an eSpeak direct link nor an embedded loader search
  path. The pinned Inflect defaults use the library basename when explicit
  session paths are absent. The prerequisite was therefore **not validated**.
  No private paths, submitted text, audio bytes, credentials, or generated
  configuration contents are recorded here.
- Path-free retry package: `audio-cpp-supertonic-3-f16`, artifact variant
  `f16`, package variant `supertonic_3_f16`, at the same pinned inventory
  commit. Declared and downloaded size: 312,784,196 bytes. Verified SHA-256:
  `b312b57797d40ac5c09d915893dbdbaf6405b7dc043f544776c5c95712dff88c`.
  This recipe has no external runtime-library prerequisite.

## Results

- **PASS — curated install:** the real shared consent, downloader, size, and
  checksum path installed the exact package with activation disabled. No
  readiness marker, default-selection mutation, server launch, or socket
  appeared during install.
- **PASS — detached draft and return authority:** unrelated dirty Speech/TTS
  fields remained byte-for-byte equivalent in the detached snapshot. The exact
  installed root was rescanned under lease and merged as one unsaved package.
  A mounted slow-lease reproduction exposed a publish-before-ack recompose race;
  the acknowledgement ordering regression is now covered. A fresh full-app run
  waited for both exact remount and lease release, enabled the current mounted
  Save control, and exercised the actual Save action. The exact managed identity
  and three unrelated dirty draft families persisted; no autosave occurred
  before the click.
- **PASS — server lifecycle and catalog:** a deliberate Managed Test/Start
  launched one owned child. Health and catalog checks passed and exposed exactly
  one `inflect_v2` TTS model. The runtime lease was busy while the child ran and
  available after stop.
- **PARTIAL — Inflect sample generation:** the real server returned structured
  HTTP 500 for the pinned Inflect recipe on both automatic Metal selection and
  explicit CPU. Chatbook reported bounded `generation_failed`; no playable WAV
  existed.
  The retained privacy-safe server output did not identify a diagnostic cause.
  Because basename loading failed independently, the HTTP 500 may reflect an
  eSpeak loader failure, but that inference is not a demonstrated server cause.
  This run was not a prerequisite-complete compatibility test and does not
  establish audible success.
- **PASS — path-free Guided sample generation:** in a new isolated root, the
  production acquisition service preflighted and provisioned the exact
  Supertonic artifact with activation disabled. Settings returned and merged
  that exact managed identity, the real Save action persisted its recipe,
  variant, public model ID, and canonical installed root, and a fresh app
  process reloaded those saved values. Deliberate Test/Start then exposed only
  `supertonic-3-f16` and generated a 319,904-byte PCM16 mono 44.1 kHz WAV through
  the production TTS service. Human playback confirmed the speech was
  intelligible.
- **PASS — removal authority:** removal was blocked while referenced, remained
  blocked without acknowledgement, and completed only after explicit resolution
  and acknowledgement. Runtime handles were released only after child stop and
  the final package fingerprint was rechecked before public service deletion.

## Automated and mutation evidence

- An unrestricted exact 17-file matrix reproduced four mount/readiness races:
  `test_audio_cpp_presentation_reveals_slow_load_once_and_keeps_error_retry`,
  `test_curated_install_click_reaches_the_shared_consent_modal`,
  `test_external_rail_mounts_through_the_existing_deferred_view_pattern`, and
  `test_external_copy_uses_task6_plan_and_stop_uses_the_shared_service`
  (1,284 passed, 4 failed). Each now synchronizes on the exact composed control
  or visible mounted action rather than a zero-time pause; the four-test
  reproduction passed after restoration. A later unrestricted matrix isolated
  `test_real_settings_return_acknowledges_before_draft_remount` (1,287 passed,
  1 failed): merge could recompose while the initial Guided `Select` was still
  mounting. The production transaction now brackets merge with the current
  Save action's mount signal before and after recompose. The complete ordered
  handoff file then passed 122 tests.
- Earlier exact sandbox matrix: 1,270 passed and 18 failed. Every failure was the
  expected sandbox `PermissionError` while binding an ephemeral loopback
  fixture; no product/test-synchronization failure remained. A final
  unrestricted invocation was requested but rejected before process start by
  the runner's approval-usage limit. That host gate was retried successfully
  below.
- Fresh unrestricted retry: **1,288 passed**, 5 existing dependency/deprecation
  warnings, in 712.79 seconds. This is the exact 17-file release matrix with
  loopback fixtures enabled; no test was deselected and no failure remained.
- All five required mutations produced the named red test: provisioning without
  `activate=False`; stale-draft acceptance; runtime-handle release before child
  stop; bypassing public deletion; and skipping the removal fingerprint recheck.
  Restored batch: 5 passed.

## Release gate

UAT is **PASS**. The path-free Supertonic run covers install → exact-root return
→ Save → structurally and audibly valid sample generation, and the fresh
unrestricted matrix is green. The earlier advanced `server.json` guidance
remains diagnostic for Inflect only and is not needed to close the Guided path.
