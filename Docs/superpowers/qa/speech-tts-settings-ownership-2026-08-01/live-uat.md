# Live External audio.cpp UAT

Date: 2026-08-01
Task: TASK-1989
Branch: `codex/task-1985-audio-cpp-settings-experience`
Starting product commit: `f2251128a`
Tested repository origin: `https://github.com/rmusser01/tldw_chatbook.git`
Tested working-tree base: `f2251128acfceb4b81edd397a66a85db0a4e1dd2`
Decision: [ADR-039](../../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md)
Speech Lab remediation: [ADR-040](../../../../backlog/decisions/040-speech-lab-current-result-and-auto-play.md)

## Evidence boundary

- Application: real Chatbook process served by the repository's
  `tldw-serve` entry point with `TLDW_CONFIG_PATH` plus isolated task-specific
  XDG/data roots.
- UI driver: Playwright CLI and the Codex in-app Browser controller against
  the same live browser-served Textual app.
- audio.cpp ownership: user-supplied, already-running external server.
  Chatbook and this harness do not launch, adopt, restart, supervise, signal,
  or stop it.
- Tested provider: canonical `audio_cpp`; configured origin is loopback
  `http://127.0.0.1:8080`.
- Initial read-only server observation: health `ok`, one TTS model, ten voices.
  The final UAT-03 retest at the same configured origin reported health `ok`
  with two TTS models. Catalog identifiers are permitted evidence; user model
  paths and model contents are not recorded.
- Synthesis text, characters, credentials, screenshots, and diagnostics are
  synthetic. The environment credential is a non-secret sentinel.
- Complete-WAV structure and playback-control handoff are technical evidence.
  The user separately confirmed hearing the synthetic line through headphones
  after the response control was invoked with audio.cpp selected globally.

## Journey ledger

| Journey | Result | Evidence |
| --- | --- | --- |
| UAT-01 first-time setup and audible response | Passed | Settings search completed in 22.661 seconds; `settings-discovery-timed-pass.png`; `settings-saved-not-checked-final.png`; explicit Test and Refresh at configuration revision 1/runtime revision 2/catalog revision 3; `speech-lab-generation-complete.png`; synthetic Console response and response-control handoff in `console-synthetic-response-pass.png` and `console-response-playback-invoked.png`; user explicitly confirmed hearing the synthetic line through headphones; the redesigned current-result retest generated `Ready · WAV · 0:11`, kept Play and Export visible, entered `Playing current result…`, and received separate audible confirmation in `speech-lab-current-result-ready.png` and `speech-lab-current-result-playing.png`; the repaired canonical OpenAI Console endpoint reached the task-owned loopback stub and returned its unique deterministic response in `uat-01-openai-canonical-session.png` and `uat-01-openai-canonical-response.png` |
| UAT-02 offline save and recovery | Passed | With the listener confirmed offline, explicit Test retained `Saved`, reported runtime `Unavailable` and catalog `Stale`, kept `supertonic-3` and `F1` visibly pinned as `Unverified`, and left generation blocked; after the user-owned restart, explicit Test returned runtime and catalog to `Ready`, retained the exact IDs without rewriting settings, and removed both stale trust qualifiers; `uat-02-offline-no-substitution-pass.png`; `uat-02-recovery-ready-pass.png` |
| UAT-03 exact and dynamic choices | Passed after P1 fix | Explicit Refresh accepted `supertonic-3` and `F1`; exact values survived Settings → Lab → Settings in `uat-03-exact-save-complete.png` and `uat-03-exact-navigation-roundtrip.png`; defaults-only dynamic save completed in `uat-03-dynamic-save-complete.png`; the post-generation config retained only `first_available` and `server_default` with no resolved identifiers; missing-choice retention remains covered by UAT-02; the final two-model catalog exposed `pocket-tts-en` and `supertonic-3`, and the same Chatbook process generated and entered playback for distinct 3-second and 2-second WAV results without changing or restarting the external server in `uat-03-pocket-tts-playing.png` and `uat-03-supertonic-playing.png` |
| UAT-04 Studio persistence and isolation | Passed | An exact Studio-only `audio_cpp` / `supertonic-3` / `F1` selection survived both Playground and preference-pane remounts in `uat-04-studio-effective-after-remount.png` and `uat-04-studio-fields-after-remount.png`; the bounded TOML change was confined to `[speech_studio]` while global `[app_tts]` remained dynamic; `uat-04-studio-exact-save-complete.png`; UAT-06's unassigned canonical-character response completed through the normal global path without changing the Studio config hash |
| UAT-05 Reset Studio to global | Passed after P1 fix | A first global change left the exact Studio `F1` override effective in `uat-05-studio-override-wins-after-global-change.png`; reset deleted `[speech_studio.selection]`, the preference editor displayed inherited values in `uat-05-reset-to-global.png`, and a second global change appeared without copied Studio fields in `uat-05-inherits-second-global-change.png`; live UAT then exposed and fixed a stale process-local Playground draft: `uat-05-regression-before-reset-exact-f1.png` became `uat-05-regression-after-reset-server-default.png` in the same Speech session |
| UAT-06 character roleplay precedence | Passed after two P1 fixes | Created exact profile `Amber Watch F1` (`audio_cpp` / `supertonic-3` / `F1` / WAV), assigned it only to canonical character `Amber Warden`, and left `Quiet Cartographer` visibly on `Use global default`; the isolated assignment row maps only character ID 2 to the profile while character ID 3 has no assignment; assigned and unassigned response controls produced non-empty RIFF/WAVE artifacts of 883,538 and 303,394 bytes respectively; the Studio config SHA-256 remained `845f9cf434f46c7b62cc3b6bc1ebccd75e0aef778c6ef996b72d3a095a90f7dd`; `uat-06-profile-saved.png`, `uat-06-assigned-profile.png`, `uat-06-assigned-response-selected.png`, `uat-06-unassigned-global-default.png`, and `uat-06-unassigned-response-selected.png` |
| UAT-07 character preview safety | Passed | Previewing `Amber Watch F1` loaded its exact persisted selection, generated `Ready · WAV · 0:11`, and entered `Playing current result…`; leaving without adoption kept the config byte-for-byte identical at SHA-256 `845f9cf434f46c7b62cc3b6bc1ebccd75e0aef778c6ef996b72d3a095a90f7dd`; a second preview plus explicit Adopt remained unsaved until Save, then changed only `[speech_studio]` from revision 4 to 5 and added the exact audio.cpp selection; the profile and one-character assignment stayed at revision/count 1; `uat-07-profile-preview-loaded.png`, `uat-07-unadopted-preview-ready.png`, `uat-07-unadopted-preview-playing.png`, `uat-07-adopted-unsaved.png`, and `uat-07-adopted-saved.png` |
| UAT-08 environment-managed credential | Passed after P1 fix | OpenAI showed only `Environment (OPENAI_API_KEY, read-only)` after restart and explicit local-fallback Clear; the action returned to `Set credential` with no shadowed-local or Clear state; ordinary Save added only the synthetic Base URL and Organization ID, and a structural scan found zero credential-key entries in the isolated TOML; `uat-08-environment-only-after-clear.png` |
| UAT-09 retained legacy providers | Passed for compatibility; provider live smoke unavailable/not live-tested | Visited OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk global setup forms and observed their retained connection/init fields; Studio exposed the supported ElevenLabs model, Chatterbox exaggeration/CFG, and AllTalk voice/format controls; passive visits left config SHA-256 `24acfb78c9f8b071de7f7554ad834c78f98ba7350cd76f0f7c1e6f7ada3e3d2a`; the six-provider TASK-1988 bridge fixture passed; no credential or local runtime was configured for a safe provider live smoke; detailed artifact names are enumerated in the UAT-09 section below |
| UAT-10 independent dependency status | Passed | The Settings inspector simultaneously showed audio.cpp runtime/catalog `Ready`, local STT `Ready`, and Kokoro, Chatterbox, and Higgs independently `Unavailable`; the same live process then generated `Ready · WAV · 0:11` through audio.cpp and entered `Playing current result…`; `uat-10-independent-dependency-status.png`, `uat-10-audio-cpp-ready-wav.png`, and `uat-10-audio-cpp-playing.png` |

## Evidence privacy review

- Passed visual review of all 50 curated PNG artifacts under
  `screenshots/`; every retained artifact is referenced by
  exact basename in this record.
- The screenshots contain only synthetic prompts and characters, permitted
  model and voice identifiers, loopback origins, bounded revisions, and safe
  UI status text. They expose no credential value, user model path or model
  contents, submitted private text, raw provider body, or provider process
  detail.
- The environment credential is shown only as the variable name
  `OPENAI_API_KEY` and read-only source provenance. The synthetic sentinel
  value is absent from every curated artifact and from the isolated TOML.
- Redundant diagnostic screenshots were removed from the curated repository
  inventory and retained only in the task-owned temporary directory for
  recoverability. Raw Playwright session state is not release evidence.
- Complete-WAV validation, playback-control handoff, and explicit human
  audible confirmation remain separately labelled throughout this record.

## UAT-01 technical observations

- Visible Settings search for `audio.cpp` reached Speech & TTS in 22.661
  seconds, below the 60-second limit.
- Saving the external origin produced separate `Saved` and `Not checked`
  states. The surface exposed no binary, server configuration, launch,
  adoption, restart, supervision, or stop control.
- Explicit Speech Lab Test reported Ready at saved configuration revision 1,
  runtime revision 2, and catalog revision 2. Explicit Refresh remained Ready
  and advanced only the catalog revision to 3.
- Direct Speech Lab synthesis of the synthetic line produced a 266,410-byte
  RIFF/WAVE artifact: 16-bit mono PCM at 44.1 kHz, 3.020023 seconds, with a
  44-byte header and 266,366 audio bytes. This is complete-WAV evidence, not
  audible-playback evidence.
- The Console used a task-owned loopback OpenAI-compatible response stub only
  to create the synthetic assistant line. The assistant response was selected,
  audio.cpp was selected as the global TTS default, and its visible speech
  control was invoked. Console intentionally auto-posts the playback event
  after complete synthesis because it has no second per-message Play button.
  `console-response-playback-invoked.png` records the response control's
  post-invocation state; it does not independently prove sound was heard.
- The user separately and explicitly confirmed hearing
  `Lantern check complete; the harbor is ready.` through headphones. Together
  with the deterministic WAV and control-handoff evidence, this passes UAT-01.

## Findings

### F-1700-05 — Completed Speech Lab WAV had no visible playback action (P0, fixed and audibly verified)

Live generation completed and exposed a valid WAV status, but the Result pane
showed neither Play nor Export in the supported viewport. The controls existed
and completion enabled them; the result/history composition and action-strip
geometry placed them outside the visible result region. The same state also
displayed `0:00 / 0:00` without a known duration, mounted an unpopulated take
history, and gave raw runtime diagnostics more authority than the artifact.

Speech Lab now presents one honest current result. Successful delivery keeps
Play and Export inside both the Result pane and the viewport at 120x40 and
80x24, focuses and reveals Play by default, reports only known format/duration
facts, and explains that the result is temporary until exported. The dead take
history is removed; connection details and the generation log are compact
collapsed rows; audio.cpp's non-applicable Language axis is hidden. A separate
Studio-only auto-play preference defaults off, persists sparsely, resets to
off, and never changes global or character settings. The focused remediation
suite passes 115 tests. The preceding related UI sweep passed 619 tests; its
sole failure was the old status-copy assertion, and that complete test module
passes after updating the assertion to the approved current-result contract.
All 2,088 TTS tests pass with 14 expected optional-dependency skips;
Ruff/compileall/CSS integrity/diff checks pass; and the one-time Impeccable
mechanical scan reports no findings.

For the live closeout, the user restarted only the Chatbook harness while the
external audio.cpp server remained independently owned and running. The
redesigned screen loaded with Play and Export visible before generation, no
audio.cpp Language axis, and compact collapsed parameter, connection, and log
rows. Generating the seeded synthetic playground sentence produced
`Ready · WAV · 0:11` plus the temporary-result ownership copy. Invoking the
visible Play action changed the status to `Playing current result…`; the user
then explicitly confirmed hearing the sentence. Evidence:
`speech-lab-current-result-ready.png` and
`speech-lab-current-result-playing.png`. This closes the P0.

### F-1700-02 — Offline voice discovery silently selected Server default (P1, fixed)

The first offline Test correctly retained the saved origin and reported the
runtime unavailable, but the Speech Playground replaced the visible exact
voice `F1` with `Server default`. The global Settings projector already
implemented ADR-039 by pinning missing or unverified exact identifiers; the
Playground's older pure projector instead selected the first model or server
default voice, and its voice worker collapsed an unverified observation into
an authoritative empty voice list.

The audio.cpp-only projection now keeps exact model and voice identifiers
visible as `Missing` or `Unverified`, blocks generation while either exact
choice lacks authority, and preserves exact choices through failed or
unverified voice discovery. Regression tests cover the pure model/voice
projection, the rebuilt Speech pane, the retained legacy widget, pending
rediscovery, authoritative removal, and discovery failure. After restarting
only the task-owned Chatbook harness while leaving audio.cpp stopped, the live
explicit Test showed `Saved`, runtime `Unavailable`, catalog `Stale`, model
`supertonic-3 (Unverified)`, and voice `F1 (Unverified)` without substitution.
The persisted origin and exact defaults remained unchanged.

### F-1700-03 — Recovered model retained a stale Unverified label (P1, fixed)

After the listener recovered, the runtime and catalog truthfully reported
`Ready` and the exact voice returned to plain `F1`, but the closed model
selector still painted `supertonic-3 (Unverified)`. Replacing a Textual
`Select`'s options resets a non-blank selector to the first value. When that
value is unchanged, its reactive watcher does not run, so the displayed prompt
can retain an obsolete label even though `Select.value` and the new option
catalog are correct.

Catalog application now forces the current prompt watcher after programmatic
option replacement while suppressing the resulting synthetic
`Select.Changed` message. A render-level regression test exercises the exact
same-ID label transition. The related pane/catalog suite passes 31 tests and
the retained audio.cpp Playground suite passes 94 tests. In the live recovery
retest, explicit Test advanced catalog revision 1 to 2, runtime and catalog
remained `Ready`, model displayed plain `supertonic-3`, voice displayed `F1`,
and the isolated config still contained exact modes and identifiers unchanged.

### UAT-03 live retest — exact, dynamic, and multi-model choices passed

An explicit Speech Lab Refresh on the recovered listener exposed the accepted
`supertonic-3` model and `F1` voice. Settings then saved the exact pair, left
the category with no unsaved changes, and returned through Lab with the exact
case-sensitive values intact. Evidence: `uat-03-exact-save-complete.png` and
`uat-03-exact-navigation-roundtrip.png`.

The same surface was restored to `First available` and `Server default`. The
application-owned Save completed and cleared the draft state, closing the live
retest for F-1700-04. Speech Lab then generated a complete temporary WAV and
reported `Ready · WAV · 0:11`. After generation, the isolated `[app_tts]`
section still contained only `default_model_mode = "first_available"` and
`default_voice_mode = "server_default"`; the exact model and voice keys were
absent. Evidence: `uat-03-dynamic-save-complete.png` and
`uat-03-dynamic-generation-ready.png`.

The final read-only listener check reported health `ok` with two offline TTS
models: `pocket-tts-en` and `supertonic-3`. In one restarted Chatbook harness
and without changing, restarting, signalling, or inspecting the user-owned
audio.cpp process, Speech Lab selected each exact model in turn. Pocket used
the advertised `Server default` voice, generated the unique synthetic phrase
`Pocket TTS live model test.` as `Ready · WAV · 0:03`, and entered
`Playing current result…`. The same live session then switched to Supertonic,
generated `Supertonic live model test.` as a separate `Ready · WAV · 0:02`
result, and entered playback. Evidence: `uat-03-pocket-tts-playing.png` and
`uat-03-supertonic-playing.png`. These are complete-result and playback-state
observations; no additional human audible confirmation is claimed. The
synthetic missing-exact no-substitution case remains covered by UAT-02.

### F-1700-11 — Server default could not recover from an unverified exact voice (P1, fixed and live verified)

Switching the Studio Playground from Supertonic to Pocket retained the exact
Studio voice `F1`, correctly marked it unavailable for Pocket, and disabled
Generate. Selecting `Server default` changed the visible choice but left the
pending exact-voice pin and the prior catalog admission result in place, so
Generate stayed disabled even though Pocket supports omission-based server
default synthesis.

The audio.cpp voice-change path now clears that pending exact pin when the
server-default sentinel is chosen and reprojects the already loaded catalog so
admission is recalculated from the new selection. The keyboard-driven
`test_unverified_exact_voice_blocks_until_server_default_is_selected`
regression failed before the change and passes afterward, including the
painted `Server default` prompt. The live Pocket and Supertonic generation and
playback observations above close the finding.

### UAT-04/UAT-05 live retest — Studio persistence, isolation, and reset

Studio saved an exact `audio_cpp` / `supertonic-3` / `F1` override. The
isolated config added only the sparse `[speech_studio.selection]` fields;
global `[app_tts]` remained `first_available` plus `server_default`. Leaving
and remounting both Playground and Studio restored the exact Studio choice.
Evidence: `uat-04-studio-exact-save-complete.png`,
`uat-04-studio-effective-after-remount.png`, and
`uat-04-studio-fields-after-remount.png`.

Changing the global voice policy to `server_default` did not displace the
saved Studio `F1` override. `Reset to Global` then removed the entire Studio
selection table, leaving only the schema/revision envelope. After a second
global change, the Studio editor truthfully showed inherited
`first_available` and `server_default` with no copied provider, model, or
voice field. Evidence: `uat-05-studio-override-wins-after-global-change.png`,
`uat-05-reset-to-global.png`, and
`uat-05-inherits-second-global-change.png`.

The reset also exposed F-1700-06: Playground still held `F1` as a bounded
session axis, so its next generated draft would have outranked the newly
inherited global. After the focused fix, a same-process live regression saved
the exact override, verified `F1` in Playground, reset it, and returned to a
visible `Server default`. The final config still had no
`[speech_studio.selection]`. Evidence:
`uat-05-regression-before-reset-exact-f1.png` and
`uat-05-regression-after-reset-server-default.png`.

UAT-04's normal non-Studio request comparison is now closed by UAT-06's
unassigned canonical-character response. It resolved through the global path,
produced a complete WAV, and left the Studio config hash unchanged.

### UAT-06 live retest — canonical character profile precedence and global fallback

Speech Lab generated an exact `audio_cpp` / `supertonic-3` / `F1` / WAV result
and saved it as the local profile `Amber Watch F1` at revision 1. The isolated
canonical character database maps `Amber Warden` to character ID 2 and
`Quiet Cartographer` to character ID 3. The profile repository contains one
assignment row only: source `local`, character ID 2, profile
`Amber Watch F1`. The Roleplay surface rendered that profile as available and
used by one character for Amber; Quiet visibly rendered `Use global default`
and has no assignment row. Evidence: `uat-06-profile-saved.png`,
`uat-06-assigned-profile.png`, and
`uat-06-unassigned-global-default.png`.

Invoking the selected Amber response's Speak action admitted the assigned
profile request and produced an 883,538-byte artifact whose first twelve bytes
decode as `RIFF....WAVE`. Invoking the same response control for Quiet admitted
the unassigned global path and produced a 303,394-byte artifact with the same
RIFF/WAVE structure. The Console completion route automatically handed both
complete artifacts to playback. This is deterministic complete-WAV and
playback-handoff evidence; no separate human audible confirmation was recorded
for these two roleplay clips. Evidence:
`uat-06-assigned-response-selected.png` and
`uat-06-unassigned-response-selected.png`.

The isolated Studio/global config SHA-256 was
`845f9cf434f46c7b62cc3b6bc1ebccd75e0aef778c6ef996b72d3a095a90f7dd`
before these roleplay flows and remained identical afterward. Character
assignment persistence therefore did not mutate global or Studio preferences.

### F-1700-07 — Delivered result never exposed Save result as profile (P1, fixed and live verified)

The redesigned current-result path stored a delivered artifact but omitted the
legacy profile-action lifecycle sync. Consequently a successful exact
audio.cpp result remained unable to expose `Save result as profile`, blocking
profile creation for UAT-06. Generation start and failure also omitted the
matching suppression transitions.

The playback and synthesis mixins now suppress and synchronize the action at
generation start/failure, then clear suppression and synchronize it only after
successful artifact delivery. A focused lifecycle regression failed before
the fix and passes afterward. The live retest generated
`Ready · WAV · 0:04` and visibly exposed the save action.

### F-1700-08 — Visible Save result as profile action was inert (P1, fixed and live verified)

After F-1700-07 exposed the action, activating it did nothing. The redesigned
playback mixin's button dispatcher had omitted the legacy
`audio-save-profile-btn` branch, so it never started the bounded profile-save
worker.

The dispatcher now starts the existing exclusive
`save_tts_result_as_profile` worker. A focused mounted-pane regression failed
when the name dialog did not open and passes after the branch was restored.
The live retest opened the modal, saved `Amber Watch F1`, rendered
`Voice profile saved.`, and persisted the exact audio.cpp request at revision
1. Evidence: `uat-06-profile-saved.png`.

### F-1700-09 — Environment projection was misclassified as a saved local fallback (P1, fixed and live verified)

After an explicit local-fallback Clear, OpenAI still claimed that a saved local
fallback was shadowed by `OPENAI_API_KEY`, retained `Replace credential`, and
offered another Clear. The isolated TOML contained no credential key. The
loader had correctly exposed the effective environment value through its
normalized compatibility projection, but the Settings credential-state helper
rescanned that runtime projection after inspecting the raw persisted config
and misclassified the effective environment value as local persistence.

When `COMPREHENSIVE_CONFIG_RAW` is available, credential provenance now treats
that raw mapping as the sole persistence authority. The normalized settings
mapping remains the fallback for focused/direct callers that do not have raw
config. A regression first failed with `local_saved=True` and
`local_shadowed=True`; the four focused credential-source and mutation tests
then passed after the fix.

The live app was restarted against the same isolated profile and synthetic
environment sentinel. OpenAI then displayed exactly
`Environment (OPENAI_API_KEY, read-only)`, `Set credential`, and no saved-local,
shadowed, Replace, or Clear state. Ordinary Save had changed the pre-UAT file
only by adding the synthetic Base URL and Organization ID; a case-insensitive
structural scan found no `OPENAI_API_KEY`, `api_key`, or credential entry. The
post-check config SHA-256 was
`24acfb78c9f8b071de7f7554ad834c78f98ba7350cd76f0f7c1e6f7ada3e3d2a`.
Evidence: `uat-08-environment-only-after-clear.png`.

### F-1700-10 — A delivered result could orphan prior playback controls (P1, fixed, independently reviewed, and live verified)

Independent closeout review found that publishing a newly completed result
while the prior result was playing could replace the authoritative card before
the old playback worker and artifact lease retired. That could hide the only
reachable Stop action, reject Studio auto-play of the new result, allow the old
result to be replayed during a slow stop, or publish a result after exact
profile navigation had retired it. A follow-up review also found that deferred
native delivery could keep `Save result as profile` hidden by publishing before
the generation lifecycle returned to idle.

Result delivery now owns one bounded transition token. It disables Play but
keeps Stop reachable, retires the prior playback worker and player before
replacing the card, fences profile-retired operations immediately before
publication, prevents timer cancellation from lying about the transition, and
clears the matching generation token before publishing. The replacement group
is also cancelled on unmount. Studio auto-play can therefore start only after
the new artifact becomes authoritative, and native profile-save eligibility is
restored in the same idle lifecycle as immediate delivery.

Four gated regressions cover active-playback replacement, pending-start
auto-play takeover, profile navigation during a slow stop, and replay attempts
during transition ownership. The full pane suite passes 36 tests and the
surrounding playback/catalog/navigation suite passes 195 tests. Ruff formatting
and lint, compile checks, diff checks, and final focused independent review are
clean; the reviewer reported no remaining actionable findings.

The live retest generated an 11-second first result, entered
`Playing current result…`, then generated the synthetic two-second line
`Second live take.` before the first result completed. Chatbook visibly
reported `Playback stopped`, published only the second result as
`Ready · WAV · 0:02`, restored Play, and kept `Save result as profile`
visible. The Studio auto-play variant remains explicitly automated rather than
mislabelled as live. Evidence: `f-1700-10-old-result-playing.png` and
`f-1700-10-replacement-ready.png`.

### UAT-09 live compatibility check — retained providers remain available without unsafe live claims

The global Configure Provider selector was visited for OpenAI, ElevenLabs,
Kokoro, Chatterbox, Higgs, and AllTalk. The forms rendered their retained
connection and initialization values: OpenAI source/Base URL/Organization ID;
ElevenLabs credential source, output format, stability, similarity, and style;
Kokoro device/ONNX/model/voice/max-token inputs; Chatterbox device, resource
directory, temperature, chunking, seed, and candidates; Higgs model/resource
paths, device, flash-attention, data type, and reference-duration controls;
and AllTalk server URL/language plus configuration-source inspector. Evidence:
`uat-08-environment-only-after-clear.png`, `uat-09-elevenlabs-setup.png`,
`uat-09-elevenlabs-tuning.png`, `uat-09-kokoro-setup.png`,
`uat-09-chatterbox-setup.png`, `uat-09-higgs-setup.png`, and
`uat-09-alltalk-setup.png`.

Speech Playground then exposed the deliberately supported Studio request
values: ElevenLabs model, Chatterbox exaggeration and CFG weight, and AllTalk
voice and WAV format. Evidence: `uat-09-studio-elevenlabs-model.png`,
`uat-09-studio-chatterbox-tuning.png`, and
`uat-09-studio-alltalk-voice-format.png`. Merely visiting these forms and
changing transient Playground provider axes left the isolated TOML unchanged
at SHA-256
`24acfb78c9f8b071de7f7554ad834c78f98ba7350cd76f0f7c1e6f7ada3e3d2a`.

No retained provider had both a safe credential/runtime and approved live
target in this scratch profile. Their network generation is therefore
`Unavailable/not live-tested`, not a live pass. The required compatibility
gate was rerun instead:
`test_every_legacy_provider_retains_its_accepted_request_shape` passed all six
provider cases and confirmed each remains behind the temporary legacy bridge.

### UAT-10 live retest — external readiness remains independent of local dependencies

With audio.cpp selected, the global Settings inspector reported configuration
`Saved`, provider runtime `Ready`, and catalog/voices `Ready` at the same time
as independent local dependency rows reported STT `Ready`, Kokoro
`Unavailable`, Chatterbox `Unavailable`, and Higgs `Unavailable`. Evidence:
`uat-10-independent-dependency-status.png`.

Without changing that process or external-server ownership, Speech Playground
showed `audio.cpp is ready`, generated the synthetic sample as
`Ready · WAV · 0:11`, kept the visible Play action available, and entered
`Playing current result…` when invoked. This is deterministic complete-WAV and
playback-handoff evidence; no separate human audible confirmation was recorded
for this specific UAT-10 run. Evidence: `uat-10-audio-cpp-ready-wav.png` and
`uat-10-audio-cpp-playing.png`. The focused runtime-projection and mounted
Settings-inspector regressions also passed, including the all-local-
dependencies-missing fixture.

### UAT-07 live retest — profile preview remains transient until explicit adoption and Save

Selecting `Preview` for `Amber Watch F1` opened the Playground with the visible
banner `Profile preview — exact saved selection.` and exact
`audio.cpp` / `supertonic-3` / `F1` / WAV controls. Generating the synthetic
Playground sentence produced `Ready · WAV · 0:11`; the documented Play action
then entered `Playing current result…`. Navigating away without selecting
Adopt left `/tmp/tldw-task-1989-uat/config.toml` byte-for-byte identical to its
pre-preview copy at SHA-256
`845f9cf434f46c7b62cc3b6bc1ebccd75e0aef778c6ef996b72d3a095a90f7dd`.
This is deterministic complete-WAV and playback-handoff evidence; no separate
human audible confirmation was recorded for this preview. Evidence:
`uat-07-profile-preview-loaded.png`,
`uat-07-unadopted-preview-ready.png`, and
`uat-07-unadopted-preview-playing.png`.

Repeating Preview and selecting `Adopt as Studio Preferences` opened the
Studio-only editor with an explicit unsaved-adoption banner and exact values.
The config was still byte-for-byte identical before Save. Selecting
`Save Studio Preferences` changed only the Studio namespace: revision 4 became
5 and `[speech_studio.selection]` gained `audio_cpp`, exact
`supertonic-3`, and exact `F1`. No `[app_tts]`, connection, credential, or
Console field changed. The profile repository still contained one profile at
revision 1 and one assignment for character ID 2. Evidence:
`uat-07-adopted-unsaved.png` and `uat-07-adopted-saved.png`.

### F-1700-06 — Reset retained a stale exact Playground draft (P1, fixed and live verified)

`Reset to Global` correctly deleted persisted Studio overrides and repainted
the preference editor as inherited, but the Lab window intentionally keeps
bounded Playground axes across internal view switches. The old `F1` axis was
therefore remounted as a current exact Studio control. Studio generation
freezes those visible controls into an exact `TTSStudioDraftSelection`, so the
stale axis would still outrank both the empty Studio store and the global
`server_default` policy. The reset status copy claiming that values now
inherited global was behaviorally false.

The Studio persistence notification now marks reset-to-global explicitly.
Only that successful reset clears the process-local Playground axis draft;
ordinary Save and Revert retain their existing behavior, and no global or
character owner is touched. A focused regression failed on the stale draft,
passes after the change, and the full Studio preference, Playground pane,
Studio storage, and effective-settings suites pass. The live same-session
retest visibly changed `F1` to `Server default` after reset while the TOML
retained only the Studio schema/revision envelope.

### F-1700-04 — Global defaults-only Save never reached the application (P1, fixed and live verified)

Selecting `First available` and `Server default` in the live Settings surface
showed the local `Saving…` state indefinitely and left the isolated TOML
unchanged. A copied-config probe proved the atomic mutation writer could apply
the exact set/delete proposal, so the failure was above persistence.

A production `TldwCli` regression reproduced the live boundary precisely: the
mounted panel posted its save, but custom messages from a widget inside the
pushed `SettingsScreen` did not bubble to the application owner. The existing
destination-only harness hid this because it mounted the screen beneath a test
App that directly captured the message. Global Settings now posts application-
owned save, credential, and Lab-navigation actions through `self.app`, matching
the existing speech surfaces. The regression proves a defaults-only dynamic
save completes, deletes the obsolete exact model and voice keys, clears the
pending request, and reaches the Lab route. The full Settings panel suite
passes 66 tests and the cross-surface closeout suite passes 9 tests. The live
retest completed the exact-to-dynamic transition, cleared the Settings draft,
removed both exact keys from the isolated TOML, and generated a complete WAV
without rewriting either resolved identifier.

### F-1700-01 — OpenAI Console endpoint ownership mismatch (P1, fixed and live verified)

The isolated Console initially used OpenAI with a model and base URL shown in
the session modal. The URL matched the value persisted under
`api_settings.openai.api_base_url`, so Console readiness accepted the request,
but the legacy OpenAI execution adapter independently read
`openai_api.api_base_url` and contacted the public OpenAI endpoint instead.
The task-owned stub received no request and Console truthfully displayed the
resulting authentication failure. The UAT continued with the task-owned direct
`llama_cpp` path, whose session endpoint is honored, without involving the
user-owned audio.cpp listener.

The root cause is confirmed: the OpenAI execution adapter consumed only the
legacy `openai_api` projection while global Settings and Console readiness use
canonical `api_settings.openai`. The adapter now overlays the canonical table
on the legacy compatibility values, while retaining the legacy projection's
resolved environment credential when the canonical table stores only
`api_key_env_var` or an empty local fallback. A failing regression proved that
the legacy endpoint won before the change; the canonical endpoint now wins,
the environment-resolved key survives an empty local fallback, and all 31
provider request-payload tests pass.

The live retest selected OpenAI in Console, retained the synthetic model ID,
and displayed the canonical task-owned Base URL
`http://127.0.0.1:18765/v1`. Sending the synthetic prompt returned the stub's
unique deterministic assistant response,
`Lantern check complete; the harbor is ready.`, proving the adapter reached
the canonical loopback endpoint rather than public OpenAI. Evidence:
`uat-01-openai-canonical-session.png` and
`uat-01-openai-canonical-response.png`. This closes the finding.

A pending or unavailable journey is not a pass.
