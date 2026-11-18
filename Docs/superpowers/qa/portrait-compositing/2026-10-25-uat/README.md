# PR #3244 final portrait UAT — 2026-10-25

Production subject: `2a1711ea602500a19a0bbf74bc9f2ce206a0dfa9`; dev base
`139333920d2a87cb011e4f3aa5f8319b59c44a5c`. The requested rebase was a no-op:
the branch already contains current dev. Both freshly rebuilt wheels match all
54 changed production files; see `installed-source-sha256.json`.

The executable portrait UAT matrix passed after two reviewed fixes. **Full live
image-provider qualification remains blocked; TASK-36555 AC5 stays unchecked and
In Progress.** This is not a claim that every provider, terminal or device passed.

| Journey | Result and boundary |
| --- | --- |
| Installed authoring | Passed real region selection/save, three component requests, small-mouth/eyelid retry, actual raw-pair viewer, six offline poses, guarded sample, Apply/reopen/Remove/reapply. Only image synthesis, TTS and device boundaries are deterministic fixtures in this journey. |
| Fresh/failed generation | Passed partial new-batch rejection with prior active set preserved, failed retry retaining candidate, and unaffected encoded frames preserved. Fixture responses; no live image requests. |
| Native F9 restore | Passed full-profile restore into a fresh app, one confirmed retry, equivalent encoding reuse and rejection of source/pixel changes. Original artwork preserved. |
| Ownership, motion, fallbacks | Passed mounted hidden/resume, stale sessions/epochs, Save reconciliation, retired mounts, reduced motion, manual still review, monochrome/pixel fallback and radio state/layout regressions in the feature matrix. Native graphics-terminal hardware remains unqualified. |
| Live Console Pocket TTS | Passed actual installed app, Pocket TTS 3.3.0, alba, native WAV adapter and MacBook Pro Speakers/CoreAudio. 5.4 s source and device-submitted WAV are identical; rest/small/open states and measured pauses observed. Natural completion follows final source DAC interval by 25.194 ms. |
| Live controls | Passed actual Stop near pending tail, message replacement, chat-switch retirement, two real utterances with zero failures, next sentence after preceding tail and listening intent after final tail. Capture-end input is simulated; microphone/STT hardware is not exercised. |
| Live guarded authoring sample | Passed visible destination confirmation for localhost:18765, actual synthesis/native playback and automatic preview mouth movement; draft remains ready. |
| Native timing | Passed PCM and eligible WAV full-waveform submissions. Four onset/pause/silence paint transitions each remain within 100 ms: PCM 8.892/68.292/43.925/42.862 ms; WAV 35.189/34.604/18.949/18.365 ms. Driver latency 53.25 ms is separate. Actual afplay reports state with timing invalid and mouth closed. |
| Speech content | Offline tiny.en ASR recognizes the exact source/device sentence including “The final word is complete.” Stream and guarded-sample transcripts are retained. This is not independent human audition or acoustic loopback. |
| Configured live image route | **Blocked**: actual default and sole enabled backend SwarmUI does not support reference-image edits. Zero requests, no fallback or credential changes. Historical external ChatGPT artwork is used only as the already-reviewed exact-pair import sample; it does not qualify in-app ComfyUI/Gemini/fal. |

## Fresh checks

All runs used Python 3.12.11 and feature-targeted pytest with private basetemp,
`-q -o addopts='' -p no:cacheprovider --tb=short --show-capture=no`.
Counts overlap and must not be summed as unique tests.

- Initial broad feature matrix: **451 passed** (storage/compositor, authoring, pairs, Console, envelopes/sinks, utterance and sequencer). This precedes the two small fixes; affected paths were rerun below.
- Final malformed-region storage plus mounted authoring: **54 passed**. All seven new controls fail on pre-fix source for the intended reasons (`corruption-red2.txt`).
- Final speech/native/legacy lifecycle gate: **208 passed**. Both new context controls fail on pre-fix source (`context-red.txt`).
- Final rebuilt installed authoring, native restore and live-probe safety: **12 passed** (`uat-release-installed.txt`).
- Final explicit real-device calibration: **3 passed**.
- Final boot/CSS/token gates: **24 passed**, 1030/1031 ready modules; no ceiling raised.
- Worker contract, Backlog IDs/files, focused Ruff/format and diff checks pass. The legacy TTS file retains its 59 baseline Ruff findings and adds none (`context-lint.json`).

## Issues found and fixed

1. SQL-valid scalar/object region identities escaped the optional-settings recovery
   contract. Enforce a JSON array, retain the existing revision when ignoring corrupt
   settings, and permit only an explicit source-checked CAS replacement to repair it.
   Mounted manual import/Apply/Remove stays available and active artwork stays exact.
2. Real-device UAT exposed a pre-existing native callback context-loss race that can
   crash transcript Markdown mounting during replacement. A notification thread's
   `call_soon_threadsafe` inherited an empty context. Capture the admission coroutine
   context for both native and owned-file started reports; observer-created UI tasks
   retain Textual's app context. A real notification-thread regression and a clean
   legacy-worker regression reproduce the failure and pass with the fix. An independent
   reviewer also reproduced the exact Markdown stack and validated the correction.

Two earlier added-sample harness runs had an incorrect `ready` assertion and a
synthesis-count observation taken after confirmation; corrected to `can_apply` and
before-confirmation counting. A subsequent repeat exposed the actual context-loss
bug above. Final complete installed/live journeys pass without relaxing gates or
masking that failure. Raw local intermediate logs remain under `/private/tmp`;
only final live evidence and deterministic red controls are retained here.

## Evidence and limits

`pocket-summary.json`, `pocket-observations.json`, `pocket-transcripts.json`, three
playable WAVs and original SVG/PNG captures describe the final live run. The device
capture is buffers submitted to PortAudio, not a microphone recording. It is omitted
as a duplicate of `pocket-speech.wav`; its equality was asserted before publication.
PNG renders use CairoSVG/Homebrew Cairo and Menlo substitution; original SVGs remain.
`pocket-probe.patch` records the precise opt-in historical probe adaptation for the
final wheel, corrected pairs, output isolation and real authoring sample. It adds no
provider, animation loop or per-frame speech injection. The extra HTTP stream probe
feeds real arriving WAV payloads into the existing PCM transport; the native Pocket
adapter remains complete-body WAV, not streaming. `sha256.json` hashes the evidence.

Remaining: supported configured image-adapter success/retry provenance and visual
review of its real generated mouth/eyelid semantics, alignment, lighting and seams.
No independent human audition/signoff, physical acoustic loopback, microphone/STT
hardware, other OS/device or native graphics-terminal UAT is claimed. No full
repository pytest sweep ran. Existing ADR-242/243 apply; no new ADR is required.
