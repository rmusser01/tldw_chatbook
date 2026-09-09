# Local provider qualification — 9 September 2026

| Concrete runtime tuple | Generation/content evidence | Playback/lifecycle result |
|---|---|---|
| AllTalk V2, Coqui VITS VCTK p225, CPU float32 | Complete WAV/MP3; both ASRs retain material lexical mismatches | Runtime passed: Lab, Console, active-call Stop/successor, real stopped-server failure/restart retry; all processes joined |
| Higgs Audio V2, pinned legacy assets, CPU float32, bundled PortAudio | All 4 generated clips exact under local Whisper-small | Lab passes; Console fills every PCM frame but terminal sink teardown and interpreter exit hang in both attempts; failed_cleanup |
| Higgs Audio V2, CPU float32, isolated PortAudio PR candidate | All 3 completed clips exact under local Whisper-small | Lab, Console drain, real active-call Stop and successor pass; Stop takes 96.400 s; worker exits 0 |
| AllTalk VITS VCTK p226, CPU float32, unchanged bundled PortAudio | WAV small/base.en each 1/3 exact; content remains review-required | Separate Lab plus two Console playback control passes; server and worker joined |

The sources/models are pinned below. These results qualify only the concrete host/runtime/engine tuples. No audio, private profiles, model weights, binaries, or credentials are copied into this package.

- [AllTalk details](alltalk/README.md), [WAV final runtime](alltalk/runs/live-wav-02/evidence.json), [MP3 runtime](alltalk/runs/live-mp3-01/evidence.json).
- [Higgs details](higgs/README.md), [first failed-cleanup run](higgs/runs/live-wav-01/evidence.json), [second failed-cleanup run](higgs/runs/live-wav-02/evidence.json).
- [PortAudio identity comparison](higgs/provenance/portaudio-comparison.json), [16-PID release check](higgs/provenance/provider-process-release.json), [adapter regression summary](regressions.json).
- [Curation provenance](package-provenance.json), [semantic round-trip verification](verification.json).
- [Higgs candidate result](higgs/candidate-README.md), [AllTalk p226 control](alltalk/p226-README.md), [candidate summary](candidate-summary.json).
- [Candidate provenance](candidate-package-provenance.json), [candidate verification](candidate-verification.json), [final 10-PID release check](higgs/provenance/candidate-provider-process-release.json).

Large application source manifests are factored through the existing [shared English manifest](../kokoro-english/provenance/source-hashes.json), with exact relative-reference overrides/removals in each receipt. All other runtime JSON values remain unchanged. Original artifact paths and hashes remain explicit. Content/stack/event receipts are byte-preserving copies. The later candidate results preserve every original failed/default-runtime receipt. The isolated PortAudio candidate does not establish that the default installed runtime is fixed.
