# AllTalk VITS p226 content control

The separate [runtime](runs/live-wav-p226-01/evidence.json) passes Lab and two Console WAV playbacks. Both Console sinks drain, worker 35473 exits 0, server 35488 receives normal owned shutdown and exits -15, and [all recorded processes are absent](runs/live-wav-p226-01/ownership-release.json). This small content control does not repeat the p225 cancellation/server-failure matrix.

The task-owned server maps only its alloy alias to p226 for this process. [Selection evidence](provenance/p226-engine-selection.json) pins the original source settings and mapping; its prepared_not_run field records the original preparation state and was retained to preserve the exact hash. The [actual generation events](runs/live-wav-p226-01/server-events.jsonl) assert Synthesizer.tts received native speaker p226 on all 3 calls. No production/global config or source mapping is changed.

The actual sounddevice function image remains the same bundled PortAudio binary as prior p225 validation, SHA256 `f48bedb7f4f79c0ca050776afe943458555db7341e0dae00511d109ccde907c0`; it does not use the Higgs candidate library. The shared harness field named portaudio_candidate_gate records the asserted bundled control identity here.

Both full-content ASR passes remain review-required, each 1/3 exact. Original p225 results are unchanged. There is no blanket voice remap, retake selection, or claim that p226 fixes pronunciation.

Expected text: “Silver compass rests by the quiet orchard under an amber sunrise.”

| Clip | Whisper-small | Whisper-base.en |
|---|---|---|
| speech-lab | Silver compass rests by the quiet orchard under a nambo sunrise. | Silver compass rasped by the quiet orchard under an ammo sunrise. |
| console-warmup | Silver compass rests by the quiet orchid under a number of summaries. | Silver compass rests by the quiet orchard under enamour semis. |
| console-repeat-01 | Silver compass rests by the quiet orchard under an amber sunrise. | Silver compass rests by the quiet orchard under an amber sunrise. |

Complete [small](runs/live-wav-p226-01/content.json) and [base.en](runs/live-wav-p226-01/content-base-en.json) receipts retain audio hashes, segments, exact-match flags and anchors. No acoustic loopback or human listening claim is made.
