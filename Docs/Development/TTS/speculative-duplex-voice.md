# Speculative duplex Console voice

This page is the implementation guide for the provider-neutral low-latency
speech-to-text-to-speech pipeline governed by
[ADR-098](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md).
The detailed approved design is
[the speculative duplex voice specification](../../superpowers/specs/2026-08-28-low-latency-speculative-duplex-voice-pipeline-design.md).

## Turn and interruption contract

The end-of-speech timer may dispatch a provisional provider attempt after
`dictation.response_eagerness_ms` of admitted silence. The default is 700 ms
and the accepted range is 500-3000 ms.

Speech admitted while a provisional reply is generating or being rendered
belongs to the same logical user turn. The current generation, queued TTS,
and audible output are fenced; the rolling transcript absorbs the additional
speech; and a new attempt dispatches only after fresh transcript coverage.
After the terminal render boundary, later speech is a new turn. TTS chunks
remain sequential and cancellable.

Native streaming STT is preferred. `Audio/rolling_transcript.py` provides the
bounded rolling-window fallback and reconciles revisions. Remote fallback
windows can overlap, so duplicated submitted audio duration is a first-class
metric and cost disclosure.

Parakeet MLX runs in a session-owned spawned process. Its native stream contexts
and rolling fallback share one loaded model, while only bounded PCM and text
cross the serialized pipe. Rolling recognition uses in-memory PCM, not temporary
microphone files. Inference cannot hold the audio process's Python interpreter;
shutdown reaps a stuck decoder before waiting for transcript cleanup. A native
throughput preflight still selects rolling fallback when necessary, but is not
proof of callback continuity or acoustic safety.

## Process ownership and stall limits

The [process-isolation design](../../superpowers/specs/2026-09-05-speculative-voice-process-isolation-design.md)
places capture draining, AEC/VAD, transcript orchestration, turn timing, phrase
sequencing and local output fencing in a view-session audio subprocess. Its
contained STT subprocess owns inference. The app retains original provider
requests/contexts, TTS synthesis and PCM normalization, accepted-turn handoff
and persistence. Private credit-bounded pipes carry validated records, text and
normalized PCM, never app objects, callable targets or credentials.

The boundary is designed to preserve local processing during bounded app-GIL
stalls. New preparation, synthesis and persistence may wait for the app; queued
audio can drain or underrun. Neither underrun nor synthesis completion proves
the entire reply played. Original cleanup, final data consumption/discard and
terminal settlement are separate facts, even when priority cleanup arrives first.

Native rings, startup pre-roll and acoustic admission policy are unchanged.
The app renews the child lease every 250 ms; five-second expiry or app-pipe EOF
fences the child. Native close is observed independently for two seconds, without
waiting for model cleanup or draft recovery. Process-tree shutdown has a
six-second grace period, then two-second terminate and kill observation budgets.
Forced death or uncertain native closure quarantines device ownership for that
app lifetime. Original parent cleanup and claimed persistence survive child exit.

This is not hard realtime and does not cover OS suspension, CPU starvation or
stalls inside the audio interpreter. Synthetic native/AEC tests establish only
their software scheduling and ownership contracts, not microphone recognition,
acoustic safety or the cause of the historical USB stall.

The finite native checkpoint at `8d73d45adcc8b7fa434623d0c32ddcf9333118a8`
passed 357 targeted tests. A separate audio child processed actual synthetic PCM
through AEC during 720/3000/720 ms app-GIL holds, fenced old output locally, and
completed revised and distinct follow-up turns through actual promotion. It also
verified visible off/draft recovery for native status and overflow. These checks
use fake STT/provider/TTS, not microphone recognition.

One earlier test failed during child startup before native production. Its
original stage and exit evidence were not retained, so its cause remains
**unclassified**. Later passes and the separately fixed draft-credit/mailbox races
do not diagnose or repair that incident. Test-only early-startup diagnostics now
retain bounded stage/category/exit facts. Startup reliability remains unproved;
the final source/artifact verification does not erase this limitation.

The subsequent startup investigation reproduced a concrete race: the pipe reader
could enqueue a heartbeat or close before initialization accounted for the
already-read bootstrap. Priority dequeue then selected that control in place of
bootstrap, aborting the child before session construction. Bootstrap accounting
now completes before the reader/writer threads start. Deterministic real-pipe
subprocess regressions failed before this reorder and pass afterward, preserving
normal readiness and early-close behavior without native audio or timeout changes.
The final six-file targeted startup/transport gate passed 302 tests with one
existing Requests dependency warning; narrow review accepted the repair after
the new test wrapper was made to propagate actual lifecycle exit codes.
This repairs the reproduced race; the missing original I1 evidence still prevents
conclusive attribution of that historical incident.

## Source-only development and release status

The current dev integration retains application and companion source version
**0.2.0**. It preserves the original all-unqualified build-identity and manifest
bytes. Historical checkpoints below describe their named source commits, not
new qualification of this integration checkout. The current voice trace migration
is schema **68 to 69**, retaining dev's prior trace/privacy/Canvas migrations.

From this checkout, validate its exact full HEAD without starting the app:

```bash
../../.venv/bin/python scripts/run_speculative_voice_dev.py --expect-head <full-40-character-integration-HEAD> --check
```

The launcher resolves this checkout and refuses a foreign package, detached
checkout, abbreviated hash, or mismatched HEAD. The visible **Hands-free** switch
is the normal entry; do not rely on Ctrl+Shift+H in a macOS terminal. Navigation
and screen suspension stop provisional audio; already-claimed winning saves and
ordinary accepted turns remain with the app runtime under ADR-094.

Packaged platforms remain unqualified. The supported development entry is
[run_speculative_voice_dev.py](../../../scripts/run_speculative_voice_dev.py),
bound to a Git checkout root and exact expected commit and excluded from wheel
and sdist. Use the visible Hands-free control; there is no packaged development
override. The app/child runtime-source handshake is separate from historical
release qualification authority and cannot qualify a physical route.

The handshake hashes a fixed, reviewed runtime list: package initializers,
audio/model composition, bounded transport, directly composed voice helpers and
parent voice flow-control/promotion contracts. It includes the provider gateway's
voice splitter/backpressure and dedicated voice trace authority modules. It is
not recursive attestation of generic configuration, prepared-request, trace
storage, or provider/installer dependency graphs. Separate source/lint inventories
cover the approved plans and their implementation/tests; those inventories do not
grant runtime or release authority.

The reviewed runtime/test source at `c36f89e24a1aa73cc302f30f962a6dea34e1244d` passed
the final exact 46-target software group: **1,432 tests, zero skips**, in 241.18s,
with three existing dependency/deprecation warnings. All 53 fingerprinted runtime
files were present and byte-identical in the built wheel and sdist; the development
launcher and test child were absent. Full-file lint passed 65 changed Python paths;
64 were formatter-clean, excluding pre-existing untouched `STT/executor_worker.py`
formatting. Exact commands, the retained failed fixture run and completion details
are recorded in the process-isolation plan.

Replacement admission waits for old child data retirement and separately for
returned parent credit. Local speech-input-limit recovery permits text-only
promotion only with clean/data-complete phrase settlement and no active synthesis;
uncertain or failed cleanup remains fatal. The approved N1 follow-up now omits
late-terminal bookkeeping for never-issued proposals: all three provider terminal
opcodes are rejected after their retirement, while admitted requests retain valid
late-record handling. The three rejection cases failed before the fix; the final
four-file targeted software gate passed **244 tests** with one existing Requests
dependency warning. This does not rerun or extend the broader evidence above.
The startup-race repair and the remaining historical I1 evidence limit are
described above.

The ordinary conversation has already been consumed. The process-isolation
slice is verified with software only; no additional live conversation,
qualification harness, route matrix, repetition or soak is part of this work.

## Provider readiness and preparation failures

The visible Hands-free control validates the owning conversation's selected
provider before starting STT preparation or audio. This reuses bounded provider
readiness checks; it does not send a chat request, consume staged input, or change
the selected provider. Turning Hands-free off, leaving the conversation, or
changing settings invalidates pending startup, even if settings are changed back.

Readiness is not a promise that a later request will succeed. If preparation
fails after entry, the transcript remains in the draft and a fixed, safe message
explains that no reply started. Automatic retry stays suspended. Diagnostics use
app-owned categories, not provider/exception messages or transcript content.

## Echo cancellation and fail-closed admission

`dictation.pipeline_aec_enabled` defaults to true. False is a
troubleshooting-only choice that forces half duplex; it does not weaken the
separate Realtime engine. Every route begins with playback-period admission
closed. Full duplex opens only when the shared preprocessor proves healthy AEC or,
with the native AEC processor still present and operational, sustained acoustic
isolation. Ambiguous correlation, render leakage, malformed timing, discontinuity,
overflow, or processor failure demotes immediately. On either open path, the first
five VAD-positive frames are held for the bounded residual-echo decision; only an
admitted event is replayed to rolling STT in order.

The DSP and duplex transport contracts live under `Audio/` and are qualified
separately by the source-bound release evidence described in
[voice-aec-release.md](voice-aec-release.md). The physical procedure is the live
production-path workflow in
[speculative-voice-qualification.md](speculative-voice-qualification.md); reports
from the old manual metric-entry workflow cannot qualify a device. Any change to a
listed qualification source invalidates the prior evidence set, and rollout stays
disabled until the complete platform/device matrix is rebuilt from one committed
source digest.

## Settings ownership and compatibility

The only UI owner is **F9 Settings ▸ Speech & TTS ▸ Pipeline conversation**.
Opening the panel creates an in-memory draft and performs no config write.
Save validates the complete draft and sends only changed keys through the
existing atomic settings writer.

- `dictation.response_eagerness_ms`: speculative silence-to-dispatch delay.
- `dictation.pipeline_aec_enabled`: default-on AEC; false forces half duplex.
- `dictation.acoustic_barge_in`: Realtime compatibility only.
- `dictation.handsfree_send_delay_seconds`: legacy pipeline only during the
  rollout gate.

There is no startup migration, rewrite, or inference between the old and new
keys. In particular, a countdown after final STT is not equivalent to the new
silence-to-speculative-dispatch threshold.

## Provisional state, privacy, and persistence

Draft transcript and assistant rows are provisional UI state. Obsolete
attempts may not publish messages, trace bodies, notifications, sync/export
content, or exception copy. A promotion owner commits exactly one winning
user/assistant pair. Temporary chats receive no provisional trace envelope;
saving one later cannot retroactively add capture for an earlier attempt.

Cancelled provider work may still be billed. Usage counters are split into
winning and discarded attempts so diagnostics do not imply cancellation was
free.

## Content-free metrics

`Audio/voice_metrics.py` accepts only numbers and closed enums. It retains:

- speech-end to dispatch, admitted barge-in to estimated audible stop,
  replacement dispatch, and first-audio latency;
- native-live or rolling-window mode;
- AEC state transitions and aggregate ERLE samples;
- underrun, restart, conservative-mode, and duplicated-STT-duration counts;
- provider result class; and
- winning versus discarded usage counters.

Transcript text, response text, PCM/audio bytes, provider bodies, and capture
bodies are not representable by the API. Keep this closed input surface when
adding instrumentation; do not add a generic metadata dictionary.

## Failure boundaries

Every attempt is generation-scoped. Cancellation advances its fence before
cleanup so late provider, TTS, or UI callbacks cannot repaint or publish.
Device changes rebuild the shared clock/AEC domain. STT lag waits for fresh
coverage; terminal STT failure leaves an editable draft; persistence failure
retains the exact winning text in explicit recovery without another provider
call. Navigation, Stop, Esc, the Mic control, hands-free off, unmount, and
shutdown use typed cancellation reasons and preserve the app-owned runtime
owners.
