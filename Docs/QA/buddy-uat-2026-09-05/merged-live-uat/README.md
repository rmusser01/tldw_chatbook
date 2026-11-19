# Migu live UAT after both merges — 2026-09-05

Chatbook PR2428 merged as `66a1cbf8fdabf2f54e3222e64895bea472d42d76`; server PR2902 merged as `84b6928dcfc48fdd7b424939a9ba52a82c37612c`. These fresh runs imported the merged Chatbook revision. The only source worktree change during launch/exit was TASK31585 documentation; each receipt records identical controller hashes and source revision at exit.

| Scenario | Result | Evidence |
| --- | --- | --- |
| DeepSeek response | Expected reply; completed in1.55s, Migu idle → thinking → speaking → idle | [Provider](provider-20260905-0d3335b27f/provider.json) |
| Stop during actual response | One response character observed; terminal stopped, visible Response stopped, Buddy idle | Same provider receipt |
| Reply after Stop | New assistant identity, same session, expected recovery reply completed, Buddy idle | Same provider receipt |
| Kokoro output | Real audio sink drained128000 PCM bytes in4.71s; Migu idle → speaking → idle | [Kokoro](kokoro-20260905-57a73fbbf4/kokoro.json) |
| Voice context switch | Active playback acknowledged stopped; presentation cleared; separate synthetic voice lease survived and released to idle | Same Kokoro receipt |

Provider wrapper87330/child87336 and Kokoro wrapper87001/child87013 exited0 with no app exception. Normal config byte hashes were unchanged. No microphone was opened. Credentials remained in child memory and were not written to these receipts or the scratch config. Each run used its own disposable profile and null keyring backend. These are mounted Textual `run_test` interactions with real network/audio output, not physical Terminal or human microphone acceptance.

At this initial preflight, OpenAI realtime had not been tested because neither modern nor environment OpenAI configuration was present. Subsequent human microphone and Codex OAuth checks are recorded below. TASK31585 AC9 stays open pending complete realtime interaction acceptance.

The [initial recovery probe](provider-20260905-e36c7b3c4c/provider.json) read the prior Stop terminal state immediately after queuing the new send. It did not wait for a new assistant identity, so its false recovery result is a harness defect. The corrected probe requires that identity before accepting a terminal state and passes without application changes. Both original and corrected receipts are retained.

Harness `.txt` files are frozen, non-executable audit snapshots. Their hashes match their launch receipts. The new harness validates environment-derived output/profile paths beneath its fixed scratch root before use. These snapshots are not supported portable launchers. No logs, keys, recordings, or normal-profile databases are included.

Verification: receipt JSON, identity/hash invariants, exact result assertions and whitespace checks. No production code changed; no new automated test suite or Bandit run applies. Existing ADR037 trusted speech and ADR074 Buddy leases govern this evidence.

## Intentional human microphone cycle

The user explicitly authorized up to20 seconds of capture, local transcription, recognized test text sent to DeepSeek, and Kokoro playback. In [microphone-20260905-7c50b8573a](microphone-20260905-7c50b8573a/microphone.json), the user spoke the requested test phrase. Local faster-whisper recognized blue/notebook/ready (38 characters). DeepSeek completed a nonempty response; Kokoro drained68608 PCM bytes. The user [confirmed hearing the reply clearly](microphone-20260905-7c50b8573a/human-confirmation.json).

Stop was requested after20.01s. The dictation state returned idle, then wrapper99022 reaped child99028 with exit0 and no app exception. The tested source is documentation commit969043daf on top of merged dev66a1cbf8f; production controller hashes match the earlier merged-code runs. Normal config was unchanged. No raw microphone audio or transcript content is included in the evidence; only the authorized transcript was sent to DeepSeek. This is local STT plus DeepSeek/Kokoro. The later OAuth check below uses a different authentication source.

`captured_bytes=97920` counts only VAD-forwarded speech chunks, not the entire20s microphone buffer. `dictation_session_released=false` reports the cached session wrapper, which the successful dictation path intentionally retains; it does not establish an open recorder. The controller claims/stops its active service, and final process exit closes any remaining device handles.

Migu remained idle during actual capture (`buddy_capture_states=[idle]`). TASK31812 (formerly TASK31741 before current-dev task-ID reconciliation) tracks connecting local dictation lifecycle to the existing request-owned Buddy listening state. The voice conversation works, but the listening visual is not accepted.

Two earlier attempts are preserved: [first attempt](microphone-20260905-1b48acf67c/execution.json) exited1 because the harness read `draft_text` without calling it; [second attempt](microphone-20260905-84cb282039/microphone.json) returned no transcript and sent nothing externally because the user missed its recording window. The successful run added a five-second countdown. Neither earlier attempt is counted as a microphone-transcription pass.

## Codex OAuth realtime provider check

At the user's explicit request, the existing Codex OAuth access token was passed in memory to Chatbook's production `OpenAIRealtimeSession`. The [authentication probe](codex-oauth-realtime-auth.json) received `session.created`. The [production session probe](codex-oauth-realtime-provider.json) then returned the exact requested synthetic reply, “The blue notebook is ready.”, and 105600 bytes of output PCM, with zero errors and session closure.

The [frozen harness](codex-oauth-realtime-provider-harness.txt) documents the path exercised. No microphone was opened and the returned audio was not played. These results establish current OAuth authentication and provider response compatibility; they do not establish human realtime voice acceptance, a saved Chatbook OAuth connection, or token refresh behavior. Credentials were not persisted in these artifacts or copied into Chatbook settings.


## Post-fix microphone replay: no accepted speech

User-started run `microphone-20260905-fd8d5bba24` tested Chatbook revision
`6d2d677ac3ff385d4da4c578ad196337d829ea36`. A five-second countdown preceded
a 20.04-second capture window. Buddy reported `listening` during capture and
returned to `idle`; dictation returned to idle and released its session. The
recorder delivered zero speech bytes after VAD filtering, so no transcript was
submitted to DeepSeek and no playback occurred. Device enumeration identified
MacBook Pro Microphone as the sole/default input. This does not distinguish missed
speech from an input/VAD problem; the user's observation is pending.

The [capture receipt](microphone-20260905-fd8d5bba24/microphone.json) and
[execution receipt](microphone-20260905-fd8d5bba24/execution.json) record clean process
exit, matching source identity, no app exception, and unchanged normal config.
No raw audio was saved or sent. This is partial listening/cleanup evidence, not
successful speech/provider/playback acceptance.


## Post-fix human voice acceptance

The user confirmed that no phrase was spoken in the preceding
`microphone-20260905-fd8d5bba24` window, explaining its empty speech result.

A new explicitly started run, `microphone-20260905-587d0a8874`, tested revision
`e9a1543d2774cee135abb5a989a4f1eacf5fd4e9`. Local capture lasted 20.02 seconds,
delivered 123520 VAD-accepted PCM bytes, and local faster-whisper recognized the
expected words in a 38-character transcript. DeepSeek completed a nonempty reply;
Kokoro delivered 68608 audio bytes to a drained playback sink. The user confirmed
“Yes, clearly.” when asked whether the spoken reply was heard.

Buddy reported `listening` during capture and ended `idle`. The receipt's
`dictation_session_released=false` measures the reusable session object, not the
microphone stream: ordinary successful stop intentionally retains that object,
while `stop_dictation` stops the recorder. It must not be treated as a leak
assertion. The live receipt did not directly inspect the recorder handle. The
child exited cleanly, with matching source identity, no application exception,
and unchanged normal configuration. No raw audio was saved or sent to DeepSeek.

[Capture receipt](microphone-20260905-587d0a8874/microphone.json),
[execution identity](microphone-20260905-587d0a8874/execution.json), and
[human confirmation](microphone-20260905-587d0a8874/human-confirmation.json) preserve
this evidence. This accepts the local dictation/provider/readback route and the
listening-state fix. It does not certify server browser voice or the complete
OpenAI realtime human interaction; thinking/speaking transitions were not sampled
in this harness.
