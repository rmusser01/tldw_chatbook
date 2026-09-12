# Speculative voice provider readiness and preparation errors

Date: 2026-09-05
Task: TASK-23175 (remains In Progress)
Status: Focused behavior and written spec approved by the user; independent spec
review passed.

## Evidence and scope

The approved USB diagnostic at `2955d2e82a0bd16671721633914481b7aeef0e8f`
recognized speech but failed before provider generation. The log recorded only
`attempt_prepare_failed exception_type=RuntimeError`; the transcript became a
draft without an explanation. A subsequent read-only check found the configured
local llama.cpp provider unreachable. Isolated production-controller preparation
reproduced `voice_provider_unavailable` for an unavailable provider and succeeded
for DeepSeek. One independent DeepSeek text request returned HTTP 200. Neither
check proves successful voice playback or resolves the earlier USB overflow.

Implement only a provider-readiness gate at speculative Hands-free entry and
safe, visible reporting when request preparation fails after entry. Use the
current conversation's selected provider/model; do not change saved defaults,
automatically select another provider, send a probe chat, or consume staged input.

## Proposed behavior

1. On the visible Hands-free control, capture the owning conversation and its
   selected provider configuration, then reuse the existing bounded provider
   readiness path. This runs before STT preparation, voice-worker startup, or
   microphone/transport opening. Readiness may perform the existing availability
   probe but must not prepare or send a generation request, persist a turn, create
   trace authority, execute tools, or alter the draft.
2. Missing conversation, unavailable/invalid provider configuration, readiness
   timeout, and unexpected validation failure reject entry. Reset the visible
   switch and preparing status and display fixed, actionable app-owned copy.
   Preserve the draft exactly. Reuse the existing startup failure/lifecycle path
   where it satisfies these requirements.
3. An off toggle, teardown, conversation change, or provider/model/configuration
   change while readiness is pending invalidates that result. Late success must
   not open audio; late failure must not reset or notify for a replacement
   session. Recheck ownership and selected configuration before starting audio.
   Do not automatically restart validation after a selection change. An A→B→A
   selection change also invalidates the original pending result; equality of
   the final configuration alone does not restore startup authority.
4. Entry readiness is not a guarantee of future availability. Keep per-attempt
   request preparation and its existing validation. A current attempt's
   preparation failure must visibly explain that no reply started and the
   transcript was preserved. Retain existing draft preservation, suspended
   automatic retry, turn ownership, and cancellation semantics. No provider
   failover, automatic retry, or duplicate draft append is introduced.
5. Use trusted categories for app-owned preparation failures (session unavailable,
   provider unavailable, unexpected preparation failure). Never classify arbitrary
   exception text by substring. Unknown exceptions get generic safe copy. The
   persistent diagnostic retains only fixed category and exception-class metadata;
   no transcript, prompt, credentials, endpoint, raw provider copy, or exception
   message may appear in the new notification or event. Ignore stale/cancelled
   preparation failures and avoid duplicate notifications for the same attempt.

## Alternatives

- Only keep DeepSeek selected: fixes the immediate unavailable-server setup but
  leaves misleading readiness and silent preparation failures in the app.
- Add a new health service or automatic provider fallback: unnecessary and changes
  runtime policy or user choice. Reusing the existing readiness path is preferred.

## Verification and limits

Use software-only targeted RED/GREEN tests: visible-control entry with ready,
blocked, missing-session and exceptional/timeout results; no microphone/STT/worker
startup on rejection; off/selection/conversation/teardown races and replacement
sessions; successful production request preparation; preparation failure preserves
draft once and shows one safe explanation; stale failures are ignored; sentinel
sensitive strings never enter new UI copy or persistent events.

Use mocked provider responses and audio boundaries in tests. No new live audio,
physical qualification, route switching, hardware matrix, repetition, soak,
provider generation call, or full-suite run is authorized. Leave unrelated
trace-ledger and CSS edits untouched. Preserve source-only launcher guards and
packaged release qualification; update source/lint inventories only as required
by changed files. Do not mark TASK-23175 complete or claim the USB fault fixed.

ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: This is a readiness/error-reporting correction using existing controller,
UI ownership, bounded validation and voice-bridge boundaries. It does not change
provider authority, audio safety, native transport, turn acceptance or persistence.
