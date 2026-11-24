---
id: TASK-41177
title: Coordinate controlled audio output ownership
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-24T03:54:00Z'
labels:
  - watchlists
  - tts
dependencies:
  - TASK-41176
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make article listening coexist predictably with existing speech and media output, with reliable stop, position and completion control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Windows controlled WAV playback restores a validated PCM16 stereo 48 kHz retained file with the qualified tail policy at its saved media position; unsupported platforms refuse controlled listening before synthesis.
- [ ] #2 All existing file, streaming PCM and media start paths serialize output ownership and wait for confirmed retirement before replacement output.
- [ ] #3 Late starts and old-surface cleanup cannot affect the new owner; failed termination retains a visible blocked owner.
- [ ] #4 Existing ordinary file playback and external-handoff behavior remain supported, with targeted regressions and native transfer evidence.
- [ ] #5 Exact logical-EOF Resume launches no player; Stop during padding stays an interruption and cannot publish natural completion or release a newer output owner.
- [ ] #6 Targeted ownership, controlled-adapter and affected production pytest runs use normal conftest and real-profile guards with verified private test storage; no recovery-policy bypass or existing profile/ACL mutation is needed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
Reason: existing accepted output ownership and Windows-first capability contract; no new architectural boundary.
1. Initial bounded increment: implement only AudioOutputOwnership and exact host-issued leases, disconnected from production callers. Use event-driven RED/GREEN tests for replacement ordering, revocation, same-label identities, stale releases, blocked retirement, cancellation custody and concurrent intents. Record the tested branch base.
2. Run targeted ownership tests, relevant import/architecture checks and Ruff; independent spec/quality review the increment. Keep the task In Progress and all whole-task AC open.
3. Next increment: implement the qualified Windows controlled WAV adapter with explicit retained artifact metadata, typed completion, fresh progress and exact EOF/padding boundaries. Rebase onto current dev and rerun targeted checks before production integration.
4. Integrate every actual output and exact retirement boundary, including ordinary/preview file and PCM, provider realtime, speculative duplex, Speech Lab/wizard samples, Watchlists/Library/artifact replay, media including seek, and quick/full exit. Failed physical cleanup retains custody. Preserve external handoff acknowledgement and existing platform support for ordinary playback.
5. Validate native Windows takeover and cross-surface races, configured adapter playback, shutdown and affected targeted regressions. Update architecture/docs and mark Done only after all task AC and required native evidence pass. No provider, queue, storage or UI feature work belongs to this task.
<!-- SECTION:PLAN:END -->

## Progress — initial ownership increment

Implemented the isolated stdlib `AudioOutputOwnership` / `AudioOutputLease` in `9a6e4360ec`, with the eager-callback ordering fix in `cd66a8aa10`. Exact lease identity and generation fencing prevent stale release/start authority. Replacement waits for positive retirement; cancellation retains one shielded cleanup task and failed retirement preserves blocked custody. The primitive is disconnected from all production callers.

Eleven event-driven tests pass with asyncio debug enabled, including Textual's eager task factory. Ruff lint/format and the import-closure check pass. Independent spec/quality review found a P1 publication race in the initial version; its RED/GREEN eager-loop regression and scoped re-review confirm it addressed, with no open Critical/Important finding.

The normal pytest bootstrap stopped before collection with `recovery_scope_uncertain`; its cause and baseline behavior were not established. Pure coordinator checks used `--noconftest`, without changing recovery policy or user profile data. Native playback, the controlled adapter, all production producer integration and app-wide takeover evidence remain unverified. Task stays In Progress and every whole-task AC remains open.

Existing ADR367 governs this increment. Living internals are updated in `Docs/Architecture/speech.md`; the eager-publication incident is recorded in `backlog/docs/lessons-testing-evidence.md`. The tested implementation base is `3e08f979ea`; rebase onto current dev and rerun affected targeted checks before production caller integration.

### Continuation authorization and bootstrap remedy

The user authorized the controlled WAV adapter and all production wiring. Both stacked branches are aligned with dev `af4b13ea75`; the audio ADR is now 367 because dev reserved 366 for CI. Existing coordinator code and eager-loop regression were preserved.

The normal pytest bootstrap blocker is diagnosed: the sandbox account cannot perform the native drive-root open, and default TEMP/worktree ancestors grant the sandbox group destructive access. A fresh native-verified private temp subtree under the user home, a process-local TEMP/TMP/TMPDIR plus pytest basetemp, and normal-user execution pass all eleven tests with normal conftest/guards. No recovery marker, existing ACL or product recovery policy was changed. The guarded runner and numerical diagnosis are retained in this plan's workspace for targeted verification.

ADR367's production contract and the Superpowers Task2 continuation specify preparation/pins, worker revocation, native classifier/settling qualification, full caller wiring, external handoff acknowledgement, duplex/media/shutdown policy and remaining acceptance evidence. All whole-task AC remain open.

## Implementation Notes

Implemented Windows controlled retained-WAV playback and serialized app output
admission under ADR367. The descriptor binds private PCM16 stereo 48 kHz bytes,
digest, logical/physical frames, 250 ms zero tail and saved frame. Native read
pins hold the qualified executable and physical file ancestry. A private Windows
CreateProcessW boundary retains birth/cleanup evidence. Typed results distinguish
Natural, Stopped, Failed, Uncertain and AlreadyAtEnd; exact EOF launches nothing,
padding Stop is interrupted, and premature zero exit cannot imply completion.

The app reserves exact intents before asynchronous preparation and acquires only
when ready. Exact callback-bound leases and worker revocation fence late starts,
stale cleanup and shutdown. Positive native retirement precedes replacement;
unknown birth/failed close retains blocked custody and exact retry. Ordinary
file delegation records an external handoff epoch; the canonical Cancel-focused
dialog acknowledges only the current action and epoch. File expiry never clears
that epoch. Artifact holds last until positive local retirement, including late
cleanup after a view is gone.

Production coverage includes Console manual/automatic/status speech, streaming
PCM and file fallback, provider realtime, the complete speculative duplex
session, Speech Lab, both first-run voice samples, Watchlists cast, Artifacts
and Library replay, video playback/seek/external delegation and quick/full/direct
unmount shutdown. Failed replacement leaves current output playing and retains
its exact Stop target even while its first native receipt is pending. Realtime
keeps input gated on uncertain output close. Natural frame EOF waits for the
exact audible process pair. Windows audible inherited-descriptor media refuses
before takeover; ordinary supported POSIX paths retain their compatibility.

The bootstrap failure is resolved by the documented guarded targeted runner and
normal account with native-verified private process TEMP/basetemp. Normal
conftest, recovery admission and real-profile guards remain enabled. No existing
profile, recovery marker or ACL is mutated. New PCM-to-WAV files set native
private permissions on their retained descriptor before writing audio/header.

Validation: targeted root runs passed 237 foundation cases and 346 caller cases
(three POSIX signal skips); the unchanged architecture source-read node needed
UTF-8 and passed on rerun. The later two-race integration selection passed 111
cases, including both pending-start/failing-replacement orders and concrete
versus opaque opener failure. Incremental file/PCM/media/duplex/mounted reviews
have separate overlapping tests; these counts are not a unique aggregate.
All 23 new nonmeasurement Python files pass Ruff lint/format; edited legacy files
have no introduced Ruff diagnostics and changed-range formatting. The message
pump and affected architecture checks pass. No full repository test sweep ran.

Native Windows qualification records twelve canonical controls plus an early
zero-exit negative, and eighteen controlled/PCM transfers (six conditions, three
rounds). Fifteen matched pairs have 78.317–586.411 ms positive source gaps; three
startup old channels are independently quiet. Exact EOF, same-file saved Resume
and startup/active/padding Stop pass. The one-frame remainder has transport
evidence with waveform fidelity explicitly unqualified. All measured resources
retired. The independent offline evidence suite passes 96 tests; audit fixes
reconstruct integer frame/absolute-clock evidence, enforce schema and negative
identity, and check error cleanup without changing receipts. See
[controlled-output evidence](../../Docs/Testing/watchlist-controlled-audio-qualification.md)
and [living speech architecture](../../Docs/Architecture/speech.md).

Limits: endpoint evidence applies to the recorded Windows default route/build,
not acoustic silence or all devices. Linux/macOS controlled output remain
unqualified. Separate Windows saved-video capacity tests refuse existing POSIX
pinned-save primitives before audio delegation; their methods/tests are
unchanged and this task does not repair that feature. Two older controller
teardown errors did not reproduce in two current and two baseline runs; their
cause remains unverified. Final whole-task review, base synchronization and
publication are still pending; status and AC stay open until those gates finish.
