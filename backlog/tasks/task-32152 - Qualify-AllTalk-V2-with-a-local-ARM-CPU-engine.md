---
id: TASK-32152
title: Qualify AllTalk V2 with a local ARM CPU engine
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:50'
updated_date: '2026-09-09 08:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
AllTalk has no configured server here and upstream Mac support is unqualified. Provision the smallest viable native ARM CPU engine and measure real application delivery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An exact AllTalk V2 engine and model tuple starts on this host with isolated local configuration and loopback endpoint, or a reproducible runtime blocker is retained.
- [x] #2 The production provider adapter generates full WAV and MP3 speech for Lab and two Console replies with real playback and content evidence.
- [x] #3 Actual server failure, cancellation and successor or retry settle application ownership; targeted regressions cover any application defect.
- [x] #4 OpenAI-compatible generation preserves canonical voice aliases and explicit custom IDs without inventing file suffixes; only endpoint-accepted custom voices succeed, and the pinned V2 catalog advertises its six supported aliases.
- [x] #5 Initialization awaits the actual HTTP health/discovery request and safely reports connection/status failures without leaking request data or leaving an unawaited coroutine; failure-first targeted regressions cover the reproduced contract defects.
- [x] #6 Fresh AllTalk Settings controls select a supported OpenAI alias and offer only the six pinned aliases; historical saved-default migration behavior is preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 provider/runtime ownership and ADR-039/040 Global/Studio/Lab behavior apply.
Reason: Correct request identifiers and awaited httpx use preserve the existing OpenAI-compatible provider boundary. Native AllTalk API routing and new custom-voice capability discovery are outside this fix.

1. Pin public primary-source runtime and asset requirements, preflight resources, and provision only task-owned environments, caches or containers.
2. Pin the actual V2 OpenAI request schema and reproduce voice rewriting plus the unawaited HTTP initialization request with focused failing production-adapter tests. Preserve accepted OpenAI aliases and opaque custom IDs, use the supported default/catalog aliases in coordination with the catalog owner, and await the actual httpx health request without changing privacy or resource ownership.
3. Run targeted red/green adapter, format, and lifecycle regressions. Keep the selected-engine server schema evidence independent of adapter implementation.
4. Run bounded real startup/synthesis while coordinating the exclusive inference/audio slot, then exercise production Lab and two Console replies, complete playback, cancellation, failure/retry, and successor behavior.
5. Preserve exact logs, source revisions, asset hashes, and resource joins. Retain explicit prerequisite evidence for unmet criteria; do not mark Done while required hardware/provider validation is missing.

Pinned schema evidence: erew123/alltalk_tts@f16117e95b540e9bbbd8247b49ca6c6b1350b172 tts_server.py OpenAIInput; /private/tmp/tts-macos-burndown/alltalk/voice-contract-probe.json.

6. Cover the active AllTalk Settings default and mounted voice selector with a failure-first regression, then share the pinned alias catalog there. Preserve the historical Studio migration sentinel for untouched legacy settings.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the pinned AllTalk V2 OpenAI-compatible boundary: await the real httpx health request; preserve canonical aliases and explicit opaque voice IDs without invented filename suffixes; expose the six accepted aliases in catalogs and active Settings. Historical saved-default migration remains unchanged. Targeted red/green provider, privacy, Settings and mounted default tests pass, including the corrected fresh alloy expectation. Existing ADR-023/039/040 apply; no alternate native API was introduced.

Pinned ARM CPU VITS p225 WAV/MP3 and p226 WAV runs passed production Lab/Console playback, actual server failure/restart-retry, cancellation and successor ownership. Client Stop settles the HTTP operation while already-running server inference completes; the controller waits for that real completion. All owned server/player/ASR processes exited. Two full recognizers retained lexical differences, including on p226, so content quality is tracked separately without remapping the default or changing pass criteria. Full format, source and cleanup receipts and the earlier restart-probe failure are preserved in Docs/QA/tts-macos-burndown-2026-09-09/providers/alltalk/{README.md,p226-README.md}.
<!-- SECTION:NOTES:END -->
