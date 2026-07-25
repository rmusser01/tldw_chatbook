---
id: TASK-551
title: Add external audio.cpp native TTS adapter
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 19:45'
updated_date: '2026-07-24 19:49'
labels:
  - tts
  - audio-cpp
  - backend
dependencies:
  - TASK-402
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md
  - Docs/superpowers/plans/2026-07-24-audio-cpp-external-adapter.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add audio.cpp as the first native TTS adapter so Chatbook can safely discover models and voices from one existing audiocpp_server and synthesize a validated complete WAV through the app-owned service boundary without process supervision or STTS UI changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The sealed registry exposes audio_cpp as an exact canonical native provider without aliases and leaves all legacy providers unchanged.
- [ ] #2 External configuration accepts only a bounded HTTP or HTTPS origin and enforces the approved timeout input metadata catalog identifier and audio size limits.
- [ ] #3 Readiness validates the pinned audio_cpp_http_v1 health and model shapes and exposes only bounded TTS models with safe provider health states.
- [ ] #4 Voice discovery is lazy bounded optional and cached by provider configuration plus catalog revision while a missing or invalid voices endpoint does not invalidate the provider.
- [ ] #5 Canonical audio_cpp requests accept only a known model optional safe voice WAV format speed 1.0 and no unknown options.
- [ ] #6 Speech synthesis sends one non-retried POST and exposes the validated complete PCM16 WAV as one asynchronous response chunk with safe provenance and timing metadata.
- [ ] #7 HTTP redirects compression oversized bodies malformed metadata and structurally invalid WAV responses fail closed without logging remote content or submitted text.
- [ ] #8 Busy unavailable incompatible invalid-model generation timeout and cancellation outcomes map to stable safe retryability semantics without silent provider fallback.
- [ ] #9 Normal automated tests use pinned contract fixtures and fake HTTP transport and require neither an audio.cpp binary nor model downloads.
- [ ] #10 ADR-023 the approved design and TTS module documentation describe the external adapter boundary and explicitly defer Playground managed-process binary and server.json work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the provider-neutral adapter contract with immutable response metadata, stable safe operation errors, and leased lazy voice discovery; preserve the legacy adapters through static voice responses.
2. Add immutable bounded external audio.cpp configuration and prepend one exact lazy exclusive audio_cpp provider spec while keeping all six legacy specs unchanged.
3. Add pinned audio_cpp_http_v1 fixtures and pure bounded parsers for health, TTS models, voices, timing headers, and structurally complete PCM16 RIFF/WAV responses.
4. Implement external readiness, catalog refresh, safe-GET retry, lazy per-model voice caching, redirects/compression/body bounds, and stale health semantics over fakeable async HTTP.
5. Implement validated canonical requests, one non-retried speech POST, complete-WAV one-chunk responses, safe metadata, cancellation, timeout, and stable error mapping with no fallback.
6. Harden concurrent first use, close, response lifetime, privacy, and exclusive reconfiguration using focused lifecycle regressions.
7. Update ADR-023, the approved design, module guide, pinned provenance, and this task while preserving the Slice 3 Playground and Slices 4-5 managed-process deferrals.
8. Run focused and broad tests, Ruff, formatting, compileall, scoped mypy, boundary searches, diff checks, and security/scope self-review before completing the task.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: ADR-023 already governs the native provider boundary, external audio.cpp contract, complete-WAV interface, security limits, exclusive reconfiguration, and ordered slice delivery; no new ADR is required.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Automated unit and contract tests cover every new adapter boundary and error category.
- [ ] #2 Ruff formatting static analysis compilation and scoped regressions pass.
- [ ] #3 Documentation and pinned fixture provenance are current and linked.
- [ ] #4 Self-review confirms no process supervision UI work binary handling or server.json ownership entered Slice 2.
- [ ] #5 Every acceptance criterion is checked and implementation notes summarize the shipped behavior before status moves to Done.
<!-- DOD:END -->

## Implementation Notes

Implemented the external-only `audio_cpp` native adapter behind the
application-owned TTS service. The exact alias-free provider is lazy and
exclusive, validates immutable bounded `[app_tts.audio_cpp]` configuration,
discovers the pinned `audio_cpp_http_v1` health/models and optional per-model
voices over fakeable HTTP, and exposes a fully bounded and validated PCM16 WAV
as one asynchronous response chunk. Safe errors, scalar metadata, response
lifetime, cancellation, privacy, health staleness, and no-overlap
reconfiguration are covered by focused regressions; normal tests require no
audio.cpp binary or models.

The six legacy providers remain unchanged behind their bridge. No STTS UI,
binary handling, `server.json` ownership, launch, or supervision entered this
slice. The external STTS vertical remains Slice 3 and managed runtime/UI work
remains Slices 4–5. Implementation followed
[the Slice 2 plan](../../Docs/superpowers/plans/2026-07-24-audio-cpp-external-adapter.md)
under [ADR-023](../decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md);
final verification and acceptance closure remain Task 8.
