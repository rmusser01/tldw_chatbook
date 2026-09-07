---
id: TASK-1985
title: Complete the external audio.cpp global Settings experience
status: Done
assignee: []
created_date: '2026-08-01 06:03'
updated_date: '2026-08-01 13:04'
labels:
  - tts
  - settings
  - audio-cpp
  - ui
dependencies:
  - TASK-1984
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn audio.cpp into the complete redesigned provider experience inside global Speech & TTS Settings while retaining its external-server-only native adapter contract. Users must be able to understand connection privacy and limits, reuse explicit Lab discovery, and preserve exact model and voice intent across stale or changing catalogs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The audio.cpp form is explicitly labeled External server, explains that the user starts and owns it, presents the URL and timeouts before a collapsed Advanced safety limits group, and exposes every accepted external adapter bound (CFG-004).
- [x] #2 The configured value is a canonical HTTP or HTTPS origin with no userinfo, query, fragment, or non-origin path; non-loopback HTTP receives the approved transport warning and the form discloses that submitted text is sent to the configured server (CFG-004 and SEC-003).
- [x] #3 The form and its persistence model reject every binary, server.json, bind, authentication-header, launch, adoption, restart, supervision, stop, and managed-process value or action.
- [x] #4 Global audio.cpp model and voice controls use only the latest accepted in-memory service observations and never perform discovery during Settings mount, search, provider selection, edit, Save, Revert, or default restoration (CAT-001 and CAT-002).
- [x] #5 With no accepted catalog, new choices are limited to First available and Server default; an already persisted exact value stays pinned and Unverified until an authoritative refresh rather than being erased or presented as ready (CAT-002 and CAT-003).
- [x] #6 Fresh, stale, missing, and Unverified model or voice states are visibly distinct; an approximate legacy catalog cannot establish absence, and a complete authoritative observation that lacks an exact value leaves it visible and blocks only affected generation without substitution (CFG-011, CAT-003, STATE-023).
- [x] #7 Voice options are scoped by exact model, provider configuration revision, and catalog revision; stale catalog, voice, and readiness results cannot overwrite a newer selection or status (CAT-004 and CAT-005).
- [x] #8 The form provides a scoped Open Speech Lab recovery action for explicit test and refresh, returns to the applicable exact selection when representable, and never displays a result from saved configuration as proof that a dirty URL draft works (CAT-006 and IA-005).
- [x] #9 audio.cpp format and speed remain visibly fixed to WAV and 1.0 with an explanatory disabled reason, complete-WAV artifacts remain playable after later settings or catalog changes, and no provider fallback occurs (CFG-002, STATE-014, and A11Y-004).
- [x] #10 Deterministic tests use fake revisioned catalogs and cover all external fields and bounds, URL and privacy warnings, zero hidden network, first-run dynamic modes, pinned unverified values, fresh and stale choices, authoritative missing exact values, model-scoped voices, stale result rejection, dirty-config attribution, and unchanged complete-WAV playback provenance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md and backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: TASK-1985 implements the accepted external-only audio.cpp Settings, cached capability, exact-selection, and no-fallback boundaries without introducing a new provider or runtime contract.

1. Add failing pure tests for strict external-form inventory, canonical-origin and transport-warning projection, managed-key rejection, and missing/fresh/stale/Unverified exact-choice presentation.
2. Add a bounded in-memory native capability observation to `TTSService`; publish only revision-coherent results produced by existing explicit Lab/runtime operations, expose a synchronous read-only snapshot, and reject stale catalog or model-scoped voice results without materializing or contacting an adapter.
3. Make the Lab's native audio.cpp voice refresh retain structured authority in that service observation while preserving its existing selector tuple and generation behavior.
4. Project the latest accepted observation into global Settings without initializing the service; render dynamic/exact model and voice choices, pinned missing or Unverified values, freshness and dirty-saved-config attribution, fixed WAV/speed reasons, remote-HTTP privacy warning, and a scoped Lab recovery action.
5. Preserve the complete external URL/timeout/Advanced safety-limit layout, canonical persistence, and explicit rejection/absence of every managed-process or authentication field/action.
6. Run focused service/model/Textual/race/privacy/playback tests, neighboring regressions, static checks, and independent review before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a bounded, read-only in-memory audio.cpp capability observation in `TTSService`, populated only by explicit Lab/runtime catalog and voice operations and guarded by provider configuration, catalog, model, and request generations.
- Completed the external-server-only audio.cpp Settings form with canonical-origin validation, timeout and safety limits, transport/privacy disclosure, fixed WAV/1.0 controls, strict rejection of managed/authentication values, and no Settings-side provider work.
- Added truthful cached exact-choice projection for first-run, pinned Unverified, Fresh, Stale, and Missing states; model-scoped voices; dirty-draft attribution; and scoped Speech Lab recovery without implementing the broader TASK-1987 deep-link contract.
- Updated both Lab playground paths to publish structured audio.cpp voice observations and reject mismatched catalog revisions. First-time provider transitions now discard foreign exact IDs while preserving exact values actually persisted for audio.cpp.
- ADRs `backlog/decisions/039-global-and-studio-tts-settings-ownership.md` and `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md` apply; no new architectural decision was introduced.
- Verification: 1,002 relevant TTS, Settings, Lab, privacy, race, and playback tests passed with one known unrelated package-export baseline deselected; Ruff checks, Python compilation, `git diff --check`, Impeccable UI detection, and independent review passed.
<!-- SECTION:NOTES:END -->
