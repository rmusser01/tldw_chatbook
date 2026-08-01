---
id: TASK-1696
title: Complete the external audio.cpp global Settings experience
status: To Do
assignee: []
created_date: '2026-08-01 06:03'
labels:
  - tts
  - settings
  - audio-cpp
  - ui
dependencies:
  - TASK-1695
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
- [ ] #1 The audio.cpp form is explicitly labeled External server, explains that the user starts and owns it, presents the URL and timeouts before a collapsed Advanced safety limits group, and exposes every accepted external adapter bound (CFG-004).
- [ ] #2 The configured value is a canonical HTTP or HTTPS origin with no userinfo, query, fragment, or non-origin path; non-loopback HTTP receives the approved transport warning and the form discloses that submitted text is sent to the configured server (CFG-004 and SEC-003).
- [ ] #3 The form and its persistence model reject every binary, server.json, bind, authentication-header, launch, adoption, restart, supervision, stop, and managed-process value or action.
- [ ] #4 Global audio.cpp model and voice controls use only the latest accepted in-memory service observations and never perform discovery during Settings mount, search, provider selection, edit, Save, Revert, or default restoration (CAT-001 and CAT-002).
- [ ] #5 With no accepted catalog, new choices are limited to First available and Server default; an already persisted exact value stays pinned and Unverified until an authoritative refresh rather than being erased or presented as ready (CAT-002 and CAT-003).
- [ ] #6 Fresh, stale, missing, and Unverified model or voice states are visibly distinct; an approximate legacy catalog cannot establish absence, and a complete authoritative observation that lacks an exact value leaves it visible and blocks only affected generation without substitution (CFG-011, CAT-003, STATE-023).
- [ ] #7 Voice options are scoped by exact model, provider configuration revision, and catalog revision; stale catalog, voice, and readiness results cannot overwrite a newer selection or status (CAT-004 and CAT-005).
- [ ] #8 The form provides a scoped Open Speech Lab recovery action for explicit test and refresh, returns to the applicable exact selection when representable, and never displays a result from saved configuration as proof that a dirty URL draft works (CAT-006 and IA-005).
- [ ] #9 audio.cpp format and speed remain visibly fixed to WAV and 1.0 with an explanatory disabled reason, complete-WAV artifacts remain playable after later settings or catalog changes, and no provider fallback occurs (CFG-002, STATE-014, and A11Y-004).
- [ ] #10 Deterministic tests use fake revisioned catalogs and cover all external fields and bounds, URL and privacy warnings, zero hidden network, first-run dynamic modes, pinned unverified values, fresh and stale choices, authoritative missing exact values, model-scoped voices, stale result rejection, dirty-config attribution, and unchanged complete-WAV playback provenance.
<!-- AC:END -->
