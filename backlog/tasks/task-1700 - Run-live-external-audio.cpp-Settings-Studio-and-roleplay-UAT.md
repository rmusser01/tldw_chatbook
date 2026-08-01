---
id: TASK-1700
title: Run live external audio.cpp Settings Studio and roleplay UAT
status: To Do
assignee: []
created_date: '2026-08-01 06:06'
labels:
  - tts
  - audio-cpp
  - uat
dependencies:
  - TASK-1699
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the completed Speech and TTS ownership program as a first-time user with a user-supplied running external audio.cpp server, then record audibly verified Console or Roleplay playback and the approved recovery and isolation journeys. This task is the manual release gate; it does not make Chatbook responsible for starting or supervising the server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 UAT uses a user-supplied already running external audiocpp_server and model, synthetic non-secret text, and no Chatbook download, binary path, server.json, launch, adoption, restart, supervision, or stop behavior.
- [ ] #2 From a first-run configuration, a user finds Speech & TTS through Settings search within 60 seconds without documentation or raw TOML, saves the external URL, sees Saved plus Not checked, explicitly tests and refreshes in Lab, generates a synthetic assistant character response in Console or Roleplay, and audibly plays the complete WAV through the response control (UAT-01).
- [ ] #3 With the server stopped, a locally valid URL remains Saved while explicit test reports Unavailable without fallback; after the user starts the same external server, a later test becomes Ready without rewriting configuration (UAT-02).
- [ ] #4 After refreshing a multi-model catalog, exact model and voice choices survive navigation; deliberate First available and Server default modes persist without writing ephemeral resolved identifiers, and missing exact choices remain visible without substitution (UAT-03).
- [ ] #5 Studio-only preferences survive remount without changing global or normal generation, and Reset to Global deletes overrides so later global changes are inherited rather than copied (UAT-04 and UAT-05).
- [ ] #6 An exact audio.cpp profile assigned to one canonical character wins for that character response while an unassigned response uses global defaults, and Studio preferences remain unchanged (UAT-06).
- [ ] #7 A character profile can be previewed and played in Studio without persistence; leaving unadopted keeps saved Studio preferences unchanged, while explicit Adopt as Studio Preferences plus Save changes only Studio (UAT-07).
- [ ] #8 An environment-managed supported credential is shown only by source and variable name, ordinary Save creates no local secret, masked text is never persisted, and clearing a local fallback cannot affect the environment (UAT-08).
- [ ] #9 Each retained legacy provider preserves its saved global connection or initialization values and supported Studio tuning and generation behavior, with any unavailable optional live provider recorded separately rather than silently treated as passing (UAT-09).
- [ ] #10 External audio.cpp remains independently Ready and playable when unrelated local TTS or STT dependencies are missing, and each unavailable dependency retains its own truthful status (UAT-10).
- [ ] #11 The acceptance record distinguishes deterministic complete-WAV and playback-handoff evidence from human audible-playback evidence, includes only synthetic/redacted screenshots and diagnostics, records the tested provider and configuration/catalog revisions, and exposes no credentials, model contents, submitted private text, or raw provider bodies.
- [ ] #12 No priority-zero finding remains; every priority-one finding is fixed or rejected with technical evidence and explicit user approval, and a priority-two finding is deferred only when it violates no acceptance criterion and has a separately created Backlog task.
<!-- AC:END -->
