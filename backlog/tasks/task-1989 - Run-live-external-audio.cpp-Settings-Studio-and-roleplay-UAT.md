---
id: TASK-1989
title: Run live external audio.cpp Settings Studio and roleplay UAT
status: Done
assignee: []
created_date: '2026-08-01 06:06'
updated_date: '2026-08-02 15:11'
labels:
  - tts
  - audio-cpp
  - uat
dependencies:
  - TASK-1988
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - Docs/superpowers/specs/2026-08-02-speech-lab-current-result-ux-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
  - backlog/decisions/040-speech-lab-current-result-and-auto-play.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the completed Speech and TTS ownership program as a first-time user with a user-supplied running external audio.cpp server, then record audibly verified Console or Roleplay playback and the approved recovery and isolation journeys. This task is the manual release gate; it does not make Chatbook responsible for starting or supervising the server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UAT uses a user-supplied already running external audiocpp_server and model, synthetic non-secret text, and no Chatbook download, binary path, server.json, launch, adoption, restart, supervision, or stop behavior.
- [x] #2 From a first-run configuration, a user finds Speech & TTS through Settings search within 60 seconds without documentation or raw TOML, saves the external URL, sees Saved plus Not checked, explicitly tests and refreshes in Lab, generates a synthetic assistant character response in Console or Roleplay, and audibly plays the complete WAV through the response control (UAT-01).
- [x] #3 With the server stopped, a locally valid URL remains Saved while explicit test reports Unavailable without fallback; after the user starts the same external server, a later test becomes Ready without rewriting configuration (UAT-02).
- [x] #4 After refreshing a multi-model catalog, exact model and voice choices survive navigation; deliberate First available and Server default modes persist without writing ephemeral resolved identifiers, and missing exact choices remain visible without substitution (UAT-03).
- [x] #5 Studio-only preferences survive remount without changing global or normal generation, and Reset to Global deletes overrides so later global changes are inherited rather than copied (UAT-04 and UAT-05).
- [x] #6 An exact audio.cpp profile assigned to one canonical character wins for that character response while an unassigned response uses global defaults, and Studio preferences remain unchanged (UAT-06).
- [x] #7 A character profile can be previewed and played in Studio without persistence; leaving unadopted keeps saved Studio preferences unchanged, while explicit Adopt as Studio Preferences plus Save changes only Studio (UAT-07).
- [x] #8 An environment-managed supported credential is shown only by source and variable name, ordinary Save creates no local secret, masked text is never persisted, and clearing a local fallback cannot affect the environment (UAT-08).
- [x] #9 Each retained legacy provider preserves its saved global connection or initialization values and supported Studio tuning and generation behavior, with any unavailable optional live provider recorded separately rather than silently treated as passing (UAT-09).
- [x] #10 External audio.cpp remains independently Ready and playable when unrelated local TTS or STT dependencies are missing, and each unavailable dependency retains its own truthful status (UAT-10).
- [x] #11 The acceptance record distinguishes deterministic complete-WAV and playback-handoff evidence from human audible-playback evidence, includes only synthetic/redacted screenshots and diagnostics, records the tested provider and configuration/catalog revisions, and exposes no credentials, model contents, submitted private text, or raw provider bodies.
- [x] #12 No priority-zero finding remains; every priority-one finding is fixed or rejected with technical evidence and explicit user approval, and a priority-two finding is deferred only when it violates no acceptance criterion and has a separately created Backlog task.
- [x] #13 Speech Lab presents one current generated result with Play and Export visible at supported widths, never reports an unknown duration as zero, explains that the result is temporary until exported, keeps operational diagnostics behind progressive disclosure, hides audio.cpp's non-applicable language axis, and offers a separately persisted Studio-only auto-play preference that defaults off and never changes global or character settings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md and backlog/decisions/040-speech-lab-current-result-and-auto-play.md
Reason: This live UAT verifies the accepted external-only runtime and four-owner persistence model. ADR-040 records the approved persistent Studio-only auto-play preference and the current-result interaction contract discovered during UAT.

1. Establish an isolated first-run `tldw-serve` harness against only the user-owned external audio.cpp listener and create a privacy-safe evidence ledger.
2. Run UAT-01 through visible Settings search, local Save, explicit Lab Test/Refresh, complete-WAV generation, response-control playback, and explicit human audible confirmation.
3. Coordinate the user-owned server stop/start checkpoints for UAT-02 without Chatbook lifecycle behavior.
4. Run UAT-03 through UAT-07 for exact/dynamic selection, Studio isolation/reset, character precedence, and preview/adoption safety.
5. Run UAT-08 through UAT-10 for environment credentials, retained legacy providers, and independent dependency status.
6. Replace the unusable Speech Lab result area with one responsive current-result audition flow, add optional Studio-only auto-play, and regress the exact UAT failure before repeating the live journey.
7. Privacy-review all evidence, address release-blocking findings, update the TASK-1988 evidence matrix, independently review, and close the task only when every criterion is proven.

Detailed plan: Docs/superpowers/plans/2026-08-01-task-1989-live-external-audio-cpp-uat.md
Speech Lab remediation plan: Docs/superpowers/plans/2026-08-02-speech-lab-current-result-ux.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed all live UAT journeys against the user-owned external audio.cpp listener without adding lifecycle ownership. Closed UAT-03 with live two-model generation and playback-state evidence for pocket-tts-en and supertonic-3. Fixed F-1700-11 by clearing a pending exact voice and reprojecting the audio.cpp catalog when Server default is selected, with a keyboard-driven regression test. Preserved OpenAI chat sampling defaults while moving only canonical connection fields into the global provider configuration. Renumbered this workstream from TASK-1692 through TASK-1700 to TASK-1981 through TASK-1989 after the latest dev branch claimed the original IDs, and refreshed the reviewed diagnostic inventory after comparison with a clean current-dev candidate. Updated the privacy-reviewed live and release evidence records and retained ADR-039 and ADR-040 as the governing decisions; no new ADR was required.

Post-rebase verification passed 1,530 tests across every changed test module plus the persistent-diagnostic, profile-owned-path, CSS-build, backlog-ID, OpenAI-sampling, and Console seam gates. The broader suite was exercised in ordered segments through the remaining collection; every branch-specific regression found during that run was fixed. Remaining stops reproduce on current dev or are environment/load-sensitive upstream baselines: the destination and Library worker censuses, Textual Select mount-order failures in Library and Watchlists, and the mosaic colour assertion under `NO_COLOR=1` (the exact mosaic test passes with that variable unset). Ruff, compileall, inventory, and diff checks are part of the final branch gate.
<!-- SECTION:NOTES:END -->
