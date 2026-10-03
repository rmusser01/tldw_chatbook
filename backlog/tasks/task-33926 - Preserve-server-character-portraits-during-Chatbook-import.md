---
id: TASK-33926
title: Preserve server character portraits during Chatbook import
status: Done
created_date: 2026-10-03 06:52
references:
- https://github.com/rmusser01/tldw_server/pull/3096#discussion_r4172046263
modified_files:
- tldw_chatbook/Chatbooks/chatbook_importer.py
- tldw_chatbook/Chatbooks/chatbook_models.py
- Tests/Chatbooks/test_chatbook_character_image_import.py
updated_date: 2026-10-03 07:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server Chatbook exports now include encoded character portrait bytes. The local importer drops those fields, so importing a server archive loses the portrait.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A server-shaped archive with a base64 character image stores the original portrait bytes through the public Chatbook importer.
- [x] #2 Malformed or unsupported encoded character portraits fail visibly without creating the character.
- [x] #3 Archives without an image marker retain their existing import behavior; affected tests and security checks pass.
- [x] #4 The public importer accepts actual server V1 character archive manifests (1.0.0 and 1.1.0) and preserves their binary portraits; unknown manifest versions remain rejected.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed three stages: (1) reproduce portrait loss through public server-shaped archives and real SQLite; (2) preserve explicit base64 portraits with strict validation and normalize only server 1.0.0/1.1.0 as the existing V1 core layout; (3) run focused checks and independent root review. Temporary plan completed and removed before commit.
ADR required: no. ADR path: N/A. Routine compatibility repair follows the server's documented backward-compatible V1 core layout and existing binary image column; local export versions and archive validation remain unchanged.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest remote main verified as f3aeb32fb3d230c0774c7fc349729d9f75c96366; isolated worktree preserves primary checkout WIP. Strict declared-base64 image decoding passes bytes through add_character_card, with failure before insertion for malformed records. Explicit server versions1.0.0/1.1.0 normalize to the existing V1 core layout; local export versions and archive validation remain unchanged.
Causal red:11portrait failures with2legacy controls passing; separate server semantic-version cases:2failures with17controls passing. Final portrait/manifest-model tests:39passed, including exactserver ContentItem/statistics/file-inventory shapes and unknown-version rejection. Earlier affected importer/message-image run qualified with existing bootstrap_profile marker applied during collection:52passed. Ordinary adjacent run:14passed,12failed,26errors at raw_source_selection_changed. A representative unchanged-HEAD importer constructor and the fixed importer reproduce the same default guard failure in separate processes with no profile marker. Collection binds config and normal per-test path redirection invalidates its selected source; the guard correctly refuses. Qualified checks are not an ordinary-suite pass.
New tests and changed models pass Ruff lint/format. Importer Ruff:62HEAD→62current diagnostics, no additions; whole-file formatting drift is unchanged. Bandit importer:0findings; importer+models:one LOWB101 at unchanged model line600, reproduced againstHEAD; no new findings. Compileall, git diff --check, Backlog ID/readability guards pass. Root independently reviewed and cleared the portrait and version changes. Completed owned temporary plan removed. No live server/native/provider UAT; exact fixture evidence qualifies character portrait import, not all server content types or V1.1 metadata semantics.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserve explicitly encoded server character portraits as durable SQLite bytes, reject malformed declared portraits before character creation, and accept only the server's backward-compatible 1.0.0/1.1.0 core manifest versions. Real public archive tests cover portrait readback, rename conflicts, malformed data, legacy cards, and unknown-version rejection. This repairs the Qodo cross-repository finding with the existing database abstraction.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
