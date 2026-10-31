---
id: TASK-38207
title: Preserve Library URL sources through local STT dispatch
status: In Progress
created_date: 2026-10-21 22:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library HTTP(S) ingestion through the Parakeet or transcribe-cpp lane must preserve the selected URL. Treating the URL as a filesystem path corrupts its separators before guarded retrieval and surfaces a misleading transcription failure. Retrying a job must also preserve the existing paired provenance contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] HTTP(S) source text, including path and query spelling, reaches the existing parse pipeline unchanged through Library dispatch and the spawned local STT worker.
- [x] Files remain Path-backed; buffer inputs and provider-facing transcription contracts keep their existing behavior.
- [x] URL transport validates source shape without granting network authority; existing parse egress checks remain authoritative.
- [x] Retry provenance includes both source job and source attempt identifiers, or neither when failure provenance is unavailable.
- [x] Targeted regressions exercise the real coordinator handoff and production worker path, including file and buffer controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase onto current dev, retaining its already-landed retry provenance pairing.
2. Amend ADR-025 for an explicit Library URL source at the existing executor boundary.
3. Reproduce URL corruption at the actual Library-to-coordinator handoff.
4. Transport immutable URL text without Path conversion; keep file/buffer and provider source contracts unchanged.
5. Exercise request validation, serialization, spawned worker dispatch, and existing parse routing with focused tests.
6. Forward only existing enum URL provenance through the default parser adapter under ADR-207; test genuine, unknown, forged, and missing values through the real parser.
7. Renumber the younger duplicate task 35080 after checking existing refs and registered worktrees, then obtain independent review before publishing.

ADR required: amend existing ADR
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; backlog/decisions/207-url-provenance-seeds-egress-self-trust.md
Reason: represent Library URL input explicitly at the existing executor boundary while preserving provider contracts and egress ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Library dispatch now constructs a repr-hidden, immutable URLIngestSource before any filesystem conversion. The executor transports the exact string to the existing guarded parse pipeline. FileAudioSource remains Path-backed, and BufferAudioSource and provider-facing TranscriptionRequest are unchanged.

The earlier downstream scheme repair was replaced because path conversion also changes URL spelling on Windows and cannot be reversed reliably. The shared parser has no net change against the rebased dev version. Retry provenance pairing was already present on dev; this PR retains it with regression coverage.

ADR-025 and the media-ingestion architecture reference document the source boundary. The younger duplicate TASK-35080 was renumbered to TASK-38207 after checking 428 refs and 64 registered worktrees (previous maximum 38206).

Verification on 2026-10-31:
- RED: two actual dispatch assertions failed because URLs arrived as paths; seven new protocol cases failed because the URL source did not exist. The local-file control passed.
- GREEN: 25 focused checks passed in 23.02 seconds, including both spawned URL worker paths on Windows, exact query/path text, file/buffer controls, invalid URL shapes, and video parse routing.
- A final validation subset passed 8 checks, including non-text rejection and the Library job-ID requirement.
- No live network/model transcription or full test sweep was performed. Independent review and required CI are pending.

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done

<!-- DOD:BEGIN -->
<!-- DOD:END -->


Final review correction: the default STT parser adapter now forwards only the
existing UrlProvenance enum under ADR-207. The real adapter-to-parser-to-video
regression first failed with USER_ENTERED becoming UNKNOWN; unknown, forged
string, and absent controls stayed closed. After the fix, all 41 focused URL
provenance and executor checks passed in 127.36 seconds, including existing
private-address and metadata-address guard tests. URL source/job identity
never mints trust. The caller's options mapping is copied before extraction.
