---
id: TASK-31748
title: 'Meetings: redact raw exception paths in phase-1 meeting logs'
status: Done
assignee: []
created_date: '2026-09-06 07:41'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The phase-2 diarization review found the same privacy-leak pattern in PHASE-1 code that shipped earlier: several logger calls interpolate a raw exception (which can embed a filesystem path via OSError) instead of type(exc).__name__ or redact_user_paths -- meeting_session.py around lines 300/372/380 and meeting_owner.py around line 436. Sweep these to match the redaction precedent (meeting_session.py already uses redact_user_paths elsewhere). Also add regression tests for the two rename-persist-failure redactions fixed in phase-2. Found during the phase-2 diarization SDD run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No meeting logger call interpolates a raw exception/path into a persistent sink; redaction is regression-tested
<!-- AC:END -->

## Implementation Notes

Five phase-1 meeting logger calls that interpolated a raw exception (which can embed a filesystem path via OSError) now log the exception type or a path-redacted message; regression tests use a real loguru sink and force a path-bearing OSError (`Tests/Audio/test_meeting_log_privacy.py`), plus the two phase-2 rename-failure sites. Diagnostic inventory re-pinned. Landed in PR #2471 (commit b7466ce24c).
