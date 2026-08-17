---
id: TASK-17169
title: Persist and track console selection feedback
status: To Do
assignee:
  - '@Robert'
created_date: '2026-08-16 13:57'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User request (2026-08-16): feedback sent via the selection menu (Request changes / LGTM / Comment) is currently ephemeral — dispatched as the next user message and forgotten. Track it durably, ideally in the session trace. Two candidate homes: (a) the trajectory trace sidecar (schema v38 message_trajectory_metadata, ADR-066 on dev / PR stack) as a new event_kind=user_feedback record carrying action, quoted selection, optional comment, and the anchor (message id / diff row key / turn id) — feedback becomes part of the run's auditable event ledger, queryable per turn; (b) the spec phase-4 transcript_annotations table (spec 2026-08-14 console-selection §3: (session_id, row_key)-anchored, quote + comment, soft-delete, inline badge UI) — feedback persists as row-anchored annotations with a visible marker. These serve different needs: trace events = chronological audit/history; annotations = inline review markers on rows. Recommendation to validate at planning: BOTH may be right — write the feedback event to the trajectory sidecar (audit) and upsert an annotation for Comment actions (inline marker); phase 4's row_key derivation spike (message:<id> / diff:<tool_call_id>) is the shared prerequisite. Decide the split in this task's plan; may merge with the phase-4 annotations task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Feedback events recorded durably (not ephemeral), surviving restart
- [ ] Option A evaluated: trajectory sidecar event_kind with anchor fields
- [ ] Option B evaluated: spec phase-4 annotations table
- [ ] Decision recorded (A, B, or both) with rationale + ADR if storage changes
- [ ] Feedback history viewable (trace viewer and/or inline badges per decision)
- [ ] Tests green
<!-- AC:END -->
Related: PR #1723 (phase 3 feedback actions); ADR-068 (selection system); spec 2026-08-14 console-selection §3 + §7 phase 4; dev ADR-066/067 trajectory sidecar.
