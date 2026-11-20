---
id: TASK-42111
title: "First turn after a restart may skip change review without a disclosure row"
status: To Do
assignee: []
created_date: 2026-11-20 07:20
labels:
- follow-up
- console
- workspaces
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen once during TASK-42101's live check (2026-11-19, branch fu2/lfs-filters, scratch profile, fake llama.cpp server). Not reproduced and not traced.

**What happened.** Workspace 1 had Change Review enabled, and its only folder's preparation was refused. The app was restarted, and the first Console turn in that workspace ran and answered "Done.". It showed no Activity group and no "⚠ change review skipped …" row. The turn before the restart, and the turn after it, each showed the skipped row with the reason.

**Expected.** `ChangeReviewConsentService` admission lists every non-READY root as skipped with its reason; a PREPARING root's reason is "Preparing change history; this turn continues without it". So a missing row means either the admission observation was not current (`return ChangeReviewAdmission()` with nothing to disclose), or the root had no readiness entry for the current consent revision.

**Evidence.** The pane captures are in the session scratchpad. Turn evidence: 02 (no row) against 01 and 04 (rows).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The first turn after a restart in a workspace with Change Review enabled is reproduced and shown either to disclose its skipped root, or why it does not
- [ ] #2 If a turn can skip change review silently, it now discloses the skip (test, red first)
<!-- AC:END -->
