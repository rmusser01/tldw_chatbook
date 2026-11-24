---
id: TASK-21510
title: Add primary Research Studio outputs
status: Done
assignee:
- '@codex'
created_date: 2026-08-24 05:54
updated_date: 2026-11-24 20:18
labels:
- research
- workspace
- studio
- artifacts
dependencies:
- TASK-21507
- TASK-21508
- TASK-21509
references:
- Docs/superpowers/specs/2026-08-23-research-workspace-design.md
- Docs/superpowers/plans/2026-08-23-research-workspace-primary-studio.md
- backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Summary, Flashcards, Quiz, Report, and Compare Sources as traceable Research Studio creations that save to and reopen from their existing canonical owners instead of a duplicate workspace-output database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The five primary output cards project explicit availability, owner, reason, and recovery from current authority, source readiness, processing route, and capability revision; Compare requires at least two ready sources.
- [x] #2 A field-level owner mapping covers ID, version, source snapshot/provenance, generation configuration, workspace association, reopen route, replacement/new-version behavior, and deletion for every primary output before implementation.
- [x] #3 Local Summary, Report, and Compare save as Local Chatbook artifacts plus workspace membership; local Flashcards save as Study deck/cards plus membership; local Quiz saves as Quiz record/questions plus membership.
- [x] #4 Server outputs generate and persist through the supported server workspace artifact/study contracts and never masquerade a client download as server persistence.
- [x] #5 Generation captures an immutable qualified workspace/source/version/config snapshot, fences stale UI results, supports cancel/retry, and reports exact processing route and saved destination.
- [x] #6 Studio history is derived from canonical records and memberships through `WorkspaceOutputRef`; no parallel output row becomes the version, delete, or list owner.
- [x] #7 View/edit/export, regenerate-replace, regenerate-new-version, discuss in Grounded Chat, save/append to Quick Notes, delete, safe undo, and canonical reopen are available only where the owning service supports them.
- [x] #8 Source-version changes mark existing outputs `Sources changed since generation` while preserving inspection; regeneration uses a fresh source snapshot.
- [x] #9 Targeted owner-adapter, field-mapping, generation, persistence, reopen, version, delete, no-blending, mounted Studio, and real canonical-owner round-trip tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-322 already chooses each canonical local/server owner and forbids a duplicate output store. This task implements those mappings; a new ADR is required if any primary output receives a new canonical database.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-primary-studio.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
All five primary Studio outputs now save to their canonical owners in Local and Server under ADR-322. Local Study deck/cards and quiz/questions use bounded membership metadata. Server text uses versioned artifacts and accepted-version Markdown export. Server Study uses the retry-safe workspace publication contract from tldw_server TASK-15637 and ADR-097 (local server commit dfd9ff90aa). Canonical Study owns edits and deletion, and Studio can generate a new version. Older server installations leave the Flashcards and Quiz cards unavailable.

Server Study handoffs check the active backend, server profile, account, and workspace. Both saved owner IDs reopen outside the first page, and Back to Workspace returns to Research. The client normalizes native typed quiz responses and settles the exact handoff after the Study selection succeeds.

The Server suite passed 100 targeted tests, including SQLite and PostgreSQL, and 19 final contract cases. The loopback check used real authenticated Server routes, selected Library Media, and a stub provider. Dropping each save response after commit and retrying recovered the same two owners. After reconnecting, native Study listed and resolved the deck and quiz and their entries. The final Chatbook regression run passed 572 tests. The two deselected provider configuration cases and 26 adjacent Study UI profile-bootstrap failures reproduce on the untouched base with raw_source_selection_changed. The mounted handoff regressions passed for both output kinds.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Research Studio primary outputs across Local and Server, with exact native Study reopening and retry-safe publication.
<!-- SECTION:FINAL_SUMMARY:END -->
