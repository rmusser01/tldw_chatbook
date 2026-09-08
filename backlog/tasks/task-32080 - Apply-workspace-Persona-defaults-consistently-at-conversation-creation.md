---
id: TASK-32080
title: Apply workspace Persona defaults consistently at conversation creation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:29'
updated_date: '2026-09-08 20:53'
labels:
  - buddy
  - console
dependencies: []
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make optional workspace Persona defaults consistent for every new-conversation surface while retaining explicit and existing assignments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New conversations resolve workspace defaults across creation surfaces; explicit Persona and explicit None take precedence.
- [x] #2 Existing, copied and moved conversations retain assignment; future-default edits never rewrite them.
- [x] #3 Workspace creation and settings permit explicit None without provisioning or backfill recreating a default.
- [x] #4 Persona memory-mode identity persists with the new session and matches its settings.
- [x] #5 Targeted creation, persistence and native UI tests verify precedence and failure behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: independent ownership and explicit default-selection semantics.
Follow Task 3 in Docs/superpowers/plans/2026-09-08-console-buddy-foundations.md.
Read the approved spec and existing relevant ADRs; add failing targeted regression coverage, implement shared authority/creation seams, validate targeted integration, then document evidence and review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented creation-only workspace Persona defaults under ADR-139 (amending ADR-079): Console bootstrap, direct/store/controller, ordinary/temporary new chat and workspace activation resolve the explicit target once. Explicit None/generic/Persona/Character identities and restore/fork snapshots bypass inheritance. Provider setup recovery retains the already-assigned Persona prompt, label and memory mode.

WorkspaceDB v8 records deliberate None separately from omitted defaults, so legacy omission still auto-provisions a workspace Agent while user-selected None survives restart and pending backfill. Set/clear update the flag atomically; migration rollback/retry is covered with real SQLite. Session memory mode now matches its settings and durable conversation identity.

Shared WorkspacePersonaPicker and WorkspacePersonaDefaultModal add native creation/details choices with unavailable-Persona validation, stale-default rejection, preserved tool profile, inline failures and explicit read/write-memory confirmation. Console, Settings and Library creation pass the local Persona service. User guide documents future-only defaults and preserved existing conversations.

Verification (shared main .venv with explicit worktree/package PYTHONPATH and isolated fixtures; no full suite):
- Tests/Chat/test_workspace_default_session.py + test_console_chat_store.py: 429 passed.
- Final core group (workspace_default_session, workspace_persona_creation, agent_provisioning, workspace_assistant_defaults, workspace_db): 81 passed.
- Existing new-workspace/settings-default/lifecycle native group: 29 passed.
- Context-policy lifecycle + session-controller + switch-draft group: 64 passed, 1 baseline failure. The formerly partial lifecycle fake now uses the real controller/store seam.
- Provider default-actions persistence/publication flow: 1 passed (33 deselected).
- New/owned module and test Ruff/formatter checks pass; 22 touched Python files parse; diff-scoped Ruff reports zero diagnostics on added lines; git diff --check passes. Whole pre-existing touched files retain unrelated lint debt.

Baseline limits: clean c0a42150 source extracted to /private/tmp/task-32080-baseline-G7MHRn (not the dirty main checkout) reproduces the compact Settings recovery-layout failure, two Library handoff-copy failures, and fork-validation DuplicateIds recompose failure. The broader Settings/Library chunk had 51 passes and 4 failures; its folder-binding failure passes in isolation and on baseline, so no root cause claimed for that transient result. The session fork failure reproduces independently on both baseline and implementation. Existing shared-venv RequestsDependencyWarning also remains.

No staging or commits. Task remains In Progress for parent integration review; acceptance checks reflect verified task behavior, not a claim that all pre-existing UI/static checks pass. Existing-session Persona reassignment is explicitly outside this task and awaits the separate management task.

Final integration gate: 813 targeted tests passed across Buddy/Persona_Visual, WorkspaceDB, creation defaults/provisioning and atomic Persona assignment (162.62s). Independent implementation review was clean; existing baseline UI failures above remain outside this change. User guide includes future-only defaults and explicit None precedence. ADR-139 remains the applicable amendment to ADR-079. No new diagnostics after removing the now-unused DEFAULT_WORKSPACE_ID session import.
<!-- SECTION:NOTES:END -->
