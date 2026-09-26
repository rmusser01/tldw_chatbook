---
id: TASK-32725
title: Add a redacted effective llama.cpp launch preview
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 17:25'
updated_date: '2026-09-20 17:02'
labels:
  - llamacpp
  - catapult-review
  - foundation
  - feature
dependencies:
  - TASK-32721
  - TASK-32722
  - TASK-32723
documentation:
  - Docs/superpowers/plans/2026-09-17-llamacpp-management-roadmap.md
  - Docs/superpowers/reviews/2026-09-17-catapult-llamacpp-management-review.md
  - Docs/superpowers/reviews/2026-09-20-llamacpp-launch-preview-verification.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users see the effective launch settings and conflicts before starting a server.

### Scope

A Lab-local preview and copy action driven by the same validated argv construction as Start. Show runtime defaults as omitted values and distinguish the running launch from the next-launch draft.

### Explicit exclusions

No arbitrary raw argument values in derived output; no command persistence, terminal execution, runtime installation, or Console sampling changes.

### Architecture gate

ADR required: no new ADR. ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md. Reason: implement the existing ADR-114 derived-command privacy boundary and ADR-165 tuning ownership.

### Affected areas

- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py`
- `tldw_chatbook/LLM_Management/llamacpp_profiles.py`
- `tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py`
- `tldw_chatbook/UI/LLM_Management_Window.py`

### Validation contract

Primary targeted run after implementing the scoped behavior:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_llamacpp_launch_contract.py Tests/LLM_Management/test_llamacpp_profiles.py Tests/UI/test_llamacpp_setup_view.py
```

New test paths are planned deliverables. Run Ruff on changed Python files, `git diff --check`, and mounted/CSS governance checks for affected UI. Live cases are opt-in with isolated state and explicit assets; skips are not qualification. No full suite without user opt-in.

### Execution boundary

This is a scoped To Do ticket. Read its dependencies and the linked roadmap before starting. Move it to In Progress, then add its detailed Implementation Plan (including the actual governing ADR path) before changing application code. Acceptance criteria and evidence govern completion; this planning pass authorizes no automatic execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview and Start consume equivalent validated launch values; unset fields, reserved alias, explicit ports and snapshot-owned options cannot diverge.
- [x] #2 Derived display and copy reveal only allowlisted flags and safe typed values; executable/model/projector paths, tokens and unknown expert values are suppressed, including attached-value forms.
- [x] #3 Invalid typed/raw combinations show the same recoverable reason before spawn; preview has no process, network, persistence or lease side effects.
- [x] #4 Editing a draft while a process runs labels it as next launch and does not change the active process or Console defaults.
- [x] #5 Mounted keyboard/copy checks and targeted command/privacy tests pass; the Lab guide explains omitted defaults and redacted placeholders.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md
Reason: implement the accepted ADR-114 derived-command privacy boundary with ADR-165 tuning ownership; preserve ADR-119 snapshot admission.

1. Preserve the uncommitted first-milestone baseline in the native isolated worktree and verify focused profile/pane coverage (63 tests passed before edits).
2. Add a pure shared launch-options preparation boundary for host/port defaults, quote-aware expert parsing, typed/raw/source conflicts and snapshot-owned options. Preview has no file probing, leases, directory creation, network or subprocess effects; source/executable validity is explicitly checked only at Start.
3. Render only trusted typed options and placeholders for executable/model/expert inputs. Snapshot flags remain conditional on Start admission and generated private directories; after admission freeze the actual sanitized summary on the exact claim. Never parse arbitrary raw arguments back into display values.
4. Add Preview/Copy actions to the existing pane, invalidate stale previews when inputs change, and distinguish Current launch from Next launch. Allow tuning edits as next-launch drafts while retaining source/process ownership controls and existing connection evidence for the active process.
5. Add meaningful pure privacy/conflict/equivalence tests and mounted keyboard/copy/current-versus-next tests with production CSS at compact and wide terminal sizes. Run focused launcher, profile, snapshot/lifecycle and UI regressions; compare the preview/Start shared path without starting a real model for this presentation-only change.
6. Run scoped lint/format, CSS governance and diff checks, obtain an independent review, update the Lab guide and task evidence, then copy only baseline-verified task changes back into the shared checkout. No full suite, commit or publication is part of this ticket.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented redacted Preview launch and Copy redacted with one pure Preview/Start option-preparation boundary. Derived output includes trusted typed values and placeholders only; all expert tokens are suppressed. Start retains source/executable validation, listener preflight, leases and snapshot admission, then freezes actual snapshot status on the exact claim. Active tuning/listener/expert edits are next-launch drafts and preserve the verified active target and Console defaults.

Added llama.cpp preview module, focused pure/Start tests and production-CSS mounted UI tests; updated the existing launcher/lifecycle, setup pane/Models window, source/generated CSS and Lab guide. Integrated only ten baseline-verified files from the isolated worktree. Independent review found and verified the repair for focused Preview moving off screen after expansion; regression checks now assert both Preview and subsequent Copy remain visible.

Validation: combined targeted run 259 passed; final full mounted preview/setup selection after the focus fix 12 passed. Token governance, CSS bundle sync and scoped whitespace checks pass. All seven changed Python files pass formatting; same-filename Ruff comparison shows zero added findings with 65 pre-existing findings retained in large legacy files. No full suite or new real-model/platform qualification was claimed. No commit or publication.

ADR required: no new ADR. ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md; implements ADR-114 privacy and preserves ADR-119 snapshot ownership. Evidence and exact commands: Docs/superpowers/reviews/2026-09-20-llamacpp-launch-preview-verification.md. Guide and roadmap updated; incident recorded in backlog/docs/lessons-live-verification.md. No unresolved review blockers.
<!-- SECTION:NOTES:END -->
