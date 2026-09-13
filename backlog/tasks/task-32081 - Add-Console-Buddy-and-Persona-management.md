---
id: TASK-32081
title: Add Console Buddy and Persona management
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:30'
updated_date: '2026-09-09 01:13'
labels:
  - buddy
  - console
dependencies:
  - TASK-32078
  - TASK-32079
  - TASK-32080
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Manage the current Buddy and conversation Persona without leaving Console using one shared modal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Composer Menu has a Buddy action opening the same management modal as floating Buddy settings.
- [x] #2 Apply can enable/select/import an independent Buddy, set animation/size, and choose an explicit named conversation or workspace binding; Cancel has no mutations.
- [x] #3 Persona changes including None use existing assignment boundaries and do not alter running request context.
- [x] #4 Preview, keyboard access, narrow-terminal scrolling and focus restoration work; missing targets show an actionable unavailable state.
- [x] #5 Buddy animation reflects only its selected scope; changing Console selection does not retarget it.
- [x] #6 Existing-session Persona assignment commits identity, prompt and settings atomically before live publication; stale, busy, inactive or repurposed targets retain their old assignment, and workspace changes preserve memory/tool policy.
- [x] #7 Preview, geometry, Apply and Cancel are visible and usable at normal and compact sizes, with optional import and geometry progressively disclosed.
- [x] #8 Apply failures preserve all staged values with actionable errors and truthful partial-save status; duplicate Apply and ambiguous dismissal while saving are prevented.
- [x] #9 Current conversation Persona and workspace default names, None and unavailable states are visible; Personas controls manage the independent Buddy through the shared coordinator.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Preserve the completed implementation under ADR-139 and ADR-079; follow Docs/superpowers/plans/2026-09-08-chatbook-buddy-ux-review-remediation.md for the reviewed UX corrections, focused regressions and mounted verification. ADR required: no new ADR. Reason: direct correction within existing ownership, persistence and runtime boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-32081 Persona assignment seam (workspace_persona_audit): Added Chat/console_persona_assignment.py. prepare_buddy_persona_assignment(app, binding, target, persona_choice) performs read-only admission and returns PreparedPersonaAssignment; await assignment.apply() revalidates the exact local target and active Persona record. #unchanged is skipped, same Persona is a no-op, #none removes the Persona explicitly. Character/server ownership, active/early-preparation runs, queued work, parked decisions, recovery and settings writes in flight are rejected. No conversation reset, greeting, recreation or active-tab lookup occurs.

Existing saved conversation identity, prompt, full Console settings and safe generation metadata use one optimistic SQLite update/transaction before live publication. Target/settings/identity/binding revisions and saved identity/version are checked, unrelated settings and metadata are preserved, live-only endpoint URLs remain out of durable settings, and source-fork, identity/generation/settings, provider-context and speech epochs fence subsequent work. This bounded synchronous local write has no await between final admission and publication; unsaved/temporary assignments remain staged without creating a durable row.

Workspace assignments preserve existing tool profile and memory policy, use read_only when starting from no Persona, and reject assigning a different Persona under existing read_write permission because this picker has no new write-memory confirmation. None revokes the default and records the provisioning optout. Added optional expected_record guards to registry set/clear, checked inside BEGIN IMMEDIATE; the native workspace default modal also uses them. WorkspaceDB intentionally rejects nested transactions, so the optimistic guard lives inside its existing registry mutation instead of an outer transaction.

Verification: 50 passed in Tests/Chat/test_console_persona_assignment.py, Tests/Workspaces/test_workspace_assistant_defaults.py, Tests/UI/test_workspace_persona_creation.py, and Tests/Persona_Buddy/test_buddy_management_coordinator.py (24.19s), using shared main .venv with explicit worktree/package PYTHONPATH and isolated fixtures. Covers real SQLite identity/prompt/settings round-trip, trigger-induced rollback, optimistic DB conflict, stale/deleted/inactive Persona, busy/queued/parked/early-preparation targets, repurposed sessions, workspace default CAS/profile replacement, None, no-op, temporary staging, exact-tab targeting and fork barrier. New files and picker pass Ruff/formatter; four changed Python files parse; zero Ruff diagnostics on changed lines (12 existing registry diagnostics outside changes); git diff --check passes. Shared-venv RequestsDependencyWarning remains. No commits/staging; parent owns remaining management ACs and final task status. ADR-139 is the applicable ownership/service boundary.

Root integration: added one native staged Buddy/Follow/Persona/Notifications form shared by Console Menu and floating settings, independent library preview/import, fixed Apply/Cancel footer, exact conversation/workspace binding persistence, and scoped trusted lifecycle projection. Apply batches private artwork/presentation/scope preferences with rollback; existing Persona assignment uses the prepared atomic seam above. First-save binding promotion is serialized with Apply and preserves temporary/stale-owner fences; restored bindings resolve to their exact current form choice. Body click/Enter opens interaction without writing geometry. Static and global reduced motion suppress animation. Scope replay preserves accepted run identity, including VALIDATING, and retained workspace-member tool/voice/wake leases survive scope additions. Targeted evidence: 31 adapter/scope/coordinator checks, 5 management modal checks, entry-point/body-click checks and fresh-profile actual Menu -> Apply -> Home -> pinned chat -> Close -> Cancel journey pass. Independent UI review issues fixed with regressions. Docs/User_Guide/buddies.md and ADR-139 describe behavior; final combined feature gate pending.

Final joint verification: 813 foundational tests passed and the 288-test runtime/UI gate passed all management behavior except a test-only premature Select assignment before child composition. Settling the pilot after modal publication fixes that fixture; the focused management modal, fresh-install journey, entry points and normal/compact speech-enabled inbox group now passes all 12 tests (18.11s). Root and independent UI reviews are complete; validating-state replay, saved binding promotion/restoration and stale-owner fixes are covered. All modified Python files parse, no introduced Ruff diagnostics, new files formatted, diff whitespace clean. Known unrelated baseline tests are recorded under final TASK-32084 evidence.

Native UX corrections complete: visible preview/focus and scoped geometry, progressively disclosed import/size, named current/default Persona with literal brackets, retained staged Apply errors and guarded imports/retry, and independent Personas inspector Manage/Show/Close/Disable controls.384affected tests passed; later management/layout20 and coordinator17 passed including partial-save recovery. Normal/compact native renders inspected; independent review clear. Existing ADR-139/079 boundaries, guide, plan and review evidence updated. No full suite or real provider/audio checks.
<!-- SECTION:NOTES:END -->
