---
id: TASK-19059
title: 'Import, review, and activate Actor Packs'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 18:40'
updated_date: '2026-08-23 05:02'
labels: []
dependencies:
  - TASK-19053
  - TASK-19055
  - TASK-19056
  - TASK-19057
  - TASK-19058
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users safely inspect and activate an untrusted Actor Pack as a new local actor, a copy, or an explicitly confirmed update without risking existing actor data or visual bindings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import enforces all outer/member/section budgets, canonical paths, declared-file and digest integrity, MIME/decode limits, and free-space preflight before extraction into pinned private staging; symlinks, hardlinks, other linked entries, encryption, nesting, devices, external references, undeclared files, and duplicate, Unicode, case, device, or alias path collisions are rejected.
- [x] #2 Review remains path-free and shows actor fields, portrait, visual inventory, license/provenance, warnings, UUID match, differences, and the exact effect of every activation choice; all untrusted actor, license, provenance, and archive text renders as plain text, and review actions are labelled, keyboard-operable, focus-safe, usable in compact and normal layouts, and bind no forbidden terminal-convention, reserved, or global keys.
- [x] #3 With no UUID match, Create New preserves the incoming UUID and Create Copy assigns a fresh UUID; with a same-kind exact match, Create Copy or explicitly confirmed Update Existing is offered; cross-kind reuse is rejected.
- [x] #4 Update Existing changes only reviewed portable actor fields and present visual sections; every omitted optional section visibly preserves its current local binding.
- [x] #5 Review snapshots the profile and source identity, actor kind and revision, portable UUID and registry row, both bindings and active versions, staged-file inode/digest identity, and free-space authority; all are revalidated immediately before activation, and any delete/recreate or revision ABA returns to review without auto-merge.
- [x] #6 Character activation is transactional; Persona activation consumes the cross-store coordinator; failure/cancellation preserves prior actor/bindings, drains workers, and exposes only opaque pinned cleanup eligibility.
- [x] #7 After commit, affected-only invalidation and refresh run independently for Shared Visual Identity caches, Persona runtime, mounted Buddy, and authoritative review/editor consumers; one consumer failure reports a fixed path-free category without suppressing the others or rolling back the committed activation.
- [x] #8 Verification includes born-RED-to-GREEN evidence; mutation proof for authority, archive, cancellation, cleanup, and invalidation-isolation guards; real SQLite migration and crash-recovery tests; assigned-worktree provenance; isolated HOME/XDG/config/data roots; independent golden round trips and adversarial traversal/link/collision/bomb/truncation/digest/MIME/disk/race/crash/cleanup tests; Pilot coverage at normal and 80x24 geometry plus isolated real-terminal keyboard confirm/cancel/focus checks; Impeccable review after the final visible UI change; and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement hostile-archive validation, private pinned staging, typed-section validation, path-free review, and fail-closed cleanup in `Actor_Packs.importer` using the existing V1 contracts.
2. Snapshot the exact UUID match/action matrix, actor and visual authority, staged filesystem identity, and free-space authority; revalidate every independently mutable input immediately before activation.
3. Activate Characters inside one owned SQLite transaction and extend ADR-074's Persona coordinator so actor, portable identity, present visual sections, and bindings commit or recover together while omitted bindings remain unchanged.
4. Add an app-owned cancellation/drain controller and affected-only post-commit invalidation whose consumer failures remain isolated and path-free.
5. Add a separate labelled Workbench import action plus compact keyboard-safe review/confirmation flow; keep legacy Character and Persona Visual imports distinct.
6. Prove the flow with born-RED-to-GREEN, mutation, real-SQLite crash/recovery, independent golden/round-trip, adversarial archive/race/cleanup, Pilot, real-terminal, Impeccable, privacy, architecture, static, and governance evidence.

ADR required: no

ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

Reason: ADR-074 already defines the archive trust boundary, UUID/action matrix, omitted-section policy, Character transaction, Persona coordination, cleanup, and refresh isolation implemented by this task.

Detailed execution plan: `Docs/superpowers/plans/2026-08-23-task-19059-actor-pack-import.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added bounded hostile-ZIP admission, streamed declared-file extraction, typed Shared Visual Identity and Persona runtime validation, source/staging/free-space revalidation, path-free review records, and an authenticated bounded startup staging sweep.
- Added the exact UUID action matrix and actor/registry/Shared Visual/Persona Visual authority snapshots. Character activation now commits actor, identity, and present visual rows in one SQLite transaction; Persona activation reuses ADR-074's JSON/SQLite intent coordinator and commits present visual rows in the same SQLite transaction. Omitted bindings remain byte-for-byte unchanged.
- Added one app-owned import controller for off-loop work, cancellation/drain, review leases, cleanup, and isolated post-commit callbacks. The Workbench now exposes a distinct **Import Actor Pack** action, compact scrollable review, in-memory portrait preview, inert untrusted text, explicit update confirmation, and affected-list/Buddy refresh.
- Added real-SQLite rollback, ambiguous-commit recovery, visual-preservation, authority-staleness, cleanup-sweep, disk, hostile-link/encryption/truncation, independent stdlib section archive, compact Pilot, keyboard, privacy, architecture, and app-lifecycle coverage. Impeccable's final detector reported no findings; isolated 80x24 terminal Enter and Escape checks returned `create_new` and cancellation respectively.
- Review hardening added strict nested Persona voice/setup validation at the pure contract boundary, bounded ZIP-directory and decoded-image admission, cleanup for every publication/cancellation outcome, owned sequential Persona portrait anchors hidden from normal Character lists, real Console/Buddy invalidation, isolated post-commit refresh failures, and a fresh review loop after stale activation. Follow-up review fixes route selected archive paths through the central validator, lazily load Pillow at image-inspection boundaries, clean failed candidate setup, use static parameterized Character inserts, keep UUID lookup transaction-owned, and atomically reserve Persona portrait anchors through SQLite's sequence.
- Verification: after rebasing onto current `dev`, `Tests/Actor_Packs` passed 201 tests and the scoped architecture/UI workflow set passed 72 tests; the persistent diagnostic inventory pin passed, and scoped Ruff, format, compileall, and `git diff --check` passed. The rebase retained `dev`'s deferred Actor Pack recovery lifecycle and audited the inherited diagnostic-inventory drift plus the one Actor Pack reindent: no new sink or sensitive interpolation was introduced, and the reviewed inventory pin was regenerated. Repository-wide collection reached 56,385 tests and is blocked only by the pre-existing parametrization error in `Tests/UI/test_library_file_notes_workspace.py` (`push_phase` is declared but unused by the test function).
- ADR required: no. ADR-074 already defines the implemented import trust, activation, recovery, omitted-section, and refresh boundaries; no new architectural decision was introduced.
<!-- SECTION:NOTES:END -->
