---
id: TASK-19059
title: 'Import, review, and activate Actor Packs'
status: To Do
assignee: []
created_date: '2026-08-20 18:40'
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
- [ ] #1 Import enforces all outer/member/section budgets, canonical paths, declared-file and digest integrity, MIME/decode limits, and free-space preflight before extraction into pinned private staging; symlinks, hardlinks, other linked entries, encryption, nesting, devices, external references, undeclared files, and duplicate, Unicode, case, device, or alias path collisions are rejected.
- [ ] #2 Review remains path-free and shows actor fields, portrait, visual inventory, license/provenance, warnings, UUID match, differences, and the exact effect of every activation choice; all untrusted actor, license, provenance, and archive text renders as plain text, and review actions are labelled, keyboard-operable, focus-safe, usable in compact and normal layouts, and bind no forbidden terminal-convention, reserved, or global keys.
- [ ] #3 With no UUID match, Create New preserves the incoming UUID and Create Copy assigns a fresh UUID; with a same-kind exact match, Create Copy or explicitly confirmed Update Existing is offered; cross-kind reuse is rejected.
- [ ] #4 Update Existing changes only reviewed portable actor fields and present visual sections; every omitted optional section visibly preserves its current local binding.
- [ ] #5 Review snapshots the profile and source identity, actor kind and revision, portable UUID and registry row, both bindings and active versions, staged-file inode/digest identity, and free-space authority; all are revalidated immediately before activation, and any delete/recreate or revision ABA returns to review without auto-merge.
- [ ] #6 Character activation is transactional; Persona activation consumes the cross-store coordinator; failure/cancellation preserves prior actor/bindings, drains workers, and exposes only opaque pinned cleanup eligibility.
- [ ] #7 After commit, affected-only invalidation and refresh run independently for Shared Visual Identity caches, Persona runtime, mounted Buddy, and authoritative review/editor consumers; one consumer failure reports a fixed path-free category without suppressing the others or rolling back the committed activation.
- [ ] #8 Verification includes born-RED-to-GREEN evidence; mutation proof for authority, archive, cancellation, cleanup, and invalidation-isolation guards; real SQLite migration and crash-recovery tests; assigned-worktree provenance; isolated HOME/XDG/config/data roots; independent golden round trips and adversarial traversal/link/collision/bomb/truncation/digest/MIME/disk/race/crash/cleanup tests; Pilot coverage at normal and 80x24 geometry plus isolated real-terminal keyboard confirm/cancel/focus checks; Impeccable review after the final visible UI change; and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance gates.
<!-- AC:END -->
