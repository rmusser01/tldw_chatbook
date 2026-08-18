---
id: TASK-17651
title: >-
  Project skills (.SKILLS/) folder discovery and prompt-driven import
status: To Do
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - skills
  - workspaces
  - ux
priority: high
dependencies:
  - TASK-17650
---

## Description (the why)

A user resuming work on an existing project has no way to bring that project's
skills into the app short of importing them one directory at a time through
Library ▸ Skills. Introduce a project-local `.SKILLS/` convention: on app
startup in (or under) a directory containing one, and after creating a
workspace bound to such a directory, offer a prompt-driven (never silent)
import. Imports stay quarantined behind the ADR-009 trust boundary.

Spec: `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` §5.
Plan: `Docs/superpowers/plans/2026-08-17-project-skills-import.md`.

## Acceptance Criteria (the what)

- [ ] Launching the app from a project (or subdirectory, up to the first .git ancestor) containing `.SKILLS/` offers an import prompt listing discovered skills; the first-run wizard suppresses it for that launch
- [ ] Declining re-prompts only when the skill set's fingerprint changes; "Never for this folder" is permanent; `[skills] project_skills_prompt_enabled = false` disables the feature
- [ ] Creating a workspace with a bound folder containing `.SKILLS/` chains the same offer after creation
- [ ] Imports run through the existing importer with trust_approved=False (quarantined), never overwrite existing names silently, and the modal states the one-time trust review expectation with a route to Library ▸ Skills
- [ ] Discovery refuses symlinked `.SKILLS/` dirs and entries, caps entries (50) and frontmatter reads (64 KiB), pre-flags invalid names, and renders repo-sourced text escaped
- [ ] ADR for the convention added; config key documented; User Guide updated

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-17-project-skills-import.md` (6 tasks:
discovery → ledger → import modal → startup trigger → create-modal chaining →
ADR/docs + live verification). Starts only after TASK-17650 is on dev.
