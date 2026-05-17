# Phase 6 QA Evidence

Status: TASK-13.1 through TASK-13.4 done; TASK-13.5 through TASK-13.7 not started

This directory contains release-hardening QA evidence for Product Maturity Phase 6. Phase 6 verifies release-candidate usability after the verified Phase 1 through Phase 5 product-maturity gates.

## Evidence Index

| Gate | Evidence | Status |
| --- | --- | --- |
| Phase 6.1 Release hardening planning and task breakdown | `Docs/superpowers/plans/2026-05-16-phase-6-release-hardening.md` | verified for planning only |
| Phase 6.2 Full first-time user release replay | `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-2-first-time-user-release-replay.md` | verified |
| Phase 6.3 Power-user workflow release replay | `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-3-power-user-workflow-release-replay.md` | verified |
| Phase 6.4 Keyboard/focus/accessibility and visual sweep | `Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-4-keyboard-focus-accessibility-visual-sweep.md` | verified |
| Phase 6.5 Recovery/setup/documentation alignment | pending | not-started |
| Phase 6.6 Packaging/configuration/data-safety validation | pending | not-started |
| Phase 6.7 Public roadmap release closeout | pending | not-started |

## Release-Hardening Rules

- Running-app QA is required before any Phase 6 child task can close.
- Actual rendered screenshots are required for any changed visible UI, and changed screens remain unapproved until the user approves the screenshot.
- P0/P1 release blockers must be fixed or explicitly accepted with owner, rationale, and follow-up.
- Verification commands in evidence should use portable `python -m pytest ...` forms unless documenting a local execution path separately.
