---
id: TASK-27038
title: Publish Tool Packs through a captured destination
status: Done
assignee:
  - '@codex'
created_date: '2026-09-01 00:00'
updated_date: '2026-09-01 00:00'
labels:
  - tool-packs
  - export
  - filesystem
  - security
dependencies:
  - TASK-26070
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish a completed Tool Pack archive only to the exact destination accepted by
the user, preserving atomicity and reporting uncertain durability truthfully when
the host cannot confirm a post-replace directory sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Destination capture validates the `.tldw-tool-pack` path and pins parent and existing-target identity without following symlinks or accepting nonregular targets.
- [x] #2 Publication writes and fsyncs one authenticated private same-parent temporary, revalidates the exact captured destination/overwrite state, and uses only a supported atomic no-follow replacement.
- [x] #3 Cancellation, destination races, unsupported primitives, and pre-replace failures preserve the destination, remove only the authenticated temporary, and return stable path-free export error categories.
- [x] #4 Post-replace failures reconcile exact destination identity and archive digest, returning committed-with-uncertain-durability only for the exact new archive, retaining `publication_failed` only for provable exact-old/no-commit state, and otherwise reporting `durability_uncertain`; targeted tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing destination-capture, race, overwrite-token, cancellation, unsupported-host, symlink/nonregular, parent-substitution, and authenticated-cleanup tests.
2. Implement immutable captured destination/result types and strict path/identity validation.
3. Implement same-parent mode-0600 staging, archive flush/fsync, final revalidation, supported no-follow atomic publication, and parent fsync.
4. Add failing post-replace reconciliation tests for exact-new, exact-old, and third-state outcomes, then implement truthful committed/uncertain results.
5. Run the targeted publication tests, related export tests, scoped Ruff, diff hygiene, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the captured-destination identity, no-follow atomic replacement, failure reconciliation, stable outcomes, and separate Windows-support boundary implemented here.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added Tool-Pack-specific captured destination and publication result types with
  private path/token representations, strict suffix/parent/regular-file capture,
  exact overwrite authorization, and stable path-free failures.
- Added POSIX capability admission for the exact descriptor-relative replacement
  primitive, authenticated same-parent mode-`0600` staging, file and parent fsync,
  immediate target/parent/content revalidation, cancellation, and owned-temp-only
  cleanup. Native Windows publication and non-atomic fallbacks remain excluded.
- Added post-replace reconciliation through the retained parent descriptor. Exact
  replacement identity plus archive digest can report committed with uncertain
  durability; provable exact-old/no-commit state remains a publication failure;
  every third or ambiguous state reports `durability_uncertain`.
- Independent review found three race/capability gaps. Fix round 1 added incumbent
  digest checks for in-place rewrites, probed `os.replace` rather than `os.rename`,
  and reconciled named parent/target state after replacement. Scoped re-review
  approved all findings with no new breakage; the Minor primitive-naming concern
  was naturally resolved.
- Verification: 58 targeted publication/export/catalog tests passed with one
  pre-existing Requests dependency warning; scoped Ruff and `git diff --check`
  passed. Per repository policy, the full suite was not run.
- ADR required: no new ADR. ADR-107 remains the governing decision. The generalized
  replacement-boundary race incident is recorded in
  `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
