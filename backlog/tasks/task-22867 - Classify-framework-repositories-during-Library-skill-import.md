---
id: TASK-22867
title: Classify framework repositories during Library skill import
status: In Progress
assignee:
  - codex
created_date: '2026-08-27 04:14'
updated_date: '2026-08-28 00:00'
labels:
  - skills
  - library
  - ux
  - import
dependencies:
  - TASK-613
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-library-skill-import-framework-classification.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Distinguish installable skill bundles from generic framework repositories and present accurate recovery guidance and import vocabulary without adding framework-specific product integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One classifier distinguishes an installable root skill, a repository/archive containing multiple independently installable skills, a valid non-skill framework, malformed/unsupported input, and fetch/authentication failure.
- [ ] #2 A valid repository with no accepted `SKILL.md` reports that it is a framework/repository rather than an installable Codex skill, without naming or special-casing ATHF.
- [ ] #3 Recovery offers only supported generic paths: choose an installable skill subdirectory, use project instructions when appropriate, use the external CLI, or create a separately reviewed wrapper skill.
- [ ] #4 Library consistently distinguishes “Import skill” from document/media ingestion and exposes idle, inspecting/importing, not-a-skill, trust-review, complete, and failed/retry states.
- [ ] #5 TASK-613's single in-flight import contract applies across file, folder, zip, and URL imports; leaving Library reports authoritative completion rather than pretending to cancel an accepted threaded install.
- [ ] #6 Local fixture tests cover each classification, multiple-skill selection, shared in-flight behavior, trust handoff, and redacted network failures.
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: N/A

Reason: This adds truthful classification and recovery states while preserving ADR-009's trust boundary, ADR-069's import-copy posture, TASK-613's app-owned single-flight coordinator, and the existing remote-fetch/import security contracts.

1. Define and test one pure bounded classifier for root skills, multi-skill repositories, valid framework repositories, malformed/unsupported inputs, and remote fetch/auth failures.
2. Preserve fetch/import separation: retain one bounded archive through explicit candidate choice, never execute repository code, and route successful selected imports through the existing trust-pending seam.
3. Add generic multi-skill choice and framework recovery states to Library, sharing TASK-613's app-owned operation lifecycle across local and remote paths.
4. Verify local fixtures, trust handoff, redacted failures, normal/compact Library UX, and unchanged SSRF/runtime/project-skill boundaries; update user guidance and implementation notes.

## Implementation Notes

- Added one bounded directory/zip classifier and a retained remote
  inspect-then-import seam. Multiple candidates require one explicit selection and
  reuse the exact inspected bytes/hash; framework and malformed packages import
  nothing.
- Extended TASK-613's existing app-owned coordinator rather than adding another
  pipeline. Initial inspection, choice, selected import, Cancel, Retry, navigation,
  routed replacement, and terminal cleanup share its one single-flight lifecycle.
- Added the generic framework recovery and failed/Retry states, a mounted
  normal/compact candidate modal, the explicit **Import skill…** vocabulary, and the
  existing trust-review handoff with `trust_approved=False`.
- Preserved ADR-009, ADR-069, SSRF/redirect/deadline/archive caps, and runtime-policy
  boundaries. No new ADR, dependency, framework-specific integration, repository-code
  execution, or briefing-to-hunt handoff was added.
- Exact targeted verification: 121 passed with one inherited Requests dependency
  warning. Additional focused canvas coverage: 9 passed, 136 deselected. Ruff,
  compilation, Impeccable detector, and `git diff --check` passed. See
  `.superpowers/sdd/2026-08-28-library-skill-framework-classification/task-1-report.md`.
- Status remains In Progress and the acceptance criteria remain unchecked pending
  independent review.
- Independent review round 1 remediation replays pending choice on every Skills
  entry/hydration, keeps signed URL authority private behind a host-only draft,
  revalidates local candidate identity/containment immediately before import, routes
  the public Console seam through the same retained classifier result, removes raw
  exception logging, and separates the 20-item display bound from package validity
  while applying declared archive-size caps. Exact plan verification is 135 passed;
  round-1 probes are 9 passed. Status and acceptance criteria remain unchanged for
  round 2.
