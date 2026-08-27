---
id: TASK-22867
title: Classify framework repositories during Library skill import
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:17'
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
