---
id: TASK-24532
title: Clarify and deduplicate startup diagnostics
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 00:59'
labels:
  - diagnostics
  - startup
  - privacy
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-29-startup-diagnostic-clarity-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-startup-diagnostic-clarity.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make startup diagnostics distinguish optional feature absence, unverified security posture, recoverable cache rejection, and genuine failures without duplicate or sensitive output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Missing HuggingFace evaluation support is reported once as actionable informational capability copy, while both production dataset-loader paths return the specific typed missing-dependency failure for recognized remote identifiers
- [ ] #2 Importing OpenTelemetry support is silent, and repeated or concurrent initialization emits one authoritative unavailable or success outcome with a stable boolean result and no global-provider replacement
- [ ] #3 Prometheus initialization emits one authoritative informational-unavailable or successful outcome with a stable boolean result, while server-start failures remain warnings
- [ ] #4 The alternate module startup path adds no unconditional metrics success messages and unexpected initializer diagnostics expose only bounded static text plus exception type
- [ ] #5 SQLite and runtime-policy unverified-platform diagnostics remain deduplicated warnings that explicitly say verification was unavailable and the named operation continues with an unverified posture
- [ ] #6 Model-catalog cache rejection remains a count-only warning that states accepted entries continue loading and discovery may restore missing data
- [ ] #7 Changed diagnostics exclude representative credential, path, service-name, cache-content, and exception-message sentinels under focused tests
- [ ] #8 Local dataset routing, invalid source behavior, privacy decisions, runtime-policy decisions, cache validation, recovery behavior, and installed-entry-point telemetry behavior remain unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Correct optional Evals import severity and typed feature-use routing.
2. Make OpenTelemetry initialization silent-on-import, thread-safe, idempotent, and boolean-returning.
3. Make Prometheus authoritative and remove caller overclaims.
4. Clarify existing privacy, runtime-policy, and cache warning copy without policy changes.
5. Run focused secrecy, deduplication, behavior, lint, and compilation checks.
6. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-startup-diagnostic-clarity.md
ADR required: no
ADR path: N/A
Reason: diagnostic ownership and wording within existing boundaries.
<!-- SECTION:PLAN:END -->
