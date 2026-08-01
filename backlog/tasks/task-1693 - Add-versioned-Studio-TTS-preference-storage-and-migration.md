---
id: TASK-1693
title: Add versioned Studio TTS preference storage and migration
status: Done
assignee: []
created_date: '2026-08-01 06:02'
updated_date: '2026-08-01 08:09'
labels:
  - tts
  - settings
  - persistence
dependencies:
  - TASK-1692
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give the Speech Studio a durable preference scope that can remember provider-specific request choices without becoming a second owner for global defaults, credentials, runtime initialization, or character profiles. The store must be additive and recoverable so existing users retain their current global and legacy-provider behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Studio preferences persist through the existing atomic configuration owner under schema-versioned speech_studio selection and canonical-provider option namespaces, with every selection and option represented as an optional override (CFG-020 and MIG-001).
- [x] #2 An absent Studio value means inherit at resolution time; Reset to Global deletes all Studio selection and provider-option overrides instead of copying global values, while retaining only the schema envelope (CFG-021).
- [x] #3 Studio persistence accepts only canonical provider IDs and options proven request-scoped by the TASK-1692 ownership contract; unknown providers, unknown options, and runtime-global values fail closed (CFG-022 and CFG-026).
- [x] #4 The Studio store cannot persist credentials, endpoints, environment-derived values, masked placeholders, runtime initialization paths, provider safety limits, character assignments, or submitted synthesis text (SEC-001 through SEC-003).
- [x] #5 A versioned idempotent migration copies only legacy values proven to be request-scoped Studio tuning, performs no startup write when unnecessary, preserves every existing global and legacy key, and never copies secrets, endpoints, or initialization resources (MIG-002, MIG-003, and MIG-006).
- [x] #6 Malformed fields recover independently, and an unrecoverable Studio record can be reset or quarantined without changing global settings, character profiles or assignments, credentials, or legacy-provider behavior (MIG-004 and STATE-024).
- [x] #7 Concurrent or stale writers cannot overwrite a newer successfully saved Studio snapshot without a conflict result, and persistence publishes no partial snapshot (MIG-005).
- [x] #8 Deterministic tests cover round-trip persistence, sparse inheritance representation, reset-by-deletion, provider isolation, revision conflicts, repeated migration, mixed valid and malformed fields, corruption recovery, and secret exclusion.
- [x] #9 This task adds no visible Settings or Lab editor changes, provider reconfiguration, network discovery, generation behavior, or managed audio.cpp behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: TASK-1693 implements ADR-039's accepted additive Studio persistence, sparse inheritance, migration, corruption-isolation, and compare-before-publish boundary; no new ADR is required.

Detailed plan: Docs/superpowers/plans/2026-08-01-task-1693-studio-tts-preference-storage.md

1. Add failing tests for immutable sparse Studio snapshots, canonical provider and request-option admission, round-trip persistence, reset-by-deletion, and provider isolation.
2. Add failing tests for revision conflicts and extend the existing atomic configuration owner with the smallest revision-guarded whole-section replacement primitive.
3. Add failing migration and corruption tests, then implement versioned field-by-field legacy migration, safe diagnostics, no-op startup reads, and Studio-only recovery.
4. Run focused configuration/TTS regressions, lint, diff integrity, and independent code review.
5. Record ADR conformance and verification, complete acceptance criteria, and mark Done only after all gates pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented immutable, sparse, schema-v1 Studio TTS preferences with canonical provider validation, request-scoped option allowlisting, reset-to-inheritance semantics, raw-only idempotent migration, field-level corruption recovery, and revision conflicts. Extended the atomic config owner with a private cross-process lock and revision-guarded section replacement; generic and whole-config writers preserve the revision-owned speech_studio section, and shutdown persistence now preserves bytes after unsafe reload failures. Reused ADR-039; no new ADR was required. Added focused storage, migration, concurrency, mutation, and persistence-owner coverage. Verification: focused matrix 172 passed; broad config regression 176 passed with 1 warning; full TTS suite 1997 passed, 14 skipped, and 1 unchanged pre-existing export-baseline failure; Ruff checks, compileall, diff integrity, and independent review passed. No UI, network, generation, provider reconfiguration, or managed audio.cpp behavior was added.
<!-- SECTION:NOTES:END -->
