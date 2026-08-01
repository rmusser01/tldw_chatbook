---
id: TASK-1693
title: Add versioned Studio TTS preference storage and migration
status: To Do
assignee: []
created_date: '2026-08-01 06:02'
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
- [ ] #1 Studio preferences persist through the existing atomic configuration owner under schema-versioned speech_studio selection and canonical-provider option namespaces, with every selection and option represented as an optional override (CFG-020 and MIG-001).
- [ ] #2 An absent Studio value means inherit at resolution time; Reset to Global deletes all Studio selection and provider-option overrides instead of copying global values, while retaining only the schema envelope (CFG-021).
- [ ] #3 Studio persistence accepts only canonical provider IDs and options proven request-scoped by the TASK-1692 ownership contract; unknown providers, unknown options, and runtime-global values fail closed (CFG-022 and CFG-026).
- [ ] #4 The Studio store cannot persist credentials, endpoints, environment-derived values, masked placeholders, runtime initialization paths, provider safety limits, character assignments, or submitted synthesis text (SEC-001 through SEC-003).
- [ ] #5 A versioned idempotent migration copies only legacy values proven to be request-scoped Studio tuning, performs no startup write when unnecessary, preserves every existing global and legacy key, and never copies secrets, endpoints, or initialization resources (MIG-002, MIG-003, and MIG-006).
- [ ] #6 Malformed fields recover independently, and an unrecoverable Studio record can be reset or quarantined without changing global settings, character profiles or assignments, credentials, or legacy-provider behavior (MIG-004 and STATE-024).
- [ ] #7 Concurrent or stale writers cannot overwrite a newer successfully saved Studio snapshot without a conflict result, and persistence publishes no partial snapshot (MIG-005).
- [ ] #8 Deterministic tests cover round-trip persistence, sparse inheritance representation, reset-by-deletion, provider isolation, revision conflicts, repeated migration, mixed valid and malformed fields, corruption recovery, and secret exclusion.
- [ ] #9 This task adds no visible Settings or Lab editor changes, provider reconfiguration, network discovery, generation behavior, or managed audio.cpp behavior.
<!-- AC:END -->
