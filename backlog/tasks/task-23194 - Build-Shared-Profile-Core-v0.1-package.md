---
id: TASK-23194
title: Build Shared Profile Core v0.1 package
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 06:24'
updated_date: '2026-08-29 07:06'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the independently buildable and versioned Shared Profile Core contract package for Personal Context, including immutable models, canonical serialization, schema, fixtures, and public request/tool contracts while keeping runtime concerns out of the package.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared Core package builds as version 0.1.0 and imports successfully from the monorepo pin
- [x] #2 Public models, enums, payloads, interview and tool contracts enforce the approved ADR-102 validation rules
- [x] #3 Canonical serialization, integrity tags, JSON Schema, and v1 positive/negative fixtures are deterministic and conformant
- [x] #4 Targeted tests and local wheel inspection pass, with no database, HTTP, provider, UI, or key-custody modules included
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read ADR-102 and implement the pinned package under packages/tldw_profile_core with public models, enums, payloads, canonical serialization, interview/tool contracts, schema export, and fixtures.
2. Add strict TDD coverage for models, lifecycle/immutability/privacy validation, canonical bytes/integrity tags, interview/tool rejection, schema and fixture conformance; run the required RED command before implementation.
3. Generate deterministic JSON Schema and package metadata/resources, add the portable monorepo local pin/import check, and build the v0.1.0 wheel.
4. Run targeted package tests, inspect wheel contents and hash, complete Backlog ACs/notes, write the Task 2 report, and commit the exact implementation with the requested message.
ADR required: no — ADR-102 already governs this contract.
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Shared Profile Core v0.1.0 with immutable ADR-102 contract, deterministic canonical serialization/schema/fixtures, package-local tests, and portable root setuptools discovery/package data. Exact package tests pass 16/16; focused root discovery regression passes 1/1; root wheel artifact contains core modules/schema/fixtures per follow-up verification. External publication deferred per execution ruling; isolated build dependency download unavailable offline. See task-2-report.md for both failed hypotheses and final evidence.
<!-- SECTION:NOTES:END -->
