---
id: TASK-23194
title: Build Shared Profile Core v0.1 package
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 06:24'
updated_date: '2026-08-29 17:01'
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
Implemented Shared Profile Core v0.1 as an independently buildable package governed by ADR-102. It now provides immutable canonical models, exact enums and lifecycle rules, interview and agent-tool contracts, RFC 8785 canonical bytes with UTC-millisecond timestamps and versioned HMAC integrity tags, a required custom Draft 2020-12 semantic vocabulary/reference validator, and 45 shared positive/negative conformance fixtures. Python/schema wire validation is aligned for timestamps, identities, proposal links, payload byte limits, numeric/boolean/version coercion, and confidence bounds; interview and agent boundaries reject recognized secrets while allowing benign token-budget language. Root and standalone distributions pin rfc8785==0.1.4 and package both schemas plus all fixtures without storage, HTTP, provider, UI, or key-custody modules. Final evidence: 151 targeted package tests passed, 2 root packaging tests passed, Ruff check and format-check passed, full Task 2 diff-check passed, a fresh standalone v0.1.0 wheel built successfully and contained the expected modules/resources/dependency metadata, and independent final spec and code-quality reviews approved the complete diff. ADR required: no new ADR; ADR-102 was amended to record the versioned semantic vocabulary and RFC 8785/UTC-millisecond canonical contract. No full repository sweep was run, per repository policy.
<!-- SECTION:NOTES:END -->
