---
id: TASK-23194
title: Build Shared Profile Core v0.1 package
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29 06:24'
updated_date: '2026-08-29 13:21'
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
- [ ] #2 Public models, enums, payloads, interview and tool contracts enforce the approved ADR-102 validation rules
- [ ] #3 Canonical serialization, integrity tags, JSON Schema, and v1 positive/negative fixtures are deterministic and conformant
- [ ] #4 Targeted tests and local wheel inspection pass, with no database, HTTP, provider, UI, or key-custody modules included
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
Review correction applied: exact V1 record/proposal/tool vocabularies, strict proposal and agent request shapes, explicit result status, regenerated schema, and package tests updated. Exact package-root tests pass 13/13; root packaging regression passes; diff check passes. Root wheel artifact was verified to contain core modules/schema/fixtures. See task-2-report.md for RED/GREEN and artifact evidence.

Correction loop 1 was incomplete: it changed enums and shallow request/result shapes but did not implement metadata bounds, content-free tombstones, UTF-8 byte sizing, resolvable schema/full fixtures, interview hardening, or comprehensive TDD. Reopened for correction loop 2.

Correction loop 3 completed the unapproved contract surface without closing the task. Shared Core now enforces bounded nonblank opaque IDs and payloads, immutable provenance and interview collections, ordered aware timestamps, content-free tombstones, exact proposal and agent-request shapes, UTF-8 canonical payload sizing, scoped tool authority, and conservative interview compound/secret rejection. Schema export now has one root `$defs`; eight full model-dispatched fixtures are byte-identical across package-root/source resources and validate consistently through Pydantic and `jsonschema`. TDD evidence: the Python 3.12 package suite first exposed 31 contract failures after collection was unblocked, then passed 56/56. Root packaging regression passed 1/1. Standalone/root wheels imported from the artifacts and contained only the expected core modules plus the 21,220-byte schema and eight fixtures. AC 2–4 remain unchecked for independent review.
<!-- SECTION:NOTES:END -->
