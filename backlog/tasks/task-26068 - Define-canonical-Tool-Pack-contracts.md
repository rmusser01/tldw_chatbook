---
id: TASK-26068
title: Define canonical Tool Pack contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:43'
updated_date: '2026-08-31 19:45'
labels:
  - tool-packs
  - contracts
  - security
  - serialization
dependencies:
  - TASK-26067
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide immutable, strictly validated portable Tool Pack contract objects, stable categorized errors, canonical JSON encoding/decoding, and destination-independent tool contract fingerprints for every later export/import stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict JSON decoding rejects duplicate keys, non-finite values, invalid UTF-8/surrogates, over-limit trees/strings, and non-NFC identities with stable error categories.
- [x] #2 Canonical JSON produces deterministic LF-terminated UTF-8 bytes with normalized strings, sorted object keys, and no non-finite values.
- [x] #3 Immutable manifest/profile/rule contracts reject missing, unknown, malformed, colliding, or out-of-grammar fields and states.
- [x] #4 Portable contract fingerprints include normalized tool identity, description, input schema, and deduplicated sorted policy-risk tags while differing from runtime definition hashes.
- [x] #5 Focused contract tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for strict JSON rejection, NFC/depth/node/string limits, exact-key validation, id/state grammar, and case-fold collisions.
2. Implement stable `ToolPackError` categories and immutable exact contract dataclasses.
3. Implement deterministic canonical JSON encoding and bounded strict object decoding.
4. Add portable contract fingerprint tests and implement the destination-independent digest preimage.
5. Run focused contract tests, scoped static checks, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: Accepted ADR-107 already defines the portable schema, strict validation, canonicalization, and fingerprint boundary implemented here.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict immutable Tool Pack contracts, canonical JSON admission/encoding, stable operation-aware errors, document integrity framing, and destination-independent portable fingerprints. Independent review required two fix rounds: optional Deny fingerprint omission, deep immutable constructors, stable error-table enforcement, mutation-sensitive canonical/document/fingerprint boundaries, and an isolated manifest payload-size integrity regression. Fresh verification: 82 focused tests passed; scoped Ruff and git diff/status passed; one pre-existing RequestsDependencyWarning remains. ADR: existing ADR-107; no new ADR required.
<!-- SECTION:NOTES:END -->
