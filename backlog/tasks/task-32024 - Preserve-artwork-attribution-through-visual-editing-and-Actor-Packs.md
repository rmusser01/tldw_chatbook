---
id: TASK-32024
title: Preserve artwork attribution through visual editing and Actor Packs
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 04:04'
updated_date: '2026-09-10 15:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve source artwork credits and complete notices when users edit and share characters, so Buddy conversion does not silently discard or replace third-party attribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Versioned bounded creator, license and notice data survives Actor Pack export, review, activation and database reopen without local identifiers.
- [x] #2 Unchanged expressions retain hash-bound attribution through edits and profile forks; replacement images lose stale attribution and existing licenses are never replaced with the builtin default.
- [x] #3 Tampering, invalid attribution, unsupported feature versions and oversized notice payloads fail before activation; legacy archives remain compatible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Docs/superpowers/plans/2026-09-07-artwork-attribution.md inline. ADR required: yes; amend ADR-074 for the optional checksummed attribution carrier. First validate and round-trip the contract, then preserve retained-image records and licenses through publication, then run targeted checks and document evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the versioned tldw/artwork namespace and checksummed Actor Pack carrier under ADR-074, including full plain-text notice review and output-hash validation. Retained images and profile forks preserve credits and original manifest terms; replacements drop stale records. Fixed imported-pack publication identity and long-notice stale-review hashing found by the real import/reopen/edit/export/reimport test. Final focused verification: 48 passed. Broader selection: 242 passed, 1 platform skip, 5 failures reproduced on unchanged merge-base d6d792ec; separate app import-closure environment failure also reproduced there. New code passes Ruff/format; existing changed files add no diagnostics. Evidence: Docs/superpowers/reviews/2026-09-07-artwork-attribution-verification.md. Conversion and Petdex remain separate scope.

2026-09-10 integration onto dev 16c72b5b1e: prior validation above is historical source-branch evidence; combined-code targeted qualification is in progress.

2026-09-10 integration complete under ADR-074. Reused dev native artwork validator and retained checksummed Actor Pack carrier/lineage; explicitly preserved public Petdex notice limit. Final affected Petdex/artwork/conversion selection: 206 passed. Native snapshot/roundtrip: 27 passed, 1 optional skip. Exact evidence and existing baseline architecture failure: Docs/superpowers/reviews/2026-09-10-buddy-feature-integration-verification.md. Prior results above are historical; no new diagnostics or full-suite claim.
<!-- SECTION:NOTES:END -->
