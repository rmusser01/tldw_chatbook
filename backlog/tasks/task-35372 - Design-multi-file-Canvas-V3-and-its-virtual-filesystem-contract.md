---
id: TASK-35372
title: Design multi-file Canvas V3 and its virtual filesystem contract
status: To Do
created_date: 2026-10-18 04:42
labels:
- canvas
- roadmap
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a reviewable V3 design for Canvas projects containing index.html, CSS, JavaScript and passive assets, so multi-file authoring can be implemented without weakening revision atomicity, historical reproduction or sandbox confinement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A canonical ADR and written design specify file identity, normalized paths, entry points, imports and bundling, project-wide quotas and atomic revision/update/conflict semantics.
- [ ] #2 The design specifies complete project replacement, branch projection, Temporary staging/promotion, source inspection/download, conversation lifecycle and export/import compatibility.
- [ ] #3 Malformed paths, cycles, undeclared resources, active assets and excessive projects have explicit failure outcomes; no host filesystem or network privilege is implied by the in-memory filesystem.
- [ ] #4 A compatibility matrix and concrete valid/invalid project fixtures define subsequent implementation acceptance, and the design is reviewed and approved before V3 implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
