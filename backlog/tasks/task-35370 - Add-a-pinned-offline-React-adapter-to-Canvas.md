---
id: TASK-35370
title: Add a pinned offline React adapter to Canvas
status: To Do
created_date: 2026-10-18 04:39
labels:
- canvas
- roadmap
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users build useful stateful Canvas components with a bundled React adapter while preserving offline execution, immutable historical profiles, and the existing sandbox. Limit the first release to a documented compatible subset under ADR-121 and ADR-124; this task does not enable browser-native React execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approved ADR and bounded design identify the supported React and DOM subset, compatibility profile, resource limits, licensing and reproducible build before implementation.
- [ ] #2 A documented stateful interactive example renders and updates in native and served Canvas using only bundled assets and the mediated runtime, with no network or native browser globals.
- [ ] #3 New revisions use a distinct immutable profile; historical revisions and archives retain their exact profile and source-only refusal behavior.
- [ ] #4 Targeted functional, quota and adversarial browser checks verify the promised subset and existing isolation limits; user and architecture documentation describe unsupported features.
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
