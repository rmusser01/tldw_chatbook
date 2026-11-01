---
id: TASK-35371
title: Add a pinned offline D3 subset to Canvas
status: To Do
created_date: 2026-10-18 04:41
labels:
- canvas
- roadmap
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Canvas authors build useful data visualizations using a small bundled D3 subset that works inside the existing virtual runtime. Preserve deterministic historical profiles and the current zero-egress security boundary rather than importing the full native-browser D3 API.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approved ADR and bounded design identify useful D3 modules, supported mediated DOM/SVG operations, immutable profile, dependency licenses and reproducible inputs before implementation.
- [ ] #2 At least one documented interactive chart renders and updates in native and served Canvas entirely from bundled resources.
- [ ] #3 Unsupported native-browser, network, module-loading and DOM operations fail closed within unchanged resource budgets.
- [ ] #4 Functional, historical-profile/archive and adversarial browser verification covers the supported subset, and public documentation names its limits.
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
