---
id: TASK-36942
title: 'Work stream: architecture guardrails that stop silent structural decay'
status: Done
created_date: 2026-10-26 01:06
updated_date: '2026-11-20 14:55'
labels:
- architecture-review-2026-10-25
- architecture
- tests
priority: high
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Import-time cycles are nearly gone, but function-level imports hide a 1,217-module dependency cycle, nothing checks layer direction globally, 40 modules of 4,000+ lines have no size ratchet row, the 'pure' agent runtime loads Textual and the DB at import, and string lookups on the app object grow unchecked. These guards are cheap and make every later structural task measurable.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F1, F5, F6, F16; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Layer direction, app string lookups, module size and the agent runtime's import footprint are each enforced by an architecture test that runs in the existing Architecture lane
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-20 on dev `d85e90bc9e`. AC #1: the four guards are `Tests/Architecture/test_layer_direction.py` (layer direction), `test_app_string_lookup_census.py` (app string lookups), `test_module_size_ratchet.py` (module size) and `test_agent_runtime_import_footprint.py` (the agent runtime's import footprint). `derived-artifacts.yml` runs the size ratchet in `pr-fast-lane` ("PR Fast Lane") and the other three in `pr-fast-lane-arch` ("PR Fast Lane (architecture-review tests)"). No job was named "Architecture lane" when this task was filed, so the AC is read as "a PR lane"; the later `architecture-guards` job ("Architecture guard contracts", TASK-38989) runs none of the four. All five guard files (these four plus `test_screen_size_ratchet.py`) passed serially on dev `d85e90bc9e` through the scratch wrapper: 314 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
