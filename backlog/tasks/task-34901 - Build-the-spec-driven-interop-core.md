---
id: TASK-34901
title: Build the spec-driven interop core
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
47 scope services (34k LOC) and 47 server services (~19k) hand-instantiate one scaffold: Backend enum _normalize_mode _require_service _enforce_policy (byte-identical in 44 of 49 definitions) _maybe_await and unsupported-capability constants. Kanban_Interop already proves the general form (operation-spec table plus __getattr__ dispatch). Each domain should reduce to a spec file plus genuinely domain-specific logic.

Evidence (verified on dev df4985e): Immediate ADR-177 tranches: 96 pure helper defs (~1k lines). app_service_wiring.py carries 31 except-ValueError blocks and 46 scope constructions. Kanban reference: kanban_scope_service.py:43+56-65+127-149. Blockers: ADR-177 rejects a base class and defers _normalize_mode/_enforce_policy consolidation until mode-default and fail-direction are decided (a wrong fail-direction is a security regression) so this task starts with that decision; 67 Interop test files (~39k LOC) pin behavior; Sync_Interop machinery (crypto envelopes outbox ~15.5k LOC) is NOT template and stays.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-177 constraints honored (helpers not base class and fail-direction decided before policy consolidation),Operation-spec table drives scope routing server facade and capability lists,Pilot domains migrated with their test suites green,Per-domain mechanical code reduced to one spec file,app_service_wiring per-domain try-except blocks collapse accordingly
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
