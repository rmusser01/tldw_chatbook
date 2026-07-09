---
id: TASK-137
title: Add local skill trust integrity controls
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-25 21:03'
labels:
  - skills
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect Chatbook-managed local skills from offline file tampering by adding explicit trust bootstrap, authenticated baseline verification, logical quarantine, reviewed re-trust, and visible recovery states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local skill trust bootstrap blocks uninitialized skills until the user explicitly trusts the current baseline.
- [x] #2 Authenticated manifest, snapshots, and generation marker verification detect modified, added, deleted, unsupported, replayed, or unverifiable skill files.
- [x] #3 Trust-blocked skills remain visible but cannot be staged in context or executed.
- [x] #4 Review approval updates the trusted baseline only after a captured diff still matches live files.
- [x] #5 Secure keyring convenience mode never stores the user passphrase and reduced rollback protection is explicit.
- [x] #6 Skills UI and Settings surface trust posture, quarantine reasons, and recovery actions without leaking secrets or absolute paths by default.
- [x] #7 Focused service, trust-layer, policy, and UI tests cover the new trust boundary.
- [x] #8 ADR and Superpowers implementation plan are linked from the task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/009-local-skill-trust-boundary.md
Reason: Local skill trust changes a security boundary, trust root, local storage semantics, runtime policy IDs, and the LocalSkillsService contract.

1. Register local skill trust runtime-policy actions.
2. Add trust models, crypto, scanner, store, and service substrate.
3. Integrate trust checks into LocalSkillsService and app wiring.
4. Add Skills and Settings trust posture UI.
5. Run focused trust, Skills, runtime-policy, and UI tests.

Superpowers plan: Docs/superpowers/plans/2026-06-25-local-skill-trust-integrity.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented local skill trust integrity for Chatbook-managed local skills. Added ADR-009-backed trust modules for passphrase-derived keys, authenticated manifests, encrypted trusted snapshots, generation markers, directory scanning, logical quarantine, reviewed re-trust, and deterministic trust-blocked service behavior. Wired trust state into LocalSkillsService, app construction, Skills UI, Settings Privacy posture, and runtime-policy action IDs. Focused trust, Skills, runtime-policy, and UI tests pass. ADR path: backlog/decisions/009-local-skill-trust-boundary.md. Superpowers plan: Docs/superpowers/plans/2026-06-25-local-skill-trust-integrity.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Local skill trust controls are implemented and wired through the trust layer, service integration, app construction, Skills recovery UI, Settings posture, and policy registry. Verification completed with focused trust/service, runtime-policy, Skills UI, Settings/navigation tests, plus diff hygiene.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
