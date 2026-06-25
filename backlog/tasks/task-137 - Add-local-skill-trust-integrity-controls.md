---
id: TASK-137
title: Add local skill trust integrity controls
status: In Progress
labels:
- skills
- security
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect Chatbook-managed local skills from offline file tampering by adding explicit trust bootstrap, authenticated baseline verification, logical quarantine, reviewed re-trust, and visible recovery states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local skill trust bootstrap blocks uninitialized skills until the user explicitly trusts the current baseline.
- [ ] #2 Authenticated manifest, snapshots, and generation marker verification detect modified, added, deleted, unsupported, replayed, or unverifiable skill files.
- [ ] #3 Trust-blocked skills remain visible but cannot be staged in context or executed.
- [ ] #4 Review approval updates the trusted baseline only after a captured diff still matches live files.
- [ ] #5 Secure keyring convenience mode never stores the user passphrase and reduced rollback protection is explicit.
- [ ] #6 Skills UI and Settings surface trust posture, quarantine reasons, and recovery actions without leaking secrets or absolute paths by default.
- [ ] #7 Focused service, trust-layer, policy, and UI tests cover the new trust boundary.
- [ ] #8 ADR and Superpowers implementation plan are linked from the task.
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
