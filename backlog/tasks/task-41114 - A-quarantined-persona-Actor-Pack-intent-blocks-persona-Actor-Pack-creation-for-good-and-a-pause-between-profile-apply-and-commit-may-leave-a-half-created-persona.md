---
id: TASK-41114
title: "A quarantined persona Actor Pack intent blocks persona Actor Pack creation for good, and a pause between profile apply and commit may leave a half-created persona"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- actor-packs
- personas
priority: medium
references:
- tldw_chatbook/Actor_Packs/persona_coordinator.py
- tldw_chatbook/app_service_wiring.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** After one interrupted creation and a persona edit, a user can never create a persona Actor Pack again and is not told how to recover.

**Evidence.**
- `persona_coordinator.py:100-102`: any `_blocked_intent_ids` refuses `actor_pack_creation_blocked`. Startup `recover()` (`:225-`) finishes a committed intent only while the persona store is unchanged; otherwise it quarantines it (TASK-40304 review r14act-rev2).
- Nothing clears a quarantined intent: `app_service_wiring.py:1672-1678` only sets `actor_pack_recovery_error = "actor_pack_recovery_blocked"` and logs it.
- `create_persona` (`:145-170`) applies the persona plan, then commits. A failure in between compensates (`_compensate_failed_creation`), but the compensation's except list (`:335-350`) does not include the storage pause's `RecoveryRequired`, so a pause there may leave the applied profile without its committed intent until the next start (reported by the round-14 Actor Pack review, not run).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user with a quarantined intent sees why persona Actor Pack creation is blocked and has an in-app way to resolve it; the notes name the flow
- [ ] #2 A storage pause between applying the persona plan and the commit leaves no half-created persona, shown by a test red before the fix (or the notes show it cannot happen)
<!-- AC:END -->
