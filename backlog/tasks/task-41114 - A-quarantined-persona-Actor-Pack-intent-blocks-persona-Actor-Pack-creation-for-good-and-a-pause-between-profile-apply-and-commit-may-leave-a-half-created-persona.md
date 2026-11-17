---
id: TASK-41114
title: "A quarantined persona Actor Pack intent blocks persona Actor Pack creation for good, and a pause between profile apply and commit may leave a half-created persona"
status: Done
assignee: ['@claude']
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 02:00
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
- [x] #1 A user with a quarantined intent sees why persona Actor Pack creation is blocked and has an in-app way to resolve it; the notes name the flow
- [x] #2 A storage pause between applying the persona plan and the commit leaves no half-created persona, shown by a test red before the fix (or the notes show it cannot happen)
<!-- AC:END -->

## Implementation Plan

1. AC#2 first: a test next to TASK-40304's activation tests that activates a persona pack on a worker with the app's bound persona service (`build_persona_service`), holds it right after the persona plan is applied, begins a local pause on the test thread, then lets it continue. Record what a capture would copy (the persona store, identity row, intent row, published assets) and the state after resume. Expect red at base: the commit and the compensation are both refused by the pause.
2. Fix: hold the persona store and ChaChaNotes admission (the persona service's paired chat-source operation) from the persona plan through the commit and any compensation, so a pause that begins inside the window waits for it instead of splitting the two-store change. Keep the post-commit intent cleanup outside the window (TASK-40304 C1 behaviour unchanged).
3. AC#1 (in-app resolution of a quarantined intent): ADR-074 says a quarantined intent "requires explicit recovery" but no recovery action is designed. Which resolution to offer is a product choice; report NEEDS_CONTEXT with options rather than choosing one.

## Implementation Notes

**AC#2 (done).** Reproduced first. `test_actor_pack_pause_between_persona_apply_and_commit_leaves_no_half_persona` (`Tests/Architecture/test_backup_owner_inventory.py`) activates a persona pack on a worker with the app's bound persona service (`build_persona_service`; the helper gained `bound=True`), holds it right after the persona plan is applied, begins a local pause, then lets it go and drains the way runtime maintenance does. At base (cc6006fcf5) the commit was refused (`storage_locally_paused`), the compensation was refused too, and the capture would copy the persona with no identity, no artwork and a `prepared` intent: red with `AssertionError: a capture copied a half-created persona` (`(['local-persona-…'], []) == ([], [])`).

Fix: `PersonaActorPackCoordinator.create_persona` now holds the persona service's paired admission (persona file + ChaChaNotes, the chat-source `operation()` the service's guarded methods already use; exposed as `LocalCharacterPersonaService._actor_pack_mutation_scope`) from the plan through the commit or its compensation. The body moved unchanged into `_create_persona_admitted`. Guarded calls inside reuse the held admission, so a pause that begins inside waits for it, and the compensation always runs. The commit itself still needs the shared visual tree's own admission (`visual_identity_participants.request`), which a pause refuses by design (TASK-40304 refuses an activation a pause interrupts), so the pinned outcome is: activation refused with `actor_pack_import_storage_unavailable`, `cleanup_pending=False`, and nothing left in the persona store, the identity table, the intent table or either tree, both in the copy and after resume. Green after.

Decisions:
- A failure is relayed after the scope closes instead of through it: the chat-source operation marks the service `chat_publication_incomplete` (which blocks later drains) when an exception leaves it after DB changes, and the coordinator has already reconciled both stores by then. An uncertain raw write still marks it, through the scope's own `raw_resources_not_retired` check at normal exit.
- The post-commit intent cleanup stays outside the scope, so TASK-40304's C1 behaviour (pause after commit -> `cleanup_pending=True`, assets kept) is unchanged; its test still passes.

**AC#1, first pass (NEEDS_CONTEXT, superseded by the owner decision below).** ADR-074 and the Actor Pack design say a quarantined intent "requires explicit recovery", but no recovery action is designed, and the task does not say which one the user gets. Options for the owner:
- A. "Keep my current personas": a Personas-screen notice names the blocked state and offers one action that deletes the quarantined intent rows and clears the block. Personas and identity rows stay as they are; a persona whose identity never committed stays an ordinary local persona. Non-destructive.
- B. "Undo the interrupted import": put back the intent's pre-import profile (or remove the created persona) and its identity rows. It overwrites any edit the user made since, which is what caused the quarantine. Destructive.
- C. Offer both, per intent, naming the persona and what each choice changes.
Today the user only sees the generic "activation failed" copy (`actor_pack_creation_blocked` maps to `actor_pack_import_activation_failed`).

Runs (load 34-81, two capture-campaign lanes running alongside):
- Architecture file `-k actor_pack`, serial: 10 passed.
- `Tests/Actor_Packs -n 4`: 239 passed, 2 failed (the export golden-byte pair, red at base, see TASK-41110).
- `Tests/Character_Chat/*persona*.py -n 4`: 127 passed.
- UI consumers (`test_actor_pack_import_workflow.py`, `test_roleplay_import_app.py`, `test_actor_pack_creation_workflow.py`) `-n 4`: 55 passed, 17 failed, all 17 in `test_actor_pack_creation_workflow.py` with `RecoveryRequired: raw_source_selection_changed` (the known local trap; identical at base per TASK-40304's notes).
- `ruff_gate.py`: exit 0.

Files: `tldw_chatbook/Actor_Packs/persona_coordinator.py`, `tldw_chatbook/Character_Chat/local_character_persona_service.py`, `Tests/Architecture/test_backup_owner_inventory.py`.

### Fix round 1: AC#1 (owner decision)

**Owner decision for TASK-41114 AC#1 (2026-11-16): "Clear the interrupted import (Recommended)".** A quarantined persona Actor Pack intent shows a notice in Personas: "keep my current personas and clear the interrupted import". It deletes only the stuck intent rows; personas stay exactly as they are, and new imports work again.

The flow: open Personas. When startup recovery left a quarantined intent, a notice under the purpose line says "An earlier persona Actor Pack import was interrupted and is still recorded, so persona Actor Packs can't be imported. Your personas are not affected." Its button **Keep my current personas and clear the interrupted import** (Tab, then Enter) clears it. A refused import shows the same notice and says "An earlier persona Actor Pack import was interrupted. Choose "Keep my current personas and clear the interrupted import" above, then import again." instead of "Actor Pack import failed.".

- `PersonaActorPackCoordinator.clear_blocked_intents()` deletes each blocked intent with `cleanup_persona_intent(..., expected_state="quarantined")`, each in its own ChaChaNotes transaction (its repository operation is the admission). A storage pause refuses it as `actor_pack_storage_unavailable` and the row stays blocked. It touches no persona record.
- Activation maps `actor_pack_creation_blocked` to the new `actor_pack_import_creation_blocked` instead of `actor_pack_import_activation_failed`; `personas_screen._actor_pack_import_error_copy` gives its copy.
- `Widgets/Persona_Widgets/actor_pack_recovery_notice.py`: the notice (`Static(markup=False)`, a constant Button label, `notify(..., markup=False)`) and `show_actor_pack_recovery_notice(screen)`. The clear runs in `asyncio.to_thread(run_finite_local_worker, ...)` from the notice's own worker. On success it resets the app's `actor_pack_recovery_error` and removes the notice, without awaiting the removal from inside the notice's own worker; awaiting it there deadlocked (`WaitForScreenTimeout`). Personas calls it after `ensure_actor_pack_recovery` and after a refused activation (+7 screen lines; `test_module_size_ratchet.py` row 16,882 -> 16,889 by the owner rule "raise by the change's own lines").
- Docs: `Docs/User_Guide/roleplay-chat-dictionaries.md` (the import paragraph) and a dated note in ADR-074 under "requires explicit recovery".

Tests (`Tests/Architecture/test_backup_owner_inventory.py`, on the PR "Architecture guard contracts" lane):
- `test_interrupted_persona_import_blocks_until_cleared_and_keeps_personas`: the stuck state is built the real way (prepared intent, the user creates a persona, startup recovery quarantines it). Red at base: `'actor_pack_import_activation_failed' == 'actor_pack_import_creation_blocked'`. Green: blocked with the named copy, clear returns 1, persona store bytes and the persona list unchanged, then the same import succeeds.
- `test_clearing_the_interrupted_import_refuses_during_a_storage_pause`: refused with `actor_pack_storage_unavailable`, the row still quarantined, clears after resume. Control: deleting through a raw connection (no admission) fails it with `DID NOT RAISE`.
- `test_interrupted_import_notice_clears_from_the_keyboard_off_the_ui_thread[None|actor_pack_storage_unavailable]`: the notice mounts once under the purpose line, is reached with Tab and run with Enter, clears off the main thread, every notify is `markup=False`, and a refusal keeps the notice with the button enabled. Control: calling the clear on the UI thread fails `assert MainThread not in [MainThread]`.
- `test_personas_shows_the_interrupted_import_notice_after_recovery` (real `PersonasScreen` mount, `bootstrap_profile` per-node opt-in as TASK-32873 does): red with the base screen (`'personas-actor-pack-recovery-notice' in ['personas-header', 'personas-purpose', 'personas-mode-strip', 'personas-workbench']` fails), green after.

Runs (load 55-67): the inventory file serially with `--timeout=1500`: 54 passed. Architecture guards (`test_worker_exclusive_group_inventory`, the three boundary files, the module size ratchet, `test_no_undefined_module_globals`), serially: 256 passed. `Tests/Actor_Packs` + persona Character_Chat tests + `Tests/UI/test_actor_pack_recovery_seam.py` + `test_actor_pack_import_workflow.py`, `-n 4`: 380 passed, 2 failed (the export golden-byte pair, red at base). `check_markup_interpolation.py`: OK.

Files: `tldw_chatbook/Actor_Packs/persona_coordinator.py`, `tldw_chatbook/Actor_Packs/activation.py`, `tldw_chatbook/Widgets/Persona_Widgets/actor_pack_recovery_notice.py` (new), `tldw_chatbook/UI/Screens/personas_screen.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/roleplay-chat-dictionaries.md`, `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`.
