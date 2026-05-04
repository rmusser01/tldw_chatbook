# Phase 4.4 Personas Service Adoption

Task: `TASK-5.4`
Branch: `codex/unified-shell-phase4-personas-service`

## Goal

Turn the top-level Personas destination from static legacy-route links plus generic Console copy into a service-backed behavior snapshot that tells users whether local characters and persona profiles are available and stages concrete behavior context into Console.

## Implementation Summary

- Loaded local behavior context through `character_persona_scope_service.list_characters(mode="local")` and `character_persona_scope_service.list_persona_profiles(mode="local")`.
- Rendered loading, available, empty, service-unavailable, and policy-denied recovery states with stable selectors.
- Preserved the existing `Open Personas` route to the character/persona management surface.
- Disabled `Attach to Console` until concrete local character or persona profile context exists.
- Built `ChatHandoffPayload` from actual local character/profile counts, sample names, and descriptions instead of the previous generic Personas placeholder.

## Verification

- Baseline focused command before changes: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py Tests/Character_Chat/test_character_persona_scope_service.py -q`
- Baseline focused result: `157 passed, 10 warnings in 95.87s`.
- Red command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py -q`
- Red result: `5 failed, 53 passed, 1 warning in 23.50s`.
- First green behavior command after implementation: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py -q`
- First green behavior result: `1 failed, 57 passed, 1 warning in 21.79s`; only the tracking evidence file was still missing.
- Final focused command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py Tests/Character_Chat/test_character_persona_scope_service.py -q`
- Final focused result: `161 passed, 8 warnings in 92.93s`.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: top navigation `Personas` destination.
- Visual check: Personas keeps the destination title, ownership copy, boundary copy, and existing profile-management route while adding a `Local Personas snapshot` section.
- Available-state result: local service responses render character and persona profile counts plus visible sample names and enable `Attach to Console`.
- Empty-state result: empty local service responses render `No local characters or persona profiles are available yet.` and disable Console handoff with add-profile recovery copy.
- Service-error result: scope-service exceptions render `Personas service unavailable; retry Personas later.` and disable Console handoff with retry-oriented recovery copy.
- Functional result: Console handoff stages `personas-context` with local character/profile counts, sample names, descriptions, runtime/source ownership metadata, and no generic placeholder body.

## Residual Risk

- This slice adopts local list and context staging only; persona detail, edit, import/export, archetypes, exemplars, dictionaries, and lore remain future Personas destination work.
- The walkthrough uses focused mounted-window QA, not a full clean-HOME running app session.
- Console handoff stages listed behavior summaries, not full character cards, lorebooks, prompt dictionaries, or exemplar corpora.
