# Phase 4.2 Skills Destination Service Adoption

Task: `TASK-5.2`
Branch: `codex/unified-shell-phase4-skills-destination-service`

## Goal

Turn the top-level Skills destination from a static Agent Skills placeholder into a local service-backed surface that lists installed `SKILL.md` packs and stages concrete skill context into Console.

## Implementation Summary

- Loaded local Agent Skills through `skills_scope_service.list_skills(mode="local")`.
- Rendered available, empty, loading, and service-unavailable states with stable selectors.
- Disabled `Attach local Skills to Console` until concrete local skill context exists.
- Built `ChatHandoffPayload` from actual listed skill names, descriptions, argument hints, record IDs, backend metadata, and the local skills directory.
- Preserved disabled import copy because import/detail/edit UX is still future destination work.

## Verification

- Baseline focused command before changes: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/Skills/test_skills_scope_service.py Tests/Skills/test_local_skills_service.py -q`
- Baseline focused result: `52 passed, 10 warnings in 35.54s`.
- Red command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py::test_skills_destination_lists_local_skills_from_scope_service Tests/UI/test_destination_shells.py::test_skills_destination_empty_state_disables_console_attach Tests/UI/test_destination_shells.py::test_skills_destination_service_failure_uses_recovery_copy Tests/UI/test_destination_shells.py::test_skills_attach_to_console_uses_listed_skill_context Tests/UI/test_destination_shells.py::test_skills_destination_service_adoption_tracking_evidence_exists -q`
- Red result: `5 failed, 1 warning in 7.39s`.
- First green behavior command after implementation: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py::test_skills_destination_lists_local_skills_from_scope_service Tests/UI/test_destination_shells.py::test_skills_destination_empty_state_disables_console_attach Tests/UI/test_destination_shells.py::test_skills_destination_service_failure_uses_recovery_copy Tests/UI/test_destination_shells.py::test_skills_attach_to_console_uses_listed_skill_context -q`
- First green behavior result: `4 passed, 1 warning in 7.00s`.
- Final focused command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/Skills/test_skills_scope_service.py Tests/Skills/test_local_skills_service.py -q`
- Final focused result: `106 passed, 8 warnings in 61.88s`.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: top navigation `Skills` destination.
- Visual check: Skills keeps the destination title, purpose copy, Agent Skills sections, local skills directory, and honest disabled import state.
- Available-state result: a local service response with two skills renders `Installed local skills: 2`, visible skill names, visible descriptions, and an enabled Console handoff.
- Empty-state result: an empty service response renders `No local Agent Skills are installed yet.` and disables Console handoff with install-oriented recovery copy.
- Service-error result: service exceptions render `Skills service unavailable; retry Skills later.` and disable Console handoff with retry-oriented recovery copy.
- Functional result: Console handoff stages `skills-context` with local skill names, descriptions, argument hints, record IDs, backend metadata, and no generic placeholder body.

## Residual Risk

- This slice adopts local list and context staging only; server skills, import, detail, edit, validation, and execution flows remain future Phase 4 work.
- The walkthrough uses focused mounted-window QA, not a full clean-HOME end-to-end app session.
- The handoff stages listed skill summaries, not full `SKILL.md`, script, reference, or asset file contents.
