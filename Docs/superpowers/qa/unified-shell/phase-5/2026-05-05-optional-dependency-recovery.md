# Phase 5.4 Optional-Dependency Recovery

Date: 2026-05-05
Task: `TASK-6.4`
Branch: `codex/unified-shell-phase5-optional-dependency-recovery`
Status: implemented; final Phase 5 running-app closeout still required

## Goal

Apply the Phase 5 recovery taxonomy to visible optional-dependency blockers so users can understand missing local extras, install or configure the missing capability, and avoid mistaking disabled states for broken controls.

## Applied Blockers

| Surface | Blocked workflow | Canonical state | Stable selector | Recovery |
| --- | --- | --- | --- | --- |
| Search/RAG | Search/RAG queries when embeddings extras are missing | `dependency_missing` | `#search-rag-dependency-missing` | Install `embeddings_rag` extras and restart, then use Settings > RAG if configuration is still needed. |
| Local speech | Local TTS/STT providers when local speech extras are missing | `dependency_missing` | `#speech-capability-status` | Install local TTS, transcription, and speech-recording extras, restart, then use Settings > Speech. |

## UX Result

- Search/RAG no longer relies only on a transient alert and disabled search button. The results area now shows a persistent recovery state with `Unavailable`, `Why`, `Next`, `Recovery`, and `Owner` fields.
- The disabled Search/RAG input and Search button carry the same recovery tooltip so keyboard and mouse users can diagnose the blocked state without reading logs.
- STTS local speech status now uses the same taxonomy fields instead of the terse `Local speech missing: TTS, STT` label.
- The local speech status tooltip names the install command and restart requirement.

## Verification

- Red regression before implementation: `python -m pytest Tests/UI/test_disabled_action_recovery_tooltips.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q -k "optional_dependency or missing_embeddings or missing_speech"`
- Initial red result: collection failed because `optional_dependency_recovery_state` did not exist.
- Focused green verification after implementation: `python -m pytest Tests/UI/test_disabled_action_recovery_tooltips.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q -k "missing_embeddings or missing_speech or optional_dependency_recovery_helper"`
- Focused green result: `3 passed`
- Affected UI and tracking verification: `python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_disabled_action_recovery_tooltips.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py -q`
- Affected verification result: `92 passed`

## Regression Coverage

- `test_search_rag_missing_embeddings_dependency_exposes_phase_five_recovery`
- `test_stts_missing_speech_dependencies_expose_phase_five_recovery`
- `test_phase_five_optional_dependency_recovery_helper_builds_required_fields`

## Residual Risk

- This slice verifies representative optional-dependency blocker families, not every optional extra in the app.
- Live install/restart behavior is not exercised in automated tests.
- Final Phase 5 running-app closeout QA remains required before marking `TASK-6` verified.

## What Remains

`TASK-6.4` covers the optional-dependency blocker family targeted for Phase 5. Parent `TASK-6` remains In Progress until final running-app closeout QA verifies recovery paths in the actual app.
