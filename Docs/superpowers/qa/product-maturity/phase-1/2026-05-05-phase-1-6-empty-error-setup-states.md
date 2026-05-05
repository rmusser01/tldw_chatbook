# Phase 1.6 Empty/Error/Setup State Coverage

Date: 2026-05-05
Task: TASK-8.6
Branch: codex/product-maturity-phase1-6-empty-setup
Workflow: Clean-run empty/error/setup states

## What Was Verified

Phase 1.6 verifies that clean-run setup, empty, unavailable, and blocked states expose honest recovery paths before Phase 1 closes.

Verified states:

- Home missing provider/model setup: clean-run Home shows `Model: Blocked`, `Set up Console model`, and explains that Console needs a working model before live AI tasks.
- Console missing provider/API key setup: clean-run Console empty state explains OpenAI is not ready because the API key is missing and points to `api_settings.openai`.
- Search/RAG optional dependency setup: missing `embeddings_rag` disables Search/RAG input and search action, explains the missing optional dependency, owner, install command, and recovery target.
- ACP runtime not configured: ACP launch is disabled, shows `runtime not configured`, explains no ACP-compatible runtime is configured, and points to Settings.
- Library service unavailable: Library source handoff is disabled and states that Library source services are unavailable.
- Personas service unavailable: Persona attach-to-Console is disabled and states that Personas services are unavailable.
- W+C service unavailable: W+C Console staging is disabled and states that W+C services are unavailable.
- Skills service unavailable: Skills attach-to-Console is disabled and states that Skills services are unavailable.

## Automated Evidence

Focused regression:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_empty_setup_states.py -q
```

Initial red run:

- Runtime empty/error/setup checks passed after aligning the clean-run Console expectation to the app's actual default OpenAI missing API key state.
- Evidence and tracker assertions failed because Phase 1.6 evidence and closeout links did not exist yet.

Final focused verification after PR review fixes:

- `Tests/UI/test_product_maturity_phase1_empty_setup_states.py -q`: 8 passed, 1 warning in 8.35s.
- Phase 1 product-maturity sweep including Phase 1.1-1.6 tests: 40 passed, 1 warning in 20.16s.
- Adjacent recovery regressions for Search/RAG optional dependency and Library/Personas/W+C/Skills service unavailable states: 5 passed, 1 warning in 5.70s.

## Manual Walkthrough

The automated Textual pilot exercised the running app from a clean HOME/XDG environment and navigated Home -> Console -> ACP. Widget-hosted Search/RAG and destination harness checks exercised optional dependency and service-unavailable states deterministically.

No local filesystem paths are recorded in this evidence. Temporary clean-run paths are represented only as clean HOME/XDG setup in the tests.

## Visual/Focus Notes

- Disabled actions remained visible and disabled rather than silently disappearing.
- Recovery copy used stable fields where available: status, unavailable workflow, why, next action, recovery target, and owner.
- Tooltips on disabled actions repeated the recovery action or setup requirement.

## Defects Found

No P0/P1 empty/error/setup blockers were found in this Phase 1.6 sweep.

Observed residual risks:

- Some service-unavailable copy is concise rather than fully taxonomy-expanded on every destination.
- This gate does not prove a complete core product loop.

## Residual Risk

Remaining Phase 1 gate: narrow core-loop proof. Phase 1.6 does not verify source/question -> Console -> saved/reopened output.

## Exit Decision

Pass for Phase 1.6. Empty/error/setup states checked in this slice are visible, disabled actions avoid false affordances, and recovery paths are explicit enough to continue to the Phase 1 core-loop proof gate.
