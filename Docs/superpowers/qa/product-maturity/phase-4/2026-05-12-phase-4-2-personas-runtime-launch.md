# Phase 4.2 Personas Runtime Launch

Status: verified for local Personas selection and Console handoff readiness

## Scope

TASK-11.2 verifies that the Personas destination exposes local character/persona options, reports whether the selected behavior target can be attached to Console, and emits stable Console handoff metadata from the existing character/persona scope service.

## UX Evidence

- Actual running-app empty state screenshot: `Docs/superpowers/qa/product-maturity/phase-4/personas-empty-2026-05-12.png`
- Actual textual-web fixture screenshot with selectable records: `Docs/superpowers/qa/product-maturity/phase-4/personas-selected-2026-05-12.png`
- The selected-state screenshot shows a local character, local persona profile, `Use` controls, `Selected: Research Mentor`, `Runtime target: local:character:1`, and enabled `Attach to Console`.

## Regression Evidence

```bash
python -m pytest -q Tests/UI/test_destination_shells.py Tests/Character_Chat/test_character_persona_scope_service.py --tb=short
```

Result: passed after making the unrelated Settings boundary assertion case-insensitive.

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase4_agent_execution_plan.py --tb=short
```

Result: passed.

## Findings

- Personas now defaults to the first local character when behavior context exists and no policy error is present.
- Users can switch to a persona profile target without changing the underlying registry model.
- Console handoff payload metadata includes selected kind, name, record id, and runtime target id.
- Policy-denied states now report `Console: blocked` and surface the denial reason instead of implying readiness.

## Residual Risk

- This slice does not implement a new character/persona runtime registry.
- Server Personas parity and deeper character runtime launch remain later Phase 4 or Phase 5 work.
- The selected-state screenshot uses a deterministic textual-web fixture because the clean local profile used for running-app QA had no saved characters/persona profiles.
