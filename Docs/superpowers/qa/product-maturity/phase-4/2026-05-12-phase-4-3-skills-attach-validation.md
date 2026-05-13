# Phase 4.3 Skills Attach Validation

Status: verified for local Agent Skills validation and Console handoff readiness

## Scope

TASK-11.3 verifies that the Skills destination distinguishes valid local Agent Skills from invalid `SKILL.md` metadata, keeps local skill files unchanged, and stages only the selected valid skill into Console.

## UX Evidence

- Actual textual-web fixture screenshot with valid and invalid skills: `Docs/superpowers/qa/product-maturity/phase-4/skills-valid-invalid-2026-05-12.png`
- User approval: approved.
- The approved screenshot shows the Skills top-level destination, three clearly separated columns, `Installed local skills: 2`, `Blocked: invalid SKILL.md`, validation errors for `broken-skill`, `Ready: valid SKILL.md` for `summarize-notes`, selected Console target metadata, and enabled `Attach local Skills to Console`.

## Regression Evidence

```bash
python -m pytest -q Tests/Skills/test_local_skills_service.py::test_local_skills_service_validates_agent_skill_metadata_contract Tests/Skills/test_local_skills_service.py::test_local_skills_service_reports_invalid_agent_skill_metadata_without_mutating_content Tests/UI/test_destination_shells.py::test_skills_destination_distinguishes_valid_and_invalid_skill_readiness Tests/UI/test_destination_shells.py::test_skills_attach_to_console_uses_selected_valid_skill_context_only --tb=short
```

Result: failed before implementation because Agent Skills validation metadata and selected-skill attach behavior were missing; passed after implementation.

```bash
python -m pytest -q Tests/UI/test_destination_shells.py Tests/Skills/test_local_skills_service.py Tests/Skills/test_skills_scope_service.py --tb=short
```

Result: passed.

## Findings

- Local Skills now parse Agent Skills YAML frontmatter fields, including `name`, `description`, `allowed-tools`, `license`, `compatibility`, and `metadata`.
- Validation reports `valid` or `invalid` without rejecting or mutating the skill content.
- The Skills screen defaults to the first valid local skill, exposes per-skill `Use` controls, and disables Console attach when an invalid skill is selected.
- Console handoff now stages only the selected valid skill and includes selected target metadata for downstream Console state.

## Residual Risk

- This slice does not implement skill import wiring, script execution, or a full local execution sandbox.
- Server Skills parity and sync remain Phase 5 work.
- The screenshot uses a deterministic textual-web fixture because this slice needs visible valid and invalid local skill records in the same approval capture.
