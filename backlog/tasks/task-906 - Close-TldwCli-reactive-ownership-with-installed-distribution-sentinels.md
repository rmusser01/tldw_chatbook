---
id: TASK-906
title: Close TldwCli reactive ownership with installed-distribution sentinels
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-28 01:21'
labels:
  - architecture
  - state
  - packaging
  - verification
dependencies:
  - TASK-647
  - TASK-648
  - TASK-649
  - TASK-650
  - TASK-651
  - TASK-652
  - TASK-904
  - TASK-905
references:
  - backlog/decisions/032-immutable-installed-distribution-assets.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enforce the exact remaining root reactive contract and prove the decomposed production application from a clean installed artifact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TldwCli retains exactly current_tab and splash_screen_active from the reviewed 61-descriptor inventory, with no source or dynamic access to any of the 59 removed names.
- [x] #2 Every relevant registered production route executes without access to a removed root owner.
- [x] #3 Focused source ownership, static, compile, Ruff, formatting, and diff hygiene checks pass.
- [x] #4 A wheel and sdist built from the repository install into a clean environment outside the checkout, import only from the installed artifact, and pass resource, product-maturity, and reactive-ownership sentinels.
- [x] #5 The authorized integrated suite passes without collecting surrogate-application tests, and any excluded legacy collections are explicitly documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md; backlog/decisions/033-application-session-state-ownership.md
Reason: ADR-033 defines the final root owners and ADR-032 requires clean installed-artifact proof.

1. Enforce the exact TldwCli reactive set.
2. Run every affected registered route in the production app.
3. Extend installed-wheel ownership and maturity probes.
4. Run the authorized integrated gate and reconcile TASK-647–652 and TASK-904–906.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed the reactive-ownership tranche on latest `dev` commit `6784c4ba3`.
`TldwCli` now has no obsolete `watch_current_tab` method and retains exactly
the two reviewed application-lifecycle reactives: `current_tab` and
`splash_screen_active`.

Added exact source and installed-artifact AST sentinels for the 59 retired
names. The final review corrected two sentinel defects before closeout:
transitive local mixins inherited by `TldwCli` are now part of the exact root
inventory, while destination-owned `reactive_attr` values are no longer
misclassified as root access. PR review then moved the retained/retired
inventory into one test-only contract module shared by the source,
production-app, and packaging sentinels, eliminating same-length list drift
without making the installed child import test code.

Added a production maturity suite that uses only the normal `TldwCli` and its
registered screens. It exercises LLM, Chat, Personas, Library, Media, Search,
the Ingest-to-Library alias, MCP, Evals, and Settings twice through fresh
screen construction, verifies destination-local state and restoration, rejects
all retired app attributes, recursively audits memory-only snapshots, and
statically rejects surrogate-app patterns throughout `Tests/ProductionApp`.

Extended the copied-source distribution fixture to build one wheel and one
sdist from committed source, install the wheel with `--no-deps` outside the
checkout, exclude both checkout and copied-build roots from `sys.path`, and
audit every loaded package module after a real installed `TldwCli` Home-to-Chat
run. Existing entry-point, metadata, resource, license, and before/after
installed-hash invariants remain enforced.

Verification on the rebased tree:

- `pytest Tests/ProductionApp/test_reactive_ownership_maturity.py Tests/test_application_state_ownership.py -q`: 57 passed, 2 warnings in 281.97s.
- `pytest Tests/Packaging/test_installed_distribution.py -q`: 6 passed in 21.83s.
- Authorized integrated suite (`Tests/ProductionApp`, the approved State,
  Provider and Library direct-function tests, source ownership, and installed
  distribution): 196 passed, 5 warnings in 567.81s.
- `compileall`, scoped Ruff lint, the exact zero-F841 Settings assertion, the
  37-file Ruff format gate, `git diff --check`, and the committed-source
  cleanliness checks passed.
- TASK-647–652 and TASK-904–905 are Done, have no unchecked acceptance
  criteria, contain Implementation Notes, and retain their reviewed
  invariants under the final matrix.

`Tests/UI` is explicitly excluded because its conftest imports legacy
surrogate app/widget harnesses; no raw repository-wide pytest result is
claimed. Latest `dev` had already deleted two stale format-plan paths and
removed the former two Settings F841 findings, so the plan records those
stronger current baselines rather than restoring dead code or preserving stale
expectations.

Modified files: `tldw_chatbook/app.py`,
`Tests/reactive_ownership_contract.py`,
`Tests/test_application_state_ownership.py`,
`Tests/ProductionApp/test_reactive_ownership_maturity.py`,
`Tests/Packaging/test_installed_distribution.py`, the TASK-906 plan, this
task, and the approved decomposition specification.

ADR required: yes. Existing
`backlog/decisions/033-application-session-state-ownership.md` defines the
final root ownership boundary, and
`backlog/decisions/032-immutable-installed-distribution-assets.md` defines the
installed-artifact gate. No new ADR was needed.
<!-- SECTION:NOTES:END -->
