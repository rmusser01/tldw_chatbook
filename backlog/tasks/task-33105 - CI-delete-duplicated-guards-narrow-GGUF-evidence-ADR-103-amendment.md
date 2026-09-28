---
id: TASK-33105
title: 'CI: delete duplicated guards, narrow GGUF evidence, ADR-103 amendment'
status: Done
assignee: []
created_date: '2026-09-27 20:37'
updated_date: '2026-09-28 02:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 2026-09-27-ci-conflicts-and-waste-design.md parts C1, C2 and E.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 css-bundle-guard and backlog-guard deleted; bundle and backlog-id checks pinned to run on dev/main pushes
- [x] #2 GGUF evidence paths narrowed to GGUF code and pinned in Tests/CI
- [x] #3 GGUF UI test files added to the UI census only if green on the minimal dependency set
- [x] #4 ADR-103 amended for the nightly cadence change and the removed guards
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both `css-bundle-guard.yml` and `backlog-guard.yml` were deleted -- their checks already ran
inside the required workflow on every pull request and on every `dev`/`main` push. Push
coverage for the bundle and backlog-id checks is pinned by
`test_bundle_and_backlog_checks_run_on_push_events` (`Tests/CI/test_ci_queue_pressure_contract.py`).

GGUF evidence was narrowed, then partly restored after Qodo review on PR #2860. `pyproject.toml`
and the shared test-harness paths are back in the GGUF workflows' `on.pull_request.paths`: `pyproject.toml`,
`Tests/conftest.py` and `Tests/UI/conftest.py` in both; `Tests/UI/consolidated_css.py` in 2062.1 only;
`Tests/private_profile.py` and `Tests/UI/app_factory.py` in 2062.2 only. `tldw_chatbook/app.py`, `tldw_chatbook/config.py` and
`tldw_chatbook/css/**` stay excluded, which removes the triggers behind about 244 of the PR merges
that fired these workflows in 30 days. **This is a deviation from the approved spec's C2 list**
(the spec's C2 narrowed all of them) -- decided by the controller as a partial accept of Qodo's
findings, trading back some of the narrowing for native macOS/Windows install signal and shared
harness coverage. Stating this plainly per the final review's finding.

No UI census additions. Both `Tests/UI/test_model_installed_view.py` and
`Tests/UI/test_llm_gguf_source_modes.py` are red on the fast lane's minimal dependency set: 11
failures total, all `RecoveryRequired: raw_source_selection_changed` --
`test_model_installed_view.py` 1 (`test_models_host_lazily_wires_parakeet_activation_and_deletion`),
`test_llm_gguf_source_modes.py` 10. `scripts/ui_pr_gate_census.txt` is unchanged.

Final trigger table (post-restore, `on.pull_request.paths`):

| Changed files | 2062.1 import | 2062.2 source |
|---|---|---|
| `tldw_chatbook/app.py` only | no | no |
| `tldw_chatbook/css/x.tcss` only | no | no |
| `Tests/conftest.py` only | yes | yes |
| `pyproject.toml` only | yes | yes |
| `tldw_chatbook/Model_Artifacts/x.py` | yes | yes |
| `tldw_chatbook/UI/Screens/llm_screen.py` | no | yes |

`backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md` was amended for the nightly
cadence change, the deleted guards, and the narrowed/partly-restored GGUF triggers.

Merged as #2860.
<!-- SECTION:NOTES:END -->
