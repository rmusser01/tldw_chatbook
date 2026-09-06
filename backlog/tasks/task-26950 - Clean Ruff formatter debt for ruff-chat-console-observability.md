---
id: TASK-26950
title: Clean Ruff formatter debt for ruff-chat-console-observability
status: In Progress
assignee:
  - codex
created_date: '2026-08-31 18:31'
updated_date: '2026-09-06 01:42'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-observability` Ruff formatter batch at the owner boundary recorded as: Console cost, trace, status, display, diff, and citation services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

<!-- TASK-26000-BATCH: ruff-chat-console-observability -->
<!-- TASK-26000-PATHS-SHA256: 17e8b292990312562b0e877a69501a5dd9eb25489008df95a9d12976ac13d58f -->
<!-- TASK-26000-FINAL: false -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_agent_diff_channel.py",
  "Tests/Chat/test_console_cost_estimate_cache.py",
  "Tests/Chat/test_console_cost_tracker.py",
  "Tests/Chat/test_console_diff_feedback_delivery.py",
  "Tests/Chat/test_console_diff_hunks.py",
  "Tests/Chat/test_console_display_state.py",
  "Tests/Chat/test_console_glyphs.py",
  "Tests/Chat/test_console_local_citation_boundary.py",
  "Tests/Chat/test_console_run_status_surfaces.py",
  "Tests/Chat/test_console_status_chips_cost.py",
  "Tests/Chat/test_console_trace_first_send_atomicity.py",
  "Tests/Chat/test_console_trace_fork_lineage.py",
  "Tests/Chat/test_console_trace_legacy_migration.py",
  "Tests/Chat/test_console_trace_models.py",
  "Tests/Chat/test_console_trace_projection.py",
  "tldw_chatbook/Chat/console_cost_tracker.py",
  "tldw_chatbook/Chat/console_display_state.py",
  "tldw_chatbook/Chat/console_trace_legacy.py",
  "tldw_chatbook/Chat/console_trace_projection.py",
  "tldw_chatbook/Chat/console_trace_redaction.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] #2 Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] #3 Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] #4 Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] #5 Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] #6 Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] #7 `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] #8 The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: mechanical formatting under the approved TASK-26000 contract. Reconcile the 20 assigned paths against current dev and the authority cut; capture structural and focused-test baseline; format only assigned files with Ruff 0.15.22; prove AST/comment/directive parity, lint, format, replay, focused tests, diagnostic inventory, and backlog uniqueness; independently review and integrate through Qodo and CI.
<!-- SECTION:PLAN:END -->

## Verification checkpoint (awaiting lint-contract decision)

At base `d3bf8b5397c9f92cf9fcc722193f75665cc0192a`, the 20-path manifest matches the canonical digest. Seventeen paths were formatted in commit `8a4e202291`; three cost-tracker paths were already formatted upstream by `144ac2e1083688bf10977d22540acef6943edcb1`. No assigned path was renamed or deleted. Python 3.12.11 and Ruff 0.15.22 structural/comment/directive checks and deterministic base-blob replay match all 20 paths; format check passes all 20. Independent review found no formatting or scope defect.

The unbounded focused baseline stalled in citation cancellation tests and was interrupted after 245 recorded cases. The complete paired runs use all 15 assigned test modules with `-q --timeout=5 --timeout-method=signal --show-capture=no --tb=short` and JUnit output `/tmp/task26950_bounded_before.xml` and `/tmp/task26950_bounded_after.xml`. Both recorded 648 cases: 575 passed, 72 failed, one expected failure, zero errors. All 72 failure identities and 14 timeout identities match; normalized identity/outcome digest is `2915ea68759ca9be3a290367e08d7a858a31c6ad53afc8065449995932c5c0e5`. The failures all belong to the inherited citation-boundary tests; no claim of a green behavioral suite is made. No full suite was run.

`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --diff` passed with no drift: 580 owners, 1336 TASK-492 calls, 30 TASK-31551 calls, 7617 TASK-494 calls, 12 sink files. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py --junitxml=/tmp/task26950_governance.xml` passed (3 tests, 2 warnings, 2.27s). `git diff --check` passed.

Exact paired command (repeat with `after` replacing `before` only in the XML filename):

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_agent_diff_channel.py Tests/Chat/test_console_cost_estimate_cache.py Tests/Chat/test_console_cost_tracker.py Tests/Chat/test_console_diff_feedback_delivery.py Tests/Chat/test_console_diff_hunks.py Tests/Chat/test_console_display_state.py Tests/Chat/test_console_glyphs.py Tests/Chat/test_console_local_citation_boundary.py Tests/Chat/test_console_run_status_surfaces.py Tests/Chat/test_console_status_chips_cost.py Tests/Chat/test_console_trace_first_send_atomicity.py Tests/Chat/test_console_trace_fork_lineage.py Tests/Chat/test_console_trace_legacy_migration.py Tests/Chat/test_console_trace_models.py Tests/Chat/test_console_trace_projection.py -q --timeout=5 --timeout-method=signal --timeout-disable-debugger-detection --show-capture=no --tb=short --junitxml=/tmp/task26950_bounded_before.xml
```

Remaining conflict: untouched base and formatted files have the same three Ruff violations: two E741 generator bindings named `l` in `Tests/Chat/test_console_diff_hunks.py`, and F811 for the second `CitationFingerprintCodec` import in `Tests/Chat/test_console_local_citation_boundary.py`. Passing lint requires an explicit exception to exact AST equality. Independently reviewed proposal: rename only the two generator bindings and their references to `line` (six AST identifier changes), and remove only the later redundant import alias (the repository re-exports the same identity class). Preserve every other AST node and all comments/directives. These lint corrections have not been applied; task remains In Progress and is not ready to merge.
