---
id: TASK-26950
title: Clean Ruff formatter debt for ruff-chat-console-observability
status: Done
assignee:
  - codex
created_date: '2026-08-31 18:31'
updated_date: '2026-09-06 02:08'
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
Owner-approved exception (2026-09-05): exact AST parity permits only six `l` to `line` identifier substitutions in the two generator expressions in `test_console_diff_hunks.py` and removal of the later redundant `CitationFingerprintCodec` import alias in `test_console_local_citation_boundary.py`. Every other AST node and all comments/directives must remain unchanged. Verify the precise lint delta separately, then formatter parity against the corrected baseline.
<!-- AC:BEGIN -->
- [x] #1 After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] #2 Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] #3 Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] #4 Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] #5 Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] #6 Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] #7 `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] #8 The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: mechanical formatting under the approved TASK-26000 contract. Reconcile the 20 assigned paths against current dev and the authority cut; capture structural and focused-test baseline; format only assigned files with Ruff 0.15.22; prove AST/comment/directive parity, lint, format, replay, focused tests, diagnostic inventory, and backlog uniqueness; independently review and integrate through Qodo and CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Formatted 17 assigned files with Ruff 0.15.22; three were already formatted upstream. Applied only the owner-approved two generator renames and redundant test import removal. Production behavior remains unchanged. Exact corrected-base replay and structural/comment/directive verification pass all 20 paths; independent final review found no actionable issues. ADR required: no; existing TASK-26000 formatter contract and explicit owner amendment apply. Detailed verification evidence follows.

Final approved-change verification: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26950_verify_approved.py` passes. Verifier SHA256: `b500708c8333f891b836d26fb9ace004b0ceae43fe2b9adad7525e8e2040b4e3`. It proves exactly six identifier substitutions and one import-alias removal, unchanged ASTs for the other 18 files, preserved comments/directives/fmt ranges for all 20 files, and byte-identical Ruff replay from the corrected base for all 20. Ruff lint and format checks pass; the retained and removed imports resolve to the same class object.

The exact paired pytest command above was rerun with only the JUnit path changed to `/tmp/task26950_approved_after.xml`: 648 cases in 224.032s, 575 passed, 72 inherited failures, one xfail, zero errors. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26950_compare_junit.py` confirms all 648 identity/outcome triples and all 14 timeout identities equal the complete bounded baseline. This comparator uses outcome labels `passed`, `failed`, and `xfail`; its sorted compact JSON digest is `86860a7cc5f449807c25e842111524608f5b64033cbd3d8b6b05352ed0bfcaf4`. This is regression parity, not a green behavioral suite.

Final governance rerun: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py` passed (3 tests, 2 warnings, 3.00s). `git diff --check` passed. Independent final branch review verified the 17 assigned Python files and task/plan documentation only, with no actionable findings. A fresh fetch confirmed `origin/dev` remains `d3bf8b5397c9f92cf9fcc722193f75665cc0192a`. No full test suite or unrelated behavioral remediation was performed.
<!-- SECTION:NOTES:END -->

## Historical verification checkpoint (before lint exception approval)

### Assigned-path lineage

Compared with authority cut `e555df102c950c29beed5e7119f433d35eee1f3c`, twelve paths are unchanged and eight modified (none renamed/deleted). `Tests/Chat/test_console_cost_tracker.py`, `Tests/Chat/test_console_status_chips_cost.py`, and `tldw_chatbook/Chat/console_cost_tracker.py` were already formatted by `144ac2e1083688bf10977d22540acef6943edcb1`. Other modified-path lineage:

- `Tests/Chat/test_console_trace_first_send_atomicity.py`: `acfe1a782e0abd5f85550c678748c0b5948f05a9`, `1e3b1d4bab3ff43ba54a9e0cb7179b0793292e9f`.
- `Tests/Chat/test_console_trace_fork_lineage.py`: `51b5305fa14230995faea5c34b2a1acc3e4c188a`.
- `Tests/Chat/test_console_trace_projection.py`: `b4e6570dde2718ccd67cec4df76f09ac4fef67fd`, `a5eabe7a872d1ce40bad93c660bd2692e30c6f2c`, `51b5305fa14230995faea5c34b2a1acc3e4c188a`, `3b87de430f01af2e723fc12d54a07dcb3b9b5986`, `e8ff1a24a4797912b6d623f7c122335461d14e8c`.
- `tldw_chatbook/Chat/console_trace_projection.py`: `bb40085203c1cd8bba35c8fa2ec6bbaf8ab73e82`, `a5eabe7a872d1ce40bad93c660bd2692e30c6f2c`, `51b5305fa14230995faea5c34b2a1acc3e4c188a`, `3b87de430f01af2e723fc12d54a07dcb3b9b5986`, `32814f12410d8d3281f8a771e05bca8cb03b636e`, `e8ff1a24a4797912b6d623f7c122335461d14e8c`, `34723cbb05259efd20816d10e4ddda20ccb263dd`, `84243885aafa87346800286afa4b3d13d17d547e`.
- `tldw_chatbook/Chat/console_trace_redaction.py`: `710997f5ff915e34595e8eaf2a7d1bf5d123f3aa`.

At base `d3bf8b5397c9f92cf9fcc722193f75665cc0192a`, the 20-path manifest matches the canonical digest. Seventeen paths were formatted in commit `8a4e202291`; three cost-tracker paths were already formatted upstream by `144ac2e1083688bf10977d22540acef6943edcb1`. No assigned path was renamed or deleted. Python 3.12.11 and Ruff 0.15.22 structural/comment/directive checks and deterministic base-blob replay match all 20 paths; format check passes all 20. Independent review found no formatting or scope defect.

The unbounded focused baseline stalled in citation cancellation tests and was interrupted after 245 recorded cases. The complete paired runs use all 15 assigned test modules with `-q --timeout=5 --timeout-method=signal --show-capture=no --tb=short` and JUnit output `/tmp/task26950_bounded_before.xml` and `/tmp/task26950_bounded_after.xml`. Both recorded 648 cases: 575 passed, 72 failed, one expected failure, zero errors. All 72 failure identities and 14 timeout identities match; normalized identity/outcome digest is `2915ea68759ca9be3a290367e08d7a858a31c6ad53afc8065449995932c5c0e5`. The failures all belong to the inherited citation-boundary tests; no claim of a green behavioral suite is made. No full suite was run.

`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --diff` passed with no drift: 580 owners, 1336 TASK-492 calls, 30 TASK-31551 calls, 7617 TASK-494 calls, 12 sink files. `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py --junitxml=/tmp/task26950_governance.xml` passed (3 tests, 2 warnings, 2.27s). `git diff --check` passed.

Exact paired command (repeat with `after` replacing `before` only in the XML filename):

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_agent_diff_channel.py Tests/Chat/test_console_cost_estimate_cache.py Tests/Chat/test_console_cost_tracker.py Tests/Chat/test_console_diff_feedback_delivery.py Tests/Chat/test_console_diff_hunks.py Tests/Chat/test_console_display_state.py Tests/Chat/test_console_glyphs.py Tests/Chat/test_console_local_citation_boundary.py Tests/Chat/test_console_run_status_surfaces.py Tests/Chat/test_console_status_chips_cost.py Tests/Chat/test_console_trace_first_send_atomicity.py Tests/Chat/test_console_trace_fork_lineage.py Tests/Chat/test_console_trace_legacy_migration.py Tests/Chat/test_console_trace_models.py Tests/Chat/test_console_trace_projection.py -q --timeout=5 --timeout-method=signal --timeout-disable-debugger-detection --show-capture=no --tb=short --junitxml=/tmp/task26950_bounded_before.xml
```

At this historical checkpoint, untouched base and formatted files had the same three Ruff violations: two E741 generator bindings named `l` in `Tests/Chat/test_console_diff_hunks.py`, and F811 for the second `CitationFingerprintCodec` import in `Tests/Chat/test_console_local_citation_boundary.py`. The owner subsequently approved the precise test-only AST exception recorded above: six identifier changes and removal of the redundant import alias. The repository re-exports the same identity class, so the retained canonical import preserves the binding. All other AST nodes and all comments/directives remain subject to parity checks.
<!-- AC:END -->
