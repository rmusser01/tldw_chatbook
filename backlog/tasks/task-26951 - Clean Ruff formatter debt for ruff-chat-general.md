---
id: TASK-26951
title: Clean Ruff formatter debt for ruff-chat-general
status: Done
assignee:
  - codex
created_date: '2026-08-31 18:31'
updated_date: '2026-09-06 03:31'
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
Clean the `ruff-chat-general` Ruff formatter batch at the owner boundary recorded as: Remaining cohesive Chat orchestration helpers and direct Chat tests.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

<!-- TASK-26000-BATCH: ruff-chat-general -->
<!-- TASK-26000-PATHS-SHA256: 845fb91c16932e965061e116d600f846d1b6f4cbc51468a6a859fe9509dfc3c3 -->
<!-- TASK-26000-FINAL: false -->

## Assigned Paths

```json
[
  "Tests/Chat/test__zz_probe2.py",
  "Tests/Chat/test_anthropic_caching_degrade.py",
  "Tests/Chat/test_change_notes_db.py",
  "Tests/Chat/test_change_turn_tracking.py",
  "Tests/Chat/test_chat_api_call_endpoint_redaction.py",
  "Tests/Chat/test_extract_response_content.py",
  "Tests/Chat/test_fleet_attention.py",
  "Tests/Chat/test_fleet_autowake.py",
  "Tests/Chat/test_fleet_settle_fanout.py",
  "Tests/Chat/test_local_server_discovery.py",
  "Tests/Chat/test_local_thinking_wire_formats.py",
  "Tests/Chat/test_openai_reasoning_sampling_params.py",
  "Tests/Chat/test_reply_sentence_sequencer.py",
  "Tests/Chat/test_turn_context_posture.py",
  "Tests/Chat/test_visual_renderer_decoupling.py",
  "tldw_chatbook/Chat/assistant_generation_state.py",
  "tldw_chatbook/Chat/document_generator.py",
  "tldw_chatbook/Chat/reply_sentence_sequencer.py",
  "tldw_chatbook/Chat/thinking_blocks.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
Owner-approved exception (2026-09-05): strict AST parity permits only splitting the first combined import in `test__zz_probe2.py` into three imports preserving order, replacing the unused `note_id = db.add_change_note(...)` assignment in `test_mark_notes_delivered_empty_list_is_noop` with the identical call expression, and removing the unused `threading` import from `test_fleet_autowake.py`. Verify these precise changes separately and compare formatting against the corrected baseline; all other AST structure and comments/directives remain protected.
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
ADR required: no. ADR path: N/A. Reason: mechanical formatting under the approved TASK-26000 contract. Reconcile all 19 assigned paths against latest dev and authority cut; capture Python 3.12.11 structural/comment/directive evidence and targeted baseline; apply the three owner-approved test-only lint corrections; run Ruff 0.15.22 only on assigned paths; verify parity, deterministic replay, lint/format, targeted outcomes, diagnostic inventory and backlog uniqueness; review and integrate through Qodo and CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Formatted all 19 assigned Python files with Ruff 0.15.22 in `707739747fc789b226601e4d7c1a9717283efb67`. Included only the three owner-approved test lint corrections; there are no hand-written production behavior changes or unassigned Python edits. ADR required: no; existing TASK-26000 contract plus the explicit owner amendment applies. The focused test surface is all 15 assigned direct Chat modules, not the full suite.

### Structural and formatter evidence

Python 3.12.11 is invoked as `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (`PY` below). `PATHS` denotes the exact ordered 19-path Assigned Paths JSON above. Original/corrected structural captures used `PY /tmp/task26947_format_guard.py capture /tmp/task26951_original.json PATHS` and the same command with `/tmp/task26951_corrected.json` after the approved corrections. `PY /tmp/task26951_approved_delta.py` proves only the three permitted AST transformations against immutable commit `1ba0aaae097ac3a08b462bbfab2dbde608d205f1`; it passes before and after formatting. The corrected files have no inline directives, standalone Ruff directives or fmt ranges; all original/corrected comment and directive fields match exactly.

`PY -m ruff format PATHS` reformatted 19 files. `PY /tmp/task26947_format_guard.py compare /tmp/task26951_corrected.json PATHS` passes all 19 type-comment-aware ASTs, ordered comments, directive anchors and fmt intervals. Corrected sources archived before formatting at `/tmp/task26951_corrected_sources.tar` were extracted into `/tmp/task26951-replay.45j6zk`, formatted using `PY -m ruff format --config /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-26951-ruff-chat-general/pyproject.toml PATHS`, and compared byte-for-byte with the worktree: all 19 match. `PY -m ruff check PATHS` and `PY -m ruff format --check PATHS` both pass.

Evidence SHA256:

- Guard: `3fac070e94fe91cd152f956b19093c457c48787ea5449b54945b2305386b7471`.
- Approved-delta verifier: `666ab01abd79ce433141b8cbffbb1bebd4ede5e774d8ad33a48e9948c7aa765b`.
- Original structural JSON: `61f9970a2a0772b96eb8b1d7b03e4af94f18b75ee18282b936e9d18331d539eb`.
- Corrected structural JSON: `9b56f704a4875b93a4b8cc3bf4bd3cd37d9e595522bfb6e872681300835ff43f`.
- Corrected source archive: `5ead2350ab6db3085ee693e2256d46667927915291a5e6719ed764d0b5396dee`.

### Targeted tests

Exact before command (repeat with only the output filename changed to `/tmp/task26951_after.xml`):

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test__zz_probe2.py Tests/Chat/test_anthropic_caching_degrade.py Tests/Chat/test_change_notes_db.py Tests/Chat/test_change_turn_tracking.py Tests/Chat/test_chat_api_call_endpoint_redaction.py Tests/Chat/test_extract_response_content.py Tests/Chat/test_fleet_attention.py Tests/Chat/test_fleet_autowake.py Tests/Chat/test_fleet_settle_fanout.py Tests/Chat/test_local_server_discovery.py Tests/Chat/test_local_thinking_wire_formats.py Tests/Chat/test_openai_reasoning_sampling_params.py Tests/Chat/test_reply_sentence_sequencer.py Tests/Chat/test_turn_context_posture.py Tests/Chat/test_visual_renderer_decoupling.py -q --timeout=10 --timeout-method=signal --timeout-disable-debugger-detection --show-capture=no --tb=short --junitxml=/tmp/task26951_before.xml
```

Before: 336 passed, one failed, 3 warnings in 148.15s. After: 336 passed, the same one failed, 2 warnings in 166.54s. All 337 identity/outcome triples match; sorted compact JSON with `passed`/`failed` labels hashes to `cbe80fc35e680cc8d0a5fb15674daf7ca2aac314e390dc8c0468236559dfc792`. The inherited failure is `test_fleet_settle_fanout.py::test_drain_row_is_terminal_at_fire_time_on_the_raise_path_too`: its `raising_persist(self, run_id, outcome)` mock receives an additional durable-handle argument. That behavioral repair is outside this formatter scope. This is exact regression parity, not a green test suite. The warning difference is a pre-existing invalid-escape SyntaxWarning emitted on baseline import; both runs include dependency/deprecation warnings and macOS temporary-directory cleanup noise.

### Governance

Independent task review found no actionable findings and verified exact scope, approved AST delta, comment/directive parity, and all 337 before/after test outcomes. No new architecture decisions or behavior changes were introduced. The temporary proof scripts and detailed implementation report remain available through PR integration.

`PY -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py --junitxml=/tmp/task26951_governance.xml` passed (3 tests, 2 warnings, 2.97s). `PY scripts/check_persistent_diagnostic_inventory.py --diff` passes on both original and formatted trees with no drift: 580 owners, 1336 TASK-492 calls, 30 TASK-31551 calls, 7617 TASK-494 calls, 12 sink files. `git diff --check` passes. No full test suite was run.
<!-- SECTION:NOTES:END -->

## Historical preflight checkpoint (before approval)

The owner subsequently approved the three precise test-only exceptions listed under Acceptance Criteria. The checkpoint below records the original state, not a pending approval.

Started from current `origin/dev` at `c0fa6639a1fd294bf2bfbdc043c0dcb70782a689` in an isolated worktree; the dirty main checkout is untouched. No Python files have been changed and no tests have been run yet.

Ruff 0.15.22 lint on the exact 19 assigned paths reports three inherited violations: E401 at `Tests/Chat/test__zz_probe2.py:1` (combined imports), F841 at `Tests/Chat/test_change_notes_db.py:351` (unused `note_id` assignment), and F401 at `Tests/Chat/test_fleet_autowake.py:14` (unused `threading` import). Formatting alone cannot satisfy both strict AST parity and clean lint. Proposed test-only exception, pending owner approval: split the combined import into three imports preserving order; retain the `db.add_change_note(...)` call but remove its unused assignment; remove the unused `threading` import. All other AST structure and all comments/directives remain under the existing parity contract. Do not implement these exceptions without approval.

## Assigned-path reconciliation

At integration base `c0fa6639a1fd294bf2bfbdc043c0dcb70782a689`, the compact JSON digest for all 19 assigned paths remains `845fb91c16932e965061e116d600f846d1b6f4cbc51468a6a859fe9509dfc3c3`. Compared with authority cut `e555df102c950c29beed5e7119f433d35eee1f3c`, 17 files are unchanged and two are modified; none are renamed or deleted and ownership is unchanged:

- `Tests/Chat/test_anthropic_caching_degrade.py`: `ed6c31a4687b3fc2532c71f6b70922a4de664f93`, `c2db1f7054cadbabd03fe7b92b25dda97fc12280`, and `75c69f2f34bbbbd0be853bfed6574a8085cd8b1a` added prompt-cache behavior and review corrections.
- `Tests/Chat/test_fleet_attention.py`: `b3860842c0f74cb0c6a5a1d37e7d997cf30c9aa9` completed the Console activity switchboard.
<!-- AC:END -->
