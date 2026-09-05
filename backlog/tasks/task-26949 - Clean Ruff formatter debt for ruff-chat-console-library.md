---
id: TASK-26949
title: Clean Ruff formatter debt for ruff-chat-console-library
status: Done
assignee:
  - codex
created_date: '2026-08-31 18:31'
updated_date: '2026-09-05 16:49'
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

<!-- TASK-26000-BATCH: ruff-chat-console-library -->
<!-- TASK-26000-PATHS-SHA256: cd664f7c1688da479c38f4c47b8c5b2c997aab2320c9ceef1e2a73bd12476b90 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-library` Ruff formatter batch at the owner boundary recorded as: Console library activity, policy, and destination services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat", "Tests/Library"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_chat_store_library_policy.py",
  "Tests/Chat/test_console_library_activity_buffer.py",
  "Tests/Chat/test_console_library_destination.py",
  "Tests/Chat/test_console_library_policy_coordinator.py",
  "Tests/Chat/test_console_library_runtime_policy.py",
  "tldw_chatbook/Chat/console_library_activity_buffer.py",
  "tldw_chatbook/Chat/console_library_destination.py",
  "tldw_chatbook/Chat/console_library_policy.py",
  "tldw_chatbook/Chat/console_library_policy_repository.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same nearest logical owner and significant-token position, using a uniquely fail-closed same-line `except` clause for an `ExceptHandler` header and otherwise the nearest containing AST statement; exclude only AST-neutral parentheses proven by shadow parse/dump comparison, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile the nine-path batch manifest and its canonical digest against the exact current `origin/dev` base, including upstream lineage since TASK-26000's authority cut.
2. Capture Python 3.12.11 structural evidence and an exact focused-test baseline before making any source edit.
3. Run Ruff 0.15.22 formatting once with all nine assigned paths supplied explicitly, then require structural, comment/directive, scope, lint, format, focused-test, deterministic-replay, diagnostic, and governance parity.
4. Obtain independent task and whole-branch reviews, close the task with exact evidence, rebase onto the latest `dev`, and repeat every base-sensitive gate before integration.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's existing contract and changes no architecture, schema, storage, security, dependency, or long-lived UX boundary.

## Implementation Notes

Ruff 0.15.22 formatted exactly the nine assigned paths in commit `5102a54f71b4d7f2b5fa3543ab76b2cf70a52726` (`style(chat): format console library batch`), producing a nine-file diff of 140 insertions and 86 deletions. Provenance is TASK-26000 authority cut `e555df102c950c29beed5e7119f433d35eee1f3c`, branch authority base `2b4973971e5dcf101c5a6ddcc55aa082ff22f814`, review/planning base `dfafa72930433b95c8eb86f0a8e1496757827c79`, and formatter commit `5102a54f71b4d7f2b5fa3543ab76b2cf70a52726`. The only upstream lineage was `1e3b1d4bab3ff43ba54a9e0cb7179b0793292e9f` modifying `tldw_chatbook/Chat/console_library_destination.py`; the other eight assigned paths were upstream-clean, with no rename or deletion. The manifest result was `task_count=9 task_unique=9 tests=5 production=4`, `canonical_batches=1 canonical_count=9 exact_match=True`, `digest=cd664f7c1688da479c38f4c47b8c5b2c997aab2320c9ceef1e2a73bd12476b90`, and `all_exist=True`.

- Tool identity: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python --version` -> `Python 3.12.11`; `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff --version` -> `ruff 0.15.22`; `shasum -a 256 /tmp/task26947_format_guard.py` -> `3fac070e94fe91cd152f956b19093c457c48787ea5449b54945b2305386b7471  /tmp/task26947_format_guard.py` (all exit 0).
- Drift and lineage: `git diff --name-status --find-renames e555df102c950c29beed5e7119f433d35eee1f3c 2b4973971e5dcf101c5a6ddcc55aa082ff22f814 -- Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `M\ttldw_chatbook/Chat/console_library_destination.py`; `git log --format='%H %ad %s' --date=iso-strict --name-status e555df102c950c29beed5e7119f433d35eee1f3c..2b4973971e5dcf101c5a6ddcc55aa082ff22f814 -- Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `1e3b1d4bab3ff43ba54a9e0cb7179b0793292e9f 2026-09-04T05:53:57-07:00 fix(console): keep vllm session endpoints process-local` and `M\ttldw_chatbook/Chat/console_library_destination.py` (exit 0).
- Formatter inventory and application: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `9 files would be reformatted` (exit 1, expected read-only debt inventory); `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `9 files reformatted` (exit 0, the only modifying Ruff invocation).
- Structural parity: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26947_format_guard.py capture /tmp/task26949_before.json Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `captured structural evidence for 9 paths`; `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26947_format_guard.py compare /tmp/task26949_before.json Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `structural evidence matches for 9 paths` (both exit 0). This proves equal type-comment-aware AST dumps with only `ast.TypeIgnore.lineno` normalized, ordered comments, inline directive owners/significant-token positions, standalone Ruff adjacency, and `fmt` enclosed-node intervals; the guard found no ambiguity.
- Focused baseline: `LOGURU_LEVEL=ERROR /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=line --disable-warnings --junitxml=/tmp/task26949_before.xml Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py` -> `Running 103 items in this shard` and `103 passed, 2 warnings in 15.37s` (exit 0).
- Focused post-format: `LOGURU_LEVEL=ERROR /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=line --disable-warnings --junitxml=/tmp/task26949_after.xml Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py` -> `Running 103 items in this shard` and `103 passed, 2 warnings in 15.50s` (exit 0). Normalized JUnit results were `/tmp/task26949_before.xml totals={"errors":0,"failures":0,"skipped":0,"tests":103} normalized_failure_error_keys=[]`, `/tmp/task26949_after.xml totals={"errors":0,"failures":0,"skipped":0,"tests":103} normalized_failure_error_keys=[]`, and `normalized_keys_equal=True`.
- Focused-test rationale: these exact five assigned `Tests/Chat` modules directly exercise the formatted Console library store, activity-buffer, destination, coordinator, and runtime-policy behavior. They are the task-owned behavioral surface and avoid unrelated suites. No full suite was run because repository policy requires explicit opt-in.
- Ruff gates: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `All checks passed!`; `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py` -> `9 files already formatted` (both exit 0).
- Deterministic replay: `set -o pipefail; for source_path in Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_library_activity_buffer.py tldw_chatbook/Chat/console_library_destination.py tldw_chatbook/Chat/console_library_policy.py tldw_chatbook/Chat/console_library_policy_repository.py; do git show "2b4973971e5dcf101c5a6ddcc55aa082ff22f814:$source_path" | /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --stdin-filename "$source_path" - | cmp -s - "$source_path"; replay_rc=$?; if [ "$replay_rc" -ne 0 ]; then echo "replay_mismatch $source_path rc=$replay_rc"; exit "$replay_rc"; fi; echo "replay_match $source_path"; done` -> `replay_match` for each of the nine assigned paths (exit 0). No lint-only commit was needed.
- Persistent diagnostics: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --diff` -> `no drift: the committed inventory matches the rebuild exactly.` and `diagnostic inventory verified: 580 owners, 1336 TASK-492 calls, 30 TASK-31551 calls, 7615 TASK-494 calls, 12 sink files` (exit 0); therefore `Docs/security/production-diagnostic-inventory.json` was not changed.
- Governance: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py` -> `Running 3 items in this shard` and `3 passed, 2 warnings in 2.19s` (exit 0); `git diff --check` and `git diff --cached --check` -> no output (both exit 0).

Independent Task 1 review found zero Critical, Important, or Minor findings. Self-review found deterministic Ruff-only changes, no unassigned path, and no handwritten production behavior change. ADR required: no. ADR path: N/A. This task directly implements TASK-26000's existing formatter contract and changes no architecture, schema, storage, security, dependency, or long-lived UX boundary. `TASK-26000-FINAL` remains `false`.
