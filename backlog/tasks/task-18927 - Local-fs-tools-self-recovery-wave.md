---
id: TASK-18927
title: 'fs_* local tools: self-recovery wave'
status: Done
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-10-11 00:00'
labels:
  - agents
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's tool self-recovery philosophy (2026-08-19 hermes-release review) to chatbook's local fs_* tools (Tools/local_tool_impls.py): when a tool call almost-succeeds, return an actionable recovery result instead of a bare error, so the agent wastes fewer turns — and fewer approval cycles — on friction. Specifics, mirroring hermes's wave: fs_edit detects already-applied edits (success no-op) and diagnoses whitespace mismatches with nearest-match candidates; fs_write verifies on-disk content after writing; fs_grep/fs_glob zero-match results probe near-misses (closest filenames/patterns); large outputs always carry pre-truncation size plus the spill path. Recovery messages are bounded, clearly machine-generated hints — never instructions the model must obey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fs_edit: an edit whose change is already present in the target returns success with a no-op note instead of a match error — evidence-gated: new_string present exactly once (a single replace's signature) or any presence under replace_all; short/coincidental presence falls through to an honest ambiguity note. `test_fs_edit_already_applied_returns_success_noop`
- [x] #2 fs_edit: a failed match returns a bounded diagnosis that visualizes whitespace differences against the nearest candidate locations in the file — whitespace-insensitive containment probe + `_closest_lines` (exact bodies first, then difflib ≥0.6) rendered with `·`/`\t` visualization, capped at 600 chars. `test_fs_edit_no_match_diagnoses_whitespace_mismatch`, `test_fs_edit_no_match_names_nearest_line_with_visible_whitespace`, `test_fs_edit_hint_is_bounded_on_a_huge_file`
- [x] #3 fs_write: post-write verification compares the written bytes against the intended content and reports any mismatch honestly — read-back under the same write lock; mismatch raises `write verification failed` naming both byte counts. `test_fs_write_verifies_on_disk_bytes_and_reports_mismatch`
- [x] #4 fs_grep/fs_glob: a zero-match result includes bounded near-miss suggestions (e.g. closest existing filenames) rather than a bare empty result — glob: bounded supplementary walk (entry-capped, same admissibility) + difflib over segments/paths; grep: single-pass case-insensitive-only hit count with a file:line example. Bare results are byte-identical when there is nothing close (pins preserved). `test_fs_glob_zero_match_suggests_closest_existing_names`, `test_fs_grep_zero_match_reports_case_insensitive_near_misses`
- [x] #5 Every recovery/hint message is length-bounded and labeled as a tool-generated hint; unit tests include adversarial inputs (huge files, binary content, path edge cases) and assert bounds — `HINT_LABEL = "hint (tool-generated)"`, `_bound_hint` 600-char clamp on every hint path; adversarial tests: 5,000-line file, binary content refusal unchanged, whitespace-only old_string, no-candidate texts keep the exact legacy message
- [x] #6 No permission-model change: recovery never bypasses Allow/Ask/Off or any approval gate — verified by tests — the no-op path still resolves through `resolve_workspace_path(intent="write")` (spied in `test_fs_edit_already_applied_still_passes_the_same_admission`); near-miss pools only contain names the filtered walk already deemed admissible (`test_fs_glob_near_miss_pool_only_contains_admissible_names`, `test_fs_grep_ci_probe_respects_sensitive_exclusions`)
- [x] #7 fs_patch: a context mismatch returns a bounded diagnosis naming the nearest candidate location, not a bare patch_context_mismatch code (Tools/patch_tool_impls.py:211,407) — `FilesystemPatchError` gains an optional `detail` (reason codes unchanged, lockstep with the tldw_server reference preserved); `apply_patch_to_text` computes the nearest candidate (exact-body location first, closest lines second, CRLF/LF note for line-ending mismatches); both wrapper surfaces (`patch_files`, `patch_validated_files`) append `hint (tool-generated): <detail>`. `test_fs_patch_context_mismatch_names_nearest_candidate`, `test_fs_patch_context_mismatch_reason_code_unchanged`
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: behavior improvements within the existing local-tool boundary (ADRs 030/032/033 govern the boundary itself); no schema or interface change.

1. fs_edit: idempotence detection + whitespace-visualized no-match diagnosis
2. fs_write: read-back verification
3. fs_grep/fs_glob: near-miss probing with bounded suggestions
4. Bounds/hint-labeling tests, adversarial cases, docs note in agent-runs-and-tools.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: recovery lives entirely in tool RESULT/ERROR text — no permission, admission, or path-resolution change anywhere. Shared hint primitives in `local_tool_impls.py` (`HINT_LABEL`, `_MAX_HINT_CHARS=600`, `_bound_hint`, `_visualize_whitespace`, `_closest_lines` with a pinned-seq2 SequenceMatcher) keep every hint bounded and labeled.
- fs_edit: already-applied detection is evidence-gated (single occurrence of new_string, or any presence under replace_all) after the first test wave showed a naive `new_string in content` check false-positives on short replacements ("x" appears anywhere); ambiguous presence is reported honestly instead of silently succeeding. Deletion edits (empty new_string) keep the plain error — no positive marker exists.
- fs_glob near-miss needed a bounded supplementary walk: `Path.glob` yields only matches, so the zero-match case collects nothing during the main walk; the probe walk is entry-capped (`MAX_SCAN_ENTRIES`) and applies the identical admissibility checks, so a sensitive name can never surface in a hint.
- fs_grep probes case-insensitive-only hits in the SAME single pass (every exact match is also a ci match for ordinary patterns); zero-match results without near misses stay byte-identical — the workspace executor's pinned payload test and the remote denylist pins both pass unchanged.
- fs_patch: reason codes are shared with tldw_server's reference implementation, so the diagnosis is carried as an optional `detail` attribute and appended by the chatbook-owned wrapper surfaces only; the verbatim-port contract (and the drift-guard byte-equality) is preserved by regenerating `remote_worker_bundle.py` with the committed builder.
- Tests: new `Tests/Tools/test_local_tool_self_recovery.py` (21 tests). Neighboring suites A/B at dev `9f8533686c`: local_impls/properties/patch/bundle-drift/wrong-file-hazard/sensitive-paths/glob-grep/census/import-closure = 12 pre-existing reds both sides (the `test_glob_grep_files.py` sandbox family, red at base, untouched); `test_workspace_tool_executor.py` + `test_remote_worker_denylist.py` = 159 passed.
- Modified files: `tldw_chatbook/Tools/local_tool_impls.py`, `tldw_chatbook/Tools/patch_tool_impls.py`, `tldw_chatbook/Tools/remote_worker_bundle.py` (regenerated, builder-run), `Tests/Tools/test_local_tool_self_recovery.py` (new), `Docs/User_Guide/console/agent-runs-and-tools.md`, this task file.
