---
id: TASK-617.4
title: Add exact character TTS assignment mutation service
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31 00:58'
updated_date: '2026-07-31 03:39'
labels:
  - tts
  - profiles
  - roleplay
dependencies:
  - TASK-617.2
  - TASK-617.3
  - TASK-763
  - TASK-951
references:
  - >-
    backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md
parent_task_id: TASK-617
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete TTS Slice 3A by exposing exact source-aware character assignment mutations over the existing profile service and repository so Slice 3B can add visible assignment and assigned-profile speech without weakening lifecycle or capability authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A caller can set or replace one exact `CharacterRef` assignment using a caller-held `LoadedTTSProfile` and its repository generation.
- [x] #2 A set or replace request validates the exact loaded profile revision against a fresh authoritative capability observation before repository mutation.
- [x] #3 The repository's final transaction checks expected lifecycle generation, selected profile revision, and expected current assignment state before mutation.
- [x] #4 Detach uses the caller-held assignment generation and exact assigned profile ID; it is idempotent only when already absent and refuses to remove a replacement.
- [x] #5 Stale restore, profile edit, catalog movement, assignment races, missing authority, and malformed repository results fail closed with bounded errors and no partial mutation.
- [x] #6 Deterministic service and repository tests cover success, replacement, detach, lifecycle races, capability races, and compare-and-set conflicts.
- [x] #7 The task adds no assignment UI, speech resolver, automatic speech, Persona TTS, portability, Sync changes, or managed audio.cpp behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: ADR-037 already governs lifecycle generation profile revision expected-current-assignment compare-and-set semantics and the Slice 3B deferrals; this task implements that accepted boundary without a schema or ownership change.

1. Pin mandatory repository assignment expectations and transaction-boundary lifecycle races with failing tests.
2. Implement transactional generation profile-revision expected-current-assignment and exact-detach checks.
3. Pin profile-service capability ordering forwarded expectations stale-state handling and bounded failures with failing tests.
4. Implement minimal exact set/replace and detach service operations over existing domain values.
5. Update the developer guide run focused and broad verification complete TASK-617.4 request review rebase and open one PR.

Full plan: Docs/superpowers/plans/2026-07-31-tts-assignment-mutation-service.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Slice 3A on the existing profile repository and service boundary. The repository now rechecks caller-held lifecycle generation, selected profile revision, and expected current assignment inside the final transaction. The service validates exact immutable caller state, performs a fresh authoritative audio.cpp capability check before set/replace, and forwards exact compare-and-set expectations. Detach forwards the caller-held generation and assigned profile ID, succeeds when already absent, and refuses to remove a replacement.

Files changed: `Docs/superpowers/plans/2026-07-31-tts-assignment-mutation-service.md`; `tldw_chatbook/TTS/profile_repository.py`; `tldw_chatbook/TTS/profile_service.py`; `Tests/TTS/test_profile_repository.py`; `Tests/TTS/test_profile_service.py`; `Tests/TTS/test_tts_profile_capabilities.py`; `Tests/TTS/test_tts_app_ownership.py`; `Docs/Development/TTS/TTS_MODULE_GUIDE.md`; and this Backlog task.

Tradeoffs and exclusions: reused the existing lifecycle generation and repository transaction rather than adding an assignment revision column, second store, schema change, or additional lock. Capability admission precedes mutation, while repository transaction checks remain authoritative. This slice adds no assignment UI, speech resolver, automatic speech, Persona inheritance, portability, Sync behavior, or managed audio.cpp behavior. No implementation-scope deviation occurred; rebase, push, PR creation, and final whole-range review remain parent-owned closeout steps.

ADR required: no. ADR-037 already governs the exact character identity and assignment compare-and-set boundary; the accepted design, ADR, plan, and developer guide remain linked.

Independent review: repository RED contract tests, repository implementation, service RED contract tests, and service implementation were each independently checked for specification compliance and quality. Re-reviews found no unresolved Critical, Important, or Minor issue.

Verification:
- Task-critical union: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS/test_profile_types.py Tests/TTS/test_profile_repository.py Tests/TTS/test_profile_repository_lifecycle.py Tests/TTS/test_profile_service.py Tests/TTS/test_tts_profile_capabilities.py Tests/TTS/test_tts_app_ownership.py Tests/TTS/test_console_speech_snapshot_admission.py Tests/TTS/test_console_audio_cpp_native.py -q` -> 672 passed, 4 warnings, 0 skipped, exit 0. Warnings were the existing RequestsDependencyWarning, two train-journey invalid-escape SyntaxWarnings, and the webrtcvad pkg_resources deprecation warning.
- Broader union: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS/test_profile_schema.py Tests/TTS/test_profile_store_lock.py Tests/TTS/test_profile_backup_integration.py Tests/UI/test_stts_profile_library.py -q` -> 292 passed, 10 failed, 1 warning, 0 skipped, exit 1. An untouched detached `origin/dev` at `1e0979c74` reproduced the identical 292 passed, 10 failed, 1 warning result, classifying all failures as an unchanged backup-integration baseline outside this slice. Failing nodes: `test_real_worker_cancellation_before_legacy_publication_leaves_no_artifact`; `test_real_manifest_stage_is_created_exclusively_in_worker_thread`; `test_real_manifest_replace_failure_cleans_stage_and_keeps_values_private`; `test_manifest_cleanup_control_flow_supersedes_ordinary_primary_error`; `test_manifest_cleanup_failure_does_not_mask_base_exception_or_expose_values`; `test_real_backup_all_same_clock_uses_distinct_timestamp_prefixed_directories`; `test_backup_all_awaits_profile_and_manifest_before_success`; `test_backup_all_records_only_legacy_entries_on_profile_partial_failure[unavailable]`; `test_backup_all_records_only_legacy_entries_on_profile_partial_failure[backup_failure]`; and `test_backup_all_worker_failure_is_private_and_never_reports_success[legacy]`.
- Ruff check: exact task command -> All checks passed, exit 0.
- Ruff format check: exact task command -> 4 files already formatted, exit 0.
- Compile: exact `compileall -q` command -> exit 0.
- Typing: exact mypy command -> Success: no issues found in 2 source files, exit 0.
- Diff hygiene: `git diff --check origin/dev...HEAD` and `git diff --check` -> exit 0.
- Scope audit before the documentation commit returned only approved plan, task, production, and test files; the only working-tree additions are the approved developer guide and task closeout.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and concise implementation notes record the delivered behavior and any plan deviations.
- [x] #2 Focused repository, service, lifecycle, capability, ownership, Console snapshot, and native audio.cpp regression tests pass.
- [x] #3 Task-scoped Ruff, formatting, compile, typing, and git diff checks pass or exact unchanged baselines are documented.
- [x] #4 ADR-037, the approved Slice 3A design, and the TTS developer guide remain current and linked.
- [x] #5 Independent review finds no unresolved Critical, Important, or Minor issue before the PR is merged.
<!-- DOD:END -->
