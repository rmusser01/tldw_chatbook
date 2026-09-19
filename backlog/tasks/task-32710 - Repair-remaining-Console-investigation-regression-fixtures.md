---
id: TASK-32710
title: Repair remaining Console investigation regression fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 01:39'
updated_date: '2026-09-17 01:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful passing coverage for the three previously identified Console identity, compaction-close and persistence test failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identity expectations reflect the existing persona and character ownership contract
- [x] #2 Compaction-close and reopen is exercised through a valid production screen/store fixture without changing behavior to satisfy a stale mock
- [x] #3 First persistence verifies persona template state without losing existing ownership assertions; all three failures and relevant neighboring tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce the three recorded failures; trace current ADR-backed ownership and production lifecycle behavior; repair stale fixtures/assertions without weakening their intended coverage; add meaningful lifecycle/identity checks where missing; run focused files and lint; have the coordinating task update the investigation report and close this task.

ADR required: no new ADR.
ADR paths: [ADR-095](../decisions/095-conversation-owned-console-generation-settings.md), [ADR-052](../decisions/052-console-conversation-memory-and-compaction-policy.md), and [ADR-149](../decisions/149-console-persona-session-identity.md).
Reason: test-only repairs preserving accepted generation-settings, memory-lifecycle, and persona ownership contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the three stale regression tests without changing production behavior. Session-ownership assertions now include persona display names and both trusted template sources per ADR-149 while keeping those fields out of generation settings. First-persist coverage now exercises both character and persona sessions, asserting the correct trusted template and character snapshot plus unchanged system prompt, assistant identity, speech, pinned prefill, retrieval policy, and staged generation settings.

Replaced the obsolete SimpleNamespace screen/store with the production ChatScreen harness, real Console store/controller, and SQLite-backed branch memory records. The gated compaction callback proves that closing detaches the modal waiter without cancelling provider work; after completion, reopening resolves the newly persisted branch-valid memory and fresh controls. Existing unrelated context-window assertions were preserved. Minimal import/format cleanup leaves both owned test files lint-clean.

Files: Tests/Chat/test_console_session_settings.py; Tests/Chat/test_console_settings_apply_store.py. Existing ADR-095, ADR-052, and ADR-149 apply; no new ADR or production changes. Self-review preserved each original lifecycle/ownership assertion and strengthened durable-memory and persona-template evidence. This reconfirms the existing lessons-testing-evidence guidance on superseded contract assertions; no new lesson pattern was introduced.

Verification:
- Before repair, the three recorded node IDs failed (3 failed in 1.27s).
- .venv/bin/python -m pytest -q Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_settings_apply_store.py --tb=short --show-capture=no — 242 passed in 39.74s.
- .venv/bin/python -m pytest -q Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh Tests/Chat/test_console_settings_apply_store.py::test_unsaved_apply_first_persist_preserves_existing_owner_paths --tb=short --show-capture=no — 49 passed in 6.55s.
- .venv/bin/python -m ruff check Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_settings_apply_store.py — passed.
- .venv/bin/python -m ruff format --check Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_settings_apply_store.py — 2 files already formatted.
- git diff --check -- Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_settings_apply_store.py — passed.

Targeted runs only; no full sweep or commit. Runs emit the pre-existing RequestsDependencyWarning. Coordinating task owns the investigation report update.
<!-- SECTION:NOTES:END -->
