---
id: TASK-36533
title: Five test suites stub json.loads process-wide with a one-argument function
status: To Do
created_date: 2026-10-24 00:44
labels:
- follow-up
- tests
priority: low
references:
- Tests/Sync_Interop/test_sync_crypto.py
- Tests/Prompts_DB/test_prompts_db_retained_history.py
- Tests/Agents/test_run_log_service_wiring.py
- Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
- Tests/Image_Generation/test_comfyui_image_adapter.py
dependencies:
- TASK-36381
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36381's implementation and review (branch fu/h-settingsreds, 2026-10-23). `monkeypatch.setattr(module.json, "loads", stub)` replaces the stdlib function for the whole process, not just for the module under test.

TASK-36381's red came from exactly this. A probe's storage admission called `json.loads(data, object_pairs_hook=...)` and got a `TypeError` from a stub (lesson in lessons-testing-evidence.md).

These stubs accept only one positional argument, so any guarded read that runs inside their window fails the same way:
- `Tests/Sync_Interop/test_sync_crypto.py:58-61`
- `Tests/Prompts_DB/test_prompts_db_retained_history.py:433-455`
- `Tests/Agents/test_run_log_service_wiring.py:688-704`
- `Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py:157-162`
- `Tests/Image_Generation/test_comfyui_image_adapter.py:1034-1036`

They all pass on dev today; each is a trap for the next change near it. Other process-wide stubs (MCP, Backup_Recovery, vllm, chat dictionary) take `*args, **kwargs` and are not affected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 None of the five suites replaces json.loads for the whole process with a stub that refuses json.loads' keyword arguments
- [ ] #2 Each changed test still fails when the behaviour it pins is removed
<!-- AC:END -->
