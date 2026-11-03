---
id: TASK-38685
title: "Nine test stubs still replace json.loads for the whole process, and nothing stops new ones"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- tests
priority: low
references:
- Tests/Backup_Recovery/test_mcp_source_lifetimes.py
- Tests/Backup_Recovery/test_async_file_participant_lifetimes.py
- Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py
- Tests/MCP/test_execution_log.py
- Tests/Character_Chat/test_chat_dictionary_injection_cache.py
- Tests/LLM_Management/test_vllm_profiles.py
dependencies:
- TASK-36533
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36533's review, PR #3378, 2026-11-03). Checked on dev 9bbd5c2515 and at #3378's head.

TASK-36533 scopes five one-argument stubs to the module under test. Nine `monkeypatch.setattr(<module>.json or json, "loads", ...)` sites in six files remain. `<module>.json` is the stdlib module object, so each site replaces `json.loads` for every thread in the process while it is installed:
- `Tests/Backup_Recovery/test_mcp_source_lifetimes.py`: :78, :151, :177 and :544 (parse and pause stubs);
- `Tests/Backup_Recovery/test_async_file_participant_lifetimes.py`: :403 (gated);
- `Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py`: :899 (`parse_then_pause`);
- `Tests/MCP/test_execution_log.py`: :330 (counting);
- `Tests/Character_Chat/test_chat_dictionary_injection_cache.py`: :869 (counting wrapper);
- `Tests/LLM_Management/test_vllm_profiles.py`: :309 (`unexpected_parse`, which raises on any call).

These stubs accept `*args, **kwargs`, so they do not raise the `TypeError` that TASK-36533's stubs did. They still see every other `json.loads` call in the process:
- the counting stubs count unrelated parses;
- the pausing stubs can stall another thread's parse;
- the vllm stub makes any concurrent parse raise.

No failure has been traced to them yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 None of the nine sites replaces json.loads outside the module under test, and each changed test still fails when the behaviour it pins is removed
- [ ] #2 A guard test fails on any new patch of json.loads on the stdlib json module under Tests/ (shown red with a planted stub)
<!-- AC:END -->
